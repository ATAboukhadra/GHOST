import os
import glob
import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

# pip install timm==0.9.12

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "OpenGVLab/InternVL3-14B"    # change as needed 38G fails on rtxa6000
FRAME_STRIDE = 50                        # sample every 50 frames
# PROMPT = "From this sequence of images, identify the object that the hand is interacting with. Describe only that object itself, including both its likely category and the details that justify it — such as its geometry, shape, size, material, color, textures, and distinctive features. Do not mention the hand, holding, what it is placed on, or any surrounding context. Begin directly with the object description, without phrases like 'The object is' or 'It is'. Output only the object description."  
# PROMPT = "From this sequence of images, identify the object. Describe only that object itself, including details such as its geometry, shape, size, material, color, textures, and distinctive features. Begin directly with the object description, without phrases like 'The object is' or 'It is'. Output only the object description."  

PROMPT = """From this sequence of images, identify the single object being interacted with. 
Exclude hands, people, or background. 
Forbidden words: hand, finger, holding, placed, background, table.  

Produce exactly two parts, separated by a single dot ("."):  

- The first part must describe only the object’s geometry and distinctive components (e.g., base, body, proportions, knobs, spouts, handles, openings, wheels).  
- The second part must list one to five likely category names for the object, separated by commas (e.g., mug, cup, coffee machine).  

Do not include extra dots or sentences. Output only these two parts separated by one dot."""

# PROMPT = "Describe this object, including its likely category and the details that support this — such as geometry, shape, size, material, color, textures, and distinctive features. Ignore any background or context. Start directly with the object description, without phrases like 'The object is' or 'It is'. Output only the object description."
USE_MULTI_GPU = False                    # True for multi-GPU
INPUT_SIZE = 448
MAX_NUM = 12
GENERATION_CONFIG = dict(max_new_tokens=512, do_sample=False)


# -----------------------------
# Image preprocessing
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    img = image.resize((input_size, input_size))
    pixel_values = transform(img).unsqueeze(0)
    return pixel_values

import numpy as np

def load_image(image_file, input_size=448, max_num=12):
    # Open in RGBA so alpha is preserved
    image = Image.open(image_file).convert("RGBA")
    
    # Resize
    img = image.resize((input_size, input_size))
    
    # Separate alpha
    rgb, alpha = img.split()[:3], img.split()[3]
    rgb_img = Image.merge("RGB", rgb)
    
    # Transform RGB normally
    transform = build_transform(input_size=input_size)
    pixel_values = transform(rgb_img).unsqueeze(0)  # (1,3,H,W)
    
    # Convert alpha to tensor mask
    alpha_mask = torch.tensor(
        (np.array(alpha) > 0).astype("float32")
    )  # (H,W)
    alpha_mask = alpha_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    
    # Apply mask: zero out pixels where alpha == 0
    pixel_values = pixel_values * alpha_mask
    
    return pixel_values

# -----------------------------
# Multi-GPU helper
# -----------------------------
def split_model(model_name):
    """Creates a device_map for multi-GPU inference"""
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map


# -----------------------------
# Model loading
# -----------------------------
print(f"Loading model {MODEL_NAME}...")

if USE_MULTI_GPU:
    device_map = split_model(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
else:
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)


# -----------------------------
# Collect sampled frames
# -----------------------------
seq_names = [
    "arctic_s03_box_grab_01_1",
    # "arctic_s03_capsulemachine_grab_01_1",
    # "arctic_s03_espressomachine_grab_01_1",
    # "arctic_s03_notebook_grab_01_1",
    # "arctic_s03_waffleiron_grab_01_1",
    # "arctic_s03_laptop_grab_01_1",
    # "arctic_s03_microwave_grab_01_1",
    # "arctic_s03_ketchup_grab_01_1",
    # "arctic_s03_mixer_grab_01_1",

    # "arctic_s01_box_grab_01",
    # "arctic_s01_laptop_grab_01",
    # "arctic_s04_capsulemachine_grab_01",
    # "arctic_s04_espressomachine_grab_01",
    # "arctic_s05_ketchup_grab_01",
    # "arctic_s05_notebook_grab_01",
    # "arctic_s06_microwave_grab_01",
    # "arctic_s06_mixer_grab_01",
    # "arctic_s07_scissors_grab_01",
    # "arctic_s10_waffleiron_grab_01",

]

for seq_name in seq_names:
    IMAGE_DIR = f"../data/{seq_name}/images/"                   # directory with frames
    
    frame_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    frame_files = frame_files[::FRAME_STRIDE]  # every Nth frame

    print(f"Using {len(frame_files)} sampled frames...")

    pixel_values_list = []
    num_patches_list = []
    for frame_path in frame_files:
        # pv = load_image(frame_path, input_size=INPUT_SIZE, max_num=MAX_NUM)
        pv = load_image(frame_path, input_size=INPUT_SIZE, max_num=MAX_NUM)
        pixel_values_list.append(pv)
        num_patches_list.append(pv.shape[0])

    pixel_values = torch.cat(pixel_values_list)
    if not USE_MULTI_GPU:
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

    # -----------------------------
    # Build the multi-image prompt
    # -----------------------------
    # Each <image> token corresponds to one frame in sequence
    image_tokens = "".join([f"Frame{i+1}: <image>\n" for i in range(len(frame_files))])
    question = image_tokens + PROMPT

    # -----------------------------
    # Inference
    # -----------------------------
    response, history = model.chat(
        tokenizer,
        pixel_values,
        question,
        GENERATION_CONFIG,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True
    )
    print("\n=== Sequence ===")
    print(seq_name)
    # print("\n=== Prompt ===")
    # print(question)
    print("\n=== Response ===")
    print(response)
    print("\n================\n")

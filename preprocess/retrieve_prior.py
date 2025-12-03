import os
import sys
import glob
import json
import torch
import threading
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict, Counter
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import hf_hub_download
import objaverse.xl as oxl
import trimesh
from io import BytesIO
import requests
from tqdm import tqdm
import open3d as o3d
os.environ["DGL_SKIP_GRAPHBOLT"] = "1"
import openshape
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true" if you want it

# Set float precision
f32 = np.float32
half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
import re
def _slugify_name(name: str) -> str:
    # keep it short + safe for filesystems
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:60] if name else "object"

def load_clip_bigG():
    sys.clip_move_lock = threading.Lock()
    model = CLIPModel.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        torch_dtype=half,
        low_cpu_mem_usage=True,
        offload_state_dict=True
    )
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    if torch.cuda.is_available():
        with sys.clip_move_lock:
            model.cuda()
    return model, processor

def load_objaverse_index():
    meta_path = hf_hub_download("OpenShape/openshape-objaverse-embeddings", "objaverse_meta.json", repo_type='dataset')
    meta = json.load(open(meta_path))
    meta = {x['u']: x for x in meta['entries']}
    data = torch.load(hf_hub_download("OpenShape/openshape-objaverse-embeddings", "objaverse.pt", repo_type='dataset'), map_location='cpu')
    return meta, data['us'], data['feats']

def embed_images_batch(image_paths, model, processor):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return F.normalize(features, dim=-1).to(torch.float32)

def retrieve_top_k(embedding, us, feats, meta, topk=10):
    embedding = F.normalize(embedding.detach().cpu(), dim=-1).squeeze()
    sims = torch.cat([embedding @ F.normalize(chunk.float(), dim=-1).T for chunk in torch.split(feats, 10240)])
    sims, idx = torch.sort(sims, descending=True)
    results = []
    for i, sim in zip(idx, sims):
        uid = us[i]
        if uid in meta:
            results.append(dict(meta[uid], sim=sim.item()))
            if len(results) >= topk:
                break
    return results

def download_glbs(results, out_dir):
    uids = [r['u'] for r in results if 'u' in r]
    annotations = oxl.get_annotations(download_dir=out_dir)
    annotations = annotations[(annotations['source'] == 'sketchfab') & (annotations['fileType'] == 'glb')]
    pattern = '|'.join(uids)
    matched = annotations[annotations['fileIdentifier'].str.contains(pattern)]
    oxl.download_objects(objects=matched, download_dir=out_dir)

def embed_texts(texts, model, processor):
    inputs = processor(
        text=texts, return_tensors="pt",
        padding=True, truncation=True, max_length=76
    ).to(model.device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)          # (M, D) on CUDA if available
    return F.normalize(feats, dim=-1).to(torch.float32).cpu()  # <-- move to CPU

def aggregate_scores_over_prompts(text_embs, us, feats, meta, topk=10, method="rank_vote"):
    """
    text_embs: [M, D] normalized
    method:
      - 'rank_vote': each prompt contributes (topk - rank) points
      - 'max': take max similarity across prompts for each UID
      - 'mean': average similarity across prompts
    """
    uid_scores = defaultdict(float)
    uid_best_sim = {}
    for m in range(text_embs.shape[0]):
        em = text_embs[m]
        sims = torch.cat([em @ F.normalize(chunk.float(), dim=-1).T for chunk in torch.split(feats, 10240)])
        sims, idx = torch.sort(sims, descending=True)
        # record topk for rank_vote and also remember best sims
        for rnk, (i, s) in enumerate(zip(idx[:topk], sims[:topk])):
            uid = us[i]
            if uid not in meta:
                continue
            if method == "rank_vote":
                uid_scores[uid] += (topk - rnk)
            uid_best_sim[uid] = max(uid_best_sim.get(uid, float("-inf")), float(s))

    if method == "rank_vote":
        ordered = sorted(uid_scores.items(), key=lambda x: -x[1])[:topk]
    elif method in ("max", "mean"):
        # recompute across all items if needed (here we used max tracked above)
        ordered = sorted(uid_best_sim.items(), key=lambda x: -x[1])[:topk]
    else:
        ordered = sorted(uid_scores.items(), key=lambda x: -x[1])[:topk]

    results = []
    for uid, _ in ordered:
        if uid in meta:
            # store similarity if we tracked it
            entry = dict(meta[uid])
            if uid in uid_best_sim:
                entry["sim"] = uid_best_sim[uid]
            results.append(entry)
    return results


def run_text_retrieval(text_prompts, clip_model, clip_proc, meta, us, feats,
                       output_dir_base, topk=10, aggregate="rank_vote", prefix="txt_"):
    # Use a separate directory for text results
    output_dir_txt = os.path.join(os.path.dirname(output_dir_base), "openshape_text")
    # if it exists delete it
    if os.path.exists(output_dir_txt):
        import shutil
        shutil.rmtree(output_dir_txt)

    os.makedirs(output_dir_txt, exist_ok=True)

    text_prompts = [t.strip() for t in text_prompts if t.strip()]
    if not text_prompts:
        print("No non-empty text prompts provided.")
        return []

    print(f"Running text-based retrieval on {len(text_prompts)} prompt(s)…")
    text_embs = embed_texts(text_prompts, clip_model, clip_proc)  # returns CPU tensor
    top_results_txt = aggregate_scores_over_prompts(text_embs, us, feats, meta, topk=topk, method=aggregate)

    # Save preview images with names included in filename
    for i, r in enumerate(top_results_txt):
        try:
            name = r.get("name", f"Object {i}")
            stem = _slugify_name(name)
            print(f"[TEXT #{i}] {name} | sim={r.get('sim', None)}")
            img = Image.open(BytesIO(requests.get(r['img']).content))
            img = annotate_image(img, name)
            img.save(os.path.join(output_dir_txt, f"{prefix}{i:02d}_{stem}.jpg"))
        except Exception as e:
            print(f"❌ Could not download/annotate text image {i}: {e}")

    # Download GLBs and convert to OBJs (with names in filenames)
    download_glbs(top_results_txt, out_dir=output_dir_txt)
    uid_to_rank_txt = {r["u"]: i for i, r in enumerate(top_results_txt) if "u" in r}
    uid_to_name_txt = {r["u"]: r.get("name", f"obj_{i}") for i, r in enumerate(top_results_txt) if "u" in r}

    convert_glbs(
        glb_dir=output_dir_txt,
        obj_dir=output_dir_txt,
        uid_to_rank=uid_to_rank_txt,
        uid_to_name=uid_to_name_txt,
        prefix=prefix
    )

    return top_results_txt

def convert_glbs(glb_dir, obj_dir, uid_to_rank, uid_to_name=None, prefix=""):
    """
    Convert downloaded .glb files to .obj and name them by rank, optionally including object name.
    Example outputs:
      - "0.obj" (no name)
      - "txt_00_black_laptop.obj" (with name and prefix)

    uid_to_rank: dict[uid] -> rank (0..topk-1)
    uid_to_name: optional dict[uid] -> name (used in filename if provided)
    prefix: optional string prefix, e.g., "txt_", "pc_", ""
    """
    os.makedirs(obj_dir, exist_ok=True)
    uid_set = set(uid_to_rank.keys())

    for root, _, files in os.walk(glb_dir):
        for file in files:
            if not file.endswith(".glb"):
                continue
            glb_path = os.path.join(root, file)

            match_uid = None
            for uid in uid_set:
                if uid in file:
                    match_uid = uid
                    break
            if match_uid is None:
                continue

            rank = uid_to_rank[match_uid]
            if uid_to_name and match_uid in uid_to_name:
                stem = _slugify_name(uid_to_name[match_uid])
                obj_path = os.path.join(obj_dir, f"{prefix}{rank:02d}_{stem}.obj")
            else:
                obj_path = os.path.join(obj_dir, f"{prefix}{rank}.obj")

            try:
                mesh = trimesh.load(glb_path, force='scene')
                mesh.export(obj_path)
                print(f"Converted: {glb_path} → {obj_path}")
            except Exception as e:
                print(f"❌ Failed to convert {glb_path}: {e}")

def annotate_image(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))
    return img

def embed_pointcloud(ply_path, model_pc):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(colors)
    if len(points) == 0:
        raise ValueError("Empty point cloud")
    if colors.shape[0] != points.shape[0]:
        colors = np.ones_like(points)
    pc = np.concatenate([points, colors], axis=1)  # (N,6)
    pc_tensor = torch.tensor(pc[:, [0, 2, 1, 3, 4, 5]].T[None], dtype=torch.float32)
    device = next(model_pc.parameters()).device
    pc_tensor = pc_tensor.to(device)
    with torch.no_grad():
        feature = model_pc(pc_tensor)
    return F.normalize(feature, dim=-1).cpu().squeeze()

def main(seq_name, topk=10):
    preprocess_dir = 'ghost_build'
    image_dir = f"../data/{seq_name}/{preprocess_dir}/obj_rgb/"
    output_dir = f"../data/{seq_name}/{preprocess_dir}/openshape/"

    os.makedirs(output_dir, exist_ok=True)

    clip_model, clip_proc = load_clip_bigG()
    meta, us, feats = load_objaverse_index()

    # check if dir exists
    if os.path.exists(image_dir):
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        uid_scores = defaultdict(float)

        if len(image_paths) > 6:
            step = max(1, len(image_paths) // 6)
            image_paths = image_paths[::step][:6]
            print(f"Using {len(image_paths)} sampled images from {len(image_paths)} total frames.")

        for i in tqdm(range(0, len(image_paths), 1), desc="Processing images"):
            batch_paths = image_paths[i:i+1]
            embeddings = embed_images_batch(batch_paths, clip_model, clip_proc)
            for emb in embeddings:
                results = retrieve_top_k(emb, us, feats, meta, topk)
                # print(results)
                for rank, r in enumerate(results):
                    sim, obj_name = r['sim'], r.get('name', f"Object {r['u']}")
                    # print(f"[Image] #{i + rank}: {obj_name} (Score: {sim:.4f})")
                    uid_scores[r["u"]] += (topk - rank)

        top_uids_img = sorted(uid_scores.items(), key=lambda x: -x[1])[:topk]
        top_results_img = [meta[uid] for uid, _ in top_uids_img if uid in meta]
        uid_to_rank = {uid: rank for rank, (uid, _) in enumerate(top_uids_img)}

    if getattr(args, "text", None):
        top_results_txt = run_text_retrieval(
            text_prompts=args.text,
            clip_model=clip_model,
            clip_proc=clip_proc,
            meta=meta, us=us, feats=feats,
            output_dir_base=output_dir,
            topk=topk,
            aggregate=getattr(args, "text_aggregate", "rank_vote"),
            prefix="txt_"
        )

        # Compare with RGB/PC lists if you have them in this scope:
        def names(uids): return [meta[u]["name"] for u in uids if u in meta]
        try:
            rgb_uids = [r["u"] for r in top_results_img]          # from your RGB step
            # pc_uids  = [r["u"] for r in top_results_pc]           # from your PC step
            txt_uids = [r["u"] for r in top_results_txt]
            inter_rgb_txt = set(rgb_uids).intersection(txt_uids)
            # inter_pc_txt  = set(pc_uids).intersection(txt_uids)

            print("\n== Overlap with RGB retrieval ==")
            for uid in inter_rgb_txt:
                print("  •", meta[uid].get("name", uid))

        except Exception:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_name", help="Sequence name used in data/{seq_name}/gs_preprocessing/obj_rgb/")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--text", type=str, nargs="*", default=None,
                        help='Text prompt(s) for retrieval, e.g. --text "red ceramic mug" "coffee cup with handle"')
    parser.add_argument("--text_aggregate", type=str, default="rank_vote", choices=["rank_vote","max","mean"],
                        help="How to aggregate multiple prompts.")
    
    args = parser.parse_args()
    main(args.seq_name, args.topk)

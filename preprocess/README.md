# Single Sequence Preprocessing

This directory contains the preprocessing pipeline used to prepare raw video sequences for **GHOST**.  
The script `run_single_sequence.sh` processes **one sequence at a time** and performs:

- SAM object segmentation  
- Structure-from-Motion (VGG-SfM or HLOC)  
- HAMER hand reconstruction  
- SAM hand segmentation (1–2 hands)  
- Mask combination  
- Object prior retrieval (optional)  
- Point cloud alignment (optional)  
- Scale and MANO refinement  
- Hand-Gaussian animation  

---

## 🚀 Usage

Run the script from inside this folder:

```bash
bash run_single_sequence.sh [OPTIONS]
```

Example for single hand

```
bash run_single_sequence.sh \
    --seq dfki_drill_03 \
    --obj_points "+694,316" \
    --hands 1 \
    --hand_pixels "1381,805" \
    --prompt "drill" \
    --sfm hloc \
    --window 30 \
    --use_prior false
```

Example for two hands with geometric priors

```
bash run_single_sequence.sh \
    --seq arctic_s03_box_grab_01_1 \
    --obj_points "+939,1105" \
    --hands 2 \
    --hand_pixels "470,964,1152,864" \
    --prompt "box" \
    --sfm vggsfm \
    --window 300 \ # reduce this depending on your GPU
    --use_prior true
```


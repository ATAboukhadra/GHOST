

## Installation

```
bash
git clone --recursive https://github.com/ATAboukhadra/GHOST
cd GHOST

conda create -n ghost python=3.10
conda activate ghost

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
conda install -c nvidia -c conda-forge cudatoolkit cudnn cusparselt
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# remove chumpy and detectron2 from hamer requirements
sed -i 's/.*detectron2.*/# &/' submodules/hamer/setup.py
sed -i 's/.*chumpy.*/# &/' submodules/hamer/setup.py

sudo apt-get install libosmesa6-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev

pip install -r requirements.txt
pip install --no-build-isolation submodules/diff-surfel-rasterization
pip install --no-build-isolation submodules/simple-knn
```

Optional for retrieving geometric priors and object templates from OpenShape
```
conda create -n openshape python=3.10 -y
conda activate openshape
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install torch.redstone einops objaverse timm==0.9.12 transformers==4.44.0 open3d
pip install -e submodules/openshape
```

Install 2D-Gaussian-Splatting Repo

```
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
cd 2d-gaussian-splatting
conda env create --file environment.yml
conda activate surfel_splatting
```

## Models

Download HaMeR checkpoint and place it in `preprocess/`
```
cd preprocess/
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT
tar -xvf hamer_demo_data.tar.gz
cd ../
```

Download [MANO models](https://mano.is.tue.mpg.de/) and place them in `preprocess/_DATA/data/mano/`


## Preprocessing

The `preprocess` directory contains the preprocessing pipeline used to prepare raw video sequences for **GHOST**.  
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

### 🚀 Usage

Run the script from inside `preprocess/` folder:

```
cd preprocess
bash run_single_sequence.sh [OPTIONS]
```

You can use `preprocess/internvl.py` in case you want to generate a text description for the object.

Example for single hand

```
bash run_single_sequence.sh \
    --seq dfki_drill_03 \
    --obj_points "+694,316" \
    --hands 1 \
    --hand_pixels "1381,805" \
    --prompt "drill" \ # add the text description of the object here. Leave empty if you dont want any geometric priors.
    --sfm hloc \
    --window 50 \
    --use_prior false \
    --visualize True
```


Example for two hands with geometric priors

```
# reduce window depending on your GPU memory
bash run_single_sequence.sh \
    --seq arctic_s03_box_grab_01_1 \
    --obj_points "+939,1105" \
    --hands 2 \
    --hand_pixels "470,964,1152,864" \
    --prompt "box" \
    --sfm vggsfm \
    --window 300 \ 
    --use_prior true \
    --visualize True 
```

## Gaussian Splatting
### 1) Object Reconstruction

```
bash scripts/train_object.bash arctic_s03_box_grab_01_1
```

### 2) Combined Reconstruction (Single hand and two hands)

```
bash scripts/train_combined.bash arctic_s03_box_grab_01_1
```

You can visualize output point clouds on [SuperSplat](https://superspl.at/editor)

## Acknowledgements

We thank the following projects for their open-source code:
[2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting), [HOLD](https://github.com/zc-alexfan/hold), [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
and all the other listed submodules.
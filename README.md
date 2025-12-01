

## Installation

```bash
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
pip install -r requirements.txt
```

Download HaMeR Checkpoint and place it in `preprocess/`
```
cd preprocess/
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT
tar -xvf hamer_demo_data.tar.gz
cd ../
```

Download [MANO models](https://mano.is.tue.mpg.de/) and place them in `preprocess/_DATA/data/mano/`
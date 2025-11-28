

## Installation

```bash
git clone --recursive https://github.com/ATAboukhadra/GHOST
cd GHOST

conda create -n ghost python=3.10
conda activate ghost

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
conda install -c nvidia -c conda-forge cudatoolkit cudnn cusparselt
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
```


## This is a project for Lagmemo

**some details coming soon...**

## Data

please download goat episode dataset from [here](https://drive.google.com/file/d/1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe/view?usp=sharing), and put it as ***/data/datasets/goat/hm3d/...***

```bash
# scene datasets
ln -s hm3d_path /path/to/data/scene_datasets/hm3d

```

## Installation

```bash

# create conda env
conda env create -n lagmemo -f environment.yml

conda activate lagmemo

# Install the core package
python -m pip install -e src/lagmemo

# initialize submodules
git submodule update --init --recursive src/lagmemo/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet

# dection module
cd src/third_party
python -m pip install -e detectron2 # torch2.1.2+cu118 is available if get error here, and some mistake maybe caused by cpu version torch, please pay attention, refer to Problem section
cd ../..

cd src/lagmemo/perception/detection/detic/Detic/
pip install -r requirements.txt
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth --no-check-certificate

# you should run demo if env correctly
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out2.jpg --vocabulary custom --custom_vocabulary headphone,webcam,paper,coffe --confidence-threshold 0.3 --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

mkdir -p data/checkpoints
cd data/checkpoints
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip
unzip ovmm_baseline_home_robot_challenge_2023.zip
cd LAGMEMO_ROOT # return to repo's root, should be changed to your own path

# simulation environment
conda env update -f src/environment.yml

git submodule update --init --recursive src/third_party/habitat-lab
python -m pip install -e src/third_party/habitat-lab/habitat-lab
python -m pip install -e src/third_party/habitat-lab/habitat-baselines
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git" # this is not neccessary if you have pytorch3d in your pip list

# switch to goat branch
cd src/third_party/habitat-lab
git checkout home-robot_goat_support
pip install -e habitat-lab
pip install -e habitat-baselines
cd ../../..

# really to start!
python project/habitat_lagmemo/eval_episode.py

```

## Problem

When having problem with installing detectron2:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.2 -c pytorch # insatll torch2.5.1
python -m pip install -e detectron2
conda uninstall libtorch # downgrade torch version
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu18 # install torch2.1.2 cu118 which is confirmed to be correct

```
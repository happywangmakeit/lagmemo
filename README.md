```相应的mamba命令直接替换为conda命令即可，最好不要用mamba```

## Installation

### 1、按照[home-robot readme](README_home_robot.md)配置环境，注意不是实机部署的步骤

step3：可能会漏掉一些submodule，可以检查一下，或者等报错的时候再下载，参考下面的p4

**step5(关键)** : 安装detectron2，注意此时安装的pyorch版本为1.13.1，可能会出现cuda版本不合适的情况，如A100机器，cuda版本为10.1，通过对cuda及pytorch版本同时升级可解决问题
    pytorch 2.1.2，cudatoolkit118可行。当然最好的解决方法还是升级cuda版本，这样基本不会遇到bug。ps：从源安装对应于10.1的detectron2会遇到一些奇怪的诸如import error之类的错误，不推荐
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.2 -c pytorch
python -m pip install -e detecron2
# 尝试运行demo，demo运行遇到问题参考下面
# 若显示cuda版本不对应
conda uninstall libtorch
# 继续运行demo，此时应该可以了
```

    step5：demo运行，/home/wxl/lagmemo/home-robot/src/third_party/detectron2/detectron2/data/transforms/transform.py中的
                        def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):其中Image.LINEAR改为Image.BILINEAR

### 2、按照[home-robot sim readme](src/home_robot_sim/README.md)配置环境

    step3：python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"可能会报错，可以查看pip list，只要有pytorch3d即可

### 3、推荐先配置[objectnav](projects/habitat_objectnav/README.md)感受一下(可跳过)

    在装好habitat lab后会将numpy版本进行一个升级导致依赖出现问题，推荐再次打一遍pip install -e src/home_robot_sim进行降级
    其他依据readme

### 4、配置goat

#### step1:

下载[goat](projects/habitat_goat/README.md)数据包，另外场景数据文件架构为 ```scene_datasets/hm3d/...```

#### step2: 

```bash
cd src/third_party/habitat-lab
git checkout home-robot_goat_support
pip install -e habitat-lab
pip install -e habitat-baselines
cd ../../..
pip install -e src/home_robot_sim
```

#### step3:

修改config，数据读取等代码（clone本仓库的话不需要此步骤）


### problem
    
```p1```: ModuleNotFoundError: No module named 'sophus' : instead of "import sophus as sp", use "import sophuspy as sp"

```p2```: 最终应使用numpy-1.23.5，可通过pip install -e src/home_robot_sim调整相应包

```p3```: File "/home/wxl/lagmemo/home-robot/src/third_party/habitat-lab/habitat-lab/habitat/utils/gym_definitions.py", line 99, in <module>
    if "Habitat-v0" not in registry.env_specs:
AttributeError: 'dict' object has no attribute 'env_specs'
修改相应条目为registry.keys()

```p4```: ModuleNotFoundError: No module named 'home_robot.agent.imagenav_agent.SuperGluePretrainedNetwork.models'
手动下载相应包，如git submodule update --init --recursive src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork
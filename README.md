# MTDG
Implementation of Multi Source Domain Generalization Task


### Installation

GPU Requirement: > 1 x NVIDIA GeForce RTX 4090.

The code has been tested with 
 - Python 3.8, CUDA 11.6, Pytorch 1.13.0, TorchSparse 2.0.0b0
 - IMPORTANT: This code base is not compatible with TorchSparse 2.1.0.

### Prerequisites
Ensure you have the following specifications:
- PyTorch 1.13.0
- CUDA 11.6
- Python 3.8
- TorchSparse 2.0.0b

### Installation Steps

1. **Setting up a Conda Environment**:  
   We recommend establishing a new conda environment for this installation.
```
$ conda create -n mtdg python=3.8
$ conda activate mtdg
```
2. **Installing PyTorch**:  
Install PyTorch, TorchVision with specific CUDA support.
```
$ pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
3. **Additional Dependencies**:  
Install additional utilities and dependencies.
```
$ pip install tqdm

$ sudo apt-get update
$ sudo apt-get install libsparsehash-dev

$ conda install backports
```
4. **Installing TorchSparse**:  
Update and install TorchSparse from its GitHub repository.
```
$ pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

#### Pip/Venv/Conda
In your virtual environment follow [TorchSparse](https://github.com/mit-han-lab/spvnas). This will install all the base packages.


### Data preparation

#### SynLiDAR
Download SynLiDAR dataset from [here](https://github.com/xiaoaoran/SynLiDAR), then prepare data folders as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    └──sequences/
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        └── 12/
```

#### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
```

#### SemanticPOSS
To download SemanticPOSS follow the instructions [here](http://www.poss.pku.edu.cn/semanticposs.html). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      ├── read_data.py
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            │   ├── tags/ 
            |   |      ├── 000000.tag
            |   |      ├── 000001.tag
            |   |      └── ...
            |   ├── calib.txt
            |   └── poses.txt
            └── 05/
```

#### SemanticSTF dataset
Download SemanticSTF dataset from [GoogleDrive](https://forms.gle/oBAkVJeFKNjpYgDA9), [BaiduYun](https://pan.baidu.com/s/10QqPZuzPclURZ6Niv1ch1g)(code: 6haz). Data folders are as follows:
The data should be organized in the following format:
```
/SemanticSTF/
  └── train/
    └── velodyne
      └── 000000.bin
      ├── 000001.bin
      ...
    └── labels
      └── 000000.label
      ├── 000001.label
      ...
  └── val/
      ...
  └── test/
      ...
  ...
  └── semanticstf.yaml
```
The author provides class annotations in 'semanticstf.yaml'

- Don't forget revise the data root dir in  `configs/kitti2stf/default.yaml` and `configs/synlidar2stf/default.yaml`

  
### Training

For SemanticKITTI + SynLiDAR + SemanticPOSS ->SemanticSTF, run:
```
python train.py configs/kitti_syn_poss2stf/minkunet/cr0p5.yaml
```

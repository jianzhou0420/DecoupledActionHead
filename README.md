# Decoupled Action Head  
*Confining Task Knowledge to Conditioning Layers*

[[Project Page]]() [[Paper]]() [[Data]]() 

Jian Zhou<sup>1</sup>, 
Sihao Lin<sup>1</sup>, 
Shuai Fu<sup>1</sup>, 
Qi Wu<sup>1</sup>

<sup>1</sup>University of Adelaide &nbsp;&nbsp;

## 1. Quick Start

(0). Please make sure you have mamba package manager to get faster dependency solving.
You may use conda to install it or use [miniforge](https://github.com/conda-forge/miniforge) instead.
```console
$ conda install -n base -c conda-forge mamba
```

(1) Prepare Environment
```console
$ bash env_install.sh
```

(2) Prepare Data
```console
$ bash prepare_data.sh
```

(3) Run
```console
$ bash scripts/train/icra/DP_C/Exp_DP_C_Normal.sh A 0 logging.mode=offline
```


## 2. Folder Tree
```
jiandecouple/
├── data/                          # Datasets & Outputs
│   ├── outputs/
│   └── robomimic/  
│       ├── datasets/
│       ├── datasets_abs/
│       └── zarr/
├── jiandecouple/                  # Main source code package
│   ├── codecs/
│   ├── common/
│   ├── config/
│   ├── dataset/
│   ├── env/
│   ├── env_runner/
│   ├── gym_util/
│   ├── model/
│   ├── policy/
│   ├── scripts/
│   ├── shared_memory/
│   └── z_utils/
├── scripts/                       # Scripts for training and other tools
│   ├── config/
│   └── train/
│       └── icra/
│           ├── DP_C/
│           ├── DP_MLP/
│           ├── DP_MLP_2/
│           ├── DP_T/
│           └── DP_T_FILM/
└── third_party/
│    ├── mimicgen/
│    ├── robomimic/
│    └── robosuite/
└── trainer_pl_all.py             # Trainer that uses Pytorch-Lightning 

```




## Acknowledgments

We acknowledge and give credit to the following works that support our experiments and datasets:

**Experimental Setup**  
Our experimental setup builds upon [EquiDiff](https://github.com/pointW/equidiff) and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

**Datasets & Simulator**  
Our dataset construction leverages [MimicGen](https://mimicgen.github.io/), as well as the underlying [robomimic](https://github.com/ARISE-Initiative/robomimic) and [robosuite](https://github.com/ARISE-Initiative/robosuite)

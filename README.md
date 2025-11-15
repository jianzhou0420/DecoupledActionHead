# Decoupled Action Head  
*Confining Task Knowledge to Conditioning Layers*

[[Project Page]]() [[Paper]]() [[Data]]() 

Jian Zhou<sup>1</sup>, Sihao Lin<sup>1</sup>, Shuai Fu<sup>1</sup>, Qi Wu<sup>1</sup>

<sup>1</sup>University of Adelaide &nbsp;&nbsp;

## 1.Quick Start

(0). Please make sure you have mamba package manager to get faster dependency solving.
You may use conda to install it or use [miniforge](https://github.com/conda-forge/miniforge) instead.
`conda install -n base -c conda-forge mamba` 


(1) Prepare Environment
`bash env_install.sh`

(2) Prepare Data
`bash prepare_data.sh`

(3) Run
`bash scripts/train/icra/DP_C/Exp_DP_C_Normal.sh A 0 logging.mode=offline`


## 2. File Order





## Credits
1. Diffusion Policy
2. mimicgen
3. equidiff

## Cite

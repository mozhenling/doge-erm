# Domain Generalization of ERM

The repository contains the experiment implementations of the paper entitled
_" When Empirical Risk Minimization Fails or Succeeds at Domain
Generalization in Causal View"_, which has been under peer review
in the **IEEE Transactions on Neural Networks and Learning Systems**.

The repository is mainly modified from the 
[DomainBed](https://github.com/facebookresearch/DomainBed). 
Therefore, the usage is similar and the DomainBed takes the
main credits for the code designs. Our contributions have
been clarified in the paper. 

## Data Settings
**DS 1**: 'BaseMNIST', 'BaseFashion', 'BaseCIFAR10'  
**DS 2**: 'ColoredNoiseMNIST', 'ColoredNoiseFashion', 'ColoredNoiseCIFAR10'  
**DS 3**: 'NoiseColoredMNIST', 'NoiseColoredFashion', 'NoiseColoredCIFAR10'  
**DS 4**: 'MNISTColoredNoise', 'FashionColoredNoise', 'CIFAR10ColoredNoise'  

## Visualization Results (Fig.4 & Fig. 5)
Download the trained models, place the `train_outs` folder 
under `.\outputs`, use `eg_visual_AB.sh` to obtain the same
results presented in the paper.

Alternatively, install the same packages specified in `requirements_visual.txt`
use `eg_train_VAE.sh` to retrain the models, and use 
`eg_visual_AB.sh` to obtain the similar results.

Outputs are located at `.\outputs\InOutComp_A` (train & the test envs) 
and `.\outputs\InOutComp_B` (the test & extra test envs)

**Trained models**:
+ [Google Drive](https://drive.google.com/file/d/1yPkuCi7dr4PZoAQBmQdKDXL0zN8aM3C_/view?usp=sharing)
+ or, [Baidu Netdisk](https://pan.baidu.com/s/1K_XB9EwVVrRjj9nryyrmbQ?pwd=pxt9 )
& Password: pxt9

## Domain Generalization Results (TABLE I & TABLE II)
Download the sweep records, place them under `'.\outputs\sweep_outs'`,
and use `eg_results.sh` to obtain the same results reported in the paper.   

Alternatively, install the same packages
specified in `requirements_numerical.txt`, follow the guidance of 
`eg_sweep.sh` to run all sweeps, and use `eg_results.sh` to
obtain the similar results.

**1) MNIST Records**:   
+ [Google Drive](https://drive.google.com/file/d/1V6xuP210PSgBtNMu7Jwh86kFVA-nPSG8/view?usp=sharing)  
+ or, [Baidu Netdisk](https://pan.baidu.com/s/1AGVbNHtz2_bZvEDXy-F0rA?pwd=vbb7)
& Password：vbb7

**2) Fashion MNIST Records**: 
+ [Google Drive](https://drive.google.com/file/d/1MGlA8gblGNGpAMguQejPYNFX8FmPnA5i/view?usp=sharing)  
+ or, [Baidu Netdisk](https://pan.baidu.com/s/1ZxiyYC8V8VG-x4Ew2vufSA?pwd=1fh5)
& Password：1fh5

**3) CIFAR10 Records**: 
+ [Google Drive](https://drive.google.com/file/d/1gDBklIvbB8BG5iHQ2BIqqYHtaDz1KkMM/view?usp=sharing)  
+ or, [Baidu Netdisk](https://pan.baidu.com/s/1XNsuc7mjTpsL9dorZS74vA?pwd=4h3j)
& Password：4h3j

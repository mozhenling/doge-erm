# DG Study of ERM

The repository is developed for the paper entitled
_"Domain Generalization Study of Empirical Risk Minimization from Causal
Perspectives"_, which has been accepted for publication in **IEEE Transactions on Multimedia**.

The  <font color=red>Supplementary Materials </font> of the paper
are placed in `.\additional readings`.

## Simulated Visualization Experiments for Theory Verification
(Fig.5 of the paper)

Download the trained models, place the `train_outs` folder 
under `.\outputs`, use `eg_visual_AB.sh` to obtain the same
results presented in the paper.

Alternatively, install the same packages specified in `requirements_visual.txt`
use `eg_train_VAE.sh` to retrain the models, and use 
`eg_visual_AB.sh` to obtain the similar results.

Outputs are located at `.\outputs\InOutComp_A` (train & the test envs) 
and `.\outputs\InOutComp_B` (the test & extra test envs)

**Trained models**:
+ [Google Drive](https://drive.google.com/file/d/1vnIjwPJu6UeSXa69SVJtpTxr7TKWcHNX/view?usp=sharing)
+ or, [Baidu Netdisk](https://pan.baidu.com/s/1CndW1WNe1p2i8P4RZusrbA?pwd=s8tu)
& Password: s8tu

## Simulated Numerical Experiments for Theory Verification
(TABLE I of the paper & TABLE VI of the Supplementary Materials)

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

## Experiments on Real-World Datasets for Theory Verification
(TABLE II of the paper)

The codes, data, sweep records of the experiments based on 
the real-world datasets for theory verification are available at :

+ [Google Drive](https://drive.google.com/file/d/1zTEM5_BGZo-FXncEkxE7baOKAQYj1ir_/view?usp=sharing)  
+ or, [Baidu Netdisk](https://pan.baidu.com/s/1ujjqYQaIV7s1P5757Ut7wQ?pwd=fq24)
& Password：fq24

You can employ `eg_results.sh` to collect results reported in the paper. 
Alternatively, you can install the same packages given in `requirements_practical.txt`
, delete old records `./outputs/sweep_outs`, employ ` eg_sweep.sh` to rerun the
experiments, and use `eg_results.sh` to obtain the similar results.

## Experiments on Real-World Datasets for FISIP Verification
(TABLE III of the paper)
The codes, data, sweep records of the experiments based on 
the real-world datasets for FISIP verification are available
in the following links:

+ codes (ERM & FISIP):\
[Google Drive](https://drive.google.com/file/d/1TsU5P7fOHCmP7I0TwdKtzmKRWpx8ZHOI/view?usp=sharing)\
or,
[Baidu Netdisk](https://pan.baidu.com/s/1VkStvcOAA4B1uw4nXcMSxA?pwd=24zy) 
Password: 24zy

+ datasets:\
[Google Drive](https://drive.google.com/file/d/1zHB-UuWONpQrR93WhDk9w0D-pDGG5SRS/view?usp=sharing)\
or,
[Baidu Netdisk](https://pan.baidu.com/s/17QsQnmwrdexupUUP8cfkUQ?pwd=qnbe) 
Password: qnbe

+ sweep records (ERM & FISIP) \
[Google Drive](https://drive.google.com/file/d/1t5ObDhjfXBjrOm1-J0jJzbBy8DpPAAGX/view?usp=sharing)\
or,
[Baidu Netdisk](https://pan.baidu.com/s/1GWtP1Wk6pxxvKRB8SLbTWA?pwd=kmph)
Password: kmph

+ dropout experiemnts\
[Google Drive](https://drive.google.com/file/d/1gHpb-Uvijfj_hPmuPD8ob68cNZ47FoMh/view?usp=sharing)\
or,
[Baidu Netdisk](https://pan.baidu.com/s/1cFmJEpdA0g3hBhMC6M8qaw?pwd=4wgx)
Password: 4wgx

The sweep records show the results presented in the paper. For retraining,
please download codes and the datasets. Place the datafolder, say, 
`Prostate` under `./datasets` of the codes. Then, use `eg_sweep.sh` to run one example of random search and use `eg_result.sh` to collect the results. Note that `FISIP` is for 
instance wise classification, while `FISIPseg` is for element wise classification.


## Acknowledgement
The repository is developed mainly based on the 
[DomainBed](https://github.com/facebookresearch/DomainBed). 
Therefore, the usage is similar and the DomainBed also takes
credits for the code designs. We would like to thank the authors
of the Domainbed for sharing the codes. We also want the thank all
the authors of public datasets used in this study. Our contributions have
been clarified in the paper. 
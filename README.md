# 고도화된 보이스피싱 탐지를 위한 deepvoice audio detection

> 세종대학교 2024 1학기 캡스톤디자인A 13조 프로젝트 코드용 repository입니다. 

## 👨‍👨‍👧‍👧 참여자
세종대학교 전자정보통신공학과

[임동훈](https://github.com/ldh-Hoon), [이준호](https://github.com/Lanvizu), [문한송](https://github.com/MHANSONG), [김진환](https://github.com/kjhwan0115), 윤나경(추가 필요)

## 📆 프로젝트 기간 

2024년 3월 ~ 2024년 6월

![image](https://github.com/ldh-Hoon/ko_deepfake-whisper-features/assets/139981434/85611a01-1d06-4cd1-99ce-12980923763f)

## 🔎 개요
### | 한국어 중심의 audio anti-spoofing 데이터셋 구축 시도
국내에서도 다양한 딥보이스 피싱 [사례](https://imnews.imbc.com/replay/2024/nwtoday/article/6598469_36523.html)
가 나타나며 새로운 방식의 범죄가 현실화되고 있다.

하지만 대부분의 공개 데이터는 영어, 중국어 등 외국어에 한정되어 있다. 

따라서 AIHUB 및 공개된 TTS 음성 수집 및 오픈소스 TTS 시스템을 활용해 데이터셋을 구축해보고자 했다.

최종적으로 가짜 오디오 데이터셋인 KoAAD(Korean Audio Anti-spoofing Dataset)를 수집 및 생성했다.

real audio는 [AIHUB 감성 및 발화스타일 동시고려 음성합성 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71349)을 사용한다.

해당 데이터셋의 음성 전사 label text를 사용해 합성된 음성을 생성한다.

>Google Cloud TTS
 
>Melo TTS

>XTTS2

>Audio Pub 

>네이버 기사 음성 요약

>클로바더빙

최종적으로 위 6개의 시스템에 대해 약 10GB, 3만개 내외의 음성 파일을 수집/생성할 수 있었다.

### | 실제 통화와 유사한 환경에서의 탐지 데모 구현 및 테스트
음성만 존재하는 데이터셋 오디오 파일들과 달리, 실제 대부분의 오디오 잡음 및 주변 환경음이 많이 포함되어 있다. 

또한 통화는 최소 두명의 화자가 대화하게 되므로 성능에 영향이 있을 수 있다. 

따라서 데이터 증강을 통한 학습, 통화녹음 데이터 사용, 화자 분리 등을 적용하여 실제 통화에 대한 탐지 데모 시스템을 구현하고자 시도했다.

원본 repository의 whisper-LCNN을 baseline 모델로 활용했다.

## demo 영상

[유튜브](https://youtu.be/BfzrxwfoHds)

## 기타 자료

[KoAAD 및 Colab 코드 자료](https://drive.google.com/drive/folders/1RihDtMWUc5sPg5fNqJ_L015U6HMgwF2u?usp=drive_link)
> 아직 정리가 잘 안 되어있습니다.

train Colab: <a href="https://colab.research.google.com/drive/1RHPZg6mdu_0X6-DfjZDKjZHUu5LJWElN?usp=sharing"><img src="https://img.shields.io/badge/open in Colab-F9AB00?style=flat&logo=Google Colab&logoColor=white" /></a>

demo Colab: <a href="https://colab.research.google.com/drive/1Xph13KuqHoydh7Blj7qN-ccuROHHmi2d?usp=sharing"><img src="https://img.shields.io/badge/open in Colab-F9AB00?style=flat&logo=Google Colab&logoColor=white" /></a>

다국어(한국어 미포함) fake audio 데이터셋인 [MLAAD](https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP) 와 real audio 데이터셋인 [MAILABS](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/) 를 함께 사용하면 좋다.



# 참고 논문

Kawa, Piotr, et al. "Improved DeepFake Detection Using Whisper Features." arXiv preprint arXiv:2306.01428 (2023).
[paper](https://arxiv.org/abs/2306.01428)

Müller, Nicolas M., et al. "MLAAD: The Multi-Language Audio Anti-Spoofing Dataset." arXiv preprint arXiv:2401.09512 (2024).
[paper](https://arxiv.org/abs/2401.09512)




# 아래는 기존 repository의 README.md입니다.





# Improved DeepFake Detection Using Whisper Features

The following repository contains code for our paper called "Improved DeepFake Detection Using Whisper Features".

The paper is available [here](https://www.isca-speech.org/archive/interspeech_2023/kawa23b_interspeech.html).


## Before you start

### Whisper
To download Whisper encoder used in training run `download_whisper.py`.

### Datasets

Download appropriate datasets:
* [ASVspoof2021 DF subset](https://zenodo.org/record/4835108) (**Please note:** we use this [keys&metadata file](https://www.asvspoof.org/resources/DF-keys-stage-1.tar.gz), directory structure is explained [here](https://github.com/piotrkawa/deepfake-whisper-features/issues/7#issuecomment-1830109945)),
* [In-The-Wild dataset](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild).



### Dependencies
Install required dependencies using (we assume you're using conda and the target env is active):
```bash
bash install.sh
```

List of requirements:
```
python=3.8
pytorch==1.11.0
torchaudio==0.11
asteroid-filterbanks==0.4.0
librosa==0.9.2
openai whisper (git+https://github.com/openai/whisper.git@7858aa9c08d98f75575035ecd6481f462d66ca27)
```

### Supported models

The following list concerns models and its names to select it supported by this repository:
* SpecRNet - `specrnet`,
* (Whisper) SpecRNet - `whisper_specrnet`,
* (Whisper + LFCC/MFCC) SpecRNet - `whisper_frontend_specrnet`,
* LCNN - `lcnn`,
* (Whisper) LCNN - `whisper_lcnn`,
* (Whisper + LFCC/MFCC) LCNN -`whisper_frontend_lcnn`,
* MesoNet - `mesonet`,
* (Whisper) MesoNet - `whisper_mesonet`,
* (Whisper + LFCC/MFCC) MesoNet - `whisper_frontend_mesonet`,
* RawNet3 - `rawnet3`.

To select appropriate front-end please specify it in the config file.

### Pretrained models

All models reported in paper are available [here](https://drive.google.com/drive/folders/1YWMC64MW4HjGUX1fnBaMkMIGgAJde9Ch?usp=sharing).

### Configs

Both training and evaluation scripts are configured with the use of CLI and `.yaml` configuration files.
e.g.:
```yaml
data:
  seed: 42

checkpoint: 
  path: "trained_models/lcnn/ckpt.pth",

model:
  name: "lcnn"
  parameters:
    input_channels: 1
    frontend_algorithm: ["lfcc"]
  optimizer:
    lr: 0.0001
    weight_decay: 0.0001
```

Other example configs are available under `configs/training/`.

## Full train and test pipeline 

To perform full pipeline of training and testing please use `train_and_test.py` script.

```
usage: train_and_test.py [-h] [--asv_path ASV_PATH] [--in_the_wild_path IN_THE_WILD_PATH] [--config CONFIG] [--train_amount TRAIN_AMOUNT] [--test_amount TEST_AMOUNT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--ckpt CKPT] [--cpu]

Arguments: 
    --asv_path          Path to the ASVSpoof2021 DF root dir
    --in_the_wild_path  Path to the In-The-Wild root dir
    --config            Path to the config file
    --train_amount      Number of samples to train on (default: 100000)
    --valid_amount      Number of samples to validate on (default: 25000)
    --test_amount       Number of samples to test on (default: None - all)
    --batch_size        Batch size (default: 8)
    --epochs            Number of epochs (default: 10)
    --ckpt              Path to saved models (default: 'trained_models')
    --cpu               Force using CPU
```

e.g.:
```bash
python train_and_test.py --asv_path ../datasets/deep_fakes/ASVspoof2021/DF --in_the_wild_path ../datasets/release_in_the_wild --config configs/training/whisper_specrnet.yaml --batch_size 8 --epochs 10 --train_amount 100000 --valid_amount 25000
```


## Finetune and test pipeline

To perform finetuning as presented in paper please use `train_and_test.py` script.

e.g.:
```
python train_and_test.py --asv_path ../datasets/deep_fakes/ASVspoof2021/DF --in_the_wild_path ../datasets/release_in_the_wild --config configs/finetuning/whisper_specrnet.yaml --batch_size 8 --epochs 5  --train_amount 100000 --valid_amount 25000
```
Please remember about decreasing the learning rate!


## Other scripts

To use separate scripts for training and evaluation please refer to respectively `train_models.py` and `evaluate_models.py`.


## Acknowledgments

We base our codebase on [Attack Agnostic Dataset repo](https://github.com/piotrkawa/attack-agnostic-dataset).
Apart from the dependencies mentioned in Attack Agnostic Dataset repository we also include: 
* [RawNet3 implementation](https://github.com/Jungjee/RawNet).



## Citation

If you use this code in your research please use the following citation:
```
@inproceedings{kawa23b_interspeech,
  author={Piotr Kawa and Marcin Plata and Michał Czuba and Piotr Szymański and Piotr Syga},
  title={{Improved DeepFake Detection Using Whisper Features}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={4009--4013},
  doi={10.21437/Interspeech.2023-1537}
}
```

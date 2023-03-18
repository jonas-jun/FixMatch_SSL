# ImageClassification with FixMatch(Semi-Supervised Learning)
- Code with pytorch-lightning

## Method
- FixMatch without EDA model

## Dataset (like torchvision.ImageFolderDataset)
- {TRAIN_DIR}
    - class 0
        - image 0-0
        - image 0-1
        - ...
    - class 1
        - image 1-0
        - image 1-1
        - ...
    - class 2
        - image 2-0
        - ...
- {VALID_DIR}
    - class 0
        - ...
    - ...
- {TEST_DIR}
    - class 0
        - ...
    - ...
- {UNLABEL_DIR}
    - image u-0
    - image u-1
    - image u-2
    - ...

## Model
- ConvNext (timm)
- PIT (Parameter-free layer)
- ResNet (Parameter-free layer)

## references
- [fixmatch-arxiv](https://arxiv.org/abs/2001.07685)
- [fixmatch-github official](https://github.com/google-research/fixmatch)
- [parameter-free layer by NaverAI](https://github.com/naver-ai/PfLayer)



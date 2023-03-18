# ImageClassification with FixMatch(Semi-Supervised Learning)
- Code with pytorch-lightning

## Method
- FixMatch without EDA model

## Dataset (like torchvision.ImageFolderDataset)
```
📦dataset
 ┣ 📂train
 ┃ ┣ 📂cls1
 ┃ ┣ 📂cls2
 ┃ ┗ 📂cls3
 ┣ 📂valid
 ┃ ┗ 📂cls1
 ┃ ┗ 📂cls2
 ┣ 📂test
 ┃ ┗ 📂cls1
 ┃ ┗ 📂cls2
 ┗ 📂unlabel
```
## Model
- ConvNext (timm)
- PIT (Parameter-free layer)
- ResNet (Parameter-free layer)

## references
- [fixmatch-arxiv](https://arxiv.org/abs/2001.07685)
- [fixmatch-github official](https://github.com/google-research/fixmatch)
- [parameter-free layer by NaverAI](https://github.com/naver-ai/PfLayer)



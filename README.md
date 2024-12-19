# TDF-Net
This repository is an official implementation of the paper "TDF-Net: Trusted Dynamic 
Feature Fusion Network for Breast Cancer Diagnosis using Incomplete Multimodal Ultrasound."

Pengfei Yan, Wushuang Gong, Minglei Li, Jiusi Zhang, Xiang Li, Yuchen Jiang, Hao Luo, and Hang Zhou. (2024). 
TDF-Net: Trusted Dynamic Feature Fusion Network for breast cancer diagnosis using incomplete multimodal ultrasound. Information Fusion, 102592.

## Dataset
We collected a multimodal ultrasound image dataset for classification in breast cancer, 
including 145 benign and 103 malignant cases. This dataset is available for only non-commercial 
use in research or educational purposes. As long as you use the dataset for these purposes, you can
edit or process images in the dataset. 
https://www.kaggle.com/datasets/timesxy/multimodal-breast-ultrasound-dataset-us3m

## Code Usage

## Installation

### Requirements

* Linux, CUDA>=11.3, GCC>=7.5.0
  
* Python>=3.8

* PyTorch>=1.11.0, torchvision>=0.12.0 (following instructions [here](https://pytorch.org/))

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
  
### Dataset preparation

Please organize the dataset as follows:

```
code_root/
└── 001/
      ├── BUS_1.jpg
      ├── DUS_1.jpg
      └── EUS_1.jpg
```

### Training

For example, the command for the training TDF-Net is as follows:

```bash
python model_train.py
```
The configs in model_train.py or other files can be changed.

### Evaluation

After obtaining the trained TDF-Net, then run the following command to evaluate it on the validation set:

```bash
python model_valid.py
```

## Notes
The code of this repository is built on
https://github.com/TimesXY/TDF_Net.

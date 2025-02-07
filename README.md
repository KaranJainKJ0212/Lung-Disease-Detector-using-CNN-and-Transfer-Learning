# Lung Disease Detection using Deep Learning

## Overview

This repository contains the implementation of deep learning models for lung disease classification using chest X-ray images. The study evaluates the classification performance of multiple deep learning models on the Lung Disease Dataset, highlighting the superior accuracy of **InceptionV3** and **Xception**. The models are trained to classify lung conditions into three categories: **Normal**, **Lung Opacity**, and **Viral Pneumonia**.

## Dataset

The dataset used in this study can be accessed here: [Lung Disease Dataset on Kaggle](https://www.kaggle.com/datasets/karanjain21/lung-disease-datasets).


The dataset used in this study consists of **3,475** X-ray images categorized into three classes:

- **Normal (1250 images)** - Healthy lung conditions.
- **Lung Opacity (1125 images)** - Lungs exhibiting abnormalities such as infections or fluid accumulation.
- **Viral Pneumonia (1100 images)** - Lungs affected by viral pneumonia.

Data augmentation techniques, including **Gaussian blur** and **Salt and Pepper noise**, were applied to increase dataset variation and improve model generalization.

## Methodology

The deep learning models were implemented using **Convolutional Neural Networks (CNNs)** with additional enhancements, including:

- **Residual connections** for better gradient flow.
- **Inception modules** to capture features at different scales.
- **Multilayer attention mechanism** to focus on key regions in X-ray images.
- **Fully connected classification layer** with three output nodes for classification.

The models were trained using **categorical cross-entropy loss** and optimized using **Stochastic Gradient Descent (SGD)** with momentum.

## Results

The classification performance of various deep learning architectures was evaluated using accuracy, precision, recall, and F1-score. The best-performing models are:

| Model           | Accuracy | Precision | Recall  | F1-Score |
| --------------- | -------- | --------- | ------- | -------- |
| VGG16           | 86%      | 86%       | 87%     | 86%      |
| VGG19           | 84%      | 85%       | 85%     | 85%      |
| DenseNet121     | 94%      | 95%       | 95%     | 95%      |
| ResNet50        | 36%      | 45%       | 33%     | 18%      |
| **InceptionV3** | **96%**  | **96%**   | **96%** | **96%**  |
| **Xception**    | **96%**  | **96%**   | **96%** | **96%**  |

Our evaluation of deep learning models for lung disease classification shows that **InceptionV3 and Xception achieve the highest accuracy (96%)**, outperforming **VGG16, VGG19, ResNet50, and DenseNet121** in terms of precision, recall, and F1-score across all classes.

## Conclusion

The study demonstrates that **InceptionV3 and Xception** outperform other architectures, achieving **96% accuracy** in lung disease classification. The results suggest that deep learning models can effectively classify chest X-ray images and assist in early disease detection. Future work may focus on further refining models, incorporating larger datasets, and integrating additional advanced techniques for improved robustness.

## Installation & Usage

To replicate the results and train the models, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Lung-Disease-Detector-using-CNN-and-Transfer-Learning.git
   cd Lung-Disease-Detector-using-CNN-and-Transfer-Learning


## Citation

My research paper:

@article{your_paper,
  title={Deep Learning Approach for Lung Disease Detection in X-Ray Images},
  author={Your Name and Others},
  journal={Your Journal},
  year={2025}
}

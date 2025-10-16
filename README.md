# Brain Tumor Classification using Transfer Learning with VGG16

**Ayesha Saleem** | Computer Science Undergraduate

[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebook-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/code/ayeshasal89/brain-tumor-classification-mri-vgg16-97-acc/notebook)
[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-success?style=flat)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

## Abstract

This project implements an automated brain tumor classification system using deep learning and transfer learning techniques. Using the VGG16 convolutional neural network pre-trained on ImageNet, the model classifies brain MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor. The system achieves 97% accuracy on the test set, demonstrating the effectiveness of transfer learning for medical image analysis with limited datasets. This work showcases practical applications of deep learning in healthcare and computer-aided diagnosis systems.

**Keywords**: Deep Learning, Transfer Learning, VGG16, Medical Image Classification, Brain Tumor Detection, Computer Vision

## Table of Contents

- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)

## Motivation

Brain tumors are among the most life-threatening diseases, and early detection is crucial for effective treatment. Manual analysis of MRI scans is time-consuming and requires specialized expertise. This project explores how artificial intelligence, specifically deep learning, can assist medical professionals in making faster and more accurate diagnoses.

As a computer science student interested in AI applications in healthcare, I wanted to understand:
- How transfer learning can be applied to real-world problems with limited data
- The practical challenges of working with medical imaging datasets
- How to build and evaluate deep learning models for classification tasks
- The potential and limitations of AI in healthcare applications

## Problem Statement

**Goal**: Develop an automated system that can accurately classify brain MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor.

**Challenges**:
1. Limited availability of labeled medical imaging data
2. High computational requirements for training deep neural networks from scratch
3. Need for high accuracy to be clinically useful
4. Class imbalance in medical datasets
5. Ensuring model generalization to unseen data

**Solution**: Implement transfer learning using VGG16 architecture, leveraging pre-trained weights from ImageNet and fine-tuning for brain tumor classification.

## Dataset

**Source**: Brain MRI Images for Brain Tumor Detection [(Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Methodology

### 1. Data Preprocessing

- **Image Preparation**
- **Data Augmentation** (The augmentations increase dataset diversity and help the model generalize better)

### 2. Transfer Learning Approach

**Why Transfer Learning?**
- Training a deep CNN from scratch requires massive datasets (millions of images)
- Medical imaging datasets are typically small due to privacy and labeling costs
- Pre-trained models on ImageNet have learned general visual features (edges, textures, shapes)
- We can adapt these features for medical imaging with fine-tuning

**Strategy**:
1. Load VGG16 pre-trained on ImageNet 
2. Freeze convolutional base
3. Add custom classification layers for our 4 classes
5. Fine-tune last convolutional block for better performance

## Results

### Classification Report

The model achieved excellent results on the test set:


| Class | Precision | Recall | F1-Score | Support |
|:------|:-----------|:--------|:----------|:----------|
| Glioma (0) | 0.97 | 0.98 | 0.98 | 300 |
| Meningioma (1) | 1.00 | 1.00 | 1.00 | 405 |
| No Tumor (2) | 0.98 | 0.90 | 0.94 | 306 |
| Pituitary (3) | 0.93 | 0.99 | 0.96 | 300 |
| **Overall Accuracy** |  |  | **0.97** | **1311** |
| **Macro Avg** | 0.97 | 0.97 | 0.97 |  |
| **Weighted Avg** | 0.97 | 0.97 | 0.97 |  |


**Interpretation:**
- The model performs exceptionally well on all classes, with near-perfect recall on *meningioma* and *pituitary*.  
- Minor misclassifications occur primarily between **glioma** and **no tumor**, likely due to visual similarity in MRI intensity patterns.  
- Overall, the VGG16 backbone delivers a **balanced, high-precision classifier** suitable for medical image analysis tasks.


**Key Findings**:
- Transfer learning accelerated training significantly
- Data augmentation prevented overfitting
- Model learned effectively with a relatively small dataset

### Quick Start with Kaggle

For the easiest experience, use the Kaggle notebook:

1. Visit: [Brain Tumor Classification Notebook](https://www.kaggle.com/code/ayeshasal89/brain-tumor-classification-mri-vgg16-97-acc/notebook)
2. Click "Copy and Edit" to create your own version
3. Turn on GPU accelerator
4. Run all cells


## Acknowledgments

I would like to thank:

- **Kaggle** for providing free GPU resources and hosting the dataset
- **TensorFlow and Keras teams** for excellent documentation and frameworks
- **Visual Geometry Group, Oxford** for the VGG16 architecture
- **Medical imaging community** for making datasets publicly available

## Citation

If you use this notebook, dataset, or model setup in your research or work, please cite as follows:

**APA Style:**
> Saleem, A. (2025). *Brain Tumor Classification using Transfer Learning with VGG16*. Kaggle.  
> Available at: [https://www.kaggle.com/code/ayeshasal89/brain-tumor-classification-mri-vgg16-97-acc](https://www.kaggle.com/code/ayeshasal89/brain-tumor-classification-mri-vgg16-97-acc)

**BibTeX:**
```bibtex
@misc{saleem2024braintumor,
  author = {Ayesha Saleem},
  title = {Brain Tumor Classification using Transfer Learning with VGG16},
  year = {2025},
  howpublished = {\url{https://www.kaggle.com/code/ayeshasal89/brain-tumor-classification-mri-vgg16-97-acc}},
  note = {Kaggle Notebook}
}

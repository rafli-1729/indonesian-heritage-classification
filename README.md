# Indonesian Traditional House Image Classification
Classifying images of five traditional Indonesian houses using a deep learning pipeline built with fastai, PyTorch, and timm. The project focuses on constructing a reliable workflow capable of handling visual variability, class imbalance, and dataset noise while achieving strong generalization across unseen samples.

This project uses data from the **Data Science Competition LOGIKA UI 2025**. Due to competition rules and dataset usage restrictions, the raw image files and official competition data cannot be publicly shared in this repository.

---

# Project Overview

The project implements an end-to-end image classification pipeline using fastai, PyTorch, timm, and various computer vision utilities. The workflow is structured to handle common challenges such as noisy datasets, duplicate images, heavy augmentation needs, and class imbalance. Each stage contributes to building a clean, reliable, and high-performing model for image classification tasks.

## Data Loading and Preparation

The process begins by importing the necessary libraries for image processing, visualization, augmentation, hashing, and model development. Dataset directories are initialized, image files are loaded, and integrity checks are performed. Perceptual hashing techniques are used to identify duplicate or near-duplicate images, ensuring the dataset remains diverse and representative. Additional exploratory checks, such as class distribution summaries, help validate that the data is ready for modeling.

## Data Augmentation

A combination of fastai transforms and Albumentations operations is used to create a strong augmentation pipeline. These augmentations include geometric transformations, color modifications, blur and noise injection, and other perturbations that encourage the model to generalize better. Augmentation previews are generated to confirm that each transformation aligns with the intended training strategy. This stage is essential for increasing the effective variability of the training data.

## Model Loading and Architecture Setup

EfficientNet-based architectures from torchvision and timm are loaded and configured as the backbone of the classifier. Pretrained weights are incorporated to accelerate convergence and improve performance on limited datasets. The model is wrapped inside fastai’s Learner for streamlined experimentation, enabling features such as mixed precision, automatic learning rate finding, and flexible callback integration. Architectural choices can be adjusted to balance accuracy and computational cost.

## Handling Class Imbalance

Class frequencies are analyzed to identify imbalance within the dataset. A WeightedRandomSampler is constructed to ensure that minority classes are sampled more frequently during training, preventing the model from being biased toward majority classes. This method improves fairness across categories and supports better macro-level performance metrics. The sampling strategy is validated before being incorporated into the training loader.

## Training Process

Training is performed using fastai’s high-level training routines. Learning rate discovery, scheduled training phases, and callback-driven monitoring help guide the optimization process. Multiple training rounds may be run with adjustments to augmentations, hyperparameters, or architecture choices. Progressive resizing techniques and freeze–unfreeze cycles are leveraged to stabilize learning and extract better performance from the chosen model backbone.

## Evaluation and Diagnostics

Performance is evaluated using metrics such as accuracy, error rate, and detailed confusion matrices. Visualization tools highlight correctly classified samples, misclassifications, and class-level weaknesses. These diagnostics support an iterative refinement process, allowing improvements to augmentations, sampling strategies, or model architecture. Additional plots and summaries provide insight into how the training data and model behavior interact.

## Inference and Export

After achieving satisfactory performance, the final model is exported for deployment or batch inference. A dedicated inference pipeline loads the exported model, preprocesses new images, and generates predictions consistently with the training setup. Utility functions enable visual inspection of predictions, batch processing of image folders, and seamless integration into external applications or competition submission workflows.

---

# Purpose of the Repository

This repository provides a complete and reproducible pipeline for image classification using a combination of fastai, PyTorch, and timm architectures. The goal is to construct a robust workflow that addresses dataset imperfections, applies domain-informed augmentation strategies, and builds a strong model capable of handling class imbalance and visual variability. The structure of this project is intended both as a practical solution and as a reference for anyone learning or experimenting with modern computer vision approaches.

The model developed through this workflow achieved final Kaggle scores of **84% (private leaderboard)** and **88% (public leaderboard)**, demonstrating strong generalization and consistent performance across unseen data splits. These results reflect the effectiveness of the methodology, especially in managing imbalanced datasets and leveraging high-quality augmentation strategies.

---

This project was created by **Muhammad Rafli Azrarsyah**, a third-year Actuarial Science student at Universitas Gadjah Mada with a deep interest in data, modeling, and the insights that emerge from careful analysis. What began as an exploration into image classification evolved into an engaging and rewarding project, reinforcing the excitement and potential that data-driven approaches bring to real-world problems.

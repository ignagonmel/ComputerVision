# Animal Classification: Custom Deep CNN Pipeline
This repository contains the final project developed for the Computer Vision course in the Robotics Engineering program at Universidad Carlos III de Madrid (UC3M). The project implements a robust, end-to-end deep learning pipeline for multi-class animal classification using a Custom Convolutional Neural Network (CNN) built from scratch.

## Description
The application focuses on the challenge of identifying 7 different animal species from high-resolution images. Unlike standard assignments, this project avoids pre-trained models (Transfer Learning) to demonstrate the capability of a custom-designed architecture to extract hierarchical features and generalize effectively on a diverse dataset.

## Key Technical Features
- Deep CNN Architecture: A 10-layer convolutional model organized into 5 dual-layer blocks with increasing filter density (32 to 512).
- Advanced Regularization: Integrated Batch Normalization after each convolutional stage to stabilize learning, combined with progressive Dropout (up to 0.5) to     mitigate overfitting.
- Data Augmentation: On-the-fly image transformations including random flips, rotations, zoom, and brightness adjustments to improve model robustness.
- High-Resolution Processing: All images are processed at 224x224 pixels to preserve critical morphological details of the species.
- Imbalance Handling: Implements dynamic Class Weighting (e.g., 1.136 for elephants) to ensure the model pays equal attention to minority classes during training.

## Objective
The primary goal of this project is to master the design and optimization of deep learning models for classification. Key technical objectives include:
- Designing a high-performance CNN architecture from scratch.
- Implementing a complete training pipeline with Early Stopping and Learning Rate Decay.
- Conducting a deep error analysis using Confusion Matrices and Classification Reports.
- Evaluating real-world generalization on a completely unseen test set.

## Performance & Results
- Test Accuracy: 81.95% achieved after 80 epochs.
- Top Class: "Dog" achieved the highest precision at 93%.
- Recall: High sensitivity for "Elephant" (93%) despite dataset imbalance.

## Requirements
- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- NumPy & Matplotlib
- Seaborn (for visualization)
- Data Folder: A dataset/ directory structured into train/, validation/, and test/ folders.

## How to Execute the Code
To train the model and generate the evaluation reports:
```bash
python main.py
```

## Generated Outputs
Upon completion, the script automatically generates:
- training_report.txt: Full breakdown of per-class metrics (Precision, Recall, F1).
- confusion_matrix.png: Visual heatmap of classification errors.
- training_graphs.png: Accuracy and Loss evolution curves.
- prediction_examples.png: A grid of 12 test images with predicted vs. true labels.
- best_model.keras: The saved weights of the most accurate iteration.
  Download: https://drive.google.com/drive/folders/17RL6uSQcTLzJetj-Q3OCMCguOd489nqw?usp=sharing

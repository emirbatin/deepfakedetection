
# Deepfake Detection with MediaPipe and Machine Learning

This repository provides an end-to-end pipeline for detecting deepfake images using facial features extracted via **MediaPipe** and classifying them with advanced machine learning models like **XGBoost**, **Random Forest**, and an ensemble Voting Classifier.

## Table of Contents

- [Project Overview](#project-overview)
- [Features Extracted](#features-extracted)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview

Deepfakes pose significant challenges in today's digital world, from misinformation to security risks. This project leverages **MediaPipe Face Mesh** for extracting facial landmarks and geometric features, which are then used to classify images as **real** or **fake** using machine learning algorithms.

### Key Features
- Facial landmark detection with **MediaPipe**.
- Extraction of geometric features (eye opening, mouth opening, etc.).
- Machine learning-based classification with models such as **XGBoost** and **Random Forest**.
- Comprehensive visualization of results, including confusion matrices and ROC curves.

## Features Extracted

The pipeline extracts the following features from images:
1. **3D Facial Landmark Coordinates**: Flattened vector of `x`, `y`, and `z` coordinates for all detected facial landmarks.
2. **Geometric Features**:
   - Left and right eye opening distances.
   - Mouth opening distance.

These features help detect inconsistencies in fake images created by GAN-based systems.

## Dataset

The project uses the [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) from Kaggle. The dataset consists of `real` and `fake` images divided into `train`, `valid`, and `test` splits.

## Model Pipeline

1. **Feature Extraction**: 
   - Images are resized to `224x224` and processed with MediaPipe.
   - Geometric and landmark features are extracted.

2. **Preprocessing**:
   - Features are scaled using `StandardScaler`.

3. **Model Training**:
   - Individual classifiers: `XGBoost` and `Random Forest`.
   - Ensemble Voting Classifier with soft voting.

4. **Evaluation**:
   - Metrics: Accuracy, ROC AUC, Confusion Matrix, Classification Report.
   - Visualizations: Confusion Matrix and ROC Curve.

## Installation

### Prerequisites

Ensure you have Python 3.9+ installed along with the following packages:

```bash
pip install numpy opencv-python mediapipe matplotlib seaborn tqdm joblib scikit-learn xgboost
```

### Additional Setup

Download the dataset and place it in the following structure:

```plaintext
real-vs-fake/
    train/
        real/
        fake/
    valid/
        real/
        fake/
    test/
        real/
        fake/
```

Update the `DATASET_PATH` variable in the code with the correct path.

## Usage

### 1. Run Feature Extraction

```python
datasets = prepare_dataset(DATASET_PATH, max_samples=100)
```

This function processes images, extracts features, and saves them as compressed `.npz` files.

### 2. Train Models

```python
results = train_models(datasets)
```

This step trains and evaluates the models on the extracted features.

### 3. Visualize Results

```python
plot_results(results, datasets['test'][0], datasets['test'][1])
```

Confusion matrices and ROC curves will be displayed.

### 4. Save the Model

```python
with open('mediapipe_model.pkl', 'wb') as f:
    pickle.dump({
        'model': results['model'],
        'scaler': results['scaler']
    }, f)
```

This saves the trained model and scaler for future inference.

## Results

### Validation Results

- **Accuracy**: `0.8076`
- **ROC AUC**: `0.8929`

### Test Results

- **Accuracy**: `0.8094`
- **ROC AUC**: ` 0.8931`

### Confusion Matrix
|               | **Predicted Real** | **Predicted Fake** |
|---------------|--------------------|--------------------|
| **Actual Real**  | 7898               | 2100               |
| **Actual Fake**  | 1700               | 8300               |

### Classification Report

| Metric         | Class 0 (Real) | Class 1 (Fake) | Macro Avg | Weighted Avg |
|----------------|----------------|----------------|-----------|--------------|
| **Precision**  | 0.82           | 0.80           | 0.81      | 0.81         |
| **Recall**     | 0.79           | 0.83           | 0.81      | 0.81         |
| **F1-Score**   | 0.81           | 0.81           | 0.81      | 0.81         |
| **Support**    | 9998           | 10000          | 19998     | 19998        |


### ROC Curve

![ROC Curve](https://res.cloudinary.com/dcy6ogtc1/image/upload/v1736105128/lpqrrczdng3ak5oyb4xk.png)

## Acknowledgments

Special thanks to:
- [MediaPipe](https://mediapipe.dev) for the powerful face mesh solution.
- [Kaggle](https://www.kaggle.com) for hosting the dataset.
- The open-source community for supporting the tools and libraries used in this project.

# Kidney Disease Classification on CT Scan Images using Hybrid VGG16 and XGBoost

## Project Description

This project performs an end-to-end classification of kidney disease on CT scan images using a **hybrid deep learning + machine learning model**. A pretrained **VGG16** Convolutional Neural Network (CNN) is used as the **feature extractor**, and **Extreme Gradient Boosting (XGBoost)** is used as the **classifier** to distinguish four kidney conditions: **cyst, normal, stone, and tumor**. The pipeline covers duplicate removal, preprocessing, adaptive data augmentation, feature extraction, hyperparameter tuning with Bayesian Optimization (Optuna), Stratified k-Fold Cross Validation, and evaluation on three dataset scenarios (full, axial, and coronal), each with and without data augmentation.

This repository is the implementation of an undergraduate thesis at the Department of Computer Science, Faculty of Mathematics and Natural Sciences, Universitas Lampung.

## Objectives

- Classify kidney CT scan images into four classes: cyst, normal, stone, tumor
- Build a hybrid model that combines VGG16 (feature extraction) and XGBoost (classification)
- Compare model performance on three dataset scenarios: **full**, **axial**, and **coronal**
- Evaluate the effect of **data augmentation** by testing each scenario with and without augmentation
- Evaluate the models with **Stratified k-Fold Cross Validation** (k = 5) and a held-out test set

## Tools & Libraries

- Python 3.12 (Kaggle Notebook, NVIDIA Tesla T4/P100 GPU)
- Pandas, NumPy: data manipulation
- OpenCV: image reading, resizing, and augmentation
- TensorFlow/Keras: VGG16 (ImageNet weights) for feature extraction
- XGBoost: multi-class gradient boosting classifier
- Optuna: hyperparameter tuning (Bayesian Optimization)
- Scikit-learn: train/test split, StratifiedKFold, LabelEncoder, evaluation metrics
- Matplotlib, Seaborn: visualization

## Dataset

- **Source:** [Riñones - Cyst, Stone, Tumor, Normal](https://www.kaggle.com/datasets/gonzajl/riones-cyst-stone-tumor-normal-dataset) (Kaggle, uploaded by GonzaJL)
- **Type:** 2D kidney CT scan images (JPEG and PNG), 224 x 224 pixels
- **Classes:** cyst, normal, stone, tumor (balanced: 2,939 images per class)
- **Original size:** 11,756 images (coronal: 5,931, axial: 5,825)

| Scenario | Description | Images removed as duplicates |
|----------|-------------|------------------------------|
| Full | All CT scan images (axial + coronal) | 1,225 |
| Axial | Axial-view images only (cross-sectional) | 222 |
| Coronal | Coronal-view images only (frontal plane) | 1,003 |

Duplicate images were detected using **MD5 hashing** of the file content, and only one image per duplicate group was kept, to avoid data leakage between train, validation, and test sets.

> **Note:** The dataset is not included in this repository. Please download it from the source above and follow its license terms.

## Pipeline Overview

### 1. Deduplication
- Computed the MD5 hash of every image and removed images with identical content

### 2. Preprocessing
- Converted images to RGB (3 channels, 224 x 224)
- Normalized with the VGG16 `preprocess_input` function (mean subtraction using the ImageNet channel means)

### 3. Train/Test Split & Cross-Validation
- **Hold-out:** 85% train+validation and 15% test, stratified
- **Stratified k-Fold Cross Validation** with **k = 5** on the train+validation data

| Scenario | Train+Validation | Per fold (train / validation) |
|----------|------------------|-------------------------------|
| Full | 8,953 | about 7,162 / 1,791 |
| Axial | 4,762 | about 3,809 / 953 |
| Coronal | 4,191 | about 3,352 / 839 |

### 4. Data Augmentation (augmented experiments only)
- Applied **only to the training data** of each fold (validation and test data are never augmented)
- **Adaptive minority-class augmentation**: minority classes are augmented until every class reaches the size of the majority class; the majority class is not augmented
- Three techniques applied sequentially to each generated image:
  - Random rotation between -15 and +15 degrees
  - Random brightness scaling with a factor between 0.1 and 1.5
  - Sharpening (unsharp masking with Gaussian blur, sigma = 2, weights 1.5 and -0.5)

### 5. Feature Extraction with VGG16
- VGG16 pretrained on **ImageNet**, `include_top=False`, all layers **frozen** (`trainable=False`)
- **GlobalAveragePooling2D** converts the 7 x 7 x 512 feature map into a **512-dimensional** feature vector per image

### 6. Hyperparameter Tuning
- **Bayesian Optimization with Optuna**, **30 trials** per scenario
- Objective: mean validation accuracy across the 5 folds
- Search space:

| Parameter | Range |
|-----------|-------|
| n_estimators | 100 - 1000 |
| max_depth | 3 - 10 |
| learning_rate | 0.001 - 0.3 |
| subsample | 0.5 - 1.0 |
| colsample_bytree | 0.5 - 1.0 |
| gamma | 0.0 - 1.0 |
| reg_alpha | 0.000001 - 10.0 |
| reg_lambda | 0.1 - 10.0 |
| min_child_weight | 1 - 10 |

- Selected hyperparameters (best of the top-3 combinations, chosen with Stratified k-Fold):

| Parameter | Full (Scenario 1) | Axial (Scenario 2) | Coronal (Scenario 3) |
|-----------|-------------------|--------------------|----------------------|
| n_estimators | 437 | 805 | 581 |
| max_depth | 10 | 5 | 4 |
| learning_rate | 0.065049 | 0.071404 | 0.089591 |
| subsample | 0.799329 | 0.767304 | 0.890828 |
| colsample_bytree | 0.578009 | 0.781481 | 0.840371 |
| gamma | 0.155995 | 0.00076 | 0.013091 |
| reg_alpha | 0.000003 | 0.000005 | 0.000005 |
| reg_lambda | 5.399484 | 0.877048 | 0.25841 |
| min_child_weight | 7 | 3 | 8 |

Other settings: `objective="multi:softprob"`, `num_class=4`, `eval_metric=["mlogloss","merror"]`, `tree_method="hist"`, `early_stopping_rounds=20`, `random_state=42`.

### 7. Training & Evaluation
- Stratified 5-Fold Cross Validation, followed by final training with the selected hyperparameters
- Metrics: **accuracy, precision, recall, and F1-score**
- Reported train accuracy, average validation accuracy (across folds), test accuracy, learning curves, confusion matrices, and per-class classification reports

## Results

### Accuracy per Scenario

| Scenario | Condition | Train Acc | Val Acc (avg 5-fold) | Test Acc |
|----------|-----------|-----------|----------------------|----------|
| 1. Full | Without augmentation | 0.9954 | 0.9641 | 0.9614 |
| 1. Full | With augmentation | 1.0000 | 0.9609 | **0.9690** |
| 2. Axial | Without augmentation | 0.9997 | 0.9976 | 0.9940 |
| 2. Axial | With augmentation | 0.9994 | 0.9983 | **0.9964** |
| 3. Coronal | Without augmentation | 0.9850 | 0.9262 | 0.9067 |
| 3. Coronal | With augmentation | 0.9995 | 0.9238 | **0.9175** |

### Best Model

The best model was obtained on the **axial dataset with augmentation (Scenario 2)**:

| Metric | Score |
|--------|-------|
| Train accuracy | 0.9994 |
| Average validation accuracy (5-fold) | 0.9983 |
| Test accuracy | 0.9964 |

The model made only **3 misclassifications out of 841 test images**, and the standard deviation of the validation accuracy across folds was only 0.09%.

### Per-class Performance of the Best Model per Scenario (with augmentation)

**Scenario 1: Full dataset**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Cyst | 0.9902 | 1.0000 | 0.9951 |
| Normal | 0.9649 | 0.9255 | 0.9448 |
| Stone | 0.9162 | 0.9533 | 0.9344 |
| Tumor | 0.9932 | 0.9932 | 0.9932 |
| **Average** | 0.9661 | 0.9680 | 0.9670 |

**Scenario 2: Axial dataset (best model)**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Cyst | 0.9957 | 1.0000 | 0.9978 |
| Normal | 0.9953 | 0.9953 | 0.9953 |
| Stone | 1.0000 | 0.9921 | 0.9960 |
| Tumor | 0.9963 | 0.9963 | 0.9963 |
| **Average** | 0.9968 | 0.9959 | 0.9964 |

**Scenario 3: Coronal dataset**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Cyst | 0.9773 | 1.0000 | 0.9885 |
| Normal | 0.8901 | 0.8293 | 0.8586 |
| Stone | 0.8398 | 0.8872 | 0.8628 |
| Tumor | 0.9820 | 0.9762 | 0.9791 |
| **Average** | 0.9223 | 0.9232 | 0.9223 |

## Key Insights

1. **Very High Performance on the Axial View**: The augmented axial model reaches 99.64% test accuracy with near-perfect precision, recall, and F1-score for all four classes. Axial images show the kidney more clearly, so VGG16 extracts more representative features.

2. **Coronal View is the Hardest**: The coronal dataset gives the lowest results (91.75% test accuracy). The full dataset, which mixes both orientations, falls in between (96.90%).

3. **Normal and Stone are the Most Confused Classes**: In the full and coronal scenarios, most errors happen between *normal* and *stone* (coronal: 32 normal predicted as stone and 19 stone predicted as normal; full: 28 and 12), while *cyst* and *tumor* are recognized very well.

4. **Augmentation Improves Test Accuracy in All Scenarios**: Test accuracy increased from 0.9614 to 0.9690 (full), 0.9940 to 0.9964 (axial), and 0.9067 to 0.9175 (coronal). In the full and coronal scenarios the average validation accuracy was slightly lower with augmentation, so the benefit depends on the characteristics of the dataset.

5. **Stable Generalization**: On the axial dataset, train, validation, and test accuracy are very close to each other, showing no sign of significant overfitting.

## Limitations & Future Work

- Only VGG16 is used as the feature extractor and only XGBoost as the classifier. Future work can compare other CNN architectures (ResNet, EfficientNet, DenseNet) and other classifiers (Random Forest, SVM, LightGBM).
- The train/validation/test splits are made at the image level.

## Repository Structure

```
classification-kidney-disease/
├── notebooks/      # notebooks for each scenario (full, axial, coronal) with and without augmentation
└── README.md
```

## How to Run

```bash
git clone https://github.com/AndriaLarasRamadhania/classification-kidney-disease.git
cd classification-kidney-disease
pip install tensorflow xgboost optuna scikit-learn opencv-python pandas numpy matplotlib seaborn tqdm
```

1. Download the dataset from the link above and update the `DATA_DIR` path at the top of the notebook (the notebooks use Kaggle paths)
2. Open the notebook in Kaggle, Google Colab, or Jupyter (GPU recommended)
3. Run all cells in order

## Citation

If you use this work, please cite:

```
Ramadhania, A. L. (2026). Klasifikasi Penyakit Ginjal pada Citra Medis Menggunakan
Hybrid VGG16 sebagai Feature Extraction dan Extreme Gradient Boosting (XGBoost)
sebagai Classifier [Undergraduate thesis]. Universitas Lampung.
```

## Author

**Andria Laras Ramadhania**
Department of Computer Science, Faculty of Mathematics and Natural Sciences, Universitas Lampung

Supervisors: Dewi Asiah Shofiana, S.Komp., M.Kom. and Erin Eka Citra, M.Kom.

## Disclaimer

This project is for research and educational purposes only. It is not a medical device and must not be used as a substitute for professional medical diagnosis.

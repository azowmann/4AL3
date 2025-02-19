# COMPSCI 4AL3 - Applications of Machine Learning Course

## This repository contains my assignments completed from COMPSCI 4AL3, a Machine Learning course I took during the fall 2024 term at McMaster University.

### Each Assignment is described below:

## Table of Contents
1. [Assignment 1: GDP vs Happiness and Abalone Age Prediction](#assignment-1-gdp-vs-happiness-and-abalone-age-prediction)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Implementation Details](#implementation-details)
5. [Results](#results)

---

### Assignment 1: GDP vs Happiness and Abalone Age Prediction

This project consists of two parts:
1. **Linear Regression on GDP vs Happiness Data**: A linear regression model is implemented to analyze the relationship between GDP per capita and happiness scores using gradient descent and ordinary least squares (OLS).
2. **Polynomial Regression on Abalone Age Prediction**: A polynomial regression model is used to predict the age of abalones (measured in rings) based on physical measurements, with k-fold cross-validation for evaluation.

---

### Assignment 2: Support Vector Machine (SVM) for Feature and Dataset Analysis

This project involves the implementation and evaluation of a Support Vector Machine (SVM) model to analyze and classify datasets from 2010 and 2020. The project is divided into two main experiments:

1. **Feature Experiment**: This experiment evaluates the performance of the SVM model using different combinations of feature sets (FS-I, FS-II, FS-III, FS-IV) on the 2010 dataset. The goal is to identify the best-performing feature set combination based on the True Skill Statistic (TSS) score and confusion matrices.

2. **Data Experiment**: This experiment compares the performance of the SVM model on the 2010 and 2020 datasets using the best-performing feature set combination identified in the feature experiment. The results are evaluated using TSS scores and confusion matrices.

#### Key Components of the Code:
- **Data Preprocessing**: The data is normalized and missing values are removed using `StandardScaler` from `scikit-learn`.
- **Feature Creation**: Features are created based on the specified feature set combinations (FS-I, FS-II, FS-III, FS-IV).
- **Cross-Validation**: A 10-fold cross-validation is performed to evaluate the model's performance.
- **SVM Training**: The SVM model is trained using the Radial Basis Function (RBF) kernel.
- **Evaluation Metrics**: The model's performance is evaluated using the TSS score and confusion matrices.
- **Visualization**: The results are visualized using plots of TSS scores and confusion matrices for each fold of cross-validation.

#### Implementation Details:
- **Feature Experiment**:
  - The `feature_experiment()` function evaluates all possible combinations of the four feature sets (FS-I, FS-II, FS-III, FS-IV) on the 2010 dataset.
  - The function outputs the mean and standard deviation of TSS scores for each feature set combination, along with confusion matrices and a plot of TSS scores across all folds.
  - The best-performing feature set combination is identified and printed.

- **Data Experiment**:
  - The `data_experiment()` function compares the performance of the SVM model on the 2010 and 2020 datasets using the best-performing feature set combination identified in the feature experiment.
  - The function outputs the mean and standard deviation of TSS scores for both datasets, along with confusion matrices and a plot of TSS scores across all folds.

#### Results:
- **Feature Experiment**:
  - The results of the feature experiment are printed to the console, showing the mean and standard deviation of TSS scores for each feature set combination.
  - Confusion matrices for each feature set combination are plotted, showing the true positive, false positive, true negative, and false negative rates.
  - A plot of TSS scores across all folds for each feature set combination is displayed, allowing for visual comparison of model performance.

- **Data Experiment**:
  - The results of the data experiment are printed to the console, showing the mean and standard deviation of TSS scores for the 2010 and 2020 datasets.
  - Confusion matrices for both datasets are plotted, showing the true positive, false positive, true negative, and false negative rates.
  - A plot of TSS scores across all folds for both datasets is displayed, allowing for visual comparison of model performance over time.

---

### Assignment 3: Support Vector Machine (SVM) Classifier for Breast Cancer Diagnosis

This project implements a Support Vector Machine (SVM) from scratch using Python and NumPy. The SVM is trained using two optimization techniques: Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent. Additionally, an Active Learning approach is implemented to improve the model's performance by iteratively selecting the most informative samples from an unlabeled dataset.

The project is divided into three parts:
1. **Part 1**:  Implementation of SVM using Stochastic Gradient Descent.
2. **Part 2**:  Implementation of SVM using Mini-Batch Gradient Descent.
3. **Part 3**:  Implementation of Active Learning with SVM.

#### Implementation Details:

- **Data Preprocessing**:
  - The dataset is normalized using StandardScaler from scikit-learn.
  - The data is split into training and validation sets (80% training, 20% validation).

- **SVM Class**:
  - The svm_ class implements the core SVM functionality.
  - **Initialization**: Weights are initialized to zero.
  - **Gradient Computation**: The gradient is computed using the hinge loss function.
  - **Loss Calculation**: Hinge loss with L2 regularization is used.
  - **Training**: Supports both Stochastic Gradient Descent and Mini-Batch Gradient Descent.
  - **Prediction**: Predicts labels for the test set and computes accuracy, precision, and recall.

- **Training Methods**:
  - **Stochastic Gradient Descent (SGD)**: Updates weights after each training sample.
  - **Mini-Batch Gradient Descent:** Updates weights using small batches of data.
  - **Early Stopping:** Training stops if the loss improvement falls below a threshold.

- **Active Learning**:
  - The model iteratively selects the most informative samples from the unlabeled dataset and adds them to the training set.
  - The process continues until the model's performance converges.

- **Visualization**:
  - Training and validation loss curves are plotted for each part of the project.
 
#### Results:
- **Part 1: Stochastic Gradient Descent**:
  - Training and Validation Loss: Plotted as a function of epochs.
  - Early Stopping: Triggered when the loss improvement is below the threshold.
  - Metrics: Accuracy, precision, and recall are computed on the test set.

- **Part 2: Mini-Batch Gradient Descent**:
  - Similar to Part 1, but uses mini-batches for weight updates.
  - Batch Size: Set to 5.
 
- **Part 3: Active Learning**:
  - The model starts with a small labeled dataset and iteratively adds samples from the unlabeled dataset.
  - Convergence: Training stops when the loss improvement is negligible.
  - Metrics: Final accuracy, precision, and recall are reported.
 
---

### Assignment 4: Convolutional Neural Network (CNN) Classifier for FashionMNIST Dataset

This project combines image classification and fairness analysis in predictive modeling using two distinct datasets: The FashionMNIST dataset provided by Zalando Research, and the COMPAS dataset, used for fairness analysis in machine learning. 

This project is divided into two main parts:
1. **FashionMNIST Classification**: A Convolutional Neural Network (CNN) is implemented to classify images from the FashionMNIST dataset.
2. **COMPAS Dataset Analysis**: A Logistic Regression Model (LRM) is used to predict recidivism risk scores from the COMPAS dataset, with a focus on fairness and equalized odds.

#### Implementation Details:

- **Part 1: FashionMNIST Classification**
  - Dataset: The FashionMNIST dataset consists of 60,000 training images and 10,000 test images of fashion items across 10 classes
  - Model: A Convolutional Neural Network (CNN) is implemented with the following architecture:
      - Three convolutional layers with ReLU activation.
      - Two fully connected layers.
      - Max-pooling layers for downsampling.
  - Training: The model is trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.001 for 15 epochs.
  - Evaluation: The model's performance is evaluated on the test set, and training/validation loss curves are plotted.

- **Part 2: COMPAS Dataset Analysis**:
  - Dataset: The COMPAS dataset contains features related to criminal defendants, including demographics, criminal history, and recidivism risk scores.
  - Preprocessing:
      - Continuous features are scaled using StandardScaler.
      - Categorical features are one-hot encoded.
      - The target variable is binary: 1 for high risk and 0 for low risk.
  - Model: A Logistic Regression Model (LRM) is implemented using PyTorch.
  - Training: The model is trained using SGD with a learning rate of 0.001 for 50 epochs.
  - Evaluation:
      - Accuracy is calculated on the test set.
      - Equalized odds are computed to evaluate fairness across different racial groups.
      - The dataset is balanced to ensure equal representation of classes within each racial group, and the model is retrained and evaluated.

#### Results:
- **Part 1: FashionMNIST Classification**:
  - Training and Validation Loss: Plotted as a function of epochs.
  - Test Accuracy: Reported after training.

- **Part 2: COMPAS Dataset Analysis**:
  - Model Accuracy: Reported for both unbalanced and balanced datasets.
  - Equalized Odds: True Positive Rate (TPR) and False Positive Rate (FPR) are calculated for each racial group to assess fairness.

---



---

## Installation

To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn` (optional, for additional utilities)

You can install the required libraries using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

To run the experiments, simply call the feature_experiment() and data_experiment() functions in the script. The results will be printed to the console and visualized using plots.

#### Run the feature experiment to evaluate different feature set combinations
feature_experiment()

#### Run the data experiment to compare performance on 2010 and 2020 datasets
data_experiment()

## Project Structure

- `data2010/` - Contains the 2010 dataset files.
- `data2020/` - Contains the 2020 dataset files.
- `assignment1.py` - The script for Assignment 1 (GDP vs Happiness and Abalone Age Prediction).
- `assignment2.py` - The script for Assignment 2 (SVM for Feature and Dataset Analysis).

## Implementation Details

Assignment 1:

Linear Regression: Implemented using gradient descent and OLS.

Polynomial Regression: Implemented with k-fold cross-validation for evaluation.

Assignment 2:

Feature Experiment: Evaluates all combinations of feature sets (FS-I, FS-II, FS-III, FS-IV) on the 2010 dataset.

Data Experiment: Compares model performance on 2010 and 2020 datasets using the best-performing feature set combination.

## Results

Assignment 1:

Results for GDP vs Happiness analysis and Abalone age prediction are displayed in the console and visualized using plots.

Assignment 2:

Feature Experiment: The best-performing feature set combination is identified and results are visualized using TSS scores and confusion matrices.

Data Experiment: Performance comparison between 2010 and 2020 datasets is visualized using TSS scores and confusion matrices.


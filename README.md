# COMPSCI 4AL3 - Applications of Machine Learning Course

## This repository contains my assignments completed from COMPSCI 4AL3, a Machine Learning course I took during the fall 2024 term at McMaster University.

### Each Assignment is described below:

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

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Implementation Details](#implementation-details)
5. [Results](#results)

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


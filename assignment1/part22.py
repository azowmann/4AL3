import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

abalone_data = pd.read_csv('training_data.csv')

# Normalize the features using Z-score normalization
features = ['Length', 'Diameter', 'Height', 'Whole_weight', 
            'Shucked_weight', 'Viscera_weight', 'Shell_weight']

# Normalize each feature
for feature in features:
    abalone_data[feature] = (abalone_data[feature] - abalone_data[feature].min()) / (abalone_data[feature].max() - abalone_data[feature].min())

# Normalize the target variable 'Rings'
abalone_data['Rings'] = (abalone_data['Rings'] - abalone_data['Rings'].mean()) / abalone_data['Rings'].std()


class PolynomialRegression:
    def __init__(self, x_, y_, degree=2):
        self.input = np.array(x_)
        self.target = np.array(y_)
        self.degree = degree

    def preprocess(self):
        # Normalize the input features
        x_mean = np.mean(self.input, axis=0)
        x_std = np.std(self.input, axis=0)
        x_train = (self.input - x_mean) / x_std

        # Normalize the target variable
        y_mean = np.mean(self.target)
        y_std = np.std(self.target)
        y_train = (self.target - y_mean) / y_std
        
        return X, y_train

    def train(self, X, Y):
        # Compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, X_test, beta):
        # Predict using beta
        Y_hat = X_test.dot(beta)
        return Y_hat

fig, axes = plt.subplots(len(features), len(features), figsize=(16, 16))

# Set font size for better visibility
plt.rcParams.update({'font.size': 10})

# Loop through each feature pair
for i in range(len(features)):
    for j in range(len(features)):
        if i == j:
            # On the diagonal, create a histogram
            axes[i, j].hist(abalone_data[features[i]], bins=20, alpha=0.5, color='blue')
            axes[i, j].set_title(features[i], fontsize=12)
        elif i > j:
            # Only create scatter plots for unique pairs in the bottom left
            axes[i, j].scatter(abalone_data[features[j]], abalone_data[features[i]], alpha=0.5, s=10)  # Smaller markers
            axes[i, j].set_xlabel(features[j], fontsize=10)
            axes[i, j].set_ylabel(features[i], fontsize=10)
        else:
            # Hide redundant scatterplots (top right)
            axes[i, j].axis('off')

# Adjust layout for better readability
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust to prevent overlap
plt.show()

subset_data = abalone_data.iloc[577:]

# Optional: Reset index if needed
subset_data.reset_index(drop=True, inplace=True)

# Check the size of the subset and display the first few rows
print(f"Subset size: {subset_data.shape[0]} rows")
print(subset_data.head())

X = subset_data[features].values
Y = subset_data['Rings'].values

# Instantiate the PolynomialRegression class with degree 2 (for quadratic)
poly_reg = PolynomialRegression(X, Y, degree=2)

# Preprocess the inputs
X_train, Y_train = poly_reg.preprocess()

# Compute beta
beta = poly_reg.train(X_train, Y_train)

# Use the computed beta for prediction
Y_predict = poly_reg.predict(X_train, beta)

# Plotting results
plt.figure(figsize=(15, 8))
plt.scatter(Y_train, Y_predict, alpha=0.5)
plt.xlabel("Actual Rings (Normalized)")
plt.ylabel("Predicted Rings (Normalized)")
plt.title("Polynomial Regression Predictions of Abalone Age")
plt.plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], color='r', linestyle='--')  # y=x line for reference
plt.show()




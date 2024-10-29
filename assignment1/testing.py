import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(42)

# Load the dataset
hap_gdp_data = pd.read_csv("gdp-vs-happiness.csv")

# Process the data: Keep only the 2018 data and drop unnecessary columns
by_year = (hap_gdp_data[hap_gdp_data['Year'] == 2018]).drop(columns=["Continent", "Population (historical estimates)", "Code"])
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]

# Create arrays for GDP and happiness where happiness score is above 4.5
happiness = []
gdp = []
for row in df.iterrows():
    if row[1]['Cantril ladder score'] > 4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])

class linear_regression():
 
    def __init__(self, x_:list, y_:list) -> None:
        self.input = np.array(x_)
        self.target = np.array(y_)
        self.hmean = None
        self.hstd = None
        self.gmean = None
        self.gstd = None

    def preprocess(self):
        # Normalize the GDP (input feature)
        self.gmean = np.mean(self.input)
        self.gstd = np.std(self.input)
        x_train = (self.input - self.gmean) / self.gstd

        # Arrange in matrix format (intercept + normalized GDP)
        X = np.column_stack((np.ones(len(x_train)), x_train))

        # Normalize the happiness (target)
        self.hmean = np.mean(self.target)
        self.hstd = np.std(self.target)
        y_train = (self.target - self.hmean) / self.hstd

        # Arrange in matrix format
        Y = np.column_stack(y_train).T

        return X, Y

    def train_ols(self, X, Y):
        # Compute and return beta using OLS
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def partial_derivative(self, X, Y, beta):
        # Calculate the gradients
        gradients = 2 / len(X) * (X.T).dot(X.dot(beta) - Y)
        return gradients
    
    def predict(self, X_test, beta):
        # Predict using the current beta values
        Y_hat = X_test*beta.T  # Proper matrix multiplication
        return np.sum(Y_hat,axis=1)  # Flatten array to avoid shape issues
    
    def calculate_mse(self, Y_true, Y_pred):
        # Compute Mean Squared Error (MSE)
        return np.mean((Y_true - Y_pred) ** 2)

# Instantiate the linear regression object  
lr_ols = linear_regression(gdp, happiness)

# Preprocess the inputs (returns normalized X and Y)
X, Y = lr_ols.preprocess()

# Define learning rates and iteration counts
alpha_values = [0.09, 0.07, 0.03, 0.09, 0.005]
iteration_count = [50, 500, 1500, 10000, 25000]

# Initialize beta randomly
beta_init = np.random.randn(2, 1)

# Store final beta values, learning rates, and iteration counts
final_betas = []

# Perform gradient descent and store the results in final_betas
for alpha in alpha_values:
    for iter_count in iteration_count:
        # Initialize beta with the same beta_init for each combination
        beta = beta_init.copy()

        # Perform gradient descent for the given number of iterations
        for i in range(iter_count):
            temp_beta = lr_ols.partial_derivative(X, Y, beta)
            beta = beta - (alpha * temp_beta)

        # Store the final beta after the current run
        final_betas.append((alpha, iter_count, beta))

# Selected betas list
selected_betas = []

# Select the specific five lines based on the given criteria
criteria = [
    (0.005, 50, 0.8389),
    (0.03, 50, 0.4789),
    (0.005, 500, 0.4770),
    (0.09, 50, 0.4769),
    (0.005, 25000, 0.4769)
]

# Loop through final_betas to find and add matching lines
for alpha, iter_count, beta in final_betas:
    Y_predict = lr_ols.predict(X, beta)
    mse = lr_ols.calculate_mse(Y.ravel(), Y_predict)
    
    for crit_alpha, crit_iter, crit_mse in criteria:
        if alpha == crit_alpha and iter_count == crit_iter and round(mse, 4) == crit_mse:
            selected_betas.append((alpha, iter_count, beta, mse))

# Sort selected_betas by mse in descending order (from largest to smallest)
selected_betas = sorted(selected_betas, key=lambda x: x[3], reverse=True)

# Plot the dataset scatterplot
fig, ax = plt.subplots()
fig.set_size_inches((15, 8))

# Display the dataset (normalized GDP vs normalized Happiness)
ax.scatter(X[..., 1], Y.ravel(), label="Dataset")

# Plot the selected lines
for alpha, iter_count, beta, mse in selected_betas:
    # Predict the Y values (happiness) using the current beta
    Y_predict = lr_ols.predict(X, beta)
    
    # Print beta values, iterations, learning rate, and MSE
    print(f"alpha: {alpha}, iterations: {iter_count}, beta: {beta.ravel()}, MSE: {mse:.4f}")
    
    # Get the 1st column (features) from X to plot the line
    X_ = X[..., 1].ravel()

    # Plot the regression line with normalized GDP on x-axis and normalized happiness on y-axis
    ax.plot(X_, Y_predict, label=f"alpha: {alpha}, iter: {iter_count}, MSE: {mse:.4f}")

# Set plot labels and title
ax.set_xlabel("GDP per capita")
ax.set_ylabel("Happiness (Cantril ladder score)")
ax.set_title("Selected Gradient Descent Regression Lines - GDP vs Happiness")

# Add legend to show learning rates and iteration counts for each line
ax.legend()

# Show the plot
plt.show()



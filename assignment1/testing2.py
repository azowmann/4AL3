import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Define the linear regression class
class linear_regression:
    def __init__(self, x_: list, y_: list) -> None:
        self.input = np.array(x_)
        self.target = np.array(y_)

    def preprocess(self):
        # Normalize the input (GDP)
        self.gmean = np.mean(self.input)
        self.gstd = np.std(self.input)
        x_train = (self.input - self.gmean) / self.gstd

        # Arrange in matrix format (intercept + normalized GDP)
        X = np.column_stack((np.ones(len(x_train)), x_train))

        # Normalize the target (Happiness)
        self.hmean = np.mean(self.target)
        self.hstd = np.std(self.target)
        y_train = (self.target - self.hmean) / self.hstd

        # Arrange in matrix format
        Y = np.column_stack(y_train).T

        return X, Y

    def train_ols(self, X, Y):
        # Compute and return beta using OLS
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, X_test, beta):
        # Predict using the current beta values
        Y_hat = X_test*beta.T  # Proper matrix multiplication
        return np.sum(Y_hat,axis=1) # Flatten array to avoid shape issues

    def partial_derivative(self, X, Y, beta):
        # Calculate the gradients
        gradients = 2 / len(X) * (X.T).dot(X.dot(beta) - Y)
        return gradients
    
    def calculate_mse(self, Y_true, Y_pred):
        # Compute Mean Squared Error (MSE)
        return np.mean((Y_true - Y_pred) ** 2)

# Instantiate the linear regression object
lr_ols = linear_regression(gdp, happiness)

# Preprocess the inputs (returns normalized X and Y)
X, Y = lr_ols.preprocess()

# Compute beta using OLS
beta_ols = lr_ols.train_ols(X, Y)
Y_predict_ols = lr_ols.predict(X, beta_ols)

# Gradient Descent parameters
alpha = 0.005  # Best learning rate from your experiments
iter_count = 25000  # Best iteration count from your experiments
beta_init = np.random.randn(2, 1)

# Perform Gradient Descent
beta_gd = beta_init.copy()
for i in range(iter_count):
    temp_beta = lr_ols.partial_derivative(X, Y, beta_gd)
    beta_gd = beta_gd - (alpha * temp_beta)

Y_predict_gd = lr_ols.predict(X, beta_gd)

# Print both beta values and corresponding learning rate and epoch for GD
print(f"OLS Beta: {beta_ols.ravel()}")
print(f"GD Beta: {beta_gd.ravel()}, Learning Rate: {alpha}, Iterations: {iter_count}")

# Plot the dataset scatterplot
fig, ax = plt.subplots()
fig.set_size_inches((15, 8))

# Display the dataset (normalized GDP vs normalized Happiness)
ax.scatter(X[..., 1], Y.ravel(), label="Dataset")

# Plot OLS line
ax.plot(X[..., 1], Y_predict_ols, label="OLS Line", color='r')

# Plot GD line
ax.plot(X[..., 1], Y_predict_gd, label=f"GD Line (alpha: {alpha}, iter: {iter_count})", color='b')

# Set plot labels and title
ax.set_xlabel("Normalized GDP per capita")
ax.set_ylabel("Normalized Happiness (Cantril ladder score)")
ax.set_title("Regression Lines: OLS vs Gradient Descent")

# Add legend
ax.legend()

# Show the plot
plt.show()

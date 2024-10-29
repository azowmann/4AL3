import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(42)

# import data
hap_gdp_data = pd.read_csv("gdp-vs-happiness.csv")

# drop columns that will not be used
by_year = (hap_gdp_data[hap_gdp_data['Year'] == 2018]).drop(columns=["Continent", "Population (historical estimates)", "Code"])
# remove missing values from columns 
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]

#create np.array for gdp and happiness where happiness score is above 4.5
happiness = []
gdp = []
for row in df.iterrows():
    if row[1]['Cantril ladder score'] > 4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])

class linear_regression:

    def __init__(self, x_: list, y_: list) -> None:
        self.input = np.array(x_)
        self.target = np.array(y_)
        self.hmean = None
        self.hstd = None
        self.gmean = None
        self.gstd = None

    def preprocess(self):

        #normalize the values
        self.gmean = np.mean(self.input)
        self.gstd = np.std(self.input)
        x_train = (self.input - self.gmean) / self.gstd

        #arrange in matrix format
        X = np.column_stack((np.ones(len(x_train)), x_train))

        #normalize the values
        self.hmean = np.mean(self.target)
        self.hstd = np.std(self.target)
        y_train = (self.target - self.hmean) / self.hstd

        #arrange in matrix format
        Y = np.column_stack(y_train).T

        return X, Y

    def train_ols(self, X, Y):
        #compute and return beta for OLS
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def partial_derivative(self, X, Y, beta):
        # Calculate the gradients
        gradients = 2 / len(X) * (X.T).dot(X.dot(beta) - Y)
        return gradients

    def predict(self, X_test, beta):
        #predict using beta
        Y_hat = X_test*beta.T
        return np.sum(Y_hat, axis=1)

    def calculate_mse(self, Y_true, Y_pred):
        #calcualte mse
        return np.mean((Y_true - Y_pred) ** 2)

#instantiate the linear_regression class 
lr_ols = linear_regression(gdp, happiness)

# preprocess the inputs
X, Y = lr_ols.preprocess()

#define learning rates and iteration counts
alpha_values = [0.09, 0.07, 0.03, 0.09, 0.005]
iteration_count = [50, 500, 1500, 10000, 25000]

#initialize beta randomly
beta_init = np.random.randn(2, 1)

#store final beta values, learning rates, and iteration counts
final_betas = []

#perform gradient descent and store the results in final_betas
for alpha in alpha_values:
    for iter_count in iteration_count:
        #initialize beta with the same beta_init for each combination
        beta = beta_init.copy()

        #perform gradient descent for the given number of iterations
        for i in range(iter_count):
            temp_beta = lr_ols.partial_derivative(X, Y, beta)
            beta = beta - (alpha * temp_beta)

        #store the final beta after the current run
        final_betas.append((alpha, iter_count, beta))

#selected betas list
selected_betas = []

#select the specific five lines
criteria = [
    (0.005, 50, 0.8389),
    (0.03, 50, 0.4789),
    (0.005, 500, 0.4770),
    (0.09, 50, 0.4769),
    (0.005, 25000, 0.4769)
]

#loop through final_betas to find and add lines
for alpha, iter_count, beta in final_betas:
    Y_predict = lr_ols.predict(X, beta)
    mse = lr_ols.calculate_mse(Y.ravel(), Y_predict)
    
    for crit_alpha, crit_iter, crit_mse in criteria:
        if alpha == crit_alpha and iter_count == crit_iter and round(mse, 4) == crit_mse:
            selected_betas.append((alpha, iter_count, beta, mse))

#sort selected_betas by mse from largest to smallest
selected_betas = sorted(selected_betas, key=lambda x: x[3], reverse=True)

#plot the dataset scatterplot for gradient descent results
fig, ax = plt.subplots()
fig.set_size_inches((15, 8))

#display the dataset
ax.scatter(X[..., 1], Y.ravel(), label="Dataset")

#plot the selected lines
for alpha, iter_count, beta, mse in selected_betas:
    #predict the Y values (happiness) using the current beta
    Y_predict = lr_ols.predict(X, beta)
    
    #print beta values, iterations, learning rate, and MSE
    print(f"alpha: {alpha}, iterations: {iter_count}, beta: {beta.ravel()}, MSE: {mse:.4f}")
    
    #get the 1st column (features) from X to plot the line
    X_ = X[..., 1].ravel()

    #plot the regression line
    ax.plot(X_, Y_predict, label=f"alpha: {alpha}, iter: {iter_count}, MSE: {mse:.4f}")

#set plot labels and title
ax.set_xlabel("GDP per capita")
ax.set_ylabel("Happiness (Cantril ladder score)")
ax.set_title("Selected Gradient Descent Regression Lines - GDP vs Happiness")

#add legend
ax.legend()

#show the plot
plt.show()

#use the computed beta for prediction
beta_ols = lr_ols.train_ols(X, Y)
Y_predict_ols = lr_ols.predict(X, beta_ols)

#best case:
alpha = 0.005 
iter_count = 25000  
beta_init = np.random.randn(2, 1)

#perform GD
beta_gd = beta_init.copy()
for i in range(iter_count):
    temp_beta = lr_ols.partial_derivative(X, Y, beta_gd)
    beta_gd = beta_gd - (alpha * temp_beta)

Y_predict_gd = lr_ols.predict(X, beta_gd)

#print both beta values and corresponding learning rate and epoch for GD
print(f"OLS Beta: {beta_ols.ravel()}")
print(f"GD Beta: {beta_gd.ravel()}, Learning Rate: {alpha}, Iterations: {iter_count}")

#plot the dataset scatterplot for OLS vs GD
fig, ax = plt.subplots()
fig.set_size_inches((15, 8))

#display the dataset 
ax.scatter(X[..., 1], Y.ravel(), label="Dataset")

#plot OLS line
ax.plot(X[..., 1], Y_predict_ols, label="OLS Line", color='r')

#plot GD line
ax.plot(X[..., 1], Y_predict_gd, label=f"GD Line (alpha: {alpha}, iter: {iter_count})", color='b')

#set plot labels and title
ax.set_xlabel("Normalized GDP per capita")
ax.set_ylabel("Normalized Happiness (Cantril ladder score)")
ax.set_title("Regression Lines: OLS vs Gradient Descent")

#add legend
ax.legend()

#show the plot
plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

#PART 2

np.random.seed(42)

# import data
abalone_data = pd.read_csv('training_data.csv')

# drop columns that will not be used
abalone_data = abalone_data.drop(abalone_data.columns[0], axis=1)

#display scatterplot matrix
sn.pairplot(abalone_data, y_vars='Rings', x_vars=['Length', 'Diameter', 'Height', 
                                                      'Whole_weight', 'Shucked_weight', 
                                                      'Viscera_weight', 'Shell_weight'], height=2)
plt.show()

class Polynomial_Regression():
    
    def __init__(self, x1_:list, x2_:list, x3_:list, x4_:list, x5_:list, x6_:list, x7_:list, y_:list) -> None:
        # initialize input features
        self.input1 = np.array(x1_)
        self.input2 = np.array(x2_)
        self.input3 = np.array(x3_)
        self.input4 = np.array(x4_)
        self.input5 = np.array(x5_)
        self.input6 = np.array(x6_)
        self.input7 = np.array(x7_)
        self.target = np.array(y_)

    def preprocess(self,):

        #calculate means and standard deviations for normalization
        means = [np.mean(self.input1),
                 np.mean(self.input2),
                 np.mean(self.input3),
                 np.mean(self.input4),
                 np.mean(self.input5),
                 np.mean(self.input6),
                 np.mean(self.input7)]
        
        stds = [np.std(self.input1),
                np.std(self.input2),
                np.std(self.input3),
                np.std(self.input4),
                np.std(self.input5),
                np.std(self.input6),
                np.std(self.input7),]
        
        #normalize input features and target variables
        length = (self.input1 - means[0])/stds[0]
        diameter = (self.input2 - means[1])/stds[1]
        height = (self.input3 - means[2])/stds[2]
        whole_weight = (self.input4 - means[3])/stds[3]
        shucked_weight = (self.input5 - means[4])/stds[4]
        viscera_weight = (self.input6 - means[5])/stds[5]
        shell_weight = (self.input7 - means[6])/stds[6]
        rings_mean = np.mean(self.target)
        rings_std = np.std(self.target)
        rings = (self.target - rings_mean)/rings_std

        return length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings
    
    def matrix(self, x1, x2, x3, x4, x5, x6, x7, y):
        #create design matrix for polynomial regression
        X = np.column_stack([x1, x1 ** 2, x1 ** 3,
                            x2, x2 ** 2, x2 ** 3,
                            x3, x3 ** 2, x3 ** 3,
                            x4, x4 ** 2, x4 ** 3,
                            x5, x5 ** 2, x5 ** 3,
                            x6, x6 ** 2, x6 ** 3,
                            x7, x7 ** 2, x7 ** 3])
        
        Y = (np.array([y])).T

        return X, Y
    
    def train(self, X, Y):
        #train model using ols
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def predict(self, x1, x2, x3, x4, x5, x6, x7, beta):
        #prepare polynomial features for prediction
        x_poly = np.stack([x1, x1 ** 2, x1 ** 3,
                           x2, x2 ** 2, x2 ** 3,
                           x3, x3 ** 2, x3 ** 3,
                           x4, x4 ** 2, x4 ** 3,
                           x5, x5 ** 2, x5 ** 3,
                           x6, x6 ** 2, x6 ** 3,
                           x7, x7 ** 2, x7 ** 3], axis=1)
        
        return np.sum(x_poly.dot(beta), axis=1)

    def MSE(self, Y_true, Y_predicted):
        #calculate mse
        Y_true = Y_true.ravel()
        mse = np.mean((Y_true - Y_predicted) ** 2)
        return mse

#number of folds for k-fold cross-validation
k = 5

#create empty lists to hold data for each fold
length = [[] for _ in range(k)]
diameter = [[] for _ in range(k)]
height = [[] for _ in range(k)]
whole_weight = [[] for _ in range(k)]
shucked_weight = [[] for _ in range(k)]
viscera_weight = [[] for _ in range(k)]
shell_weight = [[] for _ in range(k)]
rings = [[] for _ in range(k)]

n_samples = len(abalone_data)

split_size = n_samples // k # size of each fold

pn_samples = len(abalone_data)

split_size = n_samples // k

# distribute data into k folds
for index, row in abalone_data.iterrows():
    partition_index = index // split_size
    if partition_index >= k:
        partition_index = k - 1

    #append features and target to the appropriate fold
    length[partition_index].append(float(row['Length']))
    diameter[partition_index].append(float(row['Diameter']))
    height[partition_index].append(float(row['Height']))
    whole_weight[partition_index].append(float(row['Whole_weight']))
    shucked_weight[partition_index].append(float(row['Shucked_weight']))
    viscera_weight[partition_index].append(float(row['Viscera_weight']))
    shell_weight[partition_index].append(float(row['Shell_weight']))
    rings[partition_index].append(int(row['Rings']))


# prepare cross-validation
folds = [0, 1, 2, 3, 4]
train = []
avg_mse= []
final_beta = []

#cross validation loop
for i in range(len(folds)):

    #split for training sets and validation sets
    test_length = length[folds[i]]
    test_diameter = diameter[folds[i]]
    test_height = height[folds[i]]
    test_whole_weight = whole_weight[folds[i]]
    test_shucked_weight = shucked_weight[folds[i]]
    test_viscera_weight = viscera_weight[folds[i]]
    test_shell_weight = shell_weight[folds[i]]
    test_rings = rings[folds[i]]

    #collect training from other folds
    train_length = [item for j in range(len(folds)) if j != i for item in length[j]]
    train_diameter = [item for j in range(len(folds)) if j != i for item in diameter[j]]
    train_height = [item for j in range(len(folds)) if j != i for item in height[j]]
    train_whole_weight = [item for j in range(len(folds)) if j != i for item in whole_weight[j]]
    train_shucked_weight = [item for j in range(len(folds)) if j != i for item in shucked_weight[j]]
    train_viscera_weight = [item for j in range(len(folds)) if j != i for item in viscera_weight[j]]
    train_shell_weight = [item for j in range(len(folds)) if j != i for item in shell_weight[j]]
    train_rings = [item for j in range(len(folds)) if j != i for item in rings[j]]


    #create and train the polynomial regression model
    pr = Polynomial_Regression(train_length, train_diameter, train_height, train_whole_weight,
                               train_shucked_weight, train_viscera_weight, train_shell_weight
                               , train_rings)
    
    x1, x2, x3, x4, x5, x6, x7, y = pr.preprocess()
    X_, Y_ = pr.matrix(x1, x2, x3, x4, x5, x6, x7, y)

    #train the model and obtain coefficients
    beta = pr.train(X_, Y_)

    final_beta.append(beta)

    #preprocess the test data
    pr_test = Polynomial_Regression(test_length, test_diameter, test_height, test_whole_weight,
                                    test_shucked_weight, test_viscera_weight, test_shell_weight,
                                    test_rings)
    
    x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, y_test = pr_test.preprocess()

    #make predictions using the test data
    Y_predict = pr_test.predict(x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, beta)

    #calculate mse for this fold
    mse = pr_test.MSE(y_test, Y_predict)
    avg_mse.append(mse)

    #print results for current fold
    print(f'fold {i + 1}, mse: {mse}, beta: {beta.flatten()}') 

#calculate and print the final average MSE across all folds
final = np.mean(np.array(avg_mse))
print(f'final average mse {final}')

#p  repare data for final visualization
plot_length = [item for sublist in length for item in sublist]
plot_diameter = [item for sublist in diameter for item in sublist]
plot_height = [item for sublist in height for item in sublist]
plot_whole_weight = [item for sublist in whole_weight for item in sublist]
plot_shucked_weight = [item for sublist in shucked_weight for item in sublist]
plot_viscera_weight = [item for sublist in viscera_weight for item in sublist]
plot_shell_weight = [item for sublist in shell_weight for item in sublist]
plot_rings = [item for sublist in rings for item in sublist]

pr_plot = Polynomial_Regression(plot_length, plot_diameter, plot_height, plot_whole_weight,
                                plot_shucked_weight, plot_viscera_weight, plot_shell_weight,
                                plot_rings)

#preprocess the data for plotting
x1plot, x2plot, x3plot, x4plot, x5plot, x6plot, x7plot, yplot = pr_plot.preprocess()

#create subplots for visualization
fig, axes = plt.subplots(2, 4, figsize=(20, 10))


#titles and data for each subplot
titles = ['Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight',
          'Viscera Weight', 'Shell Weight']

x_plots = [x1plot, x2plot, x3plot, x4plot, x5plot, x6plot, x7plot]

axes = axes.flatten()

#define colors for each fold's regression line
colours = ['pink', 'blue', 'grey', 'green', 'purple']

#plot data points and regression lines for each feature
for i in range(7):
    axes[i].scatter(x_plots[i], yplot, label = "Data Points")
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(titles[i])
    axes[i].set_ylabel('Rings')

    #plot regression lines for each fold
    for u in range(5):
        axes[i].plot(x_plots[i], pr_plot.predict(x1plot, x2plot, x3plot, x4plot, x5plot, x6plot, x7plot, final_beta[u]), color=colours[u], label=f'fold {u+1} betas')

    axes[i].legend()

axes[-1].axis('off')

#set figure title
fig.suptitle('Rings of abalone vs features', fontsize=16)

plt.tight_layout()
#display plot
plt.show()



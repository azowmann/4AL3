import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

negfeatureshistorical2010 = np.load('./data2010/neg_features_historical.npy')
negfeaturesmaintimechange2010p1= np.load('./data2010/neg_features_main_timechange.npy')[:, :18]
negfeaturesmaintimechange2010p2 = np.load('./data2010/neg_features_main_timechange.npy')[:, 18:91]
negfeaturesmaxmin2010 = np.load('./data2010/neg_features_maxmin.npy')
posfeatureshistorical2010 = np.load('./data2010/pos_features_historical.npy')
posfeaturesmaintimechange2010p1 = np.load('./data2010/pos_features_main_timechange.npy')[:, :18]
posfeaturesmaintimechange2010p2 = np.load('./data2010/pos_features_main_timechange.npy')[:, 18:91]
posfeaturesmaxmin2010 = np.load('./data2010/pos_features_maxmin.npy')

negfeatureshistorical2020 = np.load('./data2020/neg_features_historical.npy')
negfeaturesmaintimechange2020p1 = np.load('./data2020/neg_features_main_timechange.npy')[:, :18]
negfeaturesmaintimechange2020p2 = np.load('./data2020/neg_features_main_timechange.npy')[:, 18:91]
negfeaturesmaxmin2020 = np.load('./data2020/neg_features_maxmin.npy')
posfeatureshistorical2020 = np.load('./data2020/pos_features_historical.npy')
posfeaturesmaintimechange2020p1 = np.load('./data2020/pos_features_main_timechange.npy')[:, :18]
posfeaturesmaintimechange2020p2 = np.load('./data2020/pos_features_main_timechange.npy')[:, 18:91]
posfeaturesmaxmin2020 = np.load('./data2020/pos_features_maxmin.npy')

posfeatures2010 = [posfeaturesmaintimechange2010p1, posfeaturesmaintimechange2010p2, posfeatureshistorical2010, posfeaturesmaxmin2010]
negfeatures2010 = [negfeaturesmaintimechange2010p1, negfeaturesmaintimechange2010p2, negfeatureshistorical2010, negfeaturesmaxmin2010]

class my_svm():
    # __init__() function should initialize all your variables
    def __init__(self, posfeatures2010: list, negfeatures2010: list):
        
        self.pfeatures = []
        self.nfeatures = []

        self.pfeatures.append(posfeatures2010[0])
        self.pfeatures.append(posfeatures2010[1])
        self.pfeatures.append(posfeatures2010[2])
        self.pfeatures.append(posfeatures2010[3])

        self.nfeatures.append(negfeatures2010[0])
        self.nfeatures.append(negfeatures2010[1])
        self.nfeatures.append(negfeatures2010[2])
        self.nfeatures.append(negfeatures2010[3])

        self.feature_map = {
            "FS-I": 0,
            "FS-II": 1,
            "FS-III": 2,
            "FS-IV": 3
        }
        
    # preprocess() function:
    #  1) normalizes the data, 
    #  2) removes missing values
    #  3) assign labels to target 
    def preprocess(self,):

        scaling = StandardScaler()

        for i in range(len(self.pfeatures)): 
            self.pfeatures[i] = self.pfeatures[i][~np.isnan(self.pfeatures[i]).any(axis=1), :]
            self.pfeatures[i] = scaling.fit_transform(self.pfeatures[i])
       
        for j in range(len(self.nfeatures)):
            self.nfeatures[j] = self.nfeatures[j][~np.isnan(self.nfeatures[j]).any(axis=1), :]
            self.nfeatures[j] = scaling.fit_transform(self.nfeatures[j])


    # feature_creation() function takes as input the features set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates 2 D array of corresponding features 
    # for both positive and negative observations.
    # this array will be input to the svm model
    # For instance, if the input is FS-I, the output is a 2-d array with features corresponding to 
    # FS-I for both negative and positive class observations
    def feature_creation(self, fs_value: list):
        pos_array = []
        neg_array = []

        for fs in fs_value:
            index = self.feature_map[fs]          
            pos_array.append(self.pfeatures[index])       #+ve features
            neg_array.append(self.nfeatures[index])       #-ve features

        pos_features = np.concatenate(pos_array, axis = 1)
        neg_features = np.concatenate(neg_array, axis = 1)

        X_ = np.concatenate([pos_features, neg_features], axis = 0)

        y_pos = np.ones(pos_features.shape[0])
        y_neg = np.zeros(neg_features.shape[0])
        Y_ = np.concatenate([y_pos, y_neg], axis=0)

        return X_, Y_
    
    # cross_validation() function splits the data into train and test splits,
    # Use k-fold with k=10
    # the svm is trained on training set and tested on test set
    # the output is the average accuracy across all train test splits.
    def cross_validation(self, x_data, y_data) -> tuple:
        kf = KFold(n_splits = 10, shuffle = True)
        tss_scores = []
        confusion_matrices = []

        for train_index, test_index in kf.split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            y_predict = self.training(x_train, y_train, x_test)

            measure = self.tss(y_test, y_predict)
            tss_scores.append(measure)

            cm = confusion_matrix(y_test, y_predict)
            confusion_matrices.append(cm)
        
        return np.mean(tss_scores), np.std(tss_scores), tss_scores, confusion_matrices
        # call training function
        # call tss function
        
    #training() function trains a SVM classification model on input features and corresponding target
    def training(self, x_train, y_train, x_test):
        svmmodel_rbf = SVC(kernel="rbf")
        svmmodel_rbf.fit(x_train, y_train)
        y_pred = svmmodel_rbf.predict(x_test)
        return y_pred
    
    # tss() function computes the accuracy of predicted outputs (i.e target prediction on test set)
    # using the TSS measure given in the document
    def tss(self, y_true, y_pred):
        true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_true, y_pred).ravel()
        tss = (true_pos / (true_pos + false_neg)) - (false_pos / (false_pos + true_neg))
        return tss
    
def power_set(s):
    if not s:
        return [[]]
    
    first = s[0]
    remaining_power_set = power_set(s[1:])

    with_first = [[first] + i for i in remaining_power_set]

    return remaining_power_set + with_first

def plot_confusion_matrices(confusion_matrices):
    plt.figure(figsize=(15, 15))

    for i, cm in enumerate(confusion_matrices, 1):
        plt.subplot(5, 2, i)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f'Confusion Matrix - Fold {i}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()


def plot_all_confusion_matrices(all_confusion_matrices, feature_combinations):
    total_combos = len(feature_combinations)
    fig, axes = plt.subplots(total_combos, 10, figsize=(25, 3 * total_combos))

    # Iterate through each feature combination and plot all 10 folds' confusion matrices
    for combo_index, (combo_name, confusion_matrices) in enumerate(all_confusion_matrices.items()):
        for fold_index, cm in enumerate(confusion_matrices):
            ax = axes[combo_index, fold_index]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            if fold_index == 0:
                ax.set_ylabel(f'{combo_name}', rotation=90, size='large')  # Label the rows with the combo name
            ax.set_title(f'Fold {fold_index + 1}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

# feature_experiment() function executes experiments with all 4 feature sets.
# svm is trained (and tested) on 2010 dataset with all 4 feature set combinations
# the output of this function is : 
#  1) TSS average scores (mean std) for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e 10)
#
# Above 3 charts are produced for all feature combinations
#  4) The function prints the best performing feature set combination
def feature_experiment():
    posfeatures2010 = [posfeaturesmaintimechange2010p1, posfeaturesmaintimechange2010p2, posfeatureshistorical2010, posfeaturesmaxmin2010]
    negfeatures2010 = [negfeaturesmaintimechange2010p1, negfeaturesmaintimechange2010p2, negfeatureshistorical2010, negfeaturesmaxmin2010]  

    svmClass = my_svm(posfeatures2010, negfeatures2010)
    svmClass.preprocess()

    combinations = power_set(["FS-I", "FS-II", "FS-III", "FS-IV"])
    combinations.remove([])

    all_confusion_matrices = {}
    tss_scores_dict = {}

    for i in combinations:
        X_, Y_ = svmClass.feature_creation(i)
        mean, std, tss, confusion_matrices = svmClass.cross_validation(X_, Y_)
        tss_scores_dict[str(i)] = tss
        all_confusion_matrices[str(i)] = confusion_matrices
    
        print(f'{i}: mean - {mean}, std - {std}')

    # Plot all confusion matrices for all feature set combinations
    plot_all_confusion_matrices(all_confusion_matrices, combinations)

    colouring = [
        '#DC143C', '#4169E1', '#228B22', '#FFD700',
        '#FF1493', '#FF8C00', '#800080', '#40E0D0',
        '#708090', '#FF6347', '#1E90FF', '#00FF7F',
        '#EE82EE', '#D2691E', '#F4A460']

    plt.figure(figsize=(15,9))

    for index, (i, tss) in enumerate(tss_scores_dict.items()):
        colour = colouring[index % len(colouring)]
        plt.plot(range(1, len(tss) + 1), tss, marker='o', label=f"{i}", color=colour)

    plt.title('TSS Scores for Each Feature Combo')
    plt.xlabel('Fold Number')
    plt.ylabel('TSS Score')
    plt.xticks(range(1, max(len(tss) for tss in tss_scores_dict.values()) + 1))
    plt.legend()
    plt.grid()
    plt.show()
# data_experiment() function executes 2 experiments with 2 data sets.
# svm is trained (and tested) on both 2010 data and 2020 data
# the output of this function is : 
#  1) TSS average scores for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e. 10)
# above 3 charts are produced for both datasets
# feature set for this experiment should be the 
# best performing feature set combination from feature_experiment()

def data_experiment():
    
    posfeatures2010 = [posfeaturesmaintimechange2010p1, posfeaturesmaintimechange2010p2, posfeatureshistorical2010, posfeaturesmaxmin2010]
    negfeatures2010 = [negfeaturesmaintimechange2010p1, negfeaturesmaintimechange2010p2, negfeatureshistorical2010, negfeaturesmaxmin2010]
    posfeatures2020 = [posfeaturesmaintimechange2020p1, posfeaturesmaintimechange2020p2, posfeatureshistorical2020, posfeaturesmaxmin2020]
    negfeatures2020 = [negfeaturesmaintimechange2020p1, negfeaturesmaintimechange2020p2, negfeatureshistorical2020, negfeaturesmaxmin2020]

    featureSet = ["FS-I", "FS-III", "FS-IV"]

    svmClass2010 = my_svm(posfeatures2010, negfeatures2010)
    svmClass2010.preprocess()

    svmClass2020 = my_svm(posfeatures2020, negfeatures2020)
    svmClass2020.preprocess()

    colouring2 = ['#8A2BE2','#FF4500']

    x_2010, y_2010 = svmClass2010.feature_creation(featureSet)
    x_2020, y_2020 = svmClass2020.feature_creation(featureSet)

    mean2010, std2010, tss2010, _ = svmClass2010.cross_validation(x_2010, y_2010)
    mean2020, std2020, tss2020, _ = svmClass2020.cross_validation(x_2020, y_2020)

    print(f'{featureSet} - 2010: mean - {mean2010}, std - {std2010}')
    print(f'{featureSet} - 2020: mean - {mean2020}, std - {std2020}')

    y_pred_2010 = svmClass2010.training(x_2010, y_2010, x_2010)
    y_pred_2020 = svmClass2020.training(x_2020, y_2020, x_2020)

    plt.figure(figsize=(15,9))
    plt.plot(range(1, len(tss2010) + 1), tss2010, marker='o', label='2010-2015', color=colouring2[0])
    plt.plot(range(1, len(tss2020) + 1), tss2020, marker='o', label='2020-2024', color=colouring2[1])

    plt.title('TSS Scores for bets feature set of 2010 - 2015 vs 2020 - 2024') # title 
    plt.xlabel('Fold Number') # fold number 
    plt.ylabel('TSS Score') # TSS Score 
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid()
    plt.show()

# below should be your code to call the above classes and functions
# with various combinations of feature sets
# and both datasets

feature_experiment()
data_experiment()








        




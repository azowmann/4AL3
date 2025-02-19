import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

class svm_():
    def __init__(self, learning_rate, epoch, C_value, X, Y):

        # initialize variables
        self.input = X
        self.target = Y
        self.learning_rate =learning_rate
        self.epoch = epoch
        self.C = C_value

        # initialize weight matrix based on number of features 
        # bias and weights are merged together as one matrix
        # you should try random initialization
     
        self.weights = np.zeros(X.shape[1])

    def pre_process(self, flag = True):

        # using StandardScaler to normalize the input
        scalar = StandardScaler().fit(self.input)
        X_ = scalar.transform(self.input)

        Y_ = self.target 

        if flag:
            print("splitting dataset into train and test sets:")
            X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=71)

            return  X_train, X_test, y_train, y_test
        else:
            return X_, Y_
    
    # the function return gradient for 1 instance -
    # stochastic gradient decent

    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y * np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    def compute_loss(self,Y,X):
        # calculate hinge loss
        loss=0.0
        regularization = 0.5 * np.dot(self.weights, self.weights)
       
        # hinge loss implementation - start

        # calculate margin
        for i in range(len(Y)):
            margin = Y[i] * np.dot(self.weights, X[i])
            hinge_loss = max(0, 1 - margin.item()) 
            loss += hinge_loss
        # apply regularization to loss
        loss = self.C * loss + regularization

        # hinge loss implementatin - end

        return loss

    def stochastic_gradient_descent(self, X, Y, X_val, Y_val, threshold):

        early_stopped = False           # flag to track if early stopping condition has been met
        previous_loss = float('inf')    # set previous loss to very big value 
        epoch_index = 0
        w = np.zeros(X.shape[1])        # capture the weights before overfitting
        validation_loss = []
        training_loss = []
        epochs = []

        # execute the stochastic gradient descent function for defined epochs
        for i in range(self.epoch):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y)

            for j, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[j])
                # update weights after each iteration
                self.weights = self.weights - (self.learning_rate * gradient)

            #  calculate loss for current interation 
            loss = self.compute_loss(Y, X) 
            vLoss = self.compute_loss(Y_val, X_val) 
 
            # every 10% of the epochs, print the loss
            if i % (self.epoch // 10) == 0:
                validation_loss.append(vLoss)
                training_loss.append(loss)
                epochs.append(i)
                print("Epoch is: {} and Loss is : {}".format(i, loss))

            # check for early stopping condition
            if not early_stopped:
                if abs(previous_loss - loss) < threshold:
                    print(f"Early stopping triggered at epoch {i}")
                    epoch_index = i
                    w = self.weights
                    early_stopped = True

            previous_loss = loss

        print("Training ended.")

        return validation_loss, training_loss, epochs, epoch_index, w

    def mini_batch_gradient_descent(self, X, Y, X_val, Y_val, threshold, batch_size):

        previous_loss = float('inf') # set previous loss to very big value 
        validation_loss = []
        training_loss = []
        epochs = []
        epoch_index = 0
        w = np.zeros(X.shape[1]) # capture the weights before overfitting
        early_stopped = False  # flag to track if early stopping condition has been met

        # execute mini-batch gradient descent for the defined epochs
        for i in range(self.epoch):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y)

            # process the data in mini-batches of size `batch_size`
            for j in range(0, len(features), batch_size):
                # extract mini-batch
                X_batch = features[j:j + batch_size]
                Y_batch = output[j:j + batch_size]

                # compute the gradient for the mini-batch
                gradient = np.zeros_like(self.weights)
                for k in range(len(X_batch)):
                    gradient += self.compute_gradient(X_batch[k], Y_batch[k])
                gradient /= len(X_batch)  # average gradient for the mini-batch

                self.weights -= self.learning_rate * gradient

            # calculate loss for the current iteration

            # calculate loss for current iteration 
            loss = self.compute_loss(Y, X) 
            vLoss = self.compute_loss(Y_val, X_val)  

            # every 10% of the epochs, print the loss
            if i % (self.epoch // 10) == 0:
                print(f"Epoch: {i}, Loss: {loss}")
                # add all loses and epochs for graphing purposes 
                validation_loss.append(vLoss)
                training_loss.append(loss)
                epochs.append(i)

            # early stopping condition (check for convergence)
            if not early_stopped:
                if abs(previous_loss - loss) < threshold:
                    print(f"Early stopping triggered at epoch {i}")
                    epoch_index = i
                    w = self.weights
                    early_stopped = True

            previous_loss = loss

        print("Training completed.")

        return validation_loss, training_loss, epochs, epoch_index, w

    def sampling_strategy(self, svm, X_labeled, Y_labeled, X_unlabeled, Y_unlabeled):
        # convert to list to allow .append
        X_labeled = list(X_labeled)
        Y_labeled = list(Y_labeled)
        
        losses = []

        # Calculate loss for each sample in the unlabeled set
        for i in range(len(X_unlabeled)):
            loss = svm.compute_loss(Y_unlabeled[i], X_unlabeled[i].reshape(1, -1))
            losses.append((loss, i)) 

        lowest_loss_index = min(losses, key=lambda x: x[0])[1]

        # add selected sample to labeled set
        X_labeled.append(X_unlabeled[lowest_loss_index])
        Y_labeled.append(Y_unlabeled[lowest_loss_index])

        # remove selected sample from unlabeled set
        X_unlabeled = np.delete(X_unlabeled, lowest_loss_index, axis=0)
        Y_unlabeled = np.delete(Y_unlabeled, lowest_loss_index, axis=0)

        # convert back to numpy arrays
        X_labeled = np.array(X_labeled)
        Y_labeled = np.array(Y_labeled)

        # return updated labeled and unlabeled sets
        return X_labeled, Y_labeled, X_unlabeled, Y_unlabeled

    def predict(self,X_test,Y_test, GD):

        print(f'Weights used for prediction (from most optimized model): {self.weights}')
        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
        
        print(f'Results for {GD}:')

        #compute accuracy
        accuracy = accuracy_score(Y_test, predicted_values)
        print("Accuracy on test dataset: {}".format(accuracy))

        #compute precision
        precision = precision_score(Y_test, predicted_values)
        print("Precision on test dataset: {}".format(precision))
        #compute precision

        #compute recall
        recall = recall_score(Y_test, predicted_values)
        print("Recall on test dataset: {}".format(recall))
        #compute recall

        return predicted_values, accuracy, precision, recall

def part_1(X, Y, X_predict, Y_predict): 

    print("Part 1")
    #model parameters - try different ones
    C = 0.01 
    learning_rate = 0.001 
    epoch = 150
    threshold = 0.001
  
    #instantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X,Y=Y)

    #pre process data
    X_Processed_Train, X_Processed_Val, Y_Processed_Train, Y_Processed_Val = my_svm.pre_process()

    # train model
    val_loss, training_loss, epoch_list, index, w = my_svm.stochastic_gradient_descent(X_Processed_Train, Y_Processed_Train, X_Processed_Val, Y_Processed_Val, threshold)

    X_predict, Y_predict = my_svm.pre_process(flag=False) # preprocess test data for prediction

    my_svm.weights = w # weights before overfitting occurs 

    _, _, _, _ = my_svm.predict(X_predict, Y_predict, 'Stochastic Gradient Descent') # predict on validation set

    val_loss = np.squeeze(val_loss)
    training_loss = np.squeeze(training_loss)
    epoch_list = np.squeeze(epoch_list)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, training_loss, label="Training Loss")
    plt.plot(epoch_list, val_loss, label="Validation Loss")
    plt.axvline(index, color='r', linestyle='--', label=f"Early Stopping at Epoch {index}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss for Stochastic Gradient Descent")
    plt.legend()
    plt.show()

def part_2(X,Y, X_predict, Y_predict):

    print("Part 2")
    #model parameters - try different ones
    C = 0.01 
    learning_rate = 0.001 
    epoch = 150
    threshold = 0.001
  
    # instantiate support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X,Y=Y)

    # pre process data
    X_Processed_Train, X_Processed_Val, Y_Processed_Train, Y_Processed_Val = my_svm.pre_process()

    # train model
    val_loss, training_loss, epoch_list, index, w = my_svm.mini_batch_gradient_descent(X_Processed_Train, Y_Processed_Train, X_Processed_Val, Y_Processed_Val, threshold, 5)

    my_svm.weights = w # weights before overfitting occurs 
     
    X_predict, Y_predict = my_svm.pre_process(flag=False) # prepreocess test data for prediction

    _, _, _, _ = my_svm.predict(X_predict, Y_predict, 'Mini-Batch Gradient Descent') # predict on validation set

    val_loss = np.squeeze(val_loss)
    training_loss = np.squeeze(training_loss)
    epoch_list = np.squeeze(epoch_list)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, training_loss, label="Training Loss")
    plt.plot(epoch_list, val_loss, label="Validation Loss")
    plt.axvline(index, color='r', linestyle='--', label=f"Early Stopping at Epoch {index}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss for Mini Batch Gradient Descent")
    plt.legend()
    plt.show()

def part_3(X,Y, X_predict, Y_predict):

    print("Part 3")
    #model parameters - try different ones
    C = 0.01 
    learning_rate = 0.001 
    epoch = 150
    threshold = 0.001

    validationList = []
    trainingList = []
  
    #instantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X,Y=Y)

    # pre process data
    X_initial_Processed, Y_initial_Processed = my_svm.pre_process(flag=False)
    X_predict, Y_predict = my_svm.pre_process(flag=False) # prepreocess test data for prediction

    #initial split - validation and training 
    X_initial = X_initial_Processed [:100]
    Y_initial = Y_initial_Processed [:100]
    X_initial_unlabeled = X_initial_Processed [100:]
    Y_initial_unlabeled = Y_initial_Processed [100:]

    X_train, X_val, y_train, y_val = train_test_split(X_initial, Y_initial, test_size=0.2, random_state=71)

    prevLoss = 1
    loss = 2
    sample = 0
    #train - figure when to stop
    while abs(prevLoss - loss) > 0.0001:

        my_svm.weights = np.zeros(X.shape[1]) # reset weights for each training iteration

        val_loss, training_loss, _, _, w = my_svm.stochastic_gradient_descent(X_train, y_train, X_val, y_val, threshold)

        validationList.append(val_loss)
        trainingList.append(training_loss)
        
        my_svm.weights = w # weights before overfitting occurs

        prevLoss = loss
        loss = my_svm.compute_loss(Y_predict, X_predict) # compute loss

        X_initial, Y_initial, X_initial_unlabeled, Y_initial_unlabeled = my_svm.sampling_strategy(my_svm, X_initial, Y_initial, X_initial_unlabeled, Y_initial_unlabeled) # get updated data

        sample += 1 # sample added

        X_train, X_val, y_train, y_val = train_test_split(X_initial, Y_initial, test_size=0.2, random_state=71) # split data again

    _, _, _, _ = my_svm.predict(X_predict, Y_predict, 'Active Learning Model')

    print(f'Initial training of 150 samples. Number of samples taken for satisfactory model: {sample}')

    # plot all losses
    plt.figure(figsize=(10, 6))

    # Loop over each list of losses in trainingList and validationList
    for i, (train_loss, val_loss) in enumerate(zip(trainingList, validationList), start=1):
        epochs = range(len(train_loss))  # Generate epochs based on the length of each loss list
        plt.plot(epochs, train_loss, label=f"Training Loss {i}")
        plt.plot(epochs, val_loss, label=f"Validation Loss {i}")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Across Multiple Sample Testing")
    plt.legend()
    plt.show()


#Load datapoints
print("Loading dataset...")
data = pd.read_csv('data1.csv')

# drop first and last column 
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

#segregate inputs and targets

#inputs
X = data.iloc[:, 1:]

#add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()

#converting categorical variables to integers 
# - this is same as using one hot encoding from sklearn
#benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
#transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

#split data, validation and training, validation will be used for prediction
X_test, X_test_predict, y_test, y_test_predict = train_test_split(X_features, Y_target, test_size=0.2, random_state=71)

part_1(X_test, y_test, X_test_predict, y_test_predict) # part 1

part_2(X_test, y_test, X_test_predict, y_test_predict) # part 2 

part_3(X_test, y_test, X_test_predict, y_test_predict) # part 3

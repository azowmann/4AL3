import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor

# Define CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv10 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv10(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv16(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to load FashionMNIST dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

# Load data using torchvision
transform = ToTensor()
train_dataset = FashionMNIST(root='./FashionMNIST', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='./FashionMNIST', train=False, download=True, transform=transform)

# Parameters
batch_size = 64
learning_rate = 0.001
random_seed = 71
test_size = 0.2
epochs = 15

# Split data into training and validation sets
num_train = len(train_dataset)
indices = torch.randperm(num_train)
split = int(np.floor(test_size * num_train))
train_idx, val_idx = indices[split:], indices[:split]

train_loader = DataLoader(torch.utils.data.Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = CNN()
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = lossFunction(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = lossFunction(outputs, labels)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

#------------------------------------------------------------------------------------#

def custom_train_test_split(X, y, test_size=0.2, random_seed=71):
    # Set the random seed for reproducibility, if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get the number of samples
    num_samples = len(X)
    
    # Shuffle indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Determine the split index
    split_index = int(num_samples * (1 - test_size))
    
    # Split the data
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


#Part 2 - Logistical Regression
data = pd.read_csv("./compas-scores.csv")

data['label'] = data['score_text'].apply(lambda s: 1 if s == 'High' else 0)

#step 2 - cleaning the data
# jail in and jail out
# Convert columns to datetime
data['c_jail_in'] = pd.to_datetime(data['c_jail_in'])
data['c_jail_out'] = pd.to_datetime(data['c_jail_out'])

# Calculate jail time in days
data['c_jail_time'] = (data['c_jail_out'] - data['c_jail_in']).dt.days

data['r_jail_in'] = pd.to_datetime(data['r_jail_in'])
data['r_jail_out'] = pd.to_datetime(data['r_jail_out'])

# Calculate jail time in days
data['r_jail_time'] = (data['r_jail_out'] - data['r_jail_in']).dt.days

columns_to_drop = [
    'name', 'first', 'last', 'id', 
    'c_case_number', 'r_case_number', 'vr_case_number', 
    'dob',  
    'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 
    'r_days_from_arrest' , 'r_offense_date' ,'r_charge_desc' , 
    'c_arrest_date', 'c_offense_date', 'compas_screening_date', 'c_charge_desc', 
    'v_type_of_assessment', 'type_of_assessment', 
    'num_r_cases', 'num_vr_cases', 
    'is_recid', 'is_violent_recid',
    'v_screening_date', 'screening_date',
    'c_jail_in', 'c_jail_out', 'r_jail_in', 'r_jail_out',
    'score_text', 
    
]

# Drop the columns and clean out na values
dC= data.drop(columns=columns_to_drop).dropna()

# scale continous variable columns
continuous_columns = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_jail_time', 'r_jail_time', 'days_b_screening_arrest', 'c_days_from_compas', 'v_decile_score', 'decile_score.1']

# apply one-hot encoding to categorical data
categorical_columns = ["sex", "age_cat", "race", "c_charge_degree", "r_charge_degree", "v_score_text"]

# ensure no duplicates in `continuous_columns`
continuous_columns = list(set(continuous_columns))  # Remove duplicates

# extract the continuous data from the DataFrame
continuous_data = dC[continuous_columns]

# apply StandardScaler
scaler = sklearn.preprocessing.StandardScaler()
scaled_continuous_data = scaler.fit_transform(continuous_data)

# convert scaled data back to DataFrame and ensure index alignment
scaled_continuous_df = pd.DataFrame(scaled_continuous_data, columns=continuous_columns, index=dC.index)

# replace the original continuous columns in the DataFrame
dC[continuous_columns] = scaled_continuous_df

# initialize OneHotEncoder
encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False, drop='first')  # `drop='first'` avoids multicollinearity by dropping the first category
encoded_array = encoder.fit_transform(dC[categorical_columns]) # fit and transform the categorical columns
encoded_columns = encoder.get_feature_names_out(categorical_columns) # convert the encoded array back into a DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns) # new encoded DataFrame

# drop original categorical columns and merge the encoded columns
dC = dC.drop(columns=categorical_columns).reset_index(drop=True)
dC = pd.concat([dC, encoded_df], axis=1) # merge the encoded columns

# dC.to_csv('cleaned_file.csv', index=False) # testing

# convert to tensor for pytorch, similar to part 1
X = dC.drop(columns='label')  # features
y = dC['label'] # labels

X_tensor = torch.tensor(X.values, dtype=torch.float32) # use .values to obtain the numpy array instead of the pandas DataFrame
y_tensor = torch.tensor(y.values, dtype=torch.long) # use .values to obtain the numpy array instead of the pandas DataFrame

# step 3, building logical regression class and parameters 
class LRM(nn.Module):
    def __init__(self, input_dim):
        super(LRM, self).__init__() # to inherit the properties of the parent class nn.Module for pytorch
        # logistic regression model with a single layer (neural network with 1 layer)
        self.fc = nn.Linear(input_dim, 1) 

    def architecture(self, x):
        return torch.sigmoid(self.fc(x))  # sigmoid function

model = LRM(input_dim=X_tensor.shape[1]) # input_dim is number of features

test_size = 0.2 # 20/80 split
learningRate = 0.001
epochs = 50

BCE = nn.BCELoss() # binary cross entropy loss, binary since dealing with 0 and 1, better optimized that way
sGD = torch.optim.SGD(model.parameters(), lr=learningRate) # stochastic gradient descent for training

# split our data using part 1's split function
train_x, test_x, train_y, test_y = custom_train_test_split(X_tensor, y_tensor, test_size=0.2, random_seed=42)

# part 4: Training the model
print("\nUnbalanced Data:")
for epoch in range(epochs):
    model.train()
    sGD.zero_grad()
    y_pred = model.architecture(train_x.clone().detach().float()) # prediction, forward pass
    loss = BCE(y_pred.squeeze(), train_y.float().detach())  # calculate loss
    loss.backward() # since technically still a nn, backpropagation
    sGD.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# part 5: Testing the model, reporting accurancy and equialized odds
def customConfusionMatrix(y_true, y_pred):

    y_true = y_true.numpy() # orginally pytorch object

    tp = ((y_true == 1) & (y_pred == 1)).sum()  # true positive
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # false negative
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # false positive
    tn = ((y_true == 0) & (y_pred == 0)).sum()  # true negative

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # true Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # false Positive Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0 # accuracy
    return tpr, fpr, accuracy

# Evaluate model on test set
model.eval()
with torch.no_grad():
    y_test_pred = model.architecture(test_x.clone().detach().float())
    y_test_pred_class = (y_test_pred.squeeze() >= 0.5).int()  # Convert probabilities to binary predictions

# Calculate accuracy
_, _, accuracy = customConfusionMatrix(test_y, y_test_pred_class.numpy())
print(f"Model Accuracy: {accuracy:.4f}")

# Calculate Equalized Odds
columns = [
    "age", "juv_fel_count", "decile_score", "juv_misd_count", "juv_other_count", 
    "priors_count", "days_b_screening_arrest", "c_days_from_compas", "v_decile_score", 
    "decile_score.1", "label", "c_jail_time", "r_jail_time", "sex_Male", 
    "age_cat_Greater than 45", "age_cat_Less than 25", "race_Asian", "race_Caucasian", 
    "race_Hispanic", "race_Native American", "race_Other", "c_charge_degree_M", 
    "c_charge_degree_O", "r_charge_degree_M", "r_charge_degree_O", 
    "v_score_text_Low", "v_score_text_Medium"
]

# Define base group (e.g., African American as base group)
base_group = 'race_African American'

# Define other races to evaluate
race_columns = [col for col in X.columns if col.startswith('race_')]

equalized_odds = {}

# Base Group: African American (i.e., all other race columns are 0)
base_group_indices = (test_x.numpy()[:, [columns.index(col) for col in race_columns]] == 0).all(axis=1)
y_true_base_group = test_y[base_group_indices].clone().detach()
y_pred_base_group = y_test_pred_class[base_group_indices].numpy()

# Calculate TPR and FPR for the base group (African American)
tpr_base_group, fpr_base_group, _ = customConfusionMatrix(y_true_base_group, y_pred_base_group)
equalized_odds['Base Group (African American)'] = {'TPR': tpr_base_group, 'FPR': fpr_base_group}

# Loop over each race column (excluding the base group, African American is assumed to be the base group)
for race_col in race_columns:
    race_col_index = columns.index(race_col)  # Get column index for the current race group

    # Filter rows for the current race group
    group_indices = (test_x.numpy()[:, race_col_index] == 1)  # Check where the race column equals 1

    # Ensure that base group is excluded by using the condition that they are not in the other race
    group_indices = group_indices & ~base_group_indices

    # Extract true labels and predicted labels for the selected race group
    y_true_group = test_y[group_indices].clone().detach()
    y_pred_group = y_test_pred_class[group_indices].numpy()

    # Calculate TPR and FPR for the selected race group
    tpr, fpr, _ = customConfusionMatrix(y_true_group, y_pred_group)
    equalized_odds[race_col] = {'TPR': tpr, 'FPR': fpr}

# Print Equalized Odds results
print("\nEqualized Odds:")
for group, rates in equalized_odds.items():
    print(f"Group {group}: TPR = {rates['TPR']:.4f}, FPR = {rates['FPR']:.4f}")


# part 6, equilizing the odds of race 
def balance(X, y, race_columns):
    # Initialize lists to store the balanced data
    X_balanced = []
    y_balanced = []

    # For each race column, balance class 0 and class 1
    for race_col in race_columns:
        # Select the rows where the current race column is equal to 1
        race_group_X = X[X[race_col] == 1]
        race_group_y = y[X[race_col] == 1]

        # Split into class 0 and class 1 for the current race group
        class_0_indices = race_group_y == 0
        class_1_indices = race_group_y == 1

        # Get the minimum number of samples between class 0 and class 1
        min_samples = min(class_0_indices.sum(), class_1_indices.sum())

        # Sample from each class to balance the dataset, random state for reproducibility
        class_0_X_balanced = race_group_X[class_0_indices].sample(n=min_samples, random_state=42)
        class_1_X_balanced = race_group_X[class_1_indices].sample(n=min_samples, random_state=42)

        class_0_y_balanced = race_group_y[class_0_indices].sample(n=min_samples, random_state=42)
        class_1_y_balanced = race_group_y[class_1_indices].sample(n=min_samples, random_state=42)

        # Add the balanced class 0 and class 1 samples to the lists
        X_balanced.append(class_0_X_balanced)
        X_balanced.append(class_1_X_balanced)
        y_balanced.append(class_0_y_balanced)
        y_balanced.append(class_1_y_balanced)

    # Concatenate the balanced data
    X_balanced = pd.concat(X_balanced)
    y_balanced = pd.concat(y_balanced)

    # Shuffle one last time
    X_balanced, y_balanced = X_balanced.sample(frac=1, random_state=42).reset_index(drop=True), y_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return X_balanced, y_balanced

# convert to tensors and train
X_balanced, y_balanced = balance(X, y, race_columns)

X_train_balance = torch.tensor(X_balanced.values, dtype=torch.float32) # use .values to obtain the numpy array instead of the pandas DataFrame
y_train_balance = torch.tensor(y_balanced.values, dtype=torch.long) # use .values to obtain the numpy array instead of the pandas DataFrame

model_balanced = LRM(input_dim=X_tensor.shape[1]) # input_dim is number of features


print("\nBalanced Data:")

# part 4: Training the model, same parameters as before it was balanced
for epoch in range(epochs):
    model_balanced.train()
    sGD.zero_grad()
    y_pred = model_balanced.architecture(X_train_balance.clone().detach().float()) # prediction, forward pass
    loss = BCE(y_pred.squeeze(), y_train_balance.float().detach())  # calculate loss
    loss.backward() # since technically still a nn, backpropagation
    sGD.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# part 7, Evaluate model on same test set as before it was balanced
model_balanced.eval()
with torch.no_grad():
    y_test_pred = model_balanced.architecture(test_x.clone().detach().float())
    y_test_pred_class = (y_test_pred.squeeze() >= 0.5).int()  # Convert probabilities to binary predictions

# Calculate accuracy
_, _, accuracy = customConfusionMatrix(test_y, y_test_pred_class.numpy())
print(f"Model Accuracy: {accuracy:.4f}")

# Calculate Equalized Odds
race_columns = [col for col in X.columns if col.startswith('race_')]
columns = [
    "age", "juv_fel_count", "decile_score", "juv_misd_count", "juv_other_count", 
    "priors_count", "days_b_screening_arrest", "c_days_from_compas", "v_decile_score", 
    "decile_score.1", "label", "c_jail_time", "r_jail_time", "sex_Male", 
    "age_cat_Greater than 45", "age_cat_Less than 25", "race_Asian", "race_Caucasian", 
    "race_Hispanic", "race_Native American", "race_Other", "c_charge_degree_M", 
    "c_charge_degree_O", "r_charge_degree_M", "r_charge_degree_O", 
    "v_score_text_Low", "v_score_text_Medium"
]

equalized_odds = {}

# Base Group: African American (i.e., all other race columns are 0)
base_group_indices = (test_x.numpy()[:, [columns.index(col) for col in race_columns]] == 0).all(axis=1)
y_true_base_group = test_y[base_group_indices].clone().detach()
y_pred_base_group = y_test_pred_class[base_group_indices].numpy()

# Calculate TPR and FPR for the base group (African American)
tpr_base_group, fpr_base_group, _ = customConfusionMatrix(y_true_base_group, y_pred_base_group)
equalized_odds['Base Group (African American)'] = {'TPR': tpr_base_group, 'FPR': fpr_base_group}

# Loop over each race column (excluding the base group, African American is assumed to be the base group)
for race_col in race_columns:
    race_col_index = columns.index(race_col)  # Get column index for the current race group

    # Filter rows for the current race group
    group_indices = (test_x.numpy()[:, race_col_index] == 1)  # Check where the race column equals 1

    # Ensure that base group is excluded by using the condition that they are not in the other race
    group_indices = group_indices & ~base_group_indices

    # Extract true labels and predicted labels for the selected race group
    y_true_group = test_y[group_indices].clone().detach()
    y_pred_group = y_test_pred_class[group_indices].numpy()

    # Calculate TPR and FPR for the selected race group
    tpr, fpr, _ = customConfusionMatrix(y_true_group, y_pred_group)
    equalized_odds[race_col] = {'TPR': tpr, 'FPR': fpr}

# Print Equalized Odds results
print("\nEqualized Odds:")
for group, rates in equalized_odds.items():
    print(f"Group {group}: TPR = {rates['TPR']:.4f}, FPR = {rates['FPR']:.4f}")


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# Load the data
abalone_data = pd.read_csv('training_data.csv')

# Normalize the features using Z-score normalization
features = ['Length', 'Diameter', 'Height', 'Whole_weight', 
            'Shucked_weight', 'Viscera_weight', 'Shell_weight']

# Normalize each feature
for feature in features:
    abalone_data[feature] = (abalone_data[feature] - abalone_data[feature].mean()) / abalone_data[feature].std()

# Normalize the target variable 'Rings'
abalone_data['Rings'] = (abalone_data['Rings'] - abalone_data['Rings'].mean()) / abalone_data['Rings'].std()

# Set up subplots
n = len(features)
fig, axes = plt.subplots(nrows=2, ncols=(n + 1) // 2, figsize=(15, 10))

# Flatten axes array for easy iteration
axes = axes.flatten()

for i, feature in enumerate(features):
    sn.scatterplot(data=abalone_data, x=feature, y='Rings', ax=axes[i])
    sn.regplot(data=abalone_data, x=feature, y='Rings', scatter=False, color='blue', ax=axes[i])
    axes[i].set_title(f'Rings vs {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Rings')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('all_scatterplots_standardized.png')
plt.show()

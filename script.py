import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the passenger data
passengers = pd.read_csv('passengers.csv')

# Updating the sex column to 0 or 1
passengers['Sex'] = passengers['Sex'].map({'female': 1, 'male': 0})

# Filling in the nan values in the age column
passengers['Age'].fillna(value=round(passengers['Age'].mean()), inplace=True)
#print(passengers.head())

# Creating 2 columns, either first class or second class
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# Selecting the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Splitting data into training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(features, survival)

# Scaling the feature data so it has mean = 0 and standard deviation = 1
norm = StandardScaler()
train_features = norm.fit_transform(train_features)
test_features = norm.transform(test_features)

model = LogisticRegression()
model.fit(train_features, train_labels)

# Scoring the model on the train data
#print(model.score(train_features, train_labels))

# Scoring the model on the test data
#print(model.score(test_features, test_labels))

# Analyzing the coefficients
#print(model.coef_)

# Sample passenger features
ps1 = np.array([0.0,21.0,0.0,0.0])
ps2 = np.array([1.0,16.0,1.0,0.0])
ps3 = np.array([0.0,19.0,0.0,0.0])

# Combining passenger arrays
sample_passengers = np.array([ps1, ps2, ps3])

# Scaling the sample passenger features
sample_passengers = norm.transform(sample_passengers)

# Seeing which of the sample passenger survives
print(model.predict(sample_passengers))

# Probabilities that resulted in the given outcome
print(model.predict_proba(sample_passengers))


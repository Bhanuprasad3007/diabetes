import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("diabetes_data_large.csv")

# Clean the 'FamilyHistory' column (convert to lowercase and remove extra spaces)
data['FamilyHistory'] = data['FamilyHistory'].str.strip().str.lower()  # Ensure uniformity

# Encode 'FamilyHistory' (Categorical) into numerical values (yes -> 1, no -> 0)
label_encoder = LabelEncoder()
data['FamilyHistory'] = label_encoder.fit_transform(data['FamilyHistory'])

# Print unique values to ensure correct encoding
print("Unique FamilyHistory values:", label_encoder.classes_)

# Split data into features and target
x = data.iloc[:, :-1]  # All columns except the last one
y = data['DiabetesType']  # Target column

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=60)

# Train RandomForest Classifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

# Predict on test data
y_pred = classifier.predict(x_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Save the trained model
filename = 'diabetes-prediction-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

# Save the label encoder for future decoding
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

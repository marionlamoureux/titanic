# Step 2 - Build the model
# On the training data, create a logistic regression model
# Model simply needs to be a function so we can deploy it.


import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('raw data/train.csv', sep=',',header=0).dropna(subset=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked', 'Survived'])

# Extract data and encode it (using label encoder)
# We're dropping PassengerID, Name, Ticket number, Cabin that are clearly irrelevant
# and label encoding the none numeric features.

df['Encoded_Sex'] = LabelEncoder().fit_transform(df.Sex)
df['Encoded_Embarked'] = LabelEncoder().fit_transform(df.Embarked)

# Select the features
features = df[[ 'Pclass', 'Encoded_Sex', 'Age', 'SibSp', 'Parch','Fare', 'Encoded_Embarked']]
survived = df[['Survived']]

# Build the model as a linear regression
regModel = LinearRegression()
regModel.fit(features,survived)


# Save Model as a pickle file
mdl = pickle.dumps(regModel)
with open('titanic_LinRegression.pickle', 'wb') as handle:
	pickle.dump(mdl, handle)

# Create the function for the model
features = [ 'Pclass', 'Encoded_Sex', 'Age', 'SibSp', 'Parch','Fare', 'Encoded_Embarked']

# == Main Function ==
def PredictFunc(args):
	# Load Data
	filtArgs = {key: [args[key]] for key in features}
	data = pd.DataFrame.from_dict(filtArgs)

	# Load Model
	with open('titanic_LinRegression.pickle', 'rb') as handle:
		mdl = pickle.load(handle)
	model = pickle.loads(mdl)

	# Get Prediction
	prediction = model.predict(data)

	# Return Prediction
	return prediction
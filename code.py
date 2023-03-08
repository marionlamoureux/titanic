import cml.data_v1 as cmldata
import os
import sys
import subprocess

from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import *

CONNECTION_NAME = "se-aw-mdl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Sample usage to run query through spark
EXAMPLE_SQL_QUERY = "show databases"
spark.sql(EXAMPLE_SQL_QUERY).show()

# Create a database
# CREATE_DATABASE = "create DATABASE titanic"
# spark.sql(CREATE_DATABASE).show()

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

#schema = StructType(
#    [
#      StructField("PassengerId", StringType(), True),
#      StructField("Survived", StringType(), True),
#      StructField("Pclass", StringType(), True),      
#      StructField("Name", StringType(), True),      
#      StructField("Sex", StringType(), True), 
#      StructField("Age", DoubleType(), True),       
#      StructField("SibSp", DoubleType(), True),       
#      StructField("Parch", DoubleType(), True),
#      StructField("Ticket", StringType(), True),      
 #     StructField("Fare", DoubleType(), True),    
#      StructField("Cabin", StringType(), True),          
#      StructField("Embarked", StringType(), True)
#    ]
#)

path = 'titanic/raw data/train.csv'

titanic_data = spark.read.csv(path, header=True, sep=",", nullValue="NA")

# ...and push csv into hive table
titanic_data.write.saveAsTable("titanic.titanic_train")

# Step 2 - Build the model
# On the training data, create a logistic regression model
# Model simply needs to be a function so we can deploy it.

!pip install scikit-learn
!pip install pandas
!pip install numpy

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pickle

# decode input(s)

df = pd.read_csv('titanic/raw data/train.csv', sep=',',header=0).dropna(subset=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked', 'Survived'])

# extract data and encode it (using label encoder)
# We're dropping PassengerID, Name, Ticket number, Cabin that are clearly irrelevant
# and label encoding the none numeric features.

df['Encoded_Sex'] = LabelEncoder().fit_transform(df.Sex)
df['Encoded_Embarked'] = LabelEncoder().fit_transform(df.Embarked)

features = df[[ 'Pclass', 'Encoded_Sex', 'Age', 'SibSp', 'Parch','Fare', 'Encoded_Embarked']]
survived = df[['Survived']]

regModel = LinearRegression()
regModel.fit(features,survived)


# Validate Model
scores = cross_val_score(regModel, features, survived, cv=3)
avgAcc = np.mean(scores)
print(avgAcc)

# Save Model
mdl = pickle.dumps(regModel)
with open('titanic_LogRegression.pickle', 'wb') as handle:
	pickle.dump(mdl, handle)
  





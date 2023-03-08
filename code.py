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

# On the training data, create a logistic regression model
# Model simply needs to be a function so we can deploy it.

import 




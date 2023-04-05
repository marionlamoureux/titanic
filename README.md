# Titanic
This is inspired by Kaggle challenge Titanic
https://www.kaggle.com/competitions/titanic

Data are available here :
https://www.kaggle.com/competitions/titanic/data

For data exploration, this workbook is extremely clear:
https://www.kaggle.com/code/seuwenfei/titanic-random-forest-cv-score-0-85


## How to deploy a model

Step 1 -  Writing in python the function for the model
Tutorials for building ML models are here :
https://www.cloudera.com/tutorials.html


Step 2 - Add requirements for model container:
- you need a requirements.txt file
- you need a cdsw-build.sh with the pip install for the requirements file.

Step 3 - Create the model
Indicate the arguments as input:
```
args = {
  "Pclass": "2",
  "Encoded_Sex": "0",
  "Age": "25.0",
  "SibSp": "1",
  "Parch": "0",
  "Fare": "7.9250",
  "Encoded_Embarked": "2"
}
```

## How to write to Hive databases

Step 1 - Code snippets
Code snippets are available, pointing to existing data lakes
![](images/CodeSnippet_HIVEConnection.png)
If the connection you're looking for is not in the snippets:
- Make sure you have a virtual warehouse created
- Add the connection under "Data connections" of the project

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program 
2. Import the required python libraries such as numpy,pandas,matplotlib
3. Read the dataset of student scores
4. Assign the column hours to x and column scores to y
5. From sklearn library select the model to train and to test the dataset
6. Plot the training set and testing set in the graph using matplotlib library
7. Stop the program


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R.SUROTHAAMAN
RegisterNumber:212222103003
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/content/student_scores .csv")
dataset.head()                              #display the top 5 rows
X = dataset.iloc[:,:-1].values              #assigning column hours to X
y = dataset.iloc[:,1].values                #assigning column scores to y
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("hr vs sec(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
y_pred = regressor.predict(X_test)
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='purple')
plt.title("hr vs sec(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
dataset.tail()                               #display last 5 rows
```

## Output:
![image](https://github.com/surothaaman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133313653/646cf9ee-63c7-4f3d-9bf2-64a0f0cb135c)
![image](https://github.com/surothaaman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133313653/aed3e3f0-76a7-4e02-81a9-74161bff0009)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

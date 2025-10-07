# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder. 
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NITHYASREE S
RegisterNumber:  212224040225
*/
```
````PYTHON

import pandas as pd
data=pd.read_csv(r"C:\Users\L390 Yoga\Downloads\Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
````
## Output:
Data Head:

<img width="390" height="265" alt="image" src="https://github.com/user-attachments/assets/b3824194-0d40-4cdb-b142-ee5a13ff7c27" />

Data Info:

<img width="603" height="237" alt="image" src="https://github.com/user-attachments/assets/8b2716d8-5a04-40a3-be11-ebb3aed5b761" />

isnull() sum():

<img width="201" height="88" alt="image" src="https://github.com/user-attachments/assets/c46faece-09ca-4452-a95e-a5bf24761380" />

Data Head for salary:

<img width="323" height="234" alt="image" src="https://github.com/user-attachments/assets/42196eb7-c2bc-4549-8729-770a754684ec" />

Mean Squared Error :

<img width="239" height="38" alt="image" src="https://github.com/user-attachments/assets/11a5da1e-84a5-4d40-83b7-8b3921202d8a" />

r2 Value:

 <img width="1065" height="41" alt="image" src="https://github.com/user-attachments/assets/43e094a8-f8d0-45e1-a342-f351e8015150" />
 
Data prediction :

<img width="311" height="38" alt="image" src="https://github.com/user-attachments/assets/7869e99c-e94d-41e7-ba6a-df9b383f4296" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

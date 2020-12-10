#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#################################################
#Description:
#
##################################################
#Sources:
#https://realpython.com/linear-regression-in-python/#multiple-linear-regression
##################################################


#Read the datasets for training
#to analyze different k features change the input csv file to the different ones available
x_train=pd.read_csv('p1_features_training.csv')
y_train=pd.read_csv('p1_outcome_training.csv')


#Read second datasets for testing, it is important to maintain a separate dataset from the training one for testing
x_test=pd.read_csv('p1_features_test.csv')
y_test=pd.read_csv('p1_outcome_test.csv')


#Now lets implement the linear regression
#make an instance of the regression model fitting the trainning data
#This is also called training the model
model = LinearRegression().fit(x_train, y_train)

#Here we can USE the trained model above with the training or the testing dataset
#We would obtain R2 which lets you know how well the regression is able to predict the data
#may change the dataset testing <--> training
r_sq = model.score(x_test, y_test)
print('coefficient of determination:', r_sq)

#Here we can make a prediction, remember to match this dataset to the R2 dataset to have the output reflect the same run
y_pred = model.predict(x_test)
#print('predicted response:', y_pred, sep='\n')

#compare to actual output by graphing the prediction and the actual output in the same graph
plt.plot(y_test, '--')
plt.plot(y_pred)
plt.show()
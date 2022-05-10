import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

companies=pd.read_csv('sample_data/LinearRegressionExample.csv')
x=companies.iloc[:,:-1].values
y=companies.iloc[:,4].values
companies.head()

print(x)

print(y)

sns.heatmap(companies.corr())

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#Encode State Column
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)
print(x)

x=x[:,1:]
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
model_fit=LinearRegression()
model_fit.fit(x_train,y_train)

y_pred=model_fit.predict(x_test)
print(y_pred)

print(model_fit.coef_)

print(model_fit.intercept_)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

x_state = companies.State
y_profit = companies.Profit

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x_state = labelencoder.fit_transform(x_state)
print(x_state)


x_state = np.array(x_state).reshape(-1,1)
y_profit = np.array(y_profit).reshape(-1,1)

print(x_state.shape)
print(y_profit.shape)

X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(x_state, y_profit, test_size = 0.2, random_state=5)

print(X_train_1.shape)
print(X_test_1.shape)
print(Y_train_1.shape)
print(Y_test_1.shape)

from sklearn.metrics import mean_squared_error
reg_1 = LinearRegression()
reg_1.fit(X_train_1, Y_train_1)

y_train_predict_1 = reg_1.predict(X_train_1)
rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))
r2 = round(reg_1.score(X_train_1, Y_train_1),2)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_pred_1 = reg_1.predict(X_test_1)
rmse = (np.sqrt(mean_squared_error(Y_test_1, y_pred_1)))
r2 = round(reg_1.score(X_test_1, Y_test_1),2)

print("The model performance for testing set")
print("--------------------------------------")
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(r2))
print("\n")

prediction_space = np.linspace(min(x_state), max(y_profit)).reshape(-1,1) 
plt.scatter(x_state,y_profit)
plt.show()
plt.plot(prediction_space, reg_1.predict(prediction_space), color = 'black', linewidth = 3)
plt.ylabel('Profit')
plt.xlabel('State')
plt.show()
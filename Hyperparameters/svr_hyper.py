from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error#RMSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error#MAE
from sklearn.preprocessing import StandardScaler
svr = SVR()
# reading dataset
data = pd.read_csv("C_PWS1000.csv")
print(data.shape)
x = data .iloc[:, 1:(data.shape[1]-1)].values
y = data .iloc[:, -1].values 
# split data
X_train,X_test,y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
# Standardizing the features:
X_train = np. delete(X_train, 0, axis=1)
X_test = np. delete(X_test, 0, axis=1)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test= sc.transform(X_test)
# grid search on kernel and C hyperparameters
parameters = {'kernel':('linear', 'rbf','poly'), 'C':[6, 10]}
clf = GridSearchCV(svr, param_grid=parameters)
clf.fit(X_train, y_train)
print('Grid best parameters (max accuracy): ', clf.best_params_)
print('Grid best score (accuracy): ', clf.best_score_)
y_test_pred = (clf.best_estimator_).predict(X_test)
r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
print('         r2 score for test data: ' + str(r2))
RMSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
print('         RMSE score for test data: ' + str(np.sqrt(RMSE)))
MAE = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
print('         MAE  score for test data: ' + str(MAE))
#####


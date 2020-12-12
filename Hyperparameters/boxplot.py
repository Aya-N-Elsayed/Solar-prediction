import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd  

# Creating plot plt.boxplot
data = pd.read_csv("r2_large.csv")
data.boxplot(column=['RF', 'DT', 'SVR','BR','LR','MLP','CNN'])
plt.title('Training data shape (48246, 15) testing data shape (20678, 15)')
plt.xlabel('Regressors')
plt.ylabel('r^2')
plt.show()

# Creating plot plt.boxplot
data = pd.read_csv("rmse_large.csv")
data.boxplot(column=['RF', 'DT', 'SVR','BR','LR','MLP','CNN'])
plt.title('Training data shape (48246, 15) testing data shape (20678, 15)')
plt.xlabel('Regressors')
plt.ylabel('RMSE')
plt.show()

# Creating plot plt.boxplot
data = pd.read_csv("mae_large.csv")
data.boxplot(column=['RF', 'DT', 'SVR','BR','LR','MLP','CNN'])
plt.title('Training data shape (48246, 15) testing data shape (20678, 15)')
plt.xlabel('Regressors')
plt.ylabel('MAE')
plt.show()


# Creating plot plt.boxplot
data = pd.read_csv("r2_small.csv")
data.boxplot(column=['RF', 'DT', 'SVR','BR','LR','MLP','CNN'])
plt.title('Training data shape (700, 15) testing data shape (300, 15)')
plt.xlabel('Regressors')
plt.ylabel('r^2')
plt.show()

# Creating plot plt.boxplot
data = pd.read_csv("rmse_small.csv")
data.boxplot(column=['RF', 'DT', 'SVR','BR','LR','MLP','CNN'])
plt.title('Training data shape (700, 15) testing data shape (300, 15)')
plt.xlabel('Regressors')
plt.ylabel('RMSE')
plt.show()

# Creating plot plt.boxplot
data = pd.read_csv("mae_small.csv")
data.boxplot(column=['RF', 'DT', 'SVR','BR','LR','MLP','CNN'])
plt.title('Training data shape (700, 15) testing data shape (300, 15)')
plt.xlabel('Regressors')
plt.ylabel('MAE')
plt.show()

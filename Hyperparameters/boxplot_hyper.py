import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd  

# Creating plot plt.boxplot
data = pd.read_csv("svrS_hyper.csv")
data.boxplot(column=['R2','RMSE','MAE', 'C','kernel'])
plt.title('SVR hyperparamter tuning results')
plt.xlabel('Metrics')
plt.show()

# Creating plot plt.boxplot
data = pd.read_csv("rfS_hyper.csv")
data.boxplot(column=['R2','RMSE','MAE','n_est'])
plt.title('RF hyperparamter tuning results for sampled data')
plt.xlabel('Metrics')
plt.show()

# Creating plot plt.boxplot
data = pd.read_csv("rfL_hyper.csv")
data.boxplot(column=['R2','RMSE','MAE','n_est'])
plt.title('RF hyperparamter tuning results for whole data')
plt.xlabel('Metrics')
plt.show()


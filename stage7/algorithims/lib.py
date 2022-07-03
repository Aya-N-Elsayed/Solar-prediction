import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error#RMSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error#MAE
from sklearn.preprocessing import StandardScaler
import time
import argparse
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
#from sklearn_MAPE import *
from scipy.signal import find_peaks
import os
from numpy import savetxt
from keras.models import Sequential
from keras.layers import *
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import plot_date
from matplotlib import dates as mpl_dates
import matplotlib.dates as mdates
import matplotlib.dates as matdates
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
#from sklearn_MAPE import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
from numpy import savetxt
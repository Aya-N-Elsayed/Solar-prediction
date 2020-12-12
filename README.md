## Investigating the best regression model for solar radiation prediction

Given a timestamped weather data, including wind, temperature, humidity, and pressure, and given historical solar radiation data, we can find a relation between weather features and solar radiation. According to that, we can predict solar radiation for the future according to the weather. We applied different regression algorithms and models to find the best regression model to predict solar radiation. The algorithm tested are Random Forest Regression, Decision Tree Regression, Support Vector Regression, Linear Regression, and Neural Network Multilayer Perceptron Regression. Also, we tried several CNN architectures. Hyperparameters Tuning is applied to increase the accuracy of the results from the algorithms tested.

### How to run
###### To run the code for regression algorithms in the small data (1000 instances)
```
python3 main.py --dataset C_PWS_1.csv --teston " " --randtest 1
```

###### To run the CNN model in the small data (1000 instances)
```
python3 ann.py --dataset C_PWS_1.csv --teston " " --randtest 1
```
To run on large data (68925), change the data file in the command line to C_PWS.csv

You should run the commands from the path "./Solar-prediction/stage7/algorithms"

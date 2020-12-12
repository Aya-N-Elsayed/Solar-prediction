## Investigating the best regression model for solar radiation prediction

Given a timestamped weather data, including wind, temperature, humidity, and pressure, and given historical solar radiation data, we can find a relation between weather features and solar radiation. According to that, we can predict solar radiation for the future according to the weather. We applied different regression algorithms and models to find the best regression model to predict solar radiation.

### How to run
###### To run the code for regression algorithms in the small data (1000 instances)
Run this command
```
python3 main.py --dataset C_PWS_1.csv --teston " " --randtest 1
```

###### To run the CNN model in the small data (1000 instances)
Run this command
```
python3 ann.py --dataset C_PWS_1.csv --teston " " --randtest 1
```
To run on large data (68925), change the data file in the command line to C_PWS.csv

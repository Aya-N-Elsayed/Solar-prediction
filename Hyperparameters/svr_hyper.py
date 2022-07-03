from  lib import *
svr = SVR()
# reading dataset
arguments = argparse.ArgumentParser()
arguments.add_argument('--dataset',type=str,help='dataset name')
args = arguments.parse_args()
dataset = args.dataset
data = pd.read_csv(dataset)
print(data.shape)
x = data .iloc[:, 0:(data.shape[1]-1)].values
y = data .iloc[:, -1].values 
# split data
X_train,X_test,y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
# Standardizing the features:
time_plt = X_test[:,0]
X_train = np. delete(X_train, 0, axis=1)
X_test = np. delete(X_test, 0, axis=1)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test= sc.transform(X_test)
# grid search on kernel and C hyperparameters
parameters = {'kernel':['linear'], 'C':[6, 10]}
clf = GridSearchCV(svr, param_grid=parameters)
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
test_time = end_time - start_time
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
#save result into csv file
col_names = ['time','y_test','svrhyper_ytest']
test_time = np.array([0,0,test_time])
print(test_time.shape)
temp = pd.DataFrame(data=time_plt, columns=["date"])
time_plt = (temp.date.str.split(":00-",expand=True))[0]
time_plt = time_plt.str.replace("T",' ')
results = np.array([time_plt,y_test,y_test_pred])
results = np.transpose(results)
print(results.shape)
results = np.insert(results, 0,test_time, axis=0)
print(results.shape)
print(results)
now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M")
#results = np.insert(results, 0,col_names, 0) 
results_ = pd.DataFrame(data = results,columns=col_names)
results_.to_csv("svr_hyper"+dataset+dt_string,index=False)


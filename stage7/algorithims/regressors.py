from lib import *

class regressors():
  
    def __init__(self,regressor,X_train =0, X_test = 0, y_train =0, y_test=0, train_on ='' , test_on='' ):
        
        self.regression = regressor
        self.X_train = X_train 
        self.X_test = X_test 
        self.y_train = y_train  
        self.y_test = y_test 
        self.train_on = train_on
        self.test_on = test_on
        
    # define base model for cnn
    def model(self):
        # create model
        model = Sequential()
        model.add(Conv1D(filters=6, kernel_size=5, activation='relu', input_shape=(15,1)))
        # model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
        model.add(Conv1D(filters=6, kernel_size=5, activation='relu'))
        model.add(Conv1D(filters=6, kernel_size=5, activation='relu'))

        
        model.add(Flatten())
        model.add(Dense(12, kernel_initializer='normal', activation='relu'))
        # model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def run_regressor(self):

        if self.regression == 'ann':
            model_ = self.model()
            model_.summary()
            X_train_ann = np.asarray(self.X_train).astype(np.float32)
            y_train_ann = np.asarray(self.y_train).astype(np.float32)
            X_train_ann = X_train_ann.reshape(X_train_ann.shape[0], X_train_ann.shape[1], 1)
            # x test
            X_test_ann = np.asarray(self.X_test).astype(np.float32)
            X_test_ann = X_test_ann.reshape(X_test_ann.shape[0], X_test_ann.shape[1], 1)

            model_.fit(X_train_ann, y_train_ann , batch_size=12,epochs=200, verbose=0)
            y_train_pred = model_.predict(X_train_ann)
            start_time = time.time()
            y_test_pred = model_.predict(X_test_ann)
            end_time = time.time()
            test_time = end_time - start_time
            return test_time,y_train_pred,y_test_pred
        else :
            if self.regression == 'RF':
                reg = RandomForestRegressor(n_estimators = 250 , random_state = 1)
            elif self.regression == 'D_tree':
                reg = DecisionTreeRegressor(max_depth=15,random_state=0)   
            elif  self.regression == 'svr':   
                reg = SVR(kernel='linear', degree=4, gamma='scale', coef0=0.0, tol=0.001,C=1.0, epsilon=0.1,
                shrinking=True, cache_size=200, verbose=False, max_iter=-1)
            elif  self.regression == 'MLP': 
                reg = MLPRegressor(hidden_layer_sizes=(200, 200), max_iter=8000,
                                random_state=0, tol=0.01, activation='tanh', solver='adam',learning_rate = 'adaptive')
            elif self.regression == 'BR':
                reg = BayesianRidge(compute_score=True)
            elif self.regression == 'LR':
                reg = LinearRegression()   
            #elif self.regression == 'GPR':
                #kernel = DotProduct() + WhiteKernel()
                #reg = GaussianProcessRegressor(kernel=kernel,random_state=0)   
            
            reg.fit(self.X_train, self.y_train)
            y_train_pred = reg.predict(self.X_train)
            start_time = time.time()
            y_test_pred = reg.predict(self.X_test)
            end_time = time.time()
            test_time = end_time - start_time
            return test_time,y_train_pred,y_test_pred


    def metrics(self,y_train_pred,y_test_pred,fit_time):
        text_file = open("Output.txt", "a")
        text_file.write(self.regression +' training on ' +self.train_on + ' '+ str(self.X_train.shape) +
        " testing on " + self.test_on + ' '+ str(self.X_test.shape) + '\n')
        text_file.write("           fitting time " + str(fit_time)+ '\n')

        r2 = r2_score(y_true=self.y_train, y_pred=y_train_pred)
        print('         r2 score for train data: ' + str(r2))
        text_file.write('           r2 score for train data: ' + str(r2)+ '\n')

        r2 = r2_score(y_true=self.y_test, y_pred=y_test_pred)
        print('         r2 score for test data: ' + str(r2))
        text_file.write('           r2 score for test data: ' + str(r2)+ '\n')

        RMSE = mean_squared_error(y_true=self.y_train, y_pred=y_train_pred)
        print('         RMSE score for train data: ' + str(np.sqrt(RMSE)))
        text_file.write('           RMSE score for train data: ' + str(np.sqrt(RMSE))+ '\n')

        RMSE = mean_squared_error(y_true=self.y_test, y_pred=y_test_pred)
        print('         RMSE score for test data: ' + str(np.sqrt(RMSE)))
        text_file.write('           RMSE score for test data: ' + str(np.sqrt(RMSE))+ '\n')

        MAE = mean_absolute_error(y_true=self.y_train, y_pred=y_train_pred)
        print('         MAE  score for train data: ' + str(MAE))
        text_file.write('           MAE  score for train data: ' + str(MAE)+ '\n')

        MAE = mean_absolute_error(y_true=self.y_test, y_pred=y_test_pred)
        print('         MAE  score for test data: ' + str(MAE))
        text_file.write('           MAE  score for test data: ' + str(MAE)+ '\n\n')   
from  lib import *


# define base model
def model():
    # create model
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=(15,1)))
    # model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
    # model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))

    
    model.add(Flatten())
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == "__main__":

    arguments = argparse.ArgumentParser()
    arguments.add_argument('--dataset',type=str,help='dataset name')
    arguments.add_argument('--teston',type=str,help='teston')
    arguments.add_argument('--randtest',type=str,help='for random testing enter 1')
    args = arguments.parse_args()
    dataset = args.dataset
    rand_test = args.randtest
    teston = args.teston
    # Loading our data set
    os.chdir("..")
    script_path = os.path.abspath("cleaned_data")
    # print(script_path )
    dataset = os.path.join(script_path,dataset)
    data = pd.read_csv(dataset)
    X = data .iloc[:, 1:(data.shape[1]-1)].values
    Y = data .iloc[:, -1].values
    
    X = np.asarray(X).astype(np.float32)
    Y = np.asarray(Y).astype(np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)

      # Splitting data into 70% training and 30% test data:
    if rand_test == '1':
        X_train,X_test,y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=1)
        test_on = ''
    else:    
        dataTs = os.path.join(script_path,teston)
        dataTs = pd.read_csv( dataTs)
        X_train = X
        X_test = dataTs .iloc[:, 0:(data.shape[1]-1)].values
        y_train = Y
        y_test = dataTs .iloc[:, -1].values
        print(x.shape, y.shape ,X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        test = teston.split("_")[1]
        test_on = test.split(".")[0]

    model = model()
    model.summary()
    model.fit(X_train, y_train, batch_size=12,epochs=200, verbose=0)

    ypred = model.predict(X_test)
    ypred_train = model.predict(X_train)
    print(model.evaluate(X_train, y_train))
    print("MSE: %.4f" % mean_squared_error(y_test, ypred))
    print("MAE: %.4f" % mean_absolute_error(y_train, ypred_train))
    print("r2: %.4f" % r2_score(y_true=y_train, y_pred=ypred_train))
    

    x_ax = range(len(ypred))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()


    with open('Output_ann.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    text_file = open("Output_ann.txt", "a")
    text_file.write("MSE: %.4f" % mean_squared_error(y_test, ypred) + "\n")
    text_file.write("MAE: %.4f" % mean_absolute_error(y_train, ypred_train) + "\n")
    text_file.write("r2: %.4f" % r2_score(y_true=y_train, y_pred=ypred_train))
    text_file.close()
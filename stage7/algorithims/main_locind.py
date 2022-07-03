#importing
from  lib import *
from regressors import *


if __name__ == "__main__":
    #taking arguments
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
    print(script_path )
    dataset = os.path.join(script_path,dataset)
    data = pd.read_csv(dataset)
    x = data .iloc[:, 0:(data.shape[1]-1)].values
    y = data .iloc[:, -1].values

    # Splitting data into 70% training and 30% test data:
    if rand_test == '1': # 70 30 same station 
        X_train,X_test,y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
        test_on = ''
    elif rand_test == '11':    #70 30
        X_train, _ ,y_train, _ = train_test_split(x,y, test_size=0.3, random_state=1)
        dataTs = os.path.join(script_path,teston)
        dataTs = pd.read_csv( dataTs)
        xTs = dataTs .iloc[:, 0:(dataTs.shape[1]-1)].values
        yTs = dataTs .iloc[:, -1].values
        _,X_test,_, y_test = train_test_split(xTs,yTs, test_size=0.3, random_state=1)
        print(x.shape, y.shape ,X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        test = teston.split("_")[1]
        test_on = test.split(".")[0]
        
    elif rand_test == '01':    #100 30
        X_train = x
        y_train = y 
        dataTs = os.path.join(script_path,teston)
        dataTs = pd.read_csv( dataTs)
        xTs = dataTs .iloc[:, 0:(dataTs.shape[1]-1)].values
        yTs = dataTs .iloc[:, -1].values
        _,X_test,_, y_test = train_test_split(xTs,yTs, test_size=0.3, random_state=1)
        print(x.shape, y.shape ,X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        test = teston.split("_")[1]
        test_on = test.split(".")[0]

    elif rand_test == '00':    #100 100
        X_train = x
        y_train = y 
        dataTs = os.path.join(script_path,teston)
        dataTs = pd.read_csv( dataTs)
        xTs = dataTs .iloc[:, 0:(dataTs.shape[1]-1)].values
        yTs = dataTs .iloc[:, -1].values
        X_test = xTs 
        y_test = yTs
        print(x.shape, y.shape ,X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        test = teston.split("_")[1]
        test_on = test.split(".")[0]
        

    # Standardizing the features:
    time_plt = X_test[:,0]
    X_train = np. delete(X_train, 0, axis=1)
    X_test = np. delete(X_test, 0, axis=1)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test= sc.transform(X_test)
    print(x.shape, y.shape ,X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    train = args.dataset.split("_")[1]
    train_on = train.split(".")[0]

    reg_ann = regressors('ann',X_train,X_test , y_train , y_test, train_on,test_on)
    annts_time,anny_train_pred,anny_test_pred = reg_ann.run_regressor()
    reg_ann.metrics(anny_train_pred,anny_test_pred,annts_time)

    reg_RF = regressors('RF',X_train,X_test , y_train , y_test, train_on,test_on)
    RFts_time,RFy_train_pred,RFy_test_pred = reg_RF.run_regressor()
    reg_RF.metrics(RFy_train_pred,RFy_test_pred,RFts_time)
    
    reg_TREE = regressors('D_tree',X_train,X_test , y_train , y_test, train_on,test_on)
    TREEts_time,TREEy_train_pred,TREEy_test_pred = reg_TREE.run_regressor()
    reg_TREE.metrics(TREEy_train_pred,TREEy_test_pred,TREEts_time)   

    reg_SVM = regressors('svr',X_train,X_test , y_train , y_test, train_on,test_on)
    SVMts_time,SVMy_train_pred,SVMy_test_pred = reg_SVM.run_regressor()
    reg_SVM.metrics(SVMy_train_pred,SVMy_test_pred,SVMts_time)
  
    reg_BR = regressors('BR',X_train,X_test , y_train , y_test, train_on,test_on)
    BRts_time,BRy_train_pred,BRy_test_pred = reg_BR.run_regressor()
    reg_BR.metrics(BRy_train_pred,BRy_test_pred,BRts_time)

    reg_LR = regressors('LR',X_train,X_test , y_train , y_test, train_on,test_on)
    LRts_time,LRy_train_pred,LRy_test_pred = reg_LR.run_regressor()
    reg_LR.metrics(LRy_train_pred,LRy_test_pred,LRts_time)
   
    reg_MPL = regressors('MLP',X_train,X_test , y_train , y_test, train_on,test_on)
    MPLts_time,MPLy_train_pred,MPLy_test_pred = reg_MPL.run_regressor()
    reg_MPL.metrics(MPLy_train_pred,MPLy_test_pred,MPLts_time)
    
    #save result into csv file
    col_names = ['time','y_test','TREEy_test_pred','RFy_test_pred','SVMy_test_pred','BRy_test_pred',
    'LRy_test_pred','MPLy_test_pred','cnn_y_test_pred']
    test_time = np.array([0,0,TREEts_time,RFts_time,SVMts_time,BRts_time,LRts_time,MPLts_time,annts_time])
    print(test_time.shape)
    temp = pd.DataFrame(data=time_plt, columns=["date"])
    time_plt = (temp.date.str.split(":00-0",expand=True))[0]
    time_plt = time_plt.str.replace("T",' ')
    results = np.array([time_plt,y_test,TREEy_test_pred,RFy_test_pred
    ,SVMy_test_pred,BRy_test_pred,LRy_test_pred,MPLy_test_pred,np.squeeze(anny_test_pred) ])
    results = np.transpose(results)
    print(results.shape)
    results = np.insert(results, 0,test_time, axis=0)
    print(results.shape)
    print(results)
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M")
    #results = np.insert(results, 0,col_names, 0) 
    results_ = pd.DataFrame(data = results,columns=col_names)
    results_.to_csv("TR"+train_on+"TS"+teston+dt_string,index=False)
    
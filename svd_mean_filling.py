#this file is modified from "骰子AI"'s code on bilibili
import numpy as np
import pandas as pd

def loadData():
    header = ['user_id','movie_id','rating','timestamp']
    train_path = "D:\machine learning\电影案例/u1.base"
    trainData = pd.read_csv(train_path,sep='\t',names=header)
    test_path = "D:\machine learning\电影案例/u1.test"
    testData = pd.read_csv(test_path, sep='\t', names=header)
    return trainData,testData


def getMatrix(trainData):
    userlist = list(set(trainData.iloc[:,0]))
    movielist =  list(set(trainData.iloc[:,1]))
    df = pd.DataFrame(0,index=userlist,columns=movielist)
    for i in trainData.values:
        df[i[1]][i[0]] = i[2]
    non_zero = (df!=0)
    num = df.sum(axis=0)
    den = non_zero.sum(axis=0)
    ave = num/den
    print(ave)
    for j in movielist:
        for i in userlist:
            if(df[j][i]==0):
                df[j][i] = ave[j]
    print(df)
    return df,userlist,movielist

def svd(df,k):
    u,s,v = np.linalg.svd(df)
    return u[:,:k],np.diag(s[0:k]),v[0:k,:]

def predict(testData,userlist,movielist,u,s,v):
    y_true,y_predict = [],[]
    for i in testData.values:
        user = i[0]
        movie = i[1]
        if user in userlist and movie in movielist:
            user_index = userlist.index(user)
            movie_index = movielist.index(movie)
            y_true.append(i[2])
            y_predict.append(float(u[user_index].dot(s).dot(v.T[movie_index].T)))
    return y_true,y_predict

def rmse(y_true,y_predict):
    return (np.average((np.array(y_true)-np.array(y_predict))**2))**0.5


if __name__ == '__main__':
    trainData, testData = loadData()
    df, userlist, movielist = getMatrix(trainData)
    k = 10
    u,s,v = svd(np.mat(df),k)
    y_true,y_predict = predict(testData,userlist,movielist,u,s,v)
    print("rmse:",rmse(y_true,y_predict))

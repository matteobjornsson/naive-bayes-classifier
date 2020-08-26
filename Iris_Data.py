import pandas as pd 
import os 

##No missing attributes found 
#Data formatted properly, Namely holding the value {X1.X2,...,Xn,Class}


class Iris:

    def PreProcess(self): 
        print("hello Worlds")
        DataFrame = pd.read_csv('Iris_Data/iris.DATA')
        EOF = len(DataFrame)
        count = 0 
        for i in DataFrame: 
            count +=1 
        for i in range(EOF): 
            for j in range(count): 
                if j == (count -1): 
                    #print('Miss')
                    continue 
                DataFrame.iloc[i][j] = int(DataFrame.iloc[i][j])  * 10 
                DataFrame['Bins']=pd.cut(DataFrame['sepal_length'],3,labels=['Poor','Below_average','Average'])

                print(DataFrame.head())


    def __init__(self): 
        self.PreProcess()
        print("Hello World!") 



if __name__ == '__main__': 
    iris = Iris() 
    #Vt.PreProcess()
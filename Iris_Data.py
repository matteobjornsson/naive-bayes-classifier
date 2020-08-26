import pandas as pd 
import os 

##No missing attributes found 
#Data formatted properly, Namely holding the value {X1.X2,...,Xn,Class}


class Iris:

    def PreProcess(self): 
        print("hello Worlds")
        DataFrame = pd.read_csv('Iris_Data/iris.DATA')
        for i in DataFrame: 
            print(i)

    def __init__(self): 
        self.PreProcess()
        print("Hello World!") 



if __name__ == '__main__': 
    iris = Iris() 
    #Vt.PreProcess()
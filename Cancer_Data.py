import pandas as pd 
import os 



class Cancer:

    def PreProcess(self): 
        print("hello Worlds")
        DataFrame = pd.read_csv('Breast_Cancer_Data/cancer.DATA')
        for i in DataFrame: 
            print(i)

    def __init__(self): 
        self.PreProcess()
        print("Hello World!")




if __name__ == '__main__': 
    Cc = Cancer()
    #Vt.PreProcess()
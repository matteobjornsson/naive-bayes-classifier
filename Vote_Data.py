import pandas as pd
import os 





class Vote:

    def PreProcess(self): 
        print("hello Worlds")
        DataFrame = pd.read_csv('Vote_Data/Votes.DATA')
        for i in DataFrame: 
            print(i)

    def __init__(self): 
        self.PreProcess()
        print("Hello World!") 







if __name__ == '__main__': 
    Vt = Vote()
    #Vt.PreProcess()
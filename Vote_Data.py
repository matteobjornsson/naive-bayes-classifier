import pandas as pd
import os 




class Vote:


    def PreProcess(self): 
        print("hello Worlds")
        DataFrame = pd.read_csv('Vote_Data/Votes.DATA')
        EOF = len(DataFrame)
        count = 0 
        for i in DataFrame: 
            count +=1 

        for i in range(EOF): 
            for j in range(count): 
                if DataFrame.iloc[i][j] == 'y':
                    DataFrame.iloc[i][j] = '1'
                if DataFrame.iloc[i][j] == 'n':
                    DataFrame.iloc[i][j] = '0'
                if DataFrame.iloc[i][j] == '?':
                    DataFrame.iloc[i][j] = '2'
                #print(DataFrame.iloc[i][j])
        return DataFrame
        

    def __init__(self): 
        #self.PreProcess()
        print("Hello World!") 







if __name__ == '__main__': 
    Vt = Vote()
    #Vt.PreProcess()

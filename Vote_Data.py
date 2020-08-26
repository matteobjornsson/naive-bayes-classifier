import pandas as pd
import os 





class Vote:

    def PreProcess(): 
        DataFrame = pd.read_csv('Vote_Data/Votes.DATA')
        for i in DataFrame: 
            print(i)

    def __init__(): 
        PreProcess()
        print("Hello World!") 







def main(): 
    Vt = Vote()
    vt.PreProcess()
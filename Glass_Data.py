import pandas
import os 



class Glass:

    def PreProcess(self): 
        print("hello Worlds")
        DataFrame = pd.read_csv('Glass_Data/Glass.DATA')
        for i in DataFrame: 
            print(i)

    def __init__(): 
        PreProcess()
        print("Hello World!") 




if __name__ == '__main__': 
    glass = Glass() 
    #Vt.PreProcess()
#Created by Nick Stone 
#Created on 8/20/2020 
##################################################################### MODULE COMMENTS #####################################################################
#
#
#
#
#
##################################################################### MODULE COMMENTS #####################################################################

import pandas as pd 
import os 
 



def main(): 
    
    print("Program Starting")
    #Create a data frame and feed the data from the file path 
    data = pd.read_csv('C:/Users/nston/Desktop/MachineLearning/Project 1/Glass Data/glass.data')
    mas = len(data)
    print(mas)
    for j in range(mas): 
        print(data.iloc[j])
        print('\n')
    
    for i in data: 
        print(i)
    #print(data.head())



    print("Program Finish")



#Call the main function
main() 
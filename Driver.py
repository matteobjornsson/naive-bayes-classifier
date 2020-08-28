#Created by Nick Stone and Matteo Bjornsson 
#Created on 8/20/2020 
##################################################################### MODULE COMMENTS #####################################################################
#
#
#
#
#
##################################################################### MODULE COMMENTS #####################################################################

import pandas as pd 



def main(): 
    
    print("Program Starting")
    Cc = cd.Cancer() 
    print("\n")
    glass = gd.Glass() 
    print("\n")
    Vt = vd.Vote()
    print("STARTING VOTING")
    df = Vt.PreProcess()
    print(df.head())
    
    print("===============================")
  

    print("\n") 
    iris = ird.Iris() 
    print("\n")
    soy = sd.Soybean() 






    print("Program Finish")



#Call the main function
main() 
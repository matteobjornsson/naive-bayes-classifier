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
 

import Soybean_Data as sd
import Iris_Data as ird 
import Vote_Data as vd 
import Glass_Data as gd
import Cancer_Data as cd 



def main(): 
    
    print("Program Starting")
    Cc = cd.Cancer() 
    glass = gd.Glass() 
    Vt = vd.Vote() 
    iris = ird.Iris() 
    soy = sd.Soybean() 






    print("Program Finish")



#Call the main function
main() 
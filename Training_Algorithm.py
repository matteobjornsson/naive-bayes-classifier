import pandas as pd
import numpy as np
import random 

class Training_Algorithm:

    def __init__(self):
        self.attr = []

    def CrossValidation(self,df: pd.DataFrame) -> pd.DataFrame:
        #Calculate the number of records to be sampled for testing 
        TestSize = len(df) * .1 
        #Count until we hit the number of records we want to sample 
        for i in range(TestSize): 
            #Set a value to be a random number from the dataset 
            TestValue = random.randint(0,len(df))
            #Append this row to a new dataframe
            df1.append(df.drop(df.Index[TestValue]))
        Temporary = list() 
        #Return the training and test set data 
        Temporary.append(df,df1)
        return temporary 

            



    # take in dataset and calculate occurence of each class
    def calculateN(self, df: pd.DataFrame) -> dict:
        n = {"class1": 0, "class2": 0}
        # init dict with keys = class names
        # iterate over all rows and increment the class associated with that row
        return n


    # take in n and create a new dict q that is each value / total rows
    def calculateQ(self, n: dict) -> dict:
        q = {"class1": 0, "class2": 0}
        # iterate over every row
            # increment a counter (key = class, value = count) associated with the class of that row
        # divide all counters by the total number of rows
        # return the dictionary
        return q

    def calculateF(self, n: dict, df: pd.DataFrame) -> dict:
        f = {"class1": {"A1": 0, "A2": 0}, "class2": {"A1": 0, "A2": 0}}

        # init nested dict where first layer keys are classes and second layer keys are each possible attribute value
        # iterate over every column that is an attribute
            # iterate over every row
                # increment counter of the class x attribute value 
        # iterate over all values in nested dict
            # add 1 and divide by the count of examples in the class (n[class]) plus the total number of examples
            # i.e. (v + 1)/(n[class] + d)
        return f
import pandas as pd
import numpy as np

"""
loss functions 

use 0/1 loss
https://stats.stackexchange.com/questions/296014/why-is-the-naive-bayes-classifier-optimal-for-0-1-loss
0/1 loss penalizes missclassification


"""


class Results: 

    
    def __init__(self): 
       #0/1 Loss Functions
       self.ClassificationCorrect = list() 
       self.ClassificationWrong = list() 

       #F1 Loss Functions 
       self.TruePositive = list() 
       self.TrueNegative = list() 
       self.FalsePositive = list() 
       self.FalseNegative  = list() 

    def LossFunctionStats(self, df: pd.DataFrame)->list(): 
        ClassificationHypothesis = len(df.columns)
        TrueClassification = len(df.columns -1)
        for i in range(len(df)): 
            if df.iloc[i][ClassificationHypothesis] == df.iloc[i][TrueClassification]: 
                self.ClassificationCorrect.append(df.iloc[i][ClassificationHypothesis])
            else: 
                self.ClassificationWrong.append(df.iloc[i][ClassificationHypothesis]
        TotalTestSet = len(self.ClassificationCorrect) + len(self.ClassificationWrong)
        TotalCorrect = (len(self.ClassificationCorrect) / TotalTestSet) * 100 
        #TotalWrong = (len(self.ClassificationWrong) / TotalTestSet) * 100 
        Statistics = list() 
        Statistics.append(TotalTestSet)
        Statistics.append(TotalCorrect)
        #Statistics.append(TotalWrong)
        return Statistics


    """
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

    
    """


    def F1FunctionBins(self,df:pd.DataFrame): 
        UniqueClassifiers = list() 
        Classes = len(df.columns)-1
        for i in range(len(df)): 
            if df.iloc[i][Classes] in UnqiqueClassifiers: 
                continue 
            UniqueClassifiers.append(df.iloc[i][Classes])
            continue 
        

        ClassificationHypothesis = len(df.columns)
        TrueClassification = len(df.columns -1)
        if df.iloc[i][ClassificationHypothesis] == df.iloc[i][TrueClassification]: 
            #If they are the same then a false positive 

        


if __name__ == '__main__':
    print("Program Start")


    print("Program Finish")




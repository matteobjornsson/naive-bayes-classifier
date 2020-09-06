import pandas as pd
import numpy as np

"""
loss functions 

use 0/1 loss
https://stats.stackexchange.com/questions/296014/why-is-the-naive-bayes-classifier-optimal-for-0-1-loss
0/1 loss penalizes missclassification
https://stats.stackexchange.com/questions/284028/0-1-loss-function-explanation/284062

multiclass confusion matrix 
https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier

multiclass precision and recall
https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

multiclass f1 score
https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

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

    def ZeroOneLossFunctionStats(self, df: pd.DataFrame)->list(): 
        ClassificationHypothesis = len(df.columns)
        TrueClassification = len(df.columns -1)
        for i in range(len(df)): 
            if df.iloc[i][ClassificationHypothesis] == df.iloc[i][TrueClassification]: 
                self.ClassificationCorrect.append(df.iloc[i][ClassificationHypothesis])
            else: 
                self.ClassificationWrong.append(df.iloc[i][ClassificationHypothesis])
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
        pass

    def recall(self, matrix):
        pass

    def ConfusionMatrix(self, df: pd.DataFrame) -> pd.DataFrame:
        UniqueClassifiers = list() 
        Classes = len(df.columns)-2
        for i in range(len(df)): 
            if df.iloc[i][Classes] in UniqueClassifiers: 
                continue 
            UniqueClassifiers.append(df.iloc[i][Classes])
            continue 
        zeroArray = np.zeros(shape=(Classes, Classes))
        matrix = pd.DataFrame(zeroArray, columns=UniqueClassifiers, index=UniqueClassifiers)
        for i in range(len(df)):
            truth = df.iloc[i][TrueClassification]
            guess = df.iloc[i][ClassificationHypothesis]
            matrix.at[truth, guess] += 1
        return matrix




        


if __name__ == '__main__':
    print("Program Start")


    print("Program Finish")




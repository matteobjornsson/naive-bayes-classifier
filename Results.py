#################################################################### MODULE COMMENTS ####################################################################
##
##
#################################################################### MODULE COMMENTS ####################################################################

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
    #Parameters: 
    #Returns: 
    #Function: 
    def ZeroOneLossFunctionStats(self, df: pd.DataFrame)->list(): 
        ClassificationHypothesis = len(df.columns)
        TrueClassification = len(df.columns) -1
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
    
    #Parameters: 
    #Returns: 
    #Function: 
    # return a list of true positive counts for each class
    def truePositive(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        tp = []
        for i in classCount:
            # true positive for each class is where truth == guess
            tp.append(cMatrix.iloc[i][i])
        return tp

    #Parameters: 
    #Returns: 
    #Function: 
    # return a list of false positive counts for each class
    def falsePositive(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        fp = []
        for i in classCount:
            count = 0
            for j in classCount:
                if i == j:
                    continue
                else:
                    # false positive is the sum of every count in the class
                    # column except the true positive count
                    count += cMatrix.iloc[j][i]
            fp.append(count)
        return fp

    #Parameters: 
    #Returns: 
    #Function: 
    # return a list of false negative counts for each class
    def falseNegative(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        fn = []
        for i in classCount:
            count = 0
            for j in classCount:
                if i == j:
                    continue
                else:
                    # false negative is the sum of every count in the class
                    # row except the true positive count
                    count += cMatrix.iloc[i][j]
            fn.append(count)
        return fn

    #Parameters: 
    #Returns: 
    #Function: 
    # return a list of true negative counts for each class
    def trueNegative(self, classCount: range, cMatrix: pd.DataFrame, tp: list, fp: list, fn: list) -> list:
        tn = []
        for i in classCount:
            count = 0
            # sum the value of every cell
            for j in classCount:
                for k in classCount:
                    count += cMatrix.iloc[j][k]
            # true negative counts are the sum of every cell minus true 
            # positive, false positive and false negative.
            count = count - tp[i] - fp[i] - fn[i]
            tn.append(count)
        return tn

    #Parameters: 
    #Returns: 
    #Function:  
    # create a stats summary matrix for all classes
    def classStats(self, cMatrix: pd.DataFrame) -> pd.DataFrame:
        # grab the class names
        ClassList = list(cMatrix.columns.values)
        classCount = range(len(ClassList))
        # init an empty matrix with class indexes labeled
        statsMatrix = pd.DataFrame(index=ClassList)

        # calculate stats
        tp = self.truePositive(classCount, cMatrix)
        fp = self.falsePositive(classCount, cMatrix)
        fn = self.falseNegative(classCount, cMatrix)
        tn = self.trueNegative(classCount, cMatrix, tp, fp, fn)

        # insert stats into matrix
        statsMatrix["TP"] = tp
        statsMatrix["FP"] = fp
        statsMatrix["FN"] = fn
        statsMatrix["TN"] = tn

        return statsMatrix
    #Parameters: 
    #Returns: 
    #Function: 
    # generate a matrix that checks classified test data against ground truth
    def ConfusionMatrix(self, df: pd.DataFrame) -> pd.DataFrame:
        # identify column index of ground truth and classification
        GroundTruthIndex = len(df.columns)- 2
        ClassifierGuessIndex = len(df.columns)-1 

        # generate a list of all unique classes
        UniqueClasses = list() 
        for i in range(len(df)): 
            if df.iloc[i][GroundTruthIndex] in UniqueClasses: 
                continue 
            UniqueClasses.append(df.iloc[i][GroundTruthIndex])
            continue 
        ClassCount = len(UniqueClasses)

        # initialize empty confusion matrix
        zeroArray = np.zeros(shape=(ClassCount, ClassCount))
        matrix = pd.DataFrame(zeroArray, columns=UniqueClasses, index=UniqueClasses)

        for i in range(len(df)):
            # for each example, increment a counter where row = truth, col = guess
            truth = df.iloc[i][GroundTruthIndex]
            guess = df.iloc[i][ClassifierGuessIndex]
            matrix.at[truth, guess] += 1
            continue
        return matrix


if __name__ == '__main__':
    print("Program Start")


    print("Program Finish")




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

    #On the creation of a Results object 
    def __init__(self): 
       #0/1 Loss Function Lists 
       self.ClassificationCorrect = list() 
       self.ClassificationWrong = list() 

       #F1 Loss Function Lists 
       self.TruePositive = list() 
       self.TrueNegative = list() 
       self.FalsePositive = list() 
       self.FalseNegative  = list() 

    #Parameters: DataFrames
    #Returns: List 
    #Function: 
    def ZeroOneLossFunctionStats(self, df: pd.DataFrame)->list(): 
        ClassificationHypothesis = len(df.columns)-1
        TrueClassification = len(df.columns) -2
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
    #Parameters: Dataframe 
    #Returns: DataFrame, Dictionary 
    #Function: 
    def statsSummary(self, df: pd.DataFrame) -> (pd.DataFrame, dict, dict):
        classStats = self.perClassStats(df)
        tpList = list(classStats["TP"])
        fpList = list(classStats["FP"])
        fnList = list(classStats["FN"])
        microStats = self.microAverageStats(tpList, fpList, fnList)
        macroStats = self.weightedMacroAverageStats(classStats)
        return classStats, microStats, macroStats
    #Parameters: DataFrame 
    #Returns: DataFrame
    #Function: 
    def perClassStats(self,df:pd.DataFrame): 
        cMatrix = self.ConfusionMatrix(df)
        classStats = self.classStats(cMatrix)
        return classStats

    def weightedMacroAverageStats(self, perClassStats) -> dict: 
        classValues = list(perClassStats.index.values)
        macroStatsDict = {}
        classCounts = self.countClassOccurence(perClassStats)

        totalClassCount = 0
        for key in classCounts.keys():
            totalClassCount += classCounts[key]

        macroRecallAverage = 0
        macroPrecisionAverage = 0
        macroF1Average = 0
        for i in range(len(classValues)):
            macroRecallAverage += (perClassStats["Recall"].iloc[i] * classCounts[classValues[i]])
            macroPrecisionAverage += (perClassStats["Precision"].iloc[i] * classCounts[classValues[i]])
            macroF1Average += (perClassStats["F1"].iloc[i] * classCounts[classValues[i]])
        
        macroStatsDict["macroRecall"] = macroRecallAverage / totalClassCount
        macroStatsDict["macroPrecision"] = macroPrecisionAverage / totalClassCount
        macroStatsDict["macroF1"] = macroF1Average / totalClassCount

        return macroStatsDict
        

    def countClassOccurence(self, perClassStats: pd.DataFrame) -> dict:
        classValues = list(perClassStats.index.values)
        classCounts = dict.fromkeys(classValues)
        # print("perclass stats from countclass occurence method: \n", perClassStats)
        for i in range(len(classValues)):
            x = perClassStats.iloc[i]
            count = 0
            for stat, statValue in x.items():
                count += statValue
            classCounts[classValues[i]] = count
        return classCounts
        

    def microAverageStats(self, truePositives: list, falsePositives: list, falseNegatives: list) -> dict:
        microStatsDict = {}

        tpSum = sum(truePositives)
        fpSum = sum(falsePositives)
        fnSum = sum(falseNegatives)

        microRecall = self.recall(tpSum, fnSum)
        microPrecision = self.precision(tpSum, fpSum)

        microStatsDict["microRecall"] = microRecall
        microStatsDict["microPrecision"] = microPrecision
        microStatsDict["microF1Score"] = self.f1Score(precision=microPrecision, recall=microRecall)

        return microStatsDict

    #Parameters: Float, Float
    #Returns: Float 
    #Function: 
    def f1Score(self, precision, recall) -> float:
            if (precision + recall) == 0:
                return 0
            else:
                return 2 * precision * recall / (precision + recall)

    #Parameters: Inter, Integer
    #Returns: Float 
    #Function: 
    def recall(self, truePositiveCount, falseNegativeCount) -> float:
        if truePositiveCount + falseNegativeCount == 0:
            return 0
        else:
            return truePositiveCount/(truePositiveCount + falseNegativeCount)
    #Parameters: Integer, Integer
    #Returns: Float
    #Function: 
    def precision(self, truePositiveCount, falsePositiveCount) -> float:
        if truePositiveCount + falsePositiveCount == 0:
            return 0
        else:
            return truePositiveCount/(truePositiveCount + falsePositiveCount)
    
    #Parameters: Integer, DataFame
    #Returns: List 
    #Function: return a list of true positive counts for each class
    def truePositive(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        tp = []
        # print("cmatrix true positive method: \n", cMatrix)
        for i in classCount:
            # true positive for each class is where truth == guess
            tp.append(cMatrix.iloc[i][i])
        return tp

    #Parameters: Integer, DataFrame 
    #Returns: List 
    #Function: return a list of false positive counts for each class
    def falsePositive(self, classCount: range, cMatrix: pd.DataFrame) -> list:
        # print("cmatrix in true positive method: \n")
        # print(cMatrix)
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

    #Parameters: Integer, DataFrame 
    #Returns: List 
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

    #Parameters: Int, DataFrame, List, List,List 
    #Returns: List 
    #Function: return a list of true negative counts for each class
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

    #Parameters: DataFrame
    #Returns: DataFrame
    #Function: create a stats summary matrix for all classes
    def classStats(self, cMatrix: pd.DataFrame) -> pd.DataFrame:
        # grab the class names
        #Print some data to the screen 
        # print("cmatrix stats method: \n")
        # print(cMatrix)
        #Create a list to the column names from the dataframe 
        ClassList = list(cMatrix.columns.values)
        #Print some data to the screen 
        # print("classList class stats method: \n")
        # print(ClassList)
        #Sent a class count to the len of the class list 
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
        #Create some empty lists 
        precisionList = []
        recallList = []
        fScoreList = []
        #For each of the values in the class count 
        for i in classCount:
            #Get the value of the given value 
            singleClassStats = statsMatrix.iloc[i]

            tp = singleClassStats["TP"]
            fp = singleClassStats["FP"]
            fn = singleClassStats["FN"]
            #Set the following values 
            prec = self.precision(tp, fp)
            rec = self.recall(tp, fn)
            #Get the value of the f1 score 
            f1 = self.f1Score(prec, rec)
            #Print some data to the screen 
            # print("i: ", i, " prec: ", prec, " recall: ", rec, " f1: ", f1)
            #Append data to the following lists 
            precisionList.append(prec)
            recallList.append(rec)
            fScoreList.append(f1)

        statsMatrix["Precision"] = precisionList
        statsMatrix["Recall"] = recallList
        statsMatrix["F1"] = fScoreList
        #Return the Dataframe 
        return statsMatrix

    #Parameters: DataFrame 
    #Returns: DataFrame
    #Function: generate a matrix that checks classified test data against ground truth
    def ConfusionMatrix(self, df: pd.DataFrame) -> pd.DataFrame:
        # identify column index of ground truth and classification
        GroundTruthIndex = len(df.columns)- 2
        ClassifierGuessIndex = len(df.columns)-1 

        # generate a list of all unique classes
        UniqueClasses = list() 
        #Loop through all of the rows in the dataframe 
        for i in range(len(df)): 
            #If the Ground truth classification not in the unique classes list 
            if str(df.iloc[i][GroundTruthIndex]) not in UniqueClasses:
                #Append the value to the list 
                UniqueClasses.append(str(df.iloc[i][GroundTruthIndex]))
            #If the classifcation is not in the unique classes list 
            if str(df.iloc[i][ClassifierGuessIndex]) not in UniqueClasses:
                #Add the value to the list 
                UniqueClasses.append(str(df.iloc[i][ClassifierGuessIndex]))
            #Go to the next one 
            continue 
        #Set the class count to the length of unique classes 
        ClassCount = len(UniqueClasses)

        # initialize empty confusion matrix
        zeroArray = np.zeros(shape=(ClassCount, ClassCount))
        #Set a variable to a dataframe that has the columns from the unique classes 
        matrix = pd.DataFrame(zeroArray, columns=UniqueClasses, index=UniqueClasses)
        #Print some data to the screen 
        # print("empty Cmatrix, cmatrix method: \n", matrix)
        #For each of the rows in the dataframe 
        for i in range(len(df)):
            # for each example, increment a counter where row = truth, col = guess
            truth = str(df.iloc[i][GroundTruthIndex])
            guess = str(df.iloc[i][ClassifierGuessIndex])
            #Increment the count 
            matrix.at[truth, guess] += 1
            #Go to the next one 
            continue
        #Return the dataframe 
        return matrix


#Unit Testing the object created above 
#Code not run on creation of object just testing function calls and logic above 
if __name__ == '__main__':
    print("Program Start")


    print("Program Finish")




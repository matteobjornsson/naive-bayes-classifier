import pandas as pd
import numpy as np
import random 
import sys 
import copy 
import pprint

class TrainingAlgorithm:

    def __init__(self):
        self.attr = []


    #ASSUMING THE ID COLUMN IS YEETED 
    def ShuffleData(self, df: pd.DataFrame) ->list(): 
        df1 = copy.deepcopy(df)
        #Calculate the number of records to be sampled for testing 
        TestSize = int((len(df.columns)-1) * .1)
        Shuffled = list() 
        for i in range(TestSize): 
            while(True): 
                Column_Shuffle = random.randint(0,len(df.columns))
                if Column_Shuffle in Shuffled :
                    continue 
                else: 
                    break 
            Shuffled.append(Column_Shuffle)
            temp = list()
            for j in range(len(df)):
                temp.append(df.iloc[j][Column_Shuffle])
            for j in range(len(df)): 
                value = random.randint(0,len(temp)) 
                
                df1.at[j,df.columns[Column_Shuffle]] = temp[value-1]
                temp.remove(temp[value-1])
        return df1

 

    def CrossValidation(self,df: pd.DataFrame) -> list():
        columnss = list() 
        for i in df.columns: 
            columnss.append(i)
        df1 = pd.DataFrame(columns = columnss)
        #Calculate the number of records to be sampled for testing 
        TestSize = len(df) * .1 
        #Count until we hit the number of records we want to sample 
        for i in range(int(TestSize)): 
            #Set a value to be a random number from the dataset 
            TestValue = random.randint(0,len(df)-1)
            #Append this row to a new dataframe
            df1.loc[i] = df.index[TestValue]
            df =  df.drop(df.index[TestValue])
            
            #df1.loc[i] = df.drop(df.loc[TestValue].index,inplace =True)
        Temporary = list() 
        #Return the training and test set data 
        Temporary.append(df)
        Temporary.append(df1)
        return Temporary 

    def BinTestData(self, df: pd.DataFrame) -> list(): 
        #10 bins
        Binsize = 10          
        BinsInList = list()     
        #Calculate the number of records to be sampled for testing 
        TestSize = len(df) * .1 
        for i in range(Binsize):
            if i == Binsize-1: 
                BinsInList.append(df)
                break
            columnss = list() 
            for i in df.columns: 
                columnss.append(i)
            df1 = pd.DataFrame(columns = columnss)
            #Count until we hit the number of records we want to sample 
            for i in range(int(TestSize)):
                #Set a value to be a random number from the dataset 
                TestValue = random.randint(0,len(df)-1)
                #Append this row to a new dataframe
                df1.loc[i] = df.index[TestValue]
                df =  df.drop(df.index[TestValue])
            BinsInList.append(df1)

        return BinsInList




    # take in dataset and count occurence of each class
    def calculateN(self, df: pd.DataFrame) -> dict:
        CountsPerClass = {}
        ClassColumn = len(df.columns)-1
        # iterate over all all examples of test data
        for i in range(len(df)):
            # grab the ground truth for the data point
            ClassValue = df.iloc[i][ClassColumn] 
            # if the class value has already been counted once before, increment
            if ClassValue in CountsPerClass: 
               CountsPerClass[ClassValue] += 1 
            # else add that class to the count
            else:
                CountsPerClass[ClassValue] = 1
            continue   
        # return a dictionary with class keys and count values         
        return CountsPerClass


    # take in n and create a new dict q that is each value / total rows
    def calculateQ(self, n: dict, TotalRows) -> dict:
        QValue = {} 
        for k in n.keys(): 
            QValue[k] = n[k] / TotalRows
        return QValue

    def calculateF(self, n: dict, df: pd.DataFrame) -> dict:
        fMatrix = {}
        print(df.head)
        columnList = list(df.columns.values)[:-1]
        for col in columnList: 
            for row in range(len(df)): 
                AttributeValue = df.at[row, col]
                ClassValue = df.iloc[row][len(df.columns)-1]

                if ClassValue in fMatrix.keys(): 
                    if col in fMatrix[ClassValue].keys():
                        if AttributeValue in fMatrix[ClassValue][col].keys(): 
                            fMatrix[ClassValue][col][AttributeValue] +=1 
                        else: 
                            fMatrix[ClassValue][col][AttributeValue] = 1 
                    else: 
                        fMatrix[ClassValue][col] = {AttributeValue:1}
                else: 
                    fMatrix[ClassValue] = {col:{AttributeValue:1}}
       
        for ClassValueI in fMatrix.keys():
            for Attribute in fMatrix[ClassValueI].keys(): 
                for AttributeValue in fMatrix[ClassValueI][Attribute].keys(): 
                    Value = fMatrix[ClassValueI][Attribute][AttributeValue] 
                    Value +=1 
                    fMatrix[ClassValueI][Attribute][AttributeValue]  = (Value/(n[ClassValueI] + (len(df.columns)-1)))
        return fMatrix
               
                
                
    #List = { 0,1,2,3,4,5,6,7,8,9,}
        
        #f = {"class1": {"A1": 0, "A2": 0}, "class2": {"A1": 0, "A2": 0}}

        # init nested dict where first layer keys are classes and second layer keys are each possible attribute value
        # iterate over every column that is an attribute
            # iterate over every row
                # increment counter of the class x attribute value 
        # iterate over all values in nested dict
            # add 1 and divide by the count of examples in the class (n[class]) plus the total number of examples
            # i.e. (v + 1)/(n[class] + d)

if __name__ == '__main__':
    import Classifier
    import Results

    print("Program Start")
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    # ta = Training_Algorithm() 
    # te = ta.CrossValidation(df)
    # for i in range(len(te)): 
    #     print("LENGTHS: ")
    #     print(len(te[i]))
    # test = ta.BinTestData(df)
    # print("=================")
    # for i in range(len(test)): 
    #     print("LENGTHS")
    #     print(len(test[i]))
    # df1 = ta.ShuffleData(df)

    #df2 = pd.DataFrame(df1)
    # count = 0 
    # for i in range(len(df)): 
    #     for j in range(len(df.columns)):
           
    #         #print(df2.iloc[i][j])
    #         if df.iloc[i][j] != df1.iloc[i][j]:
    #             print("CHANGE")
    #             count+=1 
    # print(count)
    ta = TrainingAlgorithm()
    n = ta.calculateN(df)
    print("n: ", n)

    q = ta.calculateQ(n, len(df))
    print("q: ", q)

    f = ta.calculateF(n, df)
    print("f: \n")
    pprint.pprint(f)

    c = Classifier.Classifier(n, q, f)
    classified = c.classify(df)
    print(classified)

    r = Results.Results()
    cM = r.ConfusionMatrix(classified)

    print(cM)

    print("Program Finish")
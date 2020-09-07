import pandas as pd
import numpy as np
import random 
import sys 
import copy 
import pprint

class TrainingAlgorithm:

    def __init__(self):
        self.attr = []



    def ShuffleData(self, df: pd.DataFrame) ->list(): 
        #Get a deep copy of the dataframe 
        df1 = copy.deepcopy(df)
        #Calculate the number of records to be sampled for testing 
        TestSize = int((len(df.columns)-1) * .1)
        if TestSize ==  0: 
            TestSize = 1
        #intialize an empty list to store all of the data frames
        Shuffled = list() 
        #Loop through the number of columns that need to have data shuffled 
        for i in range(TestSize): 
            #Just continue until we break 
            while(True): 
                #Set a variable to a random number for the column to be shuffled around 
                Column_Shuffle = random.randint(0,len(df.columns)-1)
                #If the column number is in the list above then it has been shuffled, try again 
                if Column_Shuffle in Shuffled :
                    #Go to the top of the loop 
                    continue 
                else: 
                    #We found a new column that needs to be shuffled, break out of the loop 
                    break 
            #Append the column number to the list to save what columns have been shuffled
            Shuffled.append(Column_Shuffle)
            #Create a temp list 
            temp = list()
            #Loop through the number of rows in the data frame
            for j in range(len(df)):
                #Append the value in a given cell for a given column to a list 
                temp.append(df.iloc[j][Column_Shuffle])
            #Loop through weach row in the data frame again 
            for j in range(len(df)): 
                #Pull a value out from the size of the list 
                value = random.randint(0,len(temp)-1) 
                #Set the dataframe value at this position to a random value from the list 
                df1.at[j,df.columns[Column_Shuffle]] = temp[value]
                #Remove the value that was radomly assigned
                temp.remove(temp[value])
        #Return the Data Frame 
        return df1
 
    def CrossValidation(self,df: pd.DataFrame) -> list():
        #Create an empty list 
        columnss = list() 
        #For each of the columns in the dataframe
        for i in df.columns: 
            #Append the column name to the list we created above 
            columnss.append(i)
        #Create a dataframe that has the same columns in the same format as the list we created
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
        #Return the List of dataframes
        return Temporary 

    def BinTestData(self, df: pd.DataFrame) -> list(): 
        #Set the binsize to be 10 
        Binsize = 10    
        #Create an empty list     
        BinsInList = list()     
        #Calculate the number of records to be sampled for testing 
        TestSize = len(df) * .1 
        #Loop through the number of bins we set the binsize to
        for i in range(Binsize):
            #If the counter is on the last bin 
            if i == Binsize-1: 
                #Append the dataframe to the list 
                BinsInList.append(df)
                #Break out of the loop 
                break
            #Create a new column list 
            columnss = list() 
            #For each of the columns in the dataframe 
            for i in df.columns: 
                #Append the column name to the list 
                columnss.append(i)
            #Create a dataframe with the proper columns as above 
            df1 = pd.DataFrame(columns = columnss)
            #Count until we hit the number of records we want to sample 
            for i in range(int(TestSize)):
                #Set a value to be a random number from the dataset 
                TestValue = random.randint(0,len(df)-1)
                #Append this row to a new dataframe
                df1.loc[i] = df.index[TestValue]
                #Remove the dataframe row we appended above from the dataframe
                df =  df.drop(df.index[TestValue])
            #Append the dataframe created to the list
            BinsInList.append(df1)
        #Return the list of dataframes 
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
        # divide each class count by total data points
        for k in n.keys(): 
            QValue[k] = n[k] / TotalRows
        return QValue

    # generate the smoothed past probability of each feature value found in a given class
    def calculateF(self, n: dict, df: pd.DataFrame) -> dict:
        fMatrix = {}
        # get list of all features
        columnList = list(df.columns.values)[:-1]

        # F(Aj = ak, C = ci) is calculated per feature, where ak is a specific 
        # value of the feature and ci is a specific class value.
        for feature in columnList: 
            for row in range(len(df)):
                # for each feature, iterate over every row (feature vector x).
                # save the value of the feature and the associated ground truth
                # for the row. 
                AttributeValue = df.at[row, feature]
                ClassValue = df.iloc[row][len(df.columns)-1]

                # for each (class value, feature, feature value) combination encountered,
                # increment the counter for that combination. If the combination
                # has not been found yet, add the necessary keys and counter.
                if ClassValue in fMatrix.keys(): 
                    if feature in fMatrix[ClassValue].keys():
                        if AttributeValue in fMatrix[ClassValue][feature].keys():
                            fMatrix[ClassValue][feature][AttributeValue] +=1 
                        else: 
                            fMatrix[ClassValue][feature][AttributeValue] = 1 
                    else: 
                        fMatrix[ClassValue][feature] = {AttributeValue:1}
                else: 
                    fMatrix[ClassValue] = {feature:{AttributeValue:1}}
        # after counting all attribute values per class per feature, modify each
        # by dividing by the count of the associated class plus the number of 
        # features, and +1 in the numerator (+1 and + features are for smoothing)
        for ClassValueI in fMatrix.keys():
            for Attribute in fMatrix[ClassValueI].keys(): 
                for AttributeValue in fMatrix[ClassValueI][Attribute].keys(): 
                    Value = fMatrix[ClassValueI][Attribute][AttributeValue] 
                    Value +=1 
                    fMatrix[ClassValueI][Attribute][AttributeValue] = (
                        (Value/(n[ClassValueI] + (len(df.columns)-1)))
                    )
        # return the complete "past feature value probability matrix"
        return fMatrix


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
    # print("n: ", n)

    q = ta.calculateQ(n, len(df))
    # print("q: ", q)

    f = ta.calculateF(n, df)
    # print("f: \n")
    # pprint.pprint(f)

    print("input dataframe: ")
    print(df.head)

    c = Classifier.Classifier(n, q, f)
    classified = c.classify(df)
    print("classified dataframe")
    print(classified)

    r = Results.Results()
    cM = r.ConfusionMatrix(classified)

    print("Confusion Matrix")
    print(cM)

    stats = r.classStats(cM)
    print(stats)

    print("Program Finish")
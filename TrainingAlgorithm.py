#################################################################### MODULE COMMENTS ####################################################################
#The Training Algorithm python object is #
##
##
#################################################################### MODULE COMMENTS ####################################################################
import pandas as pd
import numpy as np
import random 
import sys 
import copy 
import pprint

class TrainingAlgorithm:

    #Parameters: Dataframe
    #Returns: List of Dataframes
    #Function: This function takes in adataframe and breaks the dataframe down into multiple dataframes that are returned Goal of this is to separate testing and training datasets
    def ShuffleData(self, df: pd.DataFrame) ->list(): 
        #Get a deep copy of the dataframe 
        df1 = copy.deepcopy(df)
        #Calculate the number of records to be sampled for testing 
        TestSize = int((len(df.columns)-1) * .1)
        #if the test size is 0 
        if TestSize ==  0: 
            #Set it to 1 
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
    #Parameters: Dataframe 
    #Returns: List of dataframes
    #Function: Take in a given dataframe and break the dataframe down into 2 dataframes, a test and training dataframe. Append both of those to a list and return the list 
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
            #Drop the row from the dataframe 
            df =  df.drop(df.index[TestValue])
            #df1.loc[i] = df.drop(df.loc[TestValue].index,inplace =True)
        Temporary = list() 
        #Return the training and test set data 
        Temporary.append(df)
        Temporary.append(df1)
        #Return the List of dataframes
        return Temporary 
    #Parameters: DataFrame
    #Returns: List of dataframes 
    #Function: Take in a dataframe and break dataframe into 10 similar sized sets and append each of these to a list to be returned 
    def BinTestData(self, df: pd.DataFrame) -> list(): 
        #Set the bin size to 10 
        Binsize = 10
        #Create a List of column names that are in the dataframe 
        columnHeaders = list(df.columns.values)
        #Create an empty list 
        bins = []
        #Loop through the size of the bins 
        for i in range(Binsize):
            #Append the dataframe columns to the list created above 
            bins.append(pd.DataFrame(columns=columnHeaders))
        #Set a list of all rows in the in the dataframe 
        dataIndices = list(range(len(df)))
        #Shuffle the data 
        random.shuffle(dataIndices)
        #Shuffle the count to 0 
        count = 0
        #For each of the indexs in the dataIndices 
        for index in dataIndices:
            #Set the bin number to count mod the bin size 
            binNumber = count % Binsize
            bins[binNumber] = bins[binNumber].append(df.iloc[index], ignore_index=True)
            #Increment count 
            count += 1
            #Go to the next 
            continue
        #Return the list of Bins 
        return bins

    #Parameters: DataFrame 
    #Returns: Dictionary of the calculated N 
    #Function: take in dataset and count occurence of each class
    def calculateN(self, df: pd.DataFrame) -> dict:
        CountsPerClass = {}
        ClassColumn = len(df.columns)-1
        # iterate over all all examples of test data
        for i in range(len(df)):
            # grab the ground truth for the data point
            ClassValue = df.iloc[i][ClassColumn] 
            # if the class value has already been counted once before, increment
            if ClassValue in CountsPerClass:
                #Increment by 1  
               CountsPerClass[ClassValue] += 1 
            # else add that class to the count
            else:
                #Set the value to 1 
                CountsPerClass[ClassValue] = 1
            #Go to the next one 
            continue   
        # return a dictionary with class keys and count values         
        return CountsPerClass

    #Parameters: Dictionary N from function above and Total Rows in dataframe 
    #Returns: Dictionary 
    #Function: take in n and create a new dict q that is each value / total rows
    def calculateQ(self, n: dict, TotalRows) -> dict:
        #Set an empty dictionary
        QValue = {} 
        # divide each class count by total data points
        for k in n.keys(): 
            #Given Key K take the N value at Key K divided by The total number of rows 
            QValue[k] = n[k] / TotalRows
        #Return the Q value 
        return QValue
    #Parameters: Dictionary, Dataframe 
    #Returns: Dictionary 
    #Function: generate the smoothed past probability of each feature value found in a given class
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

#Unit Testing the object created above 
#Code not run on creation of object just testing function calls and logic above 
if __name__ == '__main__':
    import Classifier
    import Results

    print("Program Start")
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    ta = TrainingAlgorithm()
    cross_validation_chunks = ta.BinTestData(df)

    print("Program Finish")
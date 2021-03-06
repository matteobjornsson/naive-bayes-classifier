#################################################################### MODULE COMMENTS ############################################################################
#The following class is a python object that takes in the libraries: Nunmpy, Pandas, Sys and Random.                                                            #
#The python Object "DataProcessor" that is created below is a python object designed to take in a pandas dataframe and formats the data such that is can be     #
#Run into a Naive Bayes learning algorithm. The data processing function can discretize an entire dataset, and remove missing attribute values from a dataset   #
#The removal of missing attribute values is done first by identifying the percentage of rows that have missing data, if that percentage is less than 5% then we #
#Drop all of rows that have a missing value. A similar tactic is pursued for columns missing data, if the percentage of columns missing data is less than 5%    #   
#Then we drop the entire column. If the value is greater than 5 % then we randomly generate a new value to replace the missing attribute in the data set        #
#################################################################### MODULE COMMENTS ############################################################################
import pandas as pd
import numpy as np
import sys
import random 
import copy 
import math 
import TrainingAlgorithm 


class DataProcessor:
    #On the creation of a Dataprocessor object set the following values 
    def __init__(self, bin_count):
        #Total number of bins to bin the non-discrete values 
        self.bin_count = bin_count
        #Set the percentage of missing values to be dropped 
        self.PercentBeforeDrop = 10.00 
        #Set the missing value row index to an empty set 
        self.MissingRowIndexList = set() 
        #SEt the missing value column index to an empty set 
        self.MissingColumnNameList = set()

    #Parameters: Pandas DataFrame 
    #Returns: Clean ready to process Dataframe 
    #Function: This is the main function that should be called for each object that takes in the dataframe, processes it and returns the clean dataframe 
    def StartProcess(self, df:pd.DataFrame) -> pd.DataFrame:
        #Get a deep copy of the dataframe 
        df1 = copy.deepcopy(df)
        #SEt the count to 0 
        count = 0 
        #For each of the columns in the dataframe 
        for i in range(len(df.columns)): 
            #If the count is at the last column in the dataframe end because this is the classifier 
            if count == len(df.columns)-1: 
                #Break 
                break
            #bin Integers
           
            #If the type of the dataframe is a float then we need to discretize 
            if type(df1.iloc[0][i]) == np.float64: 
                #Find which column needs to be discretized
                df1 = self.discretize(df1,i)
                #Increment the count 
                count+=1
                #Go to the next one
                continue 
            #If the data frame has missing attributes 
            if self.has_missing_attrs(df1): 
                #Remove the missing attributes 
                df1 = self.fix_missing_attrs(df1)
            #Increment the count 
            count+=1
        #Return the cleaned dataframe 
        return df1

    #NIU
    #Parameters: Pandas DataFrame, Integer Column 
    #Returns: Dataframe -> with all values randomly assigned 
    #Function:  Take in a dataframe and weight each value in the dataframe with an occurence then fill in a missing attribute based on the weight of the value in the dataframe 
    def RandomRollInts(self, df: pd.DataFrame, col) -> pd.DataFrame: 
        #Set the min value to the first value in the dataframe 
        Min = df.iloc[1][col]
        #Set the max value to the first value in the dataframe 
        Max = df.iloc[1][col]
        #Loop through each row in the dataframe 
        for i in range(self.CountTotalRows(df)):
            #If the  Value in the dataframe is a missing value 
            if self.IsMissingAttribute(df.iloc[i][col]): 
                #Do nothing 
                continue 
            #Otherwise 
            else: 
                #If the value in the dataframe is greater than the recorded max 
                if df.iloc[i][col]  > Max: 
                    #Assign this value to the max 
                    Max = df.iloc[i][col] 
                    #Go to the next 
                    continue 
                #If the value is less than the recorded min 
                elif df.iloc[i][col] < Min: 
                    #Assign the new min value 
                    Min = df.iloc[i][col]
                    #Go to the next
                    continue 
                #Go to the next 
                continue                 
        #For each of the columns in the dataframe 
        for col in range(self.TotalNumberColumns(df)):
            #For each of the rows in the dataframe 
            for row in range(self.TotalNumberRows(df)):
                #If the value in the dataframe is a missing attribute  
                if self.IsMissingAttribute(df.iloc[col][row]): 
                    #Assign the roll to a random value between min and max 
                    roll = random.randint(Min,Max)
                    #Assign the random value about to the position in the dataframe 
                    df.loc[row,col] = roll   
        #Return the dataframe 
        return df 
   
    #Parameters: Pandas DataFrame 
    #Returns: A dataframe with all missing values filled in with a Y or N 
    #Function: Take in a dataframe and randomly assigned a Y or a N to a missing value 
    def RandomRollVotes(self, df: pd.DataFrame) -> pd.DataFrame: 
        #Loop through each of the rows in the dataframe 
         for i in range(len(df)):
            #loop through all of the columns except the classification column
            for j in range(len(df.columns)-1): 
                #If the given value in the dataframe is missing a value 
                if self.IsMissingAttribute(df.iloc[i][j]): 
                    #Randomly assign a value from 1 - 100 
                    roll = random.randint(0,99) + 1
                    #If the roll is greater than 50 
                    if roll >50: 
                        #Assign the value to a Y 
                        roll = 'y'
                    #Otherwise 
                    else: 
                        #Assign the value to a N 
                        roll = 'n'
                    #Set the position in the dataframe equal to the value in the roll  
                    df.iloc[i][j] = roll
                #Go to the next  
                continue  
         #Return the dataframe 
         return df 
  
    
    #NIU
    #Parameters: Pandas DataFrame 
    #Returns: Integer: Number of times a value appears in a dataframe 
    #Function: Take in column, a dataframe and a value and return the number of times that value appears in the given column in the dataframe 
    def Occurence(self,Column,df:pd.DataFrame,Value) -> int:
        #Set count to 0 
        count = 0  
        #Loop through each row in the dataframe 
        for i in range(len(df)): 
            #If the value in the columna and given row is equal to the value taken in as a parameter
            if df.iloc[i][Column] == Value:
                #Increment the count 
                count += 1 
            #Go to the next one 
            continue
        #Reuturn the total count 
        return count 

   
   
    #NIU
    #Parameters: Pandas DataFrame 
    #Returns: Dataframe with all of the int values filled in
    #Function: Take in a dataframe and fill in each missing value with a random value based on the weight of the occurence of the given values in the dataframe 
    def StatsFillInInts(self,df:pd.DataFrame) -> pd.DataFrame: 
        #Set a weighted vote string
        WeightedVote = ''
        #Set a unweighted vote string 
        UnweightedVote = '' 
        #For each column in the data frame
        for col in range(self.TotalNumberColumns(df)):
            #Go through each row 
            for row in range(self.TotalNumberRows(df)): 
                #If the given cell value is missing 
                if self.IsMissingAttribute(df.iloc[col][row]): 
                    #Get the total number os yes votes in the column 
                    yay = self.Occurence(col,df,'y')
                    #Get the total number of no votes in the column 
                    nay = self.occurence(col,df,'n')
                    #Get the total number of percent Yays
                    PercentYay = (yay/ len(df))
                    #Get the total percent of Nays 
                    PercentNay = (nay/len(df))
                    #If we have more yes's than nos 
                    if PercentYay > PercentNay: 
                        #Set a max value to the percent yes's
                        Max = PercentYay
                        #SEt nay to be the remaining count 
                        PercentNay = 1 - PercentYay
                        #Set the weighted vote value 
                        WeightedVote = 'y'
                        #SEt the unweighted vote value 
                        UnweightedVote ='n'
                    #Otherwise 
                    else: 
                        #Set the max equal to the percent saying no 
                        Max = PercentNay
                        #Set the percent saying yes to the inverse of percent no 
                        PercentYay = 1 - PercentNay
                        #Set the weighted vote to N 
                        WeightedVote = 'n'
                        # Set the unweighted vote to Y 
                        UnweightedVote ='y'
                    #Randomly roll a value between 0 and 1 
                    Stats = random() 
                    #If the stat rolled it greater than max 
                    if Stats > Max: 
                        #Set the dataframe equal to the weighted vote 
                        df.iloc[col][row] = WeightedVote
                    #Otherwise 
                    else: 
                        #Set the dataframe equal to the unweighted vote 
                        df.iloc[col][row] = UnweightedVote
        #Return the Dataframe 
        return df 

    #Parameters: Pandas DataFrame 
    #Returns: Bool if the dataframe has a missing attribute in it 
    #Function: Takes in a data frame and returns true if the data frame has  a ? value somewhere in the frame
    def has_missing_attrs(self, df: pd.DataFrame) -> bool:
        #For each row in the dataframe 
        for row in range(self.CountTotalRows(df)): 
            #For each column in the dataframe 
            for col in range(self.NumberOfColumns(df)): 
                #If the dataframe has a missing value in any of the cells
                if self.IsMissingAttribute(df.iloc[row][col]): 
                    #Return true 
                    return True
                #Go to the next value 
                continue  
        #We searched the entire list and never returned true so return false 
        return False
    
    #Parameters: Pandas DataFrame 
    #Returns: Cleaned Dataframe
    #Function: Take in a dataframe and an index and return a new dataframe with the row corresponding to the index removed 
    def KillRow(self, df: pd.DataFrame,index) -> pd.DataFrame: 
        return df.drop(df.Index[index])
          
    #Parameters: Attribute Value 
    #Returns: Bool -> True if the value is a missing value 
    #Function: Take in a given value from a data frame and return true if the value is a missing value false otherwise 
    def IsMissingAttribute(self, attribute) -> bool: 
        #Return true if the value is ? or NaN else return false 
        return attribute == "?" or attribute == np.nan

    #Parameters: Pandas DataFrame 
    #Returns: Clean Dataframe with not missing values 
    #Function: This function takes in a dataframe and returns a dataframe with all rows contianing missing values removed 
    def KillRows(self,df: pd.DataFrame) -> pd.DataFrame:
        # For each of the rows missing a value in the dataframe 
        for i in self.MissingRowIndexList: 
            #Set the dataframe equal to the dataframe with the row missing a value removed 
            df = df.drop(df.index[i])
        #Clear out all of the data in the set as to not try and drop these values again 
        self.MissingRowIndexList = set() 
        #Return the dataframe 
        return df

    #Parameters: Pandas DataFrame 
    #Returns: Dataframe with all columns with missing values dropped 
    #Function: This function takes in a dataframe and drops all columns with missing attributes 
    def KillColumns(self,df: pd.DataFrame) -> pd.DataFrame: 
        #For each of the columns with missing attributes which is appending into a object list 
        for i in self.MissingColumnNameList: 
            #Set the dataframe equal to the dataframe with these values dropped 
            df = df.drop(i,axis=1)
        #Set the object list back to an empty set as to not try and drop these columns again 
        self.MissingColumnNameList = set() 
        #Returnn the dataframe 
        return df
 
    #Takes in a dataframe and populates attributes based on the existing distribution of attribute values 
    #Parameters: Pandas DataFrame 
    #Returns: a Data frame with no missing attributes 
    #Function: Take in a given dataframe and replace all missing attributes with a randomly assigned value 
    def fix_missing_attrs(self, df: pd.DataFrame) -> pd.DataFrame:
        #Get the total percentage of rows missing values in the dataframe
        PercentRowsMissing = self.PercentRowsMissingValue(df)
        #Get the total number of columns missing values in the dataframe 
        PercentColumnsMissingData = self.PercentColumnsMissingData(df)
        #If the total number of rows missing data is less than the value specified in the init 
        if(PercentRowsMissing < self.PercentBeforeDrop): 
            #Return the dataframe that removes all rows with missing values 
            return self.KillRows(df)
        #If the percentage of columns missing values is less than the value specified in the init 
        elif(PercentColumnsMissingData < self.PercentBeforeDrop):
            #Return the dataframe with all columns including missing values dropped 
            return self.KillColumns(df)  
        #otherwise 
        else: 
            #If the Data frame has no missing attributes than the Data frame is ready to be processed 
            if self.has_missing_attrs(df) == False:
                #Return the dataframe 
                return df  
            #Find the Type of the first entry of data
            types = type(df.iloc[1][1])
            #If it is a string then we know it is a yes or no value 
            if types == str: 
                #Set the dataframe equal to the dataframe with all missing values randmoly generated
                df = self.RandomRollVotes(df) 
            #Else this is an integer value 
            else:
                #Set the dataframe equal to the dataframe with all missing values randmoly generated
                df =self.RandomRollInts(df) 
        #Return the dataframe 
        return df
        # https://thispointer.com/pandas-get-frequency-of-a-value-in-dataframe-column-index-find-its-positions-in-python/
        # if only small percent of examples have missing attributes, remove those examples.
            # i.e. check rowwise, calculate percentage
        # if only a small fraction of columns (e.g. 2/12) have missing attributes, remove those columns. 
            # i.e. check columnwise, calculate percentage
        # if many datapoints across many columns have missing attributes, generate at random to match column distribution. 
        #   find attribute value distribution across discrete options (find min/max?) Use pandas stats for this
       
    #Parameters: Pandas DataFrame 
    #Returns: Boolean value: true or false 
    #Function: Take in a dataframe and identify if the dataframe has non-discrete values 
    def has_continuous_values(self, df: pd.DataFrame) -> bool:
        #For each column in the data frame 
        for col in df:
            # if number of unique values is greater than threshold, consider column continuous-valued
            if df[col].nunique() > self.discrete_threshold:
                #Return true
                return True
        #If we never returned true and we are out of the loop return false 
        return False

    #Parameters: Pandas DataFrame, Integer Column Number 
    #Returns: DataFrame: New discretized values
    #Function: Takes in a dataframe and a column number of the data frame and bins all values in that column to discretize them 
    def discretize(self, df: pd.DataFrame,col) -> pd.DataFrame:
            #Set a min variable to a large number 
            Min = 100000
            #Set a max number to a small value 
            Max = -1
            #For each of the rows in the data frame 
            for i in range(self.CountTotalRows(df)):
                #Store the value at the given position in the column of the dataframe  
                Value = df.iloc[i][col]
                #If the value is a missing attribute 
                if self.IsMissingAttribute(Value): 
                    #Do nothing 
                    continue 
                #Otherwise 
                else: 
                    #If the value is bigger than the max then we need to set the new max value 
                    if Value  > Max: 
                        #Max is equal to the new value 
                        Max = Value 
                        #Go back to the top of the loop
                        continue 
                    #If the value is less than the min set the new min value 
                    elif Value < Min: 
                        #Min is now equal to the value in the given dataframe 
                        Min = Value
                        #Go back to the top of the loop 
                        continue 
                    #Go back to the top of the loop 
                    continue             
            #Set the delta to be the difference between the max and the min 
            Delta = Max - Min 
            #Set the binrange to be the delta divided by the number of mins which is set in init 
            BinRange = Delta / self.bin_count 
            #Create an empty list 
            Bins = list() 
            #Loop through the number of bins 
            for i in range(self.bin_count): 
                #If we are at the first bin 
                if i == 0: 
                    #Set the bin value to be the min + the offset between each bin 
                    Bins.append(Min + BinRange)
                #Otherwise 
                else: 
                    #Set the bin to be the position in the bin list multiplied by the bin offset + the min value 
                    Bins.append(((i+1) * BinRange) + Min)
            #Loop through all of the rows in the given dataframe 
            for row in range(self.CountTotalRows(df)): 
                #Store the value of a given position in the dataframe 
                Value = df.iloc[row][col]
                #Loop through each of the bins 
                for i in range(len(Bins)):
                    value = df.at[row,df.columns[col]]
                    #If we are at the last bin and have not been assigned a bin 
                    if i == len(Bins)-1: 
                        #Set the value to be the last bin 
                        df.at[row,df.columns[col]] = i +1 
                        #Break out 
                        break 
                    #Otherwise if the value is less than the value stored to be assigned a given bin 
                    elif Value < Bins[i]: 
                        #Set the row to be that bin value 
                        df.at[row,df.columns[col]] = i + 1
                        #Break 
                        if row % 10 == 0:
                            print("Value " +str( value) + " binned to value " + str(i+1), end="\r", flush=True)
                        break 
            print("Value ", value, " binned to value ", i+1)
            #Return the new changed dataframe 
            return df


    #Parameters: Pandas DataFrame 
    #Returns: Integer; Total number of rows in a dataframe
    #Function: Take in a dataframe and return the number of rows in the dataframe 
    def CountTotalRows(self,df: pd.DataFrame) -> int: 
        #Return the total number of rows in the data frame 
        return len(df)

  
    #Parameters: Pandas DataFrame 
    #Returns: Integer; Number of rows missing values 
    #Function: Take in a dataframe and return the number of rows in the dataframe with missing attribute values 
    def CountRowsMissingValues(self,df: pd.DataFrame ) -> int:
        #Set a Counter Variable for the number of columns in the data frame 
        Count = 0 
        #Set a counter to track the number of rows that have a missing value 
        MissingValues = 0 
        #Get the total number of rows in the data set 
        TotalNumRows = self.CountTotalRows(df)
        #For each of the columns in the data frame 
        for i in df: 
            #increment by 1 
            Count+=1 
        #For each of the records in the data frame 
        for i in range(TotalNumRows): 
            #For each column in each record 
            for j in range(Count): 
                #If the specific value in the record is a ? or a missing value 
                if self.IsMissingAttribute(df.iloc[i][j]):
                    #Increment Missing Values 
                    MissingValues+=1
                    self.MissingRowIndexList.add(i)
                    #Go to the next one 
                    continue 
                #Go to the next ones
                continue  
        #Return the number of rows missing values in the data set 
        return MissingValues 
    #Parameters: Pandas DataFrame 
    #Returns: float; Percent rows missing data
    #Function: Take in a dataframe and count the number of rows with missing attributes, return the percentage value 
    def PercentRowsMissingValue(self,df: pd.DataFrame) -> float: 
        #Get the total number of rows in the dataset
        TotalNumRows = self.CountTotalRows(df)
        #Get the total number of rows with missing values 
        TotalMissingRows = self.CountRowsMissingValues(df)
        #Return the % of rows missing values  
        return (TotalMissingRows/TotalNumRows) * 100 

    #Parameters: Pandas DataFrame 
    #Returns: Integer; Number of columns with missing attributes
    #Function: Return a count of the number of columns with atleast one missing attribute value in the data frame 
    def ColumnMissingData(self,df: pd.DataFrame) -> int: 
        #Create a counter variable to track the total number of columns missing data 
        Count = 0 
        #Store the total number of columns in the data set 
        TotalNumberColumns = self.NumberOfColumns(df)
        #Store the total number of rows in the data set 
        TotalNumberRows = self.CountTotalRows(df) 
        #For each of the columns in the dataset 
        for j in range(TotalNumberColumns): 
            #For each of the records in the data set 
            for i in range(TotalNumberRows): 
                #If the value at the specific location is ? or a missing value 
                if self.IsMissingAttribute(df.iloc[i][j]): 
                    #Increment the counter
                    Count+=1 
                    Names = df.columns
                    self.MissingColumnNameList.add(Names[j])
                    #Break out of the loop 
                    break 
                #Go to the next record 
                continue 
        #Return the count variable 
        return Count

    #Parameters: Pandas DataFrame 
    #Returns: Integer; Number of columns
    #Function: Take in a given dataframe and count the number of columns in the dataframe 
    def NumberOfColumns(self,df: pd.DataFrame) -> int: 
        #Create a counter variable 
        Count = 0 
        #For each of the columns in the dataframe 
        for i in df: 
            #Increment Count 
            Count+=1 
        #Return the total number of Columns 
        return Count 

    #Parameters: Pandas DataFrame 
    #Returns: Float; The percentage of columns with missing data 
    #Function: Take in a given dataframe and find the total number of columns divided by the number of columns with missing attribute values 
    def PercentColumnsMissingData(self,df: pd.DataFrame) -> float: 
        #Total Number of Columns in the dataset 
        TotalNumberColumns = self.NumberOfColumns(df)
        #Total number of columns missing values in the dataset
        TotalMissingColumns = self.ColumnMissingData(df)
        #Return the percent number of columns missing data
        return (TotalMissingColumns/TotalNumberColumns) * 100 

    
    #Parameters: Pandas DataFrame
    #Returns: None
    #Function: This is a test function that will print every cell to the screen that is in the dataframe
    def PrintAllData(self,df:pd.DataFrame) -> None: 
        #For each of the rows in the dataframe 
        for i in range(len(df)):
            #For each of the columns in the dataframe 
            for j in range(len(df.columns)): 
                #Print the value in that position of the dataframe 
                print(df.iloc[i][j])



#Unit Testing the object created above 
#Code not run on creation of object just testing function calls and logic above 
if __name__ == '__main__':
    NumberBins = 32
    #Location of each data file stored off into variables for later retrieval of data 
    VoteData = 'Vote_Data/Votes.data'
    IrisData = 'Iris_Data/iris.data'
    GlassData = 'Glass_Data/glass.data'
    CancerData = 'Breast_Cancer_Data/cancer.data'
    SoybeanData = 'Soybean_Data/soybean.data'
    #DataFrame With Voting data 
    df = pd.read_csv(VoteData,index_col = False )
    #print(df)
    #DataFrame With Iris data 
    df1 = pd.read_csv(IrisData,index_col = False)
    
    #DataFrame With Glass data 
    df2 = pd.read_csv(GlassData,index_col = False)
    #DataFrame With Cancer data 
    df3 = pd.read_csv(CancerData,index_col = False)
    #DataFrame With Soybean data 
    df4 = pd.read_csv(SoybeanData,index_col = False)
    print("DataFrames have been created")

    Vote = DataProcessor(NumberBins)
    #Vote.bin_count = NumberBins
    iris = DataProcessor(NumberBins) 
    #iris.bin_count = NumberBins 
    Glass = DataProcessor(NumberBins) 
    #Glass.bin_count = NumberBins
    Cancer = DataProcessor(NumberBins)
    #Cancer.bin_count = NumberBins 
    Soybean = DataProcessor(NumberBins) 
    #Soybean.bin_count = NumberBins 
    print("Pre Processor Objects Built ")
    #df2 = Glass.StartProcess(df2)
    
    df = Vote.StartProcess(df)
    df1 = iris.StartProcess(df1)
    df2 = Glass.StartProcess(df2)
    df3 = Cancer.StartProcess(df3)
    df4 = Soybean.StartProcess(df4)
    print("Printing processed data to Files...")
    df.to_csv('Demo/PreProcessedVoting' + '.csv')
    df1.to_csv('Demo/PreProcessediris' + '.csv')
    df2.to_csv('Demo/PreProcessedGlass' +  '.csv')
    df3.to_csv('Demo/PreProcessedCancer'+ '.csv')
    df4.to_csv('Demo/PreProcessedSoybean'+ '.csv')

    Ta = TrainingAlgorithm.TrainingAlgorithm() 
    print("Starting Noise")
    df = Ta.ShuffleData(df)
    print("First dataset Noisey")
    df1 = Ta.ShuffleData(df1)
    print("Second Dataset Noisey")
    df2 = Ta.ShuffleData(df2)
    print("Third Dataset Noisey")
    df3 = Ta.ShuffleData(df3)
    print("Fourth dataset Noisey")
    df4 = Ta.ShuffleData(df4)
    print("Fifth Dataset Noisey")
    print("\n")


    print("Printing Noisey Data to Files...")
    df.to_csv('Demo\PreProcessedVoting' +'_Noise'+ '.csv')
    df1.to_csv('Demo\PreProcessediris' + '_Noise'+ '.csv')
    df2.to_csv('Demo\PreProcessedGlass' + '_Noise' +  '.csv')
    df3.to_csv('Demo\PreProcessedCancer'  + '_Noise'+ '.csv')
    df4.to_csv('Demo\PreProcessedSoybean' + '_Noise'+ '.csv')
    print("Processing is complete ")
    print("File creation is complete ")



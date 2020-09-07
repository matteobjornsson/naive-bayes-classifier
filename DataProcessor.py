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


class DataProcessor:

    def __init__(self):
        self.discrete_threshold = 5
        self.bin_count = 5
        self.PercentBeforeDrop = 10.00 
        self.MissingRowIndexList = set() 
        self.MissingColumnNameList = set()

    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def StartProcess(self, df:pd.DataFrame) -> pd.DataFrame:
        count = 0 
        for i in range(len(df.columns)): 
            if count == len(df.columns)-1: 
                break
            if type(df.iloc[i][1])  == float: 
                #Find which column needs to be discretized
                df = self.discretize(df,i)
                    
            if self.has_missing_attrs(df): 
                df = self.fix_missing_attrs(df)
            count+=1
        return df 

    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function:        
    def RandomRollInts(self, df: pd.DataFrame) -> pd.DataFrame: 
        Min = df.iloc[1][col]
        Max = df.iloc[1][col]
        for i in range(self.CountTotalRows(df)): 
            if self.IsMissingAttribute(df.iloc[i][col]): 
                #Do nothing 
                continue 
            else: 
                if df.iloc[i][col]  > Max: 
                    Max = df.iloc[i][col] 
                    continue 
                elif df.iloc[i][col] < Min: 
                    Min = df.iloc[i][col]
                    continue 
                continue                 
        for col in range(self.TotalNumberColumns(df)):
            for row in range(self.TotalNumberRows(df)): 
                if self.IsMissingAttribute(df.iloc[col][row]): 
                    roll = random.randint(Min,Max)
                    df.loc[row,col] = roll   
        return df 
   
    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def RandomRollVotes(self, df: pd.DataFrame) -> pd.DataFrame: 
         for i in range(len(df)):
            for j in range(len(df.columns)-1): 
                if self.IsMissingAttribute(df.iloc[i][j]): 
                    roll = random.randint(0,99) + 1
                    if roll >50: 
                        roll = 'y'
                    else: 
                        roll = 'n' 
                    df.iloc[i][j] = roll 
                continue  
         return df 
    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def Occurence(self,Column,df:pd.DataFrame,Value) -> int:
        count = 0  
        for i in range(len(df)): 
            if df.iloc[Column][i] == Value:
                count += 1 
            continue
        return count 

    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def StatsFillInInts(self,df:pd.DataFrame): 
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
                    else: 
                        Max = PercentNay
                        PercentYay = 1 - PercentNay
                        WeightedVote = 'n'
                        UnweightedVote ='y'
                    Stats = Random.randint(0,Max)
                    if Stats > Max: 
                        df.iloc[col][row] = WeightedVote
                    else: 
                        df.iloc[col][row] = UnweightedVote
        return df 

    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    #Takes in a data frame and returns true if the data frame has  a ? value somewhere in the frame
    def has_missing_attrs(self, df: pd.DataFrame) -> bool:
        for col in range(self.NumberOfColumns(df)):
             for row in range(self.CountTotalRows(df)): 
                if self.IsMissingAttribute(df.iloc[row][col]): 
                    return True
                continue  
        return False
    
    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def KillRow(self, df: pd.DataFrame,index) -> pd.DataFrame: 
        return df.drop(df.Index[index])
          
    #Parameters: 
    #Returns: 
    #Function: 
    def IsMissingAttribute(self, attribute) -> bool: 
        return attribute == "?" or attribute == np.nan

    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def KillRows(self,df: pd.DataFrame) -> pd.DataFrame: 
        for i in self.MissingRowIndexList: 
            df = df.drop(df.index[i])
        self.MissingRowIndexList = set() 
        return df

    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function:  
    def KillColumns(self,df: pd.DataFrame) -> pd.DataFrame: 
        for i in self.MissingColumnNameList: 
            df = df.drop(i,axis=1)
        self.MissingColumnNameList = set() 
        return df

    #Parameters: 
    #Returns: 
    #Function: 
    def GenerateValue(self,Upperbounds,Lowerbounds,types): 
        return types(random.uniform(Upperbounds,Lowerbounds))    

    #Takes in a dataframe and populates attributes based on the existing distribution of attribute values 
    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def fix_missing_attrs(self, df: pd.DataFrame) -> pd.DataFrame:
        PercentRowsMissing = self.PercentRowsMissingValue(df)
        PercentColumnsMissingData = self.PercentColumnsMissingData(df)
        if(PercentRowsMissing < self.PercentBeforeDrop): 
            return self.KillRows(df)
        elif(PercentColumnsMissingData < self.PercentBeforeDrop):
            return self.KillColumns(df)  
        else: 
            #If the Data frame has no missing attributes than the Data frame is ready to be processed 
            if self.has_missing_attrs(df) == False:
                return df  
            #Find the Type of the first entry of data
            types = type(df.iloc[1][1])
            #If it is a string then we know it is a yes or no value 
            if types == str: 
                df = self.RandomRollVotes(df) 
            #Else this is an integer value 
            else:
                df =self.RandomRollInts(df) 
        return df
        # https://thispointer.com/pandas-get-frequency-of-a-value-in-dataframe-column-index-find-its-positions-in-python/
        # if only small percent of examples have missing attributes, remove those examples.
            # i.e. check rowwise, calculate percentage
        # if only a small fraction of columns (e.g. 2/12) have missing attributes, remove those columns. 
            # i.e. check columnwise, calculate percentage
        # if many datapoints across many columns have missing attributes, generate at random to match column distribution. 
        #   find attribute value distribution across discrete options (find min/max?) Use pandas stats for this
       
    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
    def has_continuous_values(self, df: pd.DataFrame) -> bool:
        for col in df:
            # if number of unique values is greater than threshold, consider column continuous-valued
            if df[col].nunique() > self.discrete_threshold:
                return True
        return False

    #Parameters: Pandas DataFrame, Integer Column Number 
    #Returns: 
    #Function: 
    def discretize(self, df: pd.DataFrame,col) -> pd.DataFrame:
 
            Min = df.iloc[col][1]
            Max = df.iloc[col][1]
            for i in range(self.CountTotalRows(df)): 
                if self.IsMissingAttribute(df.iloc[i][col]): 
                    #Do nothing 
                    continue 
                else: 
                    if df.iloc[i][col]  > Max: 
                        Max = df.iloc[i][col] 
                        continue 
                    elif df.iloc[i][col] < Min: 
                        Min = df.iloc[i][col]
                        continue 
                    continue                 
            Delta = Max - Min 
            BinRange = Delta / self.bin_count
            Bins = list(np.arange(Min, Max,BinRange))
            for row in range(self.CountTotalRows(df)): 
                Value = df.iloc[row][col]
                for i in range(len(Bins)):
                    if i == len(Bins): 
                        df.at[row,col] = Bins[i]
                         
                        continue  
                    elif Value < Bins[i]: 
                        df.at[row,col] = Bins[i]
                        continue 
            return df


    #Parameters: Pandas DataFrame 
    #Returns: Integer; Total number of rows in a dataframe
    #Function: Take in a dataframe and return the number of rows in the dataframe 
    def CountTotalRows(self,df: pd.DataFrame) -> int: 
        #Return the total number of rows in the data frame 
        return len(df)

  
    #Parameters: Pandas DataFrame 
    #Returns: 
    #Function: 
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
        for i in range(len(df)):
            for j in range(len(df.columns)): 
                print(df.iloc[i][j])

if __name__ == '__main__':
    #filename = sys.argv[1]
    df = pd.read_csv(filename)
    dp = DataProcessor()
    print(df)
    df = dp.StartProcess(df)
    dp.PrintAllData(df)
    

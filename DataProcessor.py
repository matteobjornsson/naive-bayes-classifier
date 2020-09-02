import pandas as pd
import numpy as np
import sys


#
#
#
"""
# OF = len(DataFrame)
        count = 0 
        for i in DataFrame: 
            count +=1 

        for i in range(EOF): 
            for j in range(count): 
                if DataFrame.iloc[i][j] == 'y':
                    DataFrame.iloc[i][j] = '1'
#
"""

class DataProcessor:

    def __init__(self):
        self.discrete_threshold = 5
        self.bin_count = 5

    #Takes in a data frame and returns true if the data frame has  a ? value somewhere in the frame
    def has_missing_attrs(self, df: pd.DataFrame) -> bool:
        
        # check if data has missing attributes
        
        
        return True

    #Takes in a dataframe and populates attributes based on the existing distribution of attribute values 
    def fix_missing_attrs(self, df: pd.DataFrame) -> pd.DataFrame:


        # https://thispointer.com/pandas-get-frequency-of-a-value-in-dataframe-column-index-find-its-positions-in-python/
        # if only small percent of examples have missing attributes, remove those examples.
            # i.e. check rowwise, calculate percentage
        # if only a small fraction of columns (e.g. 2/12) have missing attributes, remove those columns. 
            # i.e. check columnwise, calculate percentage
        # if many datapoints across many columns have missing attributes, generate at random to match column distribution. 
        #   find attribute value distribution across discrete options (find min/max?) Use pandas stats for this
        return df

    def has_continuous_values(self, df: pd.DataFrame) -> bool:
        for col in df:
            # if number of unique values is greater than threshold, consider column continuous-valued
            if df[col].nunique() > self.discrete_threshold:
                return True
        return False

    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df:
            if df[col].nunique() > self.discrete_threshold and col != "class":
                # split all values into @bin_count bins
                df[col] = pd.cut(df[col], bins=self.bin_count, labels=[*range(0,self.bin_count,1)])
        return df
    def CountTotalRows(self,df: pd.DataFrame) -> int: 
        #Return the total number of rows in the data frame 
        return len(df)

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
                if df.iloc[i][j] == "?":
                    #Increment Missing Values 
                    MissingValues+=1
                    #Go to the next one 
                    continue 
                #If the specific Value in the record is a nan value 
                if df.iloc[i][j] == np.nan: 
                    #Increment The missing value 
                    MissingValues+=1 
                    #Go to the next one 
                    continue 
                #Go to the next ones
                continue  
        #Return the number of rows missing values in the data set 
        return MissingValues 
    def PercentRowsMissingValue(self,df: pd.DataFrame) -> float: 
        #Get the total number of rows in the dataset
        TotalNumRows = self.CountTotalRows(df)
        #Get the total number of rows with missing values 
        TotalMissingRows = self.CountRowsMissingValues(df)
        #Return the % of rows missing values  
        return TotalMissingRows/TotalNumRows
    
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
                if df.iloc[i][j] == "?": 
                    #Increment the counter
                    Count+=1 
                    #Break out of the loop 
                    break 
                #If the value at the specific location is nan or a missing value 
                if df.iloc[i][j] == np.nan: 
                    #Incrememnt the counter
                    Count+=1 
                    #Break out of the loop 
                    break 
                #Go to the next record 
                continue 
        #Return the count variable 
        return Count


    def NumberOfColumns(self,df: pd.DataFrame) -> int: 
        #Create a counter variable 
        Count = 0 
        #For each of the columns in the dataframe 
        for i in df: 
            #Increment Count 
            Count+=1 
        #Return the total number of Columns 
        return Count 

    


if __name__ == '__main__':
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    print(df.head())
    dp = DataProcessor()
    #print(dp.CountTotalRows(df))
    #print(dp.CountRowsMissingValues(df))
    print(dp.PercentRowsMissingValue(df))
    print(dp.NumberOfColumns(df))
    print(dp.ColumnMissingData(df))
    #if dp.has_continuous_values(df):
     #   print("Attribute values continuous, discretizing...\n")
     #   df = dp.discretize(df)
    #print(df.head())
    

import pandas as pd
import numpy as np
import sys

class DataProcessor:

    def __init__(self):
        self.discrete_threshold = 5
        self.bin_count = 5

    def has_missing_attrs(self, df: pd.DataFrame) -> bool:
        # check if data has missing attributes
        return True

    def fix_missing_attrs(self, df: pd.DataFrame) -> pd.DataFrame:
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

if __name__ == '__main__':
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    print(df.head())
    dp = DataProcessor()
    if dp.has_continuous_values(df):
        print("Attribute values continuous, discretizing...\n")
        df = dp.discretize(df)
    print(df.head())
    

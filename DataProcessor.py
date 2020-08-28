import pandas as pd
import sys

class DataProcessor:

    discrete_threshold = 5

    def __init__(self):
        super().__init__()

    def has_missing_attrs(self, df: pd.DataFrame) -> bool:
        # check if data has missing attributes
        return True

    def fix_missing_attrs(self, df: pd.DataFrame) -> pd.DataFrame:
        # if only small percent of examples have missing attributes, remove those examples.
        # if only a small fraction of columns (e.g. 2/12) have missing attributes, remove those columns. 
        # if many datapoints across many columns have missing attributes, generate at random to match column distribution. 
        #   find attribute value distribution across discrete options (find min/max?) Use pandas stats for this
        return df

    def has_continuous_values(self, df: pd.DataFrame) -> bool:
        for col in df:
            # if number of unique values is greater than threshold, consider column continuous-valued
            if df[col].nunique() > discrete_threshold:
                return True
        return False

    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        # if continuous valued, bin all continous-valued columns with pandas functions
        return df

if __name__ == '__main__':
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    print(df.head())
    dp = DataProcessor()

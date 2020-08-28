import pandas as pd
import numpy

class DataProcessor:

    def __init__(self, filename: str):
        self.data = pd.read_csv(filename)

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
        # return true if all columns have only discrete values
        return True

    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        # if continuous valued, bin all continous-valued columns with pandas functions
        return df


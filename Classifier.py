import numpy as np
import pandas as pd
import sys

class Classifier:

    def __init__(self,n: dict, q: dict, f: dict):
        self.n = n
        self.q = q
        self.f = f
    
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df['estimate'] = ""
        print(df.head())
        attributes = []
        for col in df.columns:
            attributes.append(col)
        attributes = attributes[:-2]
        classes = self.f.keys()
        ClassEstimates = dict.fromkeys(classes)
        for i in range(len(df)):
            x = df.iloc[i]
            x = x.drop(['class', 'estimate'])
            for c in self.f.keys():
                probability = 1
                default_value = 1/(self.n[c] + len(attributes))
                for Aj in self.f[c].keys():
                    for ak in x:
                        try:
                            probability = probability * self.f[c][Aj][ak]
                        except KeyError:
                            probability = probability * default_value
                ClassEstimates[c] = probability
            estimate = self.argmax(ClassEstimates)
            df.at[i, 'estimate'] = estimate
        print(df.head(n=20))
        return df

    def argmax(self, d: dict):
        vals = list(d.values())
        keys = list(d.keys())
        return keys[vals.index(max(vals))]

if __name__ == '__main__':
    f = {"class1": {"A1": {2: (3+1)/(3+2)}, "A2": {1: (0+1)/(3+2)}}, "class2": {"A1": {4: (4+1)/(6+2)}, "A2": {1: (2+1)/(6+2)}}}
    print(f)
    n = {"class1": 4, "class2": 2}
    q = {"class1": .3, "class2": .5}
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    cl = Classifier(n=n, q=q, f=f)
    cl.classify(df)

import numpy as np
import pandas as pd
import sys

class Classifier:

    # init classifier with
    # n: count of examples in class ci
    # q: n/(total examples) Q(C = ci)
    # f: training matrix F(Aj = ak, C = ci)
    def __init__(self,n: dict, q: dict, f: dict):
        self.n = n
        self.q = q
        self.f = f

    # Take in a dataframe containing test data, return frame with all rows classified
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        # create new column to hold new classifications
        df['estimate'] = ""

        # collect all attributes of the test set to iterate over
        attributes = []
        for col in df.columns:
            attributes.append(col)
        attributes = attributes[:-2]

        # collect all classes to iterate over
        classes = self.f.keys()
       
        # iterate over all test examles, execute C(x) calculation for each x
        for i in range(len(df)):
            # isolate feature vector
            x = df.iloc[i]
            x = x.drop(['class', 'estimate'])
            # create blank dict to store classification estimate for each class
            ClassEstimates = dict.fromkeys(classes)
            # for each potential class, multiply the chance of that class by the
            # chance of each feature attribute appearing for that class
            for c in self.f.keys():
                probability = 1
                # assign a default value of the test vector attribute value if 
                # it was not seen in the training set (count ak = 0)
                default_value = 1/(self.n[c] + len(attributes))
                for Aj in self.f[c].keys():
                    for ak in x:
                        # reference the training matrix f for the chance of 
                        # seeing the attribute value ak
                        try:
                            probability = probability * self.f[c][Aj][ak]
                        except KeyError:
                            probability = probability * default_value
                # store the final calculated value for the class
                ClassEstimates[c] = probability
            # take the class with the highest calculated value
            estimate = self.argmax(ClassEstimates)
            # store the classification value with the feature vector
            df.at[i, 'estimate'] = estimate
        return df

    # small function to grab the key corresponding to the max value in a dict
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
    print(cl.classify(df))


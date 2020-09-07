#################################################################### MODULE COMMENTS ####################################################################
##
##
#################################################################### MODULE COMMENTS ####################################################################

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
            for ClassValue in self.f.keys():
                probability = 1
                # assign a default value of the test vector attribute value if 
                # it was not seen in the training set (count ak = 0)
                default_value = 1/(self.n[ClassValue] + len(attributes))

                for feature, featureValue in x.items():
                    # reference the training matrix f for the chance of 
                    # seeing the attribute value ak
                    try:
                        probability = probability * self.f[ClassValue][feature][featureValue]
                    except KeyError:
                        probability = probability * default_value

                # store the final calculated value for the class
                ClassEstimates[ClassValue] = probability

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
    import Results

    f = {
        "D1": {"A1": {2: (3+1)/(3+2)}, "A2": {1: (0+1)/(3+2)}}, 
        "D2": {"A1": {4: (4+1)/(6+2)}, "A2": {1: (2+1)/(6+2)}},
        "D3": {"A1": {4: (4+1)/(6+2)}, "A2": {1: (2+1)/(6+2)}},
        "D4": {"A1": {4: (4+1)/(6+2)}, "A2": {1: (2+1)/(6+2)}}
        }
    print(f)
    n = {
        "D1": 4, 
        "D2": 2,
        "D3": 8, 
        "D4": 1
        }
    q = {
        "D1": .4, 
        "D2": .2,
        "D3": .6, 
        "D4": .12
        }
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    cl = Classifier(n=n, q=q, f=f)
    Classified = cl.classify(df)
    print(Classified.head)

    r = Results.Results()
    cM = r.ConfusionMatrix(Classified)
    print(cM)


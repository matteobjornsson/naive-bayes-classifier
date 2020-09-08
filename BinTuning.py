import Results, DataProcessor, TrainingAlgorithm, Classifier
import pandas as pd
import pprint

def train(df: pd.DataFrame):
    trainingAlg = TrainingAlgorithm.TrainingAlgorithm()
    splitDataframesList = trainingAlg.BinTestData(df)

    testdata = splitDataframesList.pop()
    trainingData = pd.DataFrame()
    for i in range(len(splitDataframesList)):
        trainingData = trainingData.append(splitDataframesList[i], ignore_index=True)

    N = trainingAlg.calculateN(trainingData)
    Q = trainingAlg.calculateQ(N, len(trainingData))
    F = trainingAlg.calculateF(N, trainingData)

    return N, Q, F, testdata

dataset_names = ['Glass_Data', 'Iris_Data']
dataset_stats_list = []
iris_data = pd.read_csv('Iris_Data/iris.data')
glass_data = pd.read_csv('Glass_Data/glass.data')
datasets = [glass_data, iris_data]

binCount = 7000
trials = 4

for j in range(2):
    for i in range(6):
        bins = 2**(3+i*2)
        results = []
        for k in range(trials):
            dp = DataProcessor.DataProcessor(bin_count=bins)
            binnedDataset = dp.StartProcess(datasets[j])
            N, Q, F, testData = train(binnedDataset)

            model = Classifier.Classifier(N, Q, F)
            classifiedData = model.classify(testData)

            stats = Results.Results()
            zeroOne = stats.ZeroOneLoss(classifiedData)
            macroF1Average = stats.statsSummary(classifiedData)
            trial = {"zeroOne": zeroOne, "F1": macroF1Average}
            results.append(trial)
            print(trial)

        z1 = 0
        f1 = 0
        for n in results: 
            z1 += n["zeroOne"]
            f1 += n["F1"]
        z1 = z1/trials
        f1 = f1/trials

        datapoint = (dataset_names[j],bins,{"zeroOne": z1, "F1": f1})
        print(datapoint)
        dataset_stats_list.append(datapoint)

pprint.pprint(dataset_stats_list)









import Results, DataProcessor, TrainingAlgorithm, Classifier
import pandas as pd
import pprint, math 
import multiprocessing, time

dataset_names = ['Glass_Data', 'Iris_Data']
dataset_stats_list = []
iris_data = pd.read_csv('Iris_Data/iris.data')
glass_data = pd.read_csv('Glass_Data/glass.data')
datasets = [glass_data, iris_data]

trials = 5


def train(df):
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

def calc(datasetIndex, multiplierInt):
    csv = pd.DataFrame(columns=['dataset', 'bins', 'f1', 'zero-one'])
    exp = ((multiplierInt+1)/2)
    bins = math.ceil(2**exp)
    results = []
    for k in range(trials):
        dp = DataProcessor.DataProcessor(bin_count=bins)
        binnedDataset = dp.StartProcess(datasets[datasetIndex])
        N, Q, F, testData = train(binnedDataset)

        model = Classifier.Classifier(N, Q, F)
        classifiedData = model.classify(testData)

        stats = Results.Results()
        zeroOne = stats.ZeroOneLoss(classifiedData)
        macroF1Average = stats.statsSummary(classifiedData)
        datapoint = {
            'dataset': dataset_names[datasetIndex], 
            'bins': bins, 
            'f1': macroF1Average, 
            'zero-one':zeroOne/100
            }
        print(datapoint)
        csv = csv.append(datapoint, ignore_index=True)
        # trial = {"zeroOne": zeroOne, "F1": macroF1Average}
        # results.append(trial)
        # print(trial)
    data.append(csv)
    # z1 = 0
    # f1 = 0
    # for n in results: 
    #     z1 += n["zeroOne"]
    #     f1 += n["F1"]
    # z1 = z1/trials
    # f1 = f1/trials

    # datapoint = (dataset_names[j],bins,{"zeroOne": z1, "F1": f1})
    # print(datapoint)
    # dataset_stats_list.append(datapoint)



manager = multiprocessing.Manager()
data = manager.list()
start = time.time()
pool = multiprocessing.Pool()
for j in range(2):
    for i in range(25):
        pool.apply_async(calc, args=(j,i))
pool.close()
pool.join()

results = pd.DataFrame(columns=['dataset', 'bins', 'f1', 'zero-one'])    
for df in data:
    results = results.append(df, ignore_index=True)
end = time.time()
results.to_csv('bintuning_parallel.csv', index=False)
print("Elapsed time (s): ", end - start)
# pprint.pprint(dataset_stats_list)









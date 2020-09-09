#Created by Nick Stone and Matteo Bjornsson 
#Created on 8/20/2020 
##################################################################### MODULE COMMENTS #####################################################################
# This is the main function for the Naive Bayes project that was created by Nick Stone and Matteo Bjornsson. The purpose of this class is to import all of #
# The other classes and objects that were created and tie them together to run a series of experiments about the outcome stats on the data sets in question#
# The following program is just intended to run as an experiment and hyper parameter tuning will need to be done in each of the respective classes.        #
# It is important to note that the main datastructure that is used by these classes and objects is the pandas dataframe, and is used to pass the datasets  #
# Between all of the objects and functions that have been created. The classes are set up for easy modification for hyper parameter tuning.                #
##################################################################### MODULE COMMENTS #####################################################################

import pandas as pd 
import Results 
import TrainingAlgorithm 
import Classifier 
import DataProcessor 
import pprint
import copy
import time 

#Parameters: Dataset name, A list, a Dataframe and an Integer 
#Return: None
#Function: Take in the Result data, The data frame of data, and the trial number and print to a file 
def WriteToAFile(Setname, Results,Trial):
    #Set the file name based on the trial number 
    FileName = Setname + "2_" + str(Trial) + ".csv"
    #Open the file in write mode 
    f = open(FileName, "w")
    #Append the data set name to the csv files 
    f.write(str(Setname) + "\n")
    #write the results to file
    f.write(str(Results) + "\n")
    #Close the file 
    f.close()
#Parameters: # of Times the data was run, List of stats of the given runs such as F1 score 
#Return: List of Averages for each of the given runs 
#Function: Function that takes in the number of times a dataset was run and a list of statistical values and then averages all of the stats in the given list  
def Average(TotalRunCount, Stats ) -> list: 
    #Print out the list of stats that we are going to average
    print(Stats)
    #Set the values to be 0 and used to track stats later on 
    f1score = 0 
    ZOloss = 0 
    #For each of the stats in the array 
    for i in range(len(Stats)): 
        #If the position is in the even position it is the F1 score 
        if i % 2 == 0: 
            #Add the F1 score to the variable 
            f1score += Stats[i]
        #Otherwise it is ZO loss 
        else: 
            #Add the ZO Loss to the variable above
            ZOloss += Stats[i]
    #Average the F1 score 
    f1score = f1score / TotalRunCount
    #Average the zero one loss 
    ZOloss = ZOloss / TotalRunCount
    #Create a new list 
    Avg = list() 
    #Append the averaged scores to the list 
    Avg.append(f1score)
    Avg.append(ZOloss)
    #Return the list 
    return Avg
#Parameters: Training Algorithm Object , Dataframe 
#Return: Calculated N Dictionary, Calculated Q Dictionary, Calculated F dictonary 
#Function: Take in a training algorithm object and a training set of data and produce a series of datapoints to be used in the classification object on the test dataframe 
def train(trainingAlgorithm, trainingData: pd.DataFrame) -> (dict, dict, dict):
    #Calculate N,Q,F and store them into dictionarys to be later used in the classification 
    N = trainingAlgorithm.calculateN(trainingData)
    Q = trainingAlgorithm.calculateQ(N, len(trainingData))
    F = trainingAlgorithm.calculateF(N, trainingData)
    print("The following N value's were calculated on this data set ")
    print(N)
    time.sleep(2)
    print("The following Q Values were calculated on this data set  ")
    print(Q)
    time.sleep(2)
    print("The following is the calculated F matrix ")
    print("\n")
    print(F)
    print("\n")
    time.sleep(2)
    #Return the dictionary of stats 
    return N, Q, F

#Parameters: None
#Return: None
#Function: The main function is what combines all of the object together in this project. This is called one time and this function runs the Naive Bayes against all of the data 
def main(): 
    #What trial number we are on 
    Trial = 0 
    #Which set of the data is being used to test 
    TestData = 0 
    #Print data to the screen so the user knows the program is starting 
    print("Program Starting")
    #Prepocessed datasets stored in an array for iteration and experiments, Nosie included 
    data_sets = [
        'PreProcessedVoting.csv',
        'PreProcessedIris.csv',
        'PreProcessedGlass.csv',
        'PreProcessedCancer.csv',
        'PreProcessedSoybean.csv',
        'PreProcessedVoting_Noise.csv',
        'PreProcessedIris_Noise.csv',
        'PreProcessedGlass_Noise.csv',
        'PreProcessedCancer_Noise.csv',
        'PreProcessedSoybean_Noise.csv'
    ]
    #The 5 Data set names are stored in the array, Noise included
    dataset_names = [
        "Vote", "Iris", "Glass", "Cancer", "Soybean", 
        "Vote_Noise", "Iris_Noise", "Glass_Noise", "Cancer_Noise", "Soybean_Noise"
    ] 
    
    ####################################################### MACHINE LEARNING PROCESS #####################################################
    #Set the total number of runs to be 10 
    TotalRun = 10 
    #Create a dataframe that is going to hold key valuesf from the experiment 
    finalDataSummary = pd.DataFrame(columns=["Dataset", "F1", "ZeroOne"])
    #For each of the datasets and the data sets including noise 
    for dataset in data_sets:
        #Create an empty array 
        AvgZeroOne = []
        #Create a second empty array 
        AvgF1 = []
        #Get the name of the dataset being experimented on 
        datasetName = dataset_names[data_sets.index(dataset)]
        #Print the dataset name so the user knows what data set is being experimented 
        print(datasetName)
        time.sleep(1)
        #Load in the dataframe from the preprocessed data 
        df = pd.read_csv(dataset) 
        #Create a Training algorithm Object 
        ML = TrainingAlgorithm.TrainingAlgorithm()
        #Bin the data frame into a list of 10 similar sized dataframes for traning and one set to test 
        tenFoldDataset = ML.BinTestData(df)

        #Set the total number of runs to be 10 for now 
        for i in range(10): 
            #Set an empty dataframe object 
            TrainingDataFrame = pd.DataFrame()
            #Make One dataframe that is our test Dataframe 
            TestingDataFrame = copy.deepcopy(tenFoldDataset[i])
            #For each of the dataframes generated above 
            for j in range(10):
                #If the dataframe being accessed is the test dataframe 
                if i == j:
                    #Skip it 
                    continue    
                #Append the training dataframe to one dataframe to send to the ML algorithm 
                TrainingDataFrame = TrainingDataFrame.append(copy.deepcopy(tenFoldDataset[j]), ignore_index=True)

            # print('************************************************')
            # print(TrainingDataFrame)
            # print(TestingDataFrame)
            # print('************************************************')

            # calculate the N, Q, and F probabiliies
            N, Q, F = train(ML, TrainingDataFrame)
            #Create a Classifier Object to classify our test set 
            model = Classifier.Classifier(N, Q, F)
            #Reassign the testing dataframe to the dataframe that has our Machine learning classification guesses implemented 
            classifiedDataFrame = model.classify(TestingDataFrame)
            #Create a Results object
            Analysis = Results.Results()

            #Run the 0/1 Loss function on our results
            zeroOnePercent = Analysis.ZeroOneLoss(classifiedDataFrame)
            #Get the F1 score for the given dataset 
            macroF1Average = Analysis.statsSummary(classifiedDataFrame)
            #Print the zero one loss  and F1 calculation to the screen 
            print("Zero one loss: ", zeroOnePercent, "F1: ", macroF1Average)
            time.sleep(2)
            print("\n")
            #append the zero one loss and F1 average to the list to calculate the average score 
            AvgZeroOne.append(zeroOnePercent)
            AvgF1.append(macroF1Average)
            
            #Increment the trial number and the position in the array to use the dataframe to test on 
            Trial += 1 
            TestData +=1 
            #If we are at 10 we only have 10 dataframes 0 - 9 accessed in the array so on the 10th trial go back to the beginning 
            if TestData == 10: 
                #Set the value to 0 
                TestData = 0
        #Gather the dataset name the average scores for ZOloss and F1 score and put them into a data structure
        AvgStats = {
            "Dataset": datasetName, 
            "F1": sum(AvgF1)/len(AvgF1), 
            "ZeroOne": sum(AvgZeroOne)/len(AvgZeroOne)
            }
        #Set a variavle to hold all of the statistics for each of the trials so we can print them to one file 
        finalDataSummary = finalDataSummary.append(AvgStats, ignore_index=True)
        #Write the data set, the trial number and statistics of a trial to be printed to a file 
        WriteToAFile(datasetName, AvgStats,Trial)
    finalDataSummary.to_csv("ExperimentalSummary.csv")
    print("Program Finish")

#Call the main function
main() 
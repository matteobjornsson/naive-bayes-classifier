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


#Parameters: Dataset name, A list, a Dataframe and an Integer 
#Return: None
#Function: Take in the Result data, The data frame of data, and the trial number and print to a file 
def WriteToAFile(Setname, Results,Trial):
    #Set the file name based on the trial number 
    FileName = Setname + "_" + str(Trial) + ".csv"
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
    print("Program Starting")
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
    dataset_names = [
        "Vote", "Iris", "Glass", "Cancer", "Soybean", 
        "Vote_Noise", "Iris_Noise", "Glass_Noise", "Cancer_Noise", "Soybean_Noise"
    ] 
    
    ####################################################### MACHINE LEARNING PROCESS #####################################################

    TotalRun = 10 
    finalDataSummary = pd.DataFrame(columns=["Dataset", "F1", "ZeroOne"])
    for dataset in data_sets:
        
        AvgZeroOne = []
        AvgF1 = []
        datasetName = dataset_names[data_sets.index(dataset)]
        print(datasetName)
        df = pd.read_csv(dataset) 
        #Return a clean dataframe with missing attributes taken care of 
        # df = dp.StartProcess(df)
        ML = TrainingAlgorithm.TrainingAlgorithm()
        #Dataframe without noise Its a list of 10 mostly equal dataframes
        tenFoldDataset = ML.BinTestData(df)
        # #DataFrame with Noise 
        # NoiseDf =  ML.ShuffleData(df)
        # #Return a list of 10 mostly equal sized dataframes
        # NoiseDf = ML.BinTestData(NoiseDf)
        for i in range(10): 
            #Make One dataframe to hold all of the other Training dataframes 
            TrainingDataFrame = pd.DataFrame()
            #Make One dataframe that is our test Dataframe 
            TestingDataFrame = tenFoldDataset[i]
            for j in range(10):
                if i == j:
                    continue    
                #Append the training dataframe to one dataframe to send to the ML algorithm 
                TrainingDataFrame = TrainingDataFrame.append(tenFoldDataset[j], ignore_index=True)

            # calculate the N, Q, and F probabiliies
            N, Q, F = train(ML, TrainingDataFrame)

            #Create a Classifier Object to classify our test set 
            model = Classifier.Classifier(N, Q, F)
            #Reassign the testing dataframe to the dataframe that has our Machine learning classification guesses implemented 
            classifiedDataFrame = model.classify(TestingDataFrame)
            

            #Get some statistics on the Machine learning 
            #Create a Results object
            Analysis = Results.Results()

            #Run the 0/1 Loss function on our results
            zeroOnePercent = Analysis.ZeroOneLoss(classifiedDataFrame)
            #Run the stats summary on our results 
            macroF1Average = Analysis.statsSummary(classifiedDataFrame)
            print("Zero one loss: \n")
            print(zeroOnePercent)

            AvgZeroOne.append(zeroOnePercent)
            AvgF1.append(macroF1Average)
            
            # #Send the Data to a csv file for human checking and hyper parameter tuning 
            
            Trial += 1 
            TestData +=1 
            if TestData == 10: 
                TestData = 0
    # #Increment the Trial and Testdata Number and do it again 
        AvgStats = {
            "Dataset": datasetName, 
            "F1": sum(AvgF1)/len(AvgF1), 
            "ZeroOne": sum(AvgZeroOne)/len(AvgZeroOne)
            }
        finalDataSummary = finalDataSummary.append(AvgStats, ignore_index=True)
        WriteToAFile(datasetName, AvgStats,Trial)
    finalDataSummary.to_csv("ExperimentalSummary.csv")
    print("Program Finish")

#Call the main function
main() 
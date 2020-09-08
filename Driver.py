#Created by Nick Stone and Matteo Bjornsson 
#Created on 8/20/2020 
##################################################################### MODULE COMMENTS #####################################################################
##
##
##
##
##
##################################################################### MODULE COMMENTS #####################################################################

import pandas as pd 
import Results 
import TrainingAlgorithm 
import Classifier 
import DataProcessor 
import pprint


#Parameters: A list, a Dataframe and an Integer 
#Return: None
#Function: Take in the Result data, The data frame of data, and the trial number and print to a file 
def WriteToAFile(Results,DataFrame,Trial):
    #Set the file name based on the trial number 
    FileName = "Naive Bayes Results " + str(Trial) + ".csv"
    #Open the file in write mode 
    f = open(FileName, "w")
    #For each of the results in the list 
    for i in Results: 
        #Write with a new line character
        f.write(str(i) + "\n")
    #Append the Dataframe to the file 
    DataFrame.to_csv(FileName, header=None, index=None, sep=' ', mode='a')
    #Close the file 
    f.close()



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
    ]
    # VoteData = 'PreProcessedVoting.csv'
    # IrisData = 'PreProcessedIris.csv'
    # GlassData = 'PreProcessedGlass.csv'
    # CancerData = 'PreProcessedCancer.csv'
    # SoybeanData = 'PreProcessedSoybean.csv'
    
    ####################################################### MACHINE LEARNING PROCESS #####################################################
    for dataset in data_sets:
        df = pd.read_csv(dataset) 
        #Return a clean dataframe with missing attributes taken care of 
        # df = dp.StartProcess(df)
        ML = TrainingAlgorithm.TrainingAlgorithm()
        #Dataframe without noise Its a list of 10 mostly equal dataframes
        NoNoiseDf = ML.BinTestData(df)
        #DataFrame with Noise 
        NoiseDf =  ML.ShuffleData(df)
        #Return a list of 10 mostly equal sized dataframes
        NoiseDf = ML.BinTestData(NoiseDf)
        #Make One dataframe to hold all of the other Training dataframes 
        TrainingDataFrame = pd.DataFrame()
        #Make One dataframe that is our test Dataframe 
        TestingDataFrame = NoNoiseDf.pop(TestData) 
        for i in range(len(NoNoiseDf)):     
            #Append the training dataframe to one dataframe to send to the ML algorithm 
            TrainingDataFrame = TrainingDataFrame.append(NoNoiseDf[i], ignore_index=True)
        
        # print("Driver training data: \n", TrainingDataFrame)
        # print("Driver testing dataframe: \n", TestingDataFrame)


        #Calculate the N value for the Training set
        TrainingN = ML.calculateN(TrainingDataFrame)
        #Calculate the Q value for the Training set
        TrainingQ = ML.calculateQ(TrainingN,len(TrainingDataFrame))
        #Calculate the F Matrix for the Training set
        TrainingF = ML.calculateF(TrainingN,TrainingDataFrame)
        # print("N: \n")
        # pprint.pprint(TrainingN)
        # print("Q: \n")
        # pprint.pprint(TrainingQ)
        # print("F: \n")
        # pprint.pprint(TrainingF)

        #Create a Classifier Object to classify our test set 
        model = Classifier.Classifier(TrainingN,TrainingQ,TrainingF)
        #Reassign the testing dataframe to the dataframe that has our Machine learning classification guesses implemented 
        TestingDataFrame = model.classify(TestingDataFrame)
        
        # print("Classified test set: \n")
        # print(TestingDataFrame)

        #Get some statistics on the Machine learning 
        #Create a Results object
        Analysis = Results.Results()

        #List to hold our stats
        Stats = list()  
        #Run the 0/1 Loss function on our results
        zeroOne = Analysis.ZeroOneLossFunctionStats(TestingDataFrame)
        #Run the stats summary on our results 
        classStats, microStats, macroStats = Analysis.statsSummary(TestingDataFrame)
        Stats.append(classStats)
        Stats.append(microStats)
        Stats.append(macroStats)   
        print("Zero one loss: \n")
        print(zeroOne)
        print("F1 Matrix score: \n")
        print(
            "per-class stats: \n", classStats, 
            '\n micro-Averaged stats: \n', microStats, 
            '\n macro-Averaged stats: \n', macroStats
            )

        # #Send the Data to a csv file for human checking and hyper parameter tuning 
        WriteToAFile(Stats, TestingDataFrame,Trial)

        # #Increment the Trial and Testdata Number and do it again 
        # Trial+=1 
        # TestData +=1

        print("Program Finish")



#Call the main function
main() 
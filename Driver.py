#Created by Nick Stone and Matteo Bjornsson 
#Created on 8/20/2020 
##################################################################### MODULE COMMENTS #####################################################################
#
#
#
#
#
##################################################################### MODULE COMMENTS #####################################################################

import pandas as pd 

#Take in the Result data, The data frame of data, and the trial number and print to a file 
def WriteToAFile(Results,DataFrame,Trial):
    FileName = "Naive Bayes Results " + str(Trial) + ".csv"
    f = open(FileName, "w")
    for i in results: 
        f.write(i + "\n")
    df.to_csv(FileName, header=None, index=None, sep=' ', mode='a')
    f.close()




def main(): 
    #What trial number we are on 
    Trial = 0 
    #Which set of the data is being used to test 
    TestData = 0 
    print("Program Starting")
    VoteData = 'MachineLearning\Project 1\Vote_Data\Votes.data'
    IrisData = 'MachineLearning\Project 1\Iris_Data\iris.data'
    GlassData = 'MachineLearning\Project 1\Glass_Data\glass.data'
    CancerData = 'MachineLearning\Project 1\Breast_Cancer_Data\cancer.data'
    SoybeanData = 'MachineLearning\Project 1\Soybean_Data\soybean.data'
    
    ####################################################### MACHINE LEARNING PROCESS #####################################################
    dp = DataProcessor()
    df = pd.read_csv(VoteData) 
    #Return a clean dataframe with missing attributes taken care of 
    df = dp.StartProcess(df)
    ML = Training_Algorithm()
    #Dataframe without noise Its a list of 10 mostly equal dataframes
    NoNoiseDf = ML.BinTestData(df)
    #DataFrame with Noise 
    NoiseDf =  ML.ShuffleData(df)
    #Return a list of 10 mostly equal sized dataframes
    NoiseDf = ML.BinTestData(NoiseDf)
    #Make One dataframe to hold all of the other Training dataframes 
    TrainingDataFrame = pd.DataFrame()
    #Make One dataframe that is our test Dataframe 
    TestingDataFrame = NoNoiseDf[TestData] 
    for i in range(len(NoNoiseDf)):     
        if i == TestData: 
            continue 
        else: 
            #Append the training dataframe to one dataframe to send to the ML algorithm 
            TrainingDataFrame.append(NoNoiseDf[i])
    
    #Calculate the N value for the Training set
    TrainingN = ML.calculateN(TrainingDataFrame)
    #Calculate the Q value for the Training set
    TrainingQ = ML.calculateQ(TrainingN,len(TrainingDataFrame))
    #Calculate the F Matrix for the Training set
    TrainingF = ML.calculateF(TrainingN,TrainingDataFrame)
    #Create a Classifier Object to classify our test set 
    Classifier = Classifier(TrainingN,TrainingQ,TrainingF)


    #Increment the Trial and Testdata Number and do it again 
    Trial+=1 
    TestData +=1








    print("Program Finish")



#Call the main function
main() 
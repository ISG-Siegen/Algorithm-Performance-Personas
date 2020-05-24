from SiameseNeuralNetworkProject.PreProcessing.GraphicalVisualization import GraphicalVisualization
from SiameseNeuralNetworkProject.PreProcessing.LoanDatasetCleaning import LoanDatasetCleaning
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.CatBoost.CatBoost import CatBoost
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.MLPRegressor.MLPRegressor import MLPRegressor
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.SGDRegression.SGDRegression import SGDRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.RandomForestRegression.RandomForestRegression import RandomForestRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.LassoCVRegression.LassoCVRegression import LassoCVRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.AdaBoostRegressor.AdaBoostRegressor import AdaBoostRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.RANSACRegressor.RANSACRegressor import RANSACRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.GradientBoostingRegressor.GradientBoostingRegressor import GradientBoostingRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.PlottingFunctions.PlottingFunctions import PlottingFunctions
from SiameseNeuralNetworkProject.PerformanceMetric.PerformanceMetricNormalization import PerformanceMetricNormalization
from SiameseNeuralNetworkProject.SiameseNeuralNetwork.SiameseNeuralNetwork import SiameseNeuralNetwork
from SiameseNeuralNetworkProject.EmbeddingOutputSpace.EmbeddingClustering import EmbeddingClustering
from SiameseNeuralNetworkProject.TrainingPairingSystem.TrainingPairingSystem import TrainingPairingSystem
from SiameseNeuralNetworkProject.Utils.Utils import Utils
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from numpy import genfromtxt
import argparse
import random
from os import listdir
from os.path import isfile, join


# Visualization of the dataset, using multiple differebt approaches
def visualization():
    visualization=GraphicalVisualization()
    dataframe=visualization.read_CSV_file("outputCleaned",0)
    #Plotting process
    visualization.pandasProfileRows(dataframe,0)
    visualization.printDataframeHead(dataframe,20)
    visualization.correlationMatrix(dataframe,300)
    visualization.createScatterPlot(dataframe,"annual_inc","avg_cur_bal")
    visualization.createHistogramPlot(dataframe,"annual_inc")
    visualization.createBarChart(dataframe,"annual_inc")
    visualization.createBoxPlot(dataframe,"annual_inc")
    print("Done Plotting")


# all preprocessing steps done for the dataset will be unique to eqach dataset
def preprocessing():
    preprocessing = LoanDatasetCleaning()

    #columns to drop from the dataset
    dropColumns = ["collection_recovery_fee","funded_amnt_inv","num_rev_tl_bal_gt_0","num_sats","num_rev_tl_bal_gt_0","out_prncp","out_prncp_inv","url",'emp_title','policy_code']

    # drops the above list of columns
    #removes columns missing 5% data
    # Then drops rows with missing data
    print(dataframe.shape)
    dataframe = preprocessing.dropColumns(dataframe,dropColumns)
    dataframe = preprocessing.dropColumnsAboveMissingPercentage(dataframe,list(dataframe.columns.values),.05)
    dataframe = preprocessing.dropMissingDataRows(dataframe)
    print(dataframe.shape)

    #prints a list of the columns
    print(list(dataframe.columns.values))

    #prints the number of unique values in each column
    print(dataframe.select_dtypes('object').columns.values)
    listColumns = dataframe.select_dtypes('object').columns.values
    for column in listColumns:
        print(column+" : "+str(len(dataframe[column].unique())))


    missingDataColumnList=dataframe.columns[dataframe.isnull().any()].tolist()
    print(missingDataColumnList)

    # not used but can be used to clean data using different filling techniques

    # missing data cleaning
    # missingDataMeanFill=["dti","mths_since_rcnt_il"]#,"mths_since_rcnt_il"
    # missingDataBFill=["revol_util"]#,"next_pymnt_d","last_pymnt_d","all_util","bc_util","mo_sin_old_il_acct","last_credit_pull_d"
    # missingDataMO=["avg_cur_bal","bc_open_to_buy","mths_since_recent_bc"]#,"num_tl_120dpd_2m
    # missingDataZeroFill=["percent_bc_gt_75",]
    # dataframe=preprocessing.meanFillColumns(dataframe,missingDataMeanFill)
    # dataframe=preprocessing.backFillColumns(dataframe,missingDataBFill)
    # dataframe=preprocessing.mostOftenFillColumns(dataframe,missingDataMO)
    # dataframe=preprocessing.zeroFillColumns(dataframe,missingDataZeroFill)


    missingDataColumnListCleaned=dataframe.columns[dataframe.isnull().any()].tolist()
    print(missingDataColumnListCleaned)

    labelencodeColumns=("term","grade","sub_grade","home_ownership","verification_status","issue_d","loan_status",\
    "pymnt_plan","purpose","title","zip_code","addr_state","earliest_cr_line","initial_list_status","last_pymnt_d",\
    "last_credit_pull_d","application_type","hardship_flag","disbursement_method","debt_settlement_flag")

    visualization.pandasProfileRows(dataframe,19000)
    dataframe=preprocessing.labelEncoder(dataframe,list(dataframe.columns.values),labelencodeColumns)
    print(dataframe.head(5))

    #for one hot encoder
    #print(dataframe.head(5))
    #dataframe=preprocessing.oneHotEncoder(dataframe,("term",))
    #print(dataframe.head(5))

    # normalize using 0-1
    print("normalizing")
    print(dataframe.head(5))
    print(dataframe.shape)
    dataframe=preprocessing.normalizeDataset(dataframe)

    #output cleaned file to csv
    print(dataframe.head(5))
    print(dataframe.shape)
    dataframe.to_csv("outputCleanedAndNormalized.csv", index=False)


# splits data into training and testing
def splitData(trainingsplit=.90,testsplit=.10):
    print(trainingsplit)
    print(testsplit)

    df = pd.read_csv("./SiameseNeuralNetworkProject/PreProcessing/outputCleanedAndNormalized.csv")

    # splits roughly 50,000 instances to train the suite of algorithms
    algorithmSuiteTrainingData,SiameseTrainingData  = train_test_split(df, train_size=0.025)
    print(len(algorithmSuiteTrainingData))
    print(len(SiameseTrainingData))
    # splits siamese data into training and testing
    SiameseTrainingData,OverallTestingData  = train_test_split(SiameseTrainingData, train_size=trainingsplit)
    print(len(SiameseTrainingData))
    print(len(OverallTestingData))

    # label/target column
    LabelColumnSiamese = SiameseTrainingData['int_rate']
    LabelColumnTestData = OverallTestingData['int_rate']

    # drop it from training data
    SiameseTrainingData = SiameseTrainingData.drop(columns=['int_rate'])
    OverallTestingData = OverallTestingData.drop(columns=['int_rate'])

    algorithmSuiteTrainingData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TrainingData.csv", index=False)
    SiameseTrainingData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv", index=False)
    OverallTestingData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv", index=False)
    LabelColumnSiamese.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingTargetColumn.csv", index=False)
    LabelColumnTestData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TestDataTargetColumn.csv", index=False)



def evaluateSuiteOfAlgorithms():
    Plotting=True
    catBoost = CatBoost()
    sgdRegression = SGDRegression()
    mlpRegressor = MLPRegressor()
    randomForestRegressor = RandomForestRegression()
    lassoRegression = LassoCVRegression()
    adaBoostRegressor = AdaBoostRegression()
    baggingRegressor = BaggingRegression()
    extraTreeRegressor = ExtraTreeRegression()
    rANSACRegressor = RANSACRegression()
    gradientBoostingRegressor = GradientBoostingRegression()
    trainingData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TrainingData.csv")
    performanceData = pd.DataFrame(columns=["Algortihm","Number Of Rows","rSquared Acccuracy","Time Taken"])
    NumberOfRows = [100,500,1000,2000,5000,10000,20000,30000,40000,50000]
    AlgorithmSuiteArray=(lassoRegression,sgdRegression,mlpRegressor,catBoost,randomForestRegressor,adaBoostRegressor,rANSACRegressor,gradientBoostingRegressor)

    for rowNumber in NumberOfRows:
        trainingDataSample=trainingData.sample(rowNumber)
        for algorithm in AlgorithmSuiteArray:
            start = time.time()
            Acccuracy = algorithm.run(trainingDataSample,Plotting)
            end = time.time()
            row = [algorithm.getName(), rowNumber, Acccuracy,(end - start)]
            performanceData.loc[len(performanceData)] = row
            print(performanceData.tail())
    performanceData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/PerformanceData/PerformanceData.csv", index=False)


def runSuiteOfAlgorithms():
    Plotting=False
    # run the suite of algorithms
    # if true this will gather the data to plot performance
    trainingData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TrainingData.csv")
    catBoost = CatBoost()
    sgdRegression = SGDRegression()
    mlpRegressor = MLPRegressor()
    randomForestRegressor = RandomForestRegression()
    lassoRegression = LassoCVRegression()
    adaBoostRegressor = AdaBoostRegression()
    rANSACRegressor = RANSACRegression()
    gradientBoostingRegressor = GradientBoostingRegression()
    AlgorithmSuiteArray=(lassoRegression,sgdRegression,mlpRegressor,catBoost,randomForestRegressor,adaBoostRegressor,rANSACRegressor,gradientBoostingRegressor)

    for algorithm in AlgorithmSuiteArray:
        algorithm.run(trainingData,Plotting)

# reads performance data and plots it
def plotSuiteOfAlgorithms():
    plottingFunctions = PlottingFunctions()
    performanceData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/PerformanceData/PerformanceData.csv")
    plottingFunctions.plotPerformance(performanceData)





# calculates normalization of raw data
def CalculatePeformanceMetric():
        print("started second methhod of calculation")
        plottingFunctions=PlottingFunctions()
        performanceMetricNormalization=PerformanceMetricNormalization()


        labels = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingTargetColumn.csv",names=["labels",])
        SGDRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictions.csv",names=["SGDRegression",])
        CatBoostPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictions.csv",names=["CatBoost",])
        MLPRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictions.csv",names=["MLPRegressor",])
        RandomForestRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictions.csv",names=["RandomForestRegressor",])
        lassoCVRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictions.csv",names=["LassoCVRegression",])
        adaBoostRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictions.csv",names=["adaBoostRegressor",])
        RANSACRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictions.csv",names=["RANSACRegressor",])
        gradientBoostingRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictions.csv",names=["gradientBoostingRegressor",])

        predictionArray=[lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressor,RANSACRegressor,gradientBoostingRegressor]

        print("length of labels is "+str(len(labels)))
        print("length of predictions is "+str(len(predictionArray[0])))


        normalizedMetricDataframe = performanceMetricNormalization.CalculatePerformanceMetric(labels,predictionArray)

        plottingFunctions.plotRankNumbers(labels,1-normalizedMetricDataframe,"Converted metric performance data")


# used to find the averages of the performance and feature space, to estimate a margin for hard and easy positive and negitive
def estimateMargins():
        trainingPairingSystem = TrainingPairingSystem()
        trainingPairingSystem.estimatePositiveANdNegitiveMargin()

# pairing system for siamese neural network
def createPerformancePairs(featureSpaceClose=2.5,featureSpaceFar=4.5,performanceSpaceClose=0.2,performanceSpaceFar=1.8):
    trainingPairingSystem = TrainingPairingSystem()
    trainingPairingSystem.createPairs(featureSpaceClose,featureSpaceFar,performanceSpaceClose,performanceSpaceFar)


# runs the Siamese Neural Network
def runNetwork():
    siameseNeuralNetwork=SiameseNeuralNetwork()
    siameseNeuralNetwork.runNetwork()


# uses the embedding space and KNN to find and plot the final performance

def calculateKNN(neighbours=128):
    print(neighbours)
    embeddingClustering=EmbeddingClustering()
    embeddingClustering.ClusterEmbeddings(neighbours)
    calculateErrorArray()
    embeddingClustering.ConvertNearestTensorsToTrainingDataIndexes()
    embeddingClustering.ConvertTestDataToRankPredictions(neighbours)
    embeddingClustering.getActualPerformanceRank()

def getFinalPerformanceScore():
    embeddingClustering=EmbeddingClustering()
    embeddingClustering.getPerInstanceScore()
    embeddingClustering.getPeformanceScore()
    embeddingClustering.plotRanks()


def runUtilsMethod():
    utils=Utils()
    labels = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingTargetColumn.csv",names=["labels",])
    labelsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TestDataTargetColumn.csv",names=["labels",])
    SGDRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictions.csv",names=["SGDRegression",])
    CatBoostPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictions.csv",names=["CatBoost",])
    MLPRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictions.csv",names=["MLPRegressor",])
    RandomForestRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictions.csv",names=["RandomForestRegressor",])
    adaBoostRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictions.csv",names=["adaBoostRegressor",])
    RANSACRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictions.csv",names=["RANSACRegressor",])
    gradientBoostingRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictions.csv",names=["gradientBoostingRegressor",])
    lassoCVRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictions.csv",names=["LassoCVRegression",])
    SGDRegressionPredictionsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictionsTestData.csv",names=["SGDRegression",])
    CatBoostPredictionsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictionsTestData.csv",names=["CatBoost",])
    MLPRegressorPredictionsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictionsTestData.csv",names=["MLPRegressor",])
    RandomForestRegressorPredictionsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictionsTestData.csv",names=["RandomForestRegressor",])
    lassoCVRegressionPredictionsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictionsTestData.csv",names=["LassoCVRegression",])
    adaBoostRegressorPredictionsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictionsTestData.csv",names=["adaBoostRegressor",])
    gradientBoostingRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictionsTestData.csv",names=["RANSACRegressor",])
    RANSACRegressorPredictionsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictionsTestData.csv",names=["gradientBoostingRegressor",])

    predictionArray=[lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressorPredictions,RANSACRegressorPredictions,gradientBoostingRegressorPredictions]
    TestpredictionArray=[lassoCVRegressionPredictionsTestData,SGDRegressionPredictionsTestData,CatBoostPredictionsTestData,MLPRegressorPredictionsTestData,RandomForestRegressorPredictionsTestData,adaBoostRegressorPredictionsTestData,gradientBoostingRegressor,RANSACRegressorPredictionsTestData]#KNNPredictions,SVRPredictions


    # unnormalize the data
    for index in range(0,len(predictionArray)):
        predictionArray[index]=(predictionArray[index]*(30.99-5.31))+5.31


    labels=(labels*(30.99-5.31))+5.31
    labelsTestData=(labelsTestData*(30.99-5.31))+5.31

    #unnormalize the data
    for index in range(0,len(TestpredictionArray)):
        TestpredictionArray[index]=(TestpredictionArray[index]*(30.99-5.31))+5.31

    #utils.calculateAbsoluteError(labels,predictionArray)

    # for use in the RF regressor baseline
    utils.CreateAugmentedDataset(labels,labelsTestData,predictionArray,TestpredictionArray)

# calculates the absolute error array of all algorithms
def calculateErrorArray():
    utils=Utils()
    labels = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingTargetColumn.csv",names=["labels",])
    labelsTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TestDataTargetColumn.csv",names=["labels",])
    SGDRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictions.csv",names=["SGDRegression",])
    CatBoostPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictions.csv",names=["CatBoost",])
    MLPRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictions.csv",names=["MLPRegressor",])
    RandomForestRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictions.csv",names=["RandomForestRegressor",])
    lassoCVRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictions.csv",names=["LassoCVRegression",])
    adaBoostRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictions.csv",names=["adaBoostRegressor",])
    RANSACRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictions.csv",names=["RANSACRegressor",])
    gradientBoostingRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictions.csv",names=["gradientBoostingRegressor",])

    predictionArray=[lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressor,RANSACRegressor,gradientBoostingRegressor]
    utils.calculateAbsoluteError(labels,predictionArray)





def main():

    parser = argparse.ArgumentParser(description="Main area of command line interface")

    subparsers = parser.add_subparsers(dest='parser')

    datasetsplit = subparsers.add_parser("datasetsplit", help="Specify the dataset split between the train and test dataset")
    datasetsplit.add_argument('--traintestsplit',nargs=2, help="algorithm selection: select algorithms to run in the algorithm suite")

    algorithmselection = subparsers.add_parser("algorithmselection", help="algorithm selection: select algorithms to run in the algorithm suite")

    algorithmevaluation = subparsers.add_parser("algorithmevaluation", help="algorithm evaluation: get the performance of the selected algorithms")

    performancemetric = subparsers.add_parser("performancemetric", help="performance metric: Translates raw performance data to converted performance metric")

    marginestimation = subparsers.add_parser("marginestimation", help="margin estimation: Samples data and produces min, max and average values for feature and performance space")

    pairingsystem = subparsers.add_parser("pairingsystem", help="pairing system: Creates training pairs data based on users inputed margins or default margins")
    pairingsystem.add_argument('--featurespacemargins',type=float,nargs=2, help="Two values used to select the close and distant margin for the feature space")
    pairingsystem.add_argument('--performancespacemargins',type=float,nargs=2, help=" Two values used to select the close and distant margin for the performance space")

    snn = subparsers.add_parser("snn", help="Siamese Neural Network: Runs the Siamese neural network on the pairs produced by the pairing system")

    knn = subparsers.add_parser("knn", help="K Nearest Neighbours: Runs the knn algorithm to find closest embeddings to test data")
    knn.add_argument('--neighbournumber',nargs="?",type=int, help="Number of neighbours to be used in KNN model")

    performance = subparsers.add_parser("performance", help="performance: Calculates the final performance of the pipeline")

    runpipeline = subparsers.add_parser("runpipeline", help="run pipeline: Run the pipeline from start to finish, warning this will take several hours")

    args = parser.parse_args()


    if args.parser == 'datasetsplit':
        if int(args.traintestsplit[0])+int(args.traintestsplit[1])==100:
            splitData(int(args.traintestsplit[0])/100,int(args.traintestsplit[1])/100)

    if args.parser == 'algorithmselection':
        runSuiteOfAlgorithms()

    if args.parser == 'algorithmevaluation':
        evaluateSuiteOfAlgorithms()

    if args.parser == 'performancemetric':
        CalculatePeformanceMetric()

    if args.parser == 'marginestimation':
        estimateMargins()

    if args.parser == 'pairingsystem':
        if args.featurespacemargins is not None and args.performancespacemargins is not None:
            createPerformancePairs(args.featurespacemargins[0],args.featurespacemargins[1],args.performancespacemargins[0],args.performancespacemargins[1])
        else:
            createPerformancePairs()

    if args.parser == 'snn':
        runNetwork()

    if args.parser == 'knn':
        if args.neighbournumber is not None:
            calculateKNN(int(args.neighbournumber[0]))
        else:
            calculateKNN()


    if args.parser == 'performance':
        getFinalPerformanceScore()

    if args.parser == "runpipeline":
        #splitData()
        #evaluateSuiteOfAlgorithms()
        runSuiteOfAlgorithms()
        CalculatePeformanceMetric()
        #estimateMargins()
        createPerformancePairs()
        runNetwork()
        calculateKNN()
        getFinalPerformanceScore()




if __name__ == '__main__':
    main()

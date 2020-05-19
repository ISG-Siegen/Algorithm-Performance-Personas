from SiameseNeuralNetworkProject.PreProcessing.GraphicalVisualization import GraphicalVisualization
from SiameseNeuralNetworkProject.PreProcessing.LoanDatasetCleaning import LoanDatasetCleaning
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.CatBoost.CatBoost import CatBoost
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.KNN.KNN import KNN
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.MLPRegressor.MLPRegressor import MLPRegressor
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.SGDRegression.SGDRegression import SGDRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.SupportVectorRegressor.SupportVectorRegressor import SupportVectorRegressor
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.RandomForestRegression.RandomForestRegression import RandomForestRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.LassoCVRegression.LassoCVRegression import LassoCVRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.AdaBoostRegressor.AdaBoostRegressor import AdaBoostRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.BaggingRegressor.BaggingRegressor import BaggingRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.ExtraTreeRegressor.ExtraTreeRegressor import ExtraTreeRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.RANSACRegressor.RANSACRegressor import RANSACRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.GradientBoostingRegressor.GradientBoostingRegressor import GradientBoostingRegression
from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.PlottingFunctions.PlottingFunctions import PlottingFunctions
from SiameseNeuralNetworkProject.PerformanceMetric.PerformanceMetric import PerformanceMetric
from SiameseNeuralNetworkProject.PerformanceMetric.PerformanceMetricSecondApproach import PerformanceMetricSecondApproach
from SiameseNeuralNetworkProject.SiameseNeuralNetwork.SiameseNeuralNetwork import SiameseNeuralNetwork
from SiameseNeuralNetworkProject.SiameseNeuralNetwork.SiameseNeuralNetworkTripletLoss import SiameseNeuralNetworkTripletLoss
from SiameseNeuralNetworkProject.TrainingClustering.TrainingClustering import TrainingClustering
from SiameseNeuralNetworkProject.TensorOutputSpace.TensorClustering import TensorClustering
from SiameseNeuralNetworkProject.MarginEstimation.MarginEstimation import MarginEstimation
from SiameseNeuralNetworkProject.TrainingPairingSystem.TrainingPairingSystem import TrainingPairingSystem
from SiameseNeuralNetworkProject.Utils.Utils import Utils
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from numpy import genfromtxt
import time
import argparse
import random
from os import listdir
from os.path import isfile, join
from tensorflow.keras.utils import plot_model,normalize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras import optimizers


# uses padas profiling to get an overview look at the data
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
    dropColumns = ["collection_recovery_fee","funded_amnt_inv","num_rev_tl_bal_gt_0","num_sats","num_rev_tl_bal_gt_0","out_prncp","out_prncp_inv","url",'emp_title']

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
def splitData(trainingsplit=.80,testsplit=.20):
    print(trainingsplit)
    print(testsplit)

    #df = pd.read_csv("./SiameseNeuralNetworkProject/PreProcessing/outputCleanedAndNormalized.csv")#outputCleanedAndNormalized#outputCleaned
    df = pd.read_csv("./outputCleanedAndNormalized.csv")

    # remove this column should have been done in last section
    df = df.drop(columns=['policy_code'])

    print(len(df))

    # splits roughly 50,000 instances to train the suite of algorithms
    # algorithmSuiteTrainingData,SiameseTrainingData  = train_test_split(df, train_size=0.025)
    # print(len(algorithmSuiteTrainingData))
    # print(len(SiameseTrainingData))
    # # splits siamese data into training and testing
    # SiameseTrainingData,OverallTestingData  = train_test_split(SiameseTrainingData, train_size=trainingsplit)
    # print(len(SiameseTrainingData))
    # print(len(OverallTestingData))
    #
    # # label/target column
    # LabelColumnSiamese = SiameseTrainingData['int_rate']
    # LabelColumnTestData = OverallTestingData['int_rate']
    #
    # # drop it from training data
    # SiameseTrainingData = SiameseTrainingData.drop(columns=['int_rate'])
    # OverallTestingData = OverallTestingData.drop(columns=['int_rate'])
    #
    # algorithmSuiteTrainingData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TrainingData.csv", index=False)
    # SiameseTrainingData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv", index=False)
    # OverallTestingData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv", index=False)
    # LabelColumnSiamese.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingTargetColumn.csv", index=False)
    # LabelColumnTestData.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TestDataTargetColumn.csv", index=False)

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
    #NumberOfRows = [100,500,1000,2000,5000,10000,20000,30000,40000,50000]
    NumberOfRows = [50000]
    AlgorithmSuiteArray=(lassoRegression,sgdRegression,mlpRegressor,catBoost,randomForestRegressor,adaBoostRegressor,rANSACRegressor,gradientBoostingRegressor)#baggingRegressor,extraTreeRegressor

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
    knn = KNN()
    catBoost = CatBoost()
    sgdRegression = SGDRegression()
    mlpRegressor = MLPRegressor()
    supportVectorRegressor = SupportVectorRegressor()
    randomForestRegressor = RandomForestRegression()
    lassoRegression = LassoCVRegression()
    adaBoostRegressor = AdaBoostRegression()
    baggingRegressor = BaggingRegression()
    extraTreeRegressor = ExtraTreeRegression()
    rANSACRegressor = RANSACRegression()
    gradientBoostingRegressor = GradientBoostingRegression()
    AlgorithmSuiteArray=(lassoRegression,sgdRegression,mlpRegressor,catBoost,randomForestRegressor,adaBoostRegressor,rANSACRegressor,gradientBoostingRegressor) # not in use algorrithms: knn,supportVectorRegressor,baggingRegressor,extraTreeRegressor

    for algorithm in AlgorithmSuiteArray:
        algorithm.run(trainingData,Plotting)

# actually reads performance data and plots it
def plotSuiteOfAlgorithms():
    plottingFunctions = PlottingFunctions()
    performanceData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/PerformanceData/PerformanceData.csv")
    plottingFunctions.plotPerformance(performanceData)


# this code is used to calcuate the first performance metric (currently not in use)
def CalculatePeformanceMetric():
    plottingFunctions=PlottingFunctions()
    performanceMetric=PerformanceMetric()
    #helperFunctions=HelperFunctions()

    print("started calculation")
    labels = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingTargetColumn.csv",names=["labels",])
    #KNNPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/KNNPredictions.csv")
    #SVRPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SupportVectorRegressorPredictions.csv")
    SGDRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictions.csv",names=["SGDRegression",])
    CatBoostPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictions.csv",names=["CatBoost",])
    MLPRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictions.csv",names=["MLPRegressor",])
    RandomForestRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictions.csv",names=["RandomForestRegressor",])
    lassoCVRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictions.csv",names=["LassoCVRegression",])
    adaBoostRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictions.csv",names=["adaBoostRegressor",])
    RANSACRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictions.csv",names=["RANSACRegressor",])
    gradientBoostingRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictions.csv",names=["gradientBoostingRegressor",])

    predictionArray=(lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressor,RANSACRegressor,gradientBoostingRegressor)#KNNPredictions,SVRPredictions.baggingRegressor,extraTreeRegressor



    print("length of labels is "+str(len(labels)))
    print("length of predictions is "+str(len(predictionArray[0])))

    # can be used to calculate rsme
    #RSMEArray=performanceMetric.CalculateRSME(labels,predictionArray)

    #finalMetricDataframe = performanceMetric.CalculatePerformanceMetric(labels,predictionArray)

    # used to save a sample of the dataframe
    # finalMetricDataframe.head(1000).to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/PerformanceData/1000SamplePerformanceData.csv", index=False)

    # plot the rank of each algorithm. How amny times each algorithm comes first secend etc
    #plottingFunctions.plotRankNumbers(labels,finalMetricDataframe,"Converted metric performance data")


# improved metric: second approach to calculating the final metric
def CalculatePeformanceMetricSecondMetric():
        print("started second methhod of calculation")
        plottingFunctions=PlottingFunctions()
        performanceMetricSecondApproach=PerformanceMetricSecondApproach()


        labels = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingTargetColumn.csv",names=["labels",])
        SGDRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictions.csv",names=["SGDRegression",])
        CatBoostPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictions.csv",names=["CatBoost",])
        MLPRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictions.csv",names=["MLPRegressor",])
        RandomForestRegressorPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictions.csv",names=["RandomForestRegressor",])
        lassoCVRegressionPredictions = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictions.csv",names=["LassoCVRegression",])
        adaBoostRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictions.csv",names=["adaBoostRegressor",])
        RANSACRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictions.csv",names=["RANSACRegressor",])
        gradientBoostingRegressor = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictions.csv",names=["gradientBoostingRegressor",])



        #
        # oneKRandomSampleIndexes = random.sample(range(1, len(lassoCVRegressionPredictions)-1), 1000)
        # np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/oneKRandomSampleIndexes.csv", oneKRandomSampleIndexes, delimiter=",")
        # SGDRegressionPredictions = SGDRegressionPredictions.iloc[oneKRandomSampleIndexes]
        # CatBoostPredictions = CatBoostPredictions.iloc[oneKRandomSampleIndexes]
        # MLPRegressorPredictions = MLPRegressorPredictions.iloc[oneKRandomSampleIndexes]
        # RandomForestRegressorPredictions = RandomForestRegressorPredictions.iloc[oneKRandomSampleIndexes]
        # lassoCVRegressionPredictions = lassoCVRegressionPredictions.iloc[oneKRandomSampleIndexes]
        # adaBoostRegressor = adaBoostRegressor.iloc[oneKRandomSampleIndexes]
        # RANSACRegressor= RANSACRegressor.iloc[oneKRandomSampleIndexes]
        # gradientBoostingRegressor = gradientBoostingRegressor.iloc[oneKRandomSampleIndexes]


        predictionArray=[lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressor,RANSACRegressor,gradientBoostingRegressor]#KNNPredictions,SVRPredictions,baggingRegressor,extraTreeRegressor

        # unnormalize the data
        # for index in range(0,len(predictionArray)):
        #     predictionArray[index]=(predictionArray[index]*(30.99-5.31))+5.31
        #
        #
        # labels=(labels*(30.99-5.31))+5.31


        #PredictionIndexSampled=[0,0,0,0,0]
        #oneKRandomSampleIndexes = random.sample(range(1, len(predictionArray[1])-1), 1000)
        #print(len(oneKRandomSampleIndexes))
        #labels = labels.iloc[oneKRandomSampleIndexes]

        # for index in range(0,len(predictionArray)-1):
        #     PredictionIndexSampled[index] = predictionArray[index].iloc[oneKRandomSampleIndexes]

        print("length of labels is "+str(len(labels)))
        print("length of predictions is "+str(len(predictionArray[0])))


        finalMetricDataframe = performanceMetricSecondApproach.CalculatePerformanceMetric(labels,predictionArray)

        plottingFunctions.plotRankNumbers(labels,1-finalMetricDataframe,"Converted metric performance data")


# this was used to create training clusters using the mean shift clustering algorithm
def CreateTrainingClusters():
    print("started clustering")
    trainingClustering=TrainingClustering()
    # first and second metric: First metric commented out
    #algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetric.csv")
    algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetricSecondApproach.csv")

    print("Algorithm performance first and last 5 rows:")
    print(algorithmPerformance.head())
    print(algorithmPerformance.tail())

    #gets labels from clustering
    clusterLabels,clusterCenters=trainingClustering.ClusterData(algorithmPerformance)

    #clusters training data based on second performance metric by joining labels to training data as columns
    trainingClustering.SiameseTrainingDataGrouping()

    # groups cluster performance into clusters to measure distance in cluster space
    trainingClustering.groupFinalMetricToClusters()

    # this is a method I experimented with no currently in use
    # # second filter method based on rank
    # performanceClustersArray=trainingClustering.SiamesePerformanceDataGrouping(clusterLabels,algorithmPerformance)
    # RankDistance=8
    # SecondMetricArray = trainingClustering.filterClustersOnRank(performanceClustersArray,clusterCenters,RankDistance)

def estimateMargins():
        trainingPairingSystem = TrainingPairingSystem()
        trainingPairingSystem.estimatePositiveANdNegitiveMargin()

# new approach to pairing
def createPerformancePairs(featureSpaceClose=2.5,featureSpaceFar=4.5,performanceSpaceClose=0.2,performanceSpaceFar=1.8):
    print(featureSpaceClose)
    print(featureSpaceFar)
    print(performanceSpaceClose)
    print(performanceSpaceFar)
    trainingPairingSystem = TrainingPairingSystem()
    trainingPairingSystem.createPairs(featureSpaceClose,featureSpaceFar,performanceSpaceClose,performanceSpaceFar)


# create the training data and run the final network
def runNetwork():
    siameseNeuralNetwork=SiameseNeuralNetwork()
    #siameseNeuralNetwork.setUpNetwork()
    #marginEstimation.getMaxDistanceForCluster()
    #siameseNeuralNetwork.SetUpNetworkWithPositivePairsDistanceSelection()
    #siameseNeuralNetwork.SetUpNetworkWithNegitivePairsDistanceSelection(1.5)
    #siameseNeuralNetwork.SetUpNetworkWithNegitivePairs()
    siameseNeuralNetwork.runNetwork()

# currently workin on triplet loss still not fully working
def runTripletNetwork():
    tripletNetwork=SiameseNeuralNetworkTripletLoss()
    dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin=tripletNetwork.buildDataSet()
    triplets = tripletNetwork.get_batch_random(2,dataset_train,dataset_test)
    print("Checking batch width, should be 3 : ",len(triplets))
    print("Shapes in the batch A:{0} P:{1} N:{2}".format(triplets[0].shape, triplets[1].shape, triplets[2].shape))
    #set up network
    network=tripletNetwork.build_network(trbrightnessbbbiplets[0].shape[1:])
    network_train = tripletNetwork.build_model(triplets[0].shape[1:],network)
    optimizer = Adam()
    network_train.compile(loss=None,optimizer=optimizer)

    hardtriplets = tripletNetwork.get_batch_hard(50,20,20,network,dataset_train,dataset_test)
    print("Shapes in the hardbatch A:{0} P:{1} N:{2}".format(hardtriplets[0].shape, hardtriplets[1].shape, hardtriplets[2].shape))
    idx = np.random.randint(len(x_test_origin), size=200)
    probs,yprobs=tripletNetwork.compute_probs(network,x_test_origin[idx,:],y_test_origin[idx,:])
    fpr, tpr, thresholds,auc = tripletNetwork.compute_metrics(probs,yprobs)
    tripletNetwork.draw_roc(fpr, tpr,thresholds,auc)
    tripletNetwork.draw_interdist(network,0,10,triplets[0].shape,dataset_test)

    n_iteration=1000
    n_val=180
    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iteration):
        triplets = tripletNetwork.get_batch_hard(200,32,32,network,dataset_train,dataset_test)
        loss = network_train.train_on_batch(triplets, None)
        n_iteration += 1
        if i % 200 == 0:
            print("\n ------------- \n")
            print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration))
            idx = np.random.randint(len(x_test_origin), size=100)
            probs,yprobs=tripletNetwork.compute_probs(network,x_test_origin[idx,:],y_test_origin[idx,:])
            #probs,yprob = tripletNetwork.compute_probs(network,x_test_origin[:n_val,:],y_test_origin[:n_val,:])
            fpr, tpr, thresholds,auc = tripletNetwork.compute_metrics(probs,yprobs)
            tripletNetwork.draw_roc(fpr, tpr,thresholds,auc)
            tripletNetwork.draw_interdist(network,0,10,triplets[0].shape,dataset_test)

    idx = np.random.randint(len(x_test_origin), size=1000)
    probs,yprobs=tripletNetwork.compute_probs(network,x_test_origin[idx,:],y_test_origin[idx,:])
    #probs,yprob = tripletNetwork.compute_probs(network,x_test_origin,y_test_origin)
    fpr, tpr, thresholds,auc = tripletNetwork.compute_metrics(probs,yprobs)
    tripletNetwork.draw_roc(fpr, tpr,thresholds,auc)
    tripletNetwork.draw_interdist(network,0,10,triplets[0].shape,dataset_test)


# def indexTrainingData():
#     trainingData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
#     LabelsForData = pd.read_csv("./SiameseNeuralNetworkProject/TempFolder/labelsForPerformanceData.csv",names=["labels",])
#     print("labels size: "+str(len(LabelsForData)))
#     print("training data size: "+str(len(trainingData)))
#     trainingData["indexes"] = trainingData.index
#     trainingData.to_csv("./SiameseNeuralNetworkProject/TempFolder/TrainingDataWithIndexes.csv", index=False)

# this uses the etensor space to use KNN and plot the final performance

def calculateKNN(neighbours=12):
    print(neighbours)
    tensorClustering=TensorClustering()
    #tensorClustering.ClusterTensorsWithLabels()
    tensorClustering.ClusterTensors(neighbours)
    calculateErrorArray()
    tensorClustering.ConvertNearestTensorsToTrainingDataIndexes()
    tensorClustering.ConvertTestDataToRankPredictions(neighbours)
    tensorClustering.getActualPerformanceRank()

def getTrainingTensorIndexes():
    tensorClustering=TensorClustering()
    tensorClustering.getPerInstanceScore()
    tensorClustering.getPeformanceScore()
    #tensorClustering.plotRanks()


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
    #helperFunctions=HelperFunctions()
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

    #labelsTestData=(labelsTestData*(30.99-5.31))+5.31

    #utils.calculateInverseAbsoluteError(labels,predictionArray)
    #utils.getClusterSpread()

    utils.CreateAugmentedDataset(labels,labelsTestData,predictionArray,TestpredictionArray)


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

    #helperFunctions=HelperFunctions()
    predictionArray=[lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressor,RANSACRegressor,gradientBoostingRegressor]
    utils.calculateInverseAbsoluteError(labels,predictionArray)

def marginEstimationProcess():
    marginEstimation=MarginEstimation()
    marginEstimation.getLargestAndSmallestValues()
    #marginEstimation.getMaxDistanceForCluster()
    clusterCenters = genfromtxt("./SiameseNeuralNetworkProject/TempFolder/clusterCenters.csv",delimiter=",")
    marginEstimation.getClusterDistancesToOrigin(clusterCenters,[0.5,0.5,0.5,0.5,0.5])

def unNormalizedTarget():
    #dataframeOriginal=pd.read_csv("./SiameseNeuralNetworkProject/PreProcessing/loan.csv")
    #dfCleaned = pd.read_csv("./outputCleanedAndNormalized.csv")
    #dfCleaned = pd.read_csv("./SiameseNeuralNetworkProject/PreProcessing/outputCleanedAndNormalized.csv")
    dfCleaned = pd.read_csv("./SiameseNeuralNetworkProject/PreProcessing/outputCleaned.csv")
    #print(dataframeOriginal.head())
    #dfCleaned = dfCleaned["int_rate"]
    #dfCleaned = dfCleaned-dfCleaned.min()
    print(dfCleaned)
    dfCleanedNormalized = (dfCleaned-dfCleaned.min())/(dfCleaned.max()-dfCleaned.min())
    print(dfCleanedNormalized)
    # print("Minimum value is: "+str(dfCleaned.min()))
    # print("Maximum value is: "+str(dfCleaned.max()))


#runUtilsMethod()

#unNormalizedTarget()
#indexTrainingData()
#splitData()
#suiteOfAlgorithms()
#plotSuiteOfAlgorithms()
#CalculatePeformanceMetric()
#CalculatePeformanceMetricSecondMetric()
#CreateTrainingClusters()
#createPerformancePairs()
#runNetwork()
#runTripletNetwork()
#getTrainingTensorIndexes()



#getClusterCount()
#saveTrainingPredictions()
#marginEstimationProcess()
# runUtilsMethod()
# print("Finished")
def main():

    parser = argparse.ArgumentParser(description="Main area of command line interface")

    subparsers = parser.add_subparsers(dest='parser')

    datasetsplit = subparsers.add_parser("datasetsplit", help="Specify the dataset split between the train and test dataset")
    datasetsplit.add_argument('--traintestsplit',nargs=2, help="algorithm selection: select algorithms to run in the algorithm suite")

    algorithmselection = subparsers.add_parser("algorithmselection", help="algorithm selection: select algorithms to run in the algorithm suite")
    #algorithmselection.add_argument('--algorithms',nargs='+', help="algorithm selection: select algorithms to run in the algorithm suite")

    algorithmevaluation = subparsers.add_parser("algorithmevaluation", help="algorithm evaluation: get the performance of the selected algorithms")
    #algorithmevaluation.add_argument('--algorithms',nargs='+', help="algorithm selection: select algorithms to evaluate in the algorithm suite")

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
        print(args)
        print(args.traintestsplit)
        if int(args.traintestsplit[0])+int(args.traintestsplit[1])==100:
            splitData(int(args.traintestsplit[0])/100,int(args.traintestsplit[1])/100)

    if args.parser == 'algorithmselection':
        print(args)
        runSuiteOfAlgorithms()

    if args.parser == 'algorithmevaluation':
        print(args)
        evaluateSuiteOfAlgorithms()

    if args.parser == 'performancemetric':
        print(args)
        CalculatePeformanceMetricSecondMetric()

    if args.parser == 'marginestimation':
        print(args)
        estimateMargins()

    if args.parser == 'pairingsystem':
        print(args)
        if args.featurespacemargins is not None and args.performancespacemargins is not None:
            createPerformancePairs(args.featurespacemargins[0],args.featurespacemargins[1],args.performancespacemargins[0],args.performancespacemargins[1])
        else:
            createPerformancePairs()

    if args.parser == 'snn':
        print(args)
        runNetwork()

    if args.parser == 'knn':
        print(args)
        if args.neighbournumber is not None:
            calculateKNN(int(args.neighbournumber[0]))
        else:
            calculateKNN()


    if args.parser == 'performance':
        getTrainingTensorIndexes()

    if args.parser == "runpipeline":
        #splitData(.8,.2)
        #evaluateSuiteOfAlgorithms()
        #runSuiteOfAlgorithms()
        #CalculatePeformanceMetricSecondMetric()
        #estimateMargins()
        #createPerformancePairs()
        #runNetwork()
        calculateKNN(2)
        getTrainingTensorIndexes()
        #runUtilsMethod()




if __name__ == '__main__':
    main()

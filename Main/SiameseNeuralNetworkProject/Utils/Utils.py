import numpy as np
import pandas as pd
from numpy import genfromtxt
from os import listdir
from os.path import isfile, join

class Utils:


    def calculateAbsoluteError(self,labels,PredictionArray):
        print(labels.shape)
        result = np.concatenate((PredictionArray[0], PredictionArray[1]),axis=1)
        for i in range(2,(len(PredictionArray))):
            result = np.concatenate((result, PredictionArray[i]),axis=1)

        for i in range(0,result.shape[1]):
            result[:,i]=abs(labels.iloc[:, 0]-result[:,i])

        np.savetxt("./SiameseNeuralNetworkProject/Utils/AbsoluteErrorPerformanceRawData.csv", result, delimiter=",")



    def CreateAugmentedDataset(self,labels,TestLabels,PredictionArray,TestPredictionArray):
        # read in the dataset (training and test) and concat it with the prediction data
        SiameseTrainingData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
        SiameseTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv")


        #concat predictions
        AugmentedLoanDatasetPredictions = np.concatenate((PredictionArray[0], PredictionArray[1]),axis=1)
        for i in range(2,(len(PredictionArray))):
            AugmentedLoanDatasetPredictions = np.concatenate((AugmentedLoanDatasetPredictions, PredictionArray[i]),axis=1)

        # get absolute error
        AugmentedLoanDatasetAbsoluteError = np.copy(AugmentedLoanDatasetPredictions)
        for i in range(0,AugmentedLoanDatasetPredictions.shape[1]):
            AugmentedLoanDatasetAbsoluteError[:,i]=abs(labels.iloc[:, 0] - AugmentedLoanDatasetPredictions[:,i])

        # get ranks
        AugmentedLoanDatasetRanks = np.copy(AugmentedLoanDatasetAbsoluteError)
        AugmentedLoanDatasetRanks = np.argsort(np.argsort(AugmentedLoanDatasetRanks))

        TestAugmentedLoanDatasetPredictions = np.concatenate((TestPredictionArray[0], TestPredictionArray[1]),axis=1)
        for i in range(2,(len(PredictionArray))):
            TestAugmentedLoanDatasetPredictions = np.concatenate((TestAugmentedLoanDatasetPredictions, TestPredictionArray[i]),axis=1)

        AugmentedLoanTestDatasetAbsoluteError = np.copy(TestAugmentedLoanDatasetPredictions)
        for i in range(0,TestAugmentedLoanDatasetPredictions.shape[1]):
            AugmentedLoanTestDatasetAbsoluteError[:,i]=abs(TestLabels.iloc[:, 0]-TestAugmentedLoanDatasetPredictions[:,i])

        AugmentedLoanTestDatasetRanks = np.copy(AugmentedLoanTestDatasetAbsoluteError)
        AugmentedLoanTestDatasetRanks = np.argsort(np.argsort(AugmentedLoanTestDatasetAbsoluteError))

        # concat the label, prediction and absolute errorand rank data together for training data
        AugmentedLoanDataset = np.concatenate((labels,AugmentedLoanDatasetPredictions),axis=1)
        AugmentedLoanDataset = np.concatenate((AugmentedLoanDataset, AugmentedLoanDatasetAbsoluteError),axis=1)
        AugmentedLoanDataset = np.concatenate((AugmentedLoanDataset, AugmentedLoanDatasetRanks),axis=1)


        # concat the label, prediction and absolute error and rank data together for test data
        TestAugmentedLoanDataset = np.concatenate((TestLabels,TestAugmentedLoanDatasetPredictions),axis=1)
        TestAugmentedLoanDataset = np.concatenate((TestAugmentedLoanDataset, AugmentedLoanTestDatasetAbsoluteError),axis=1)
        TestAugmentedLoanDataset = np.concatenate((TestAugmentedLoanDataset, AugmentedLoanTestDatasetRanks),axis=1)


        #np.savetxt("./SiameseNeuralNetworkProject/Utils/deleteOncedUsed.csv", AugmentedLoanDataset, delimiter=",")

        indexNames = ["Label","Lasso Regression Predictions","SGD Regression Predictions","CatBoost Regression Predictions","MLP Regressor Predictions","Random Forest Regression Predictions","Ada Regression Predictions","RSNAC Regression Predictions","Gradient Boosting Regression Predictions",
        "Lasso Regression ABS Error","SGD Regression ABS Error","CatBoost Regression ABS Error","MLP Regressor ABS Error","Random Forest Regression ABS Error","Ada Regression ABS Error","RSNAC Regression ABS Error","Gradient Boosting Regression ABS Error",
        "Lasso Regression Rank","SGD Regression Rank","CatBoost Regression Rank","MLP Regressor Rank","Random Forest Regression Rank","Ada Regression Rank","RSNAC Regression Rank","Gradient Boosting Regression Rank"]
        AugmentedLoanDataset = pd.DataFrame(data = AugmentedLoanDataset[:,:], columns=indexNames)
        AugmentedTestLoanDataset = pd.DataFrame(data = TestAugmentedLoanDataset[:,:], columns=indexNames)

        # join the loan data and the test data together
        FinalAugmentedLoanDataset = pd.concat([SiameseTrainingData, AugmentedLoanDataset],axis=1,sort=False)
        FinalTestAugmentedLoanDataset = pd.concat([SiameseTestData, AugmentedTestLoanDataset],axis=1,sort=False)

        print(FinalAugmentedLoanDataset.head(5))
        print(FinalTestAugmentedLoanDataset.head(5))

        FinalAugmentedLoanDataset.to_csv('./SiameseNeuralNetworkProject/Utils/AugmentedLoanDataset.csv', index=False)
        FinalTestAugmentedLoanDataset.to_csv('./SiameseNeuralNetworkProject/Utils/AugmentedTestLoanDataset.csv', index=False)
        FinalAugmentedLoanDataset.to_hdf('./SiameseNeuralNetworkProject/Utils/AugmentedLoanDataset.h5', key='df',format="table",mode='w')
        FinalTestAugmentedLoanDataset.to_hdf('./SiameseNeuralNetworkProject/Utils/AugmentedTestLoanDataset.h5', key='df',format="table", mode='w')

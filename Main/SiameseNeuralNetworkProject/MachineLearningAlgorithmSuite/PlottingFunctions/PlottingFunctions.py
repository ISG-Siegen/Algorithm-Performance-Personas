import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class PlottingFunctions:
    AlgorithmNames = ["SGD Regression","Lasso Regression","CatBoost","MLP Regressor","Random Forest Regression","adaBoostRegressor","RANSACRegressor","gradientBoostingRegressor"]

    def plotPerformance(self,performanceData):
            print(performanceData.head())
            algorithmNames = performanceData["Algortihm"].unique()
            plt.figure(figsize = (16,5))
            for algorithm in algorithmNames:
                plt.plot(performanceData.loc[performanceData['Algortihm'] == algorithm, 'Number Of Rows'], performanceData.loc[performanceData['Algortihm'] == algorithm, 'rSquared Acccuracy'], label = 'Training error')
                plt.ylabel('R Squared Error', fontsize = 14)
                plt.xlabel('Training set size', fontsize = 14)
                title = 'Learning curves for different models'
                plt.title(title, fontsize = 18, y = 1.03)
                plt.legend(self.AlgorithmNames)
                plt.ylim(0,40)
            plt.show()

    def plotRankNumbers(self,labels,predictedValueArray,typeOfPlot):
        print("Plotting rank graph for "+str(typeOfPlot))
        calculatedRank = self.CalculateRankNumbers(predictedValueArray)
        ax = calculatedRank.plot(kind='bar',title="Plotting rank graph for "+str(typeOfPlot)+" for "+str(len(labels))+" instances",)
        ax.set_xlabel("Rank (0 = Best)")
        ax.set_ylabel("Amount")
        #plt.xlim(xmin=1)
        plt.legend(self.AlgorithmNames)
        plt.show()

    def CalculateRankNumbers(self,predictedValueArray):
        print("##################### Plotting algorithm rankings ######################################")
        # creates a numpy array with all the prediction values concatenated
        print(str(type(predictedValueArray)))
        if (str(type(predictedValueArray))!="<class 'pandas.core.frame.DataFrame'>"):
            print(predictedValueArray[0].head())
            concatPredictedValueArray = pd.concat([predictedValueArray[0], predictedValueArray[1]],axis=1)
            print(type(predictedValueArray))
            for i in range(2,(len(predictedValueArray))):
                concatPredictedValueArray = pd.concat([concatPredictedValueArray, predictedValueArray[i]],axis=1)
        else:
            concatPredictedValueArray=predictedValueArray
        print(concatPredictedValueArray.head())
        #copy the dataframe
        rankedConcatPredictedValueArray = concatPredictedValueArray.copy()
        #sort the algorithms based on rank
        rankedConcatPredictedValueArray = np.argsort(rankedConcatPredictedValueArray.values.argsort())
        # convert the numpy array back to a dataframe
        rankedConcatPredictedValueArray = pd.DataFrame(rankedConcatPredictedValueArray,columns=concatPredictedValueArray.columns)
        print(rankedConcatPredictedValueArray.head())
        #column names
        columnNames=concatPredictedValueArray.columns
        # empty DataFrame
        allCountResult = pd.DataFrame()
        for column in columnNames:
            print(column)
            uniqueRankCount = rankedConcatPredictedValueArray[column].value_counts()
            CountResult = pd.DataFrame(uniqueRankCount,columns=[column,])
            print(CountResult.head())
            allCountResult = pd.concat([allCountResult, CountResult],axis=1)
            print(allCountResult.head())
        allCountResult.sort_index()
        print(allCountResult.head())
        allCountResult.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/PerformanceData/RankPerformanceData.csv", index=False)
        return allCountResult

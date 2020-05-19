from SiameseNeuralNetworkProject.MachineLearningAlgorithmSuite.PlottingFunctions.PlottingFunctions import PlottingFunctions
import numpy as np
import pandas as pd


class PerformanceMetricSecondApproach:

        def CalculatePerformanceMetric(self,labels,predictedValueArray):
            plottingFunctions=PlottingFunctions()
            FinalMetricArray=[]

            print("Array type "+str(type(predictedValueArray)))
            #print(predictedValueArray[0])

            MaxLabelValue = labels.iloc[:, 0].max()
            print("max label value: ")
            print(MaxLabelValue)



            #join all the algorithm predictions together
            allPredictionColumns = np.concatenate((predictedValueArray[0], predictedValueArray[1]),axis=1)
            allPredictionColumnsDF = pd.concat([predictedValueArray[0], predictedValueArray[1]],axis=1)
            for i in range(2,(len(predictedValueArray))):
                 allPredictionColumns = np.concatenate((allPredictionColumns, predictedValueArray[i]),axis=1)
                 allPredictionColumnsDF = pd.concat([allPredictionColumnsDF, predictedValueArray[i]],axis=1)
            print(allPredictionColumns.shape)
            print(allPredictionColumnsDF.head())

            np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/TargetPerformanceSecondMetric.csv", labels, delimiter=",")
            np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/PredictionPerformanceSecondMetric.csv", allPredictionColumns, delimiter=",")
            #np.savetxt("./SiameseNeuralNetworkProject/Utils/RawPredictionArray.csv", result, delimiter=",")
            #resultDF.head(1000).to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/PerformanceData/1000pointRawPerformanceData.csv", index=False)

            # get max possible error for each row (max(label-0,maxValue-label))

            # creates a single column array of max label values
            MaxLabelValueArray = np.full((len(allPredictionColumns), 1), MaxLabelValue)
            #MaxLabelValueArray = np.full((len(allPredictionColumns), 1.0), MaxLabelValue)

            SecondValue = MaxLabelValueArray[:, 0] - labels.iloc[:, 0]
            SecondValue = SecondValue.values.reshape(-1,1)

            # labels minus 0: First value for range
            firstValue = labels.iloc[:, 0]
            firstValue = firstValue.values.reshape(-1,1)

            # concat two values to form range
            rangeForMax = np.concatenate((firstValue, SecondValue),axis=1)

            maxPossibleError = np.amax(rangeForMax,1)

            # save max possible error
            np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/maxPossibleErrorSecondMetric.csv", maxPossibleError, delimiter=",")



            print("max maxPossibleError value is: "+str(maxPossibleError))
            # get an array of absolute error values to find minimum performing value
            # second array used to allow plotting function to execute
            absoluteErrorArray=np.copy(allPredictionColumns)
            absoluteErrorForDF=np.copy(allPredictionColumnsDF)


            #cycle through each column of array to calculate absolute error
            for i in range(0,allPredictionColumns.shape[1]):
                absoluteErrorArray[:,i] = abs(labels.iloc[:, 0] - allPredictionColumns[:,i])
                absoluteErrorForDF[:,i] = abs(labels.iloc[:, 0] - allPredictionColumns[:,i])

            # data save for spreadsheet
            np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/absoluteErrorSecondMetric.csv", absoluteErrorArray, delimiter=",")

            # code to plot raw performance data
            np.savetxt("./SiameseNeuralNetworkProject/Utils/InverseErrorPerformanceRawData.csv", absoluteErrorArray, delimiter=",")
            #absoluteErrorForDF = pd.DataFrame(data=absoluteErrorForDF[0:,0:], columns=allPredictionColumnsDF.columns)
            absoluteErrorForDF=pd.DataFrame(data=absoluteErrorForDF[:,:])
            plottingFunctions.plotRankNumbers(labels,absoluteErrorForDF,"raw error data")

            # get the minimum absolute error
            minimumAbsoluteErrorValue = np.amin(absoluteErrorArray,1)

            #create an empty array for realitiveIntraDistancePerformance
            realitiveIntraDistancePerformance = np.zeros((allPredictionColumns.shape[0], allPredictionColumns.shape[1]))

            # get realitive Intra-Distance Performance
            for i in range(0,allPredictionColumns.shape[1]):
                realitiveIntraDistancePerformance[:,i] = minimumAbsoluteErrorValue[:]/absoluteErrorArray[:,i]

            # data save for spreadsheet
            np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/realitiveIntraDistancePerformanceSecondMetric.csv", realitiveIntraDistancePerformance, delimiter=",")

            #create an empty array for realitive possible error
            realitivePossibleError = np.zeros((allPredictionColumns.shape[0], allPredictionColumns.shape[1]))

            # get realitive Possible Error
            for i in range(0,allPredictionColumns.shape[1]):
                realitivePossibleError[:,i] = absoluteErrorArray[:,i]/maxPossibleError[:]

            # data save for spreadsheet
            np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/realitivePossibleErrorSecondMetric.csv", realitivePossibleError, delimiter=",")
            #create an empty array for the final output
            FinalMetric = np.zeros((allPredictionColumns.shape[0], allPredictionColumns.shape[1]))

            # get final metric
            for i in range(0,allPredictionColumns.shape[1]):
                FinalMetric[:,i] = realitiveIntraDistancePerformance[:,i] * (1-realitivePossibleError[:,i])

            # data save for spreadsheet
            np.savetxt("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetricSpreadSheetSecondMetric.csv", FinalMetric, delimiter=",")

            print(FinalMetric)
            print(FinalMetric.shape)
            print(type(FinalMetric))

            OutputDataframe = pd.DataFrame(data=FinalMetric[0:,0:],# 1st column as index
                                           columns=allPredictionColumnsDF.columns)
            print(OutputDataframe.head())
            print(OutputDataframe.tail())
            OutputDataframe.to_csv("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetricSecondApproach.csv", index=False)
            return OutputDataframe

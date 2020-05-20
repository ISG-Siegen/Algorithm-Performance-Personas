import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt
from matplotlib import pyplot as plt
import time
import math

class EmbeddingClustering:

        AlgorithmNames = ["SGD Regression","Lasso Regression","CatBoost","MLP Regressor","Random Forest Regression","ADA Boost","RANSC Regression","Gradient Boosting"]


        #Finds nearest neighbours and their indexes
        def ClusterEmbeddings(self,neighbours=12):
            pairsLeftTensors = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TensorOutputLeftSide.csv', delimiter=',')
            pairsRightTensors = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TensorOutputRightSide.csv', delimiter=',')
            TestDataTensors = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/OverallTestDataTensors.csv', delimiter=',')

            TensorTrainingData = np.concatenate((pairsLeftTensors, pairsRightTensors), axis=0)

            nbrs = NearestNeighbors(n_neighbors=neighbours, algorithm='ball_tree',n_jobs=9).fit(TensorTrainingData)
            distances, indices = nbrs.kneighbors(TestDataTensors)

            print("Finished KNN Clustering")
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/FinalIndicesOfTrainingTensorsTestData.csv", indices, delimiter=",")


        def ConvertNearestTensorsToTrainingDataIndexes(self):
            indices = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/FinalIndicesOfTrainingTensorsTestData.csv", delimiter=",")
            pairsLeftTensors = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TensorOutputLeftSide.csv', delimiter=',')
            pairsRightTensors = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TensorOutputRightSide.csv', delimiter=',')
            TestDataTensors = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/OverallTestDataTensors.csv', delimiter=',')
            # these indexs link the embeddings to the training instances
            TensorLeftSideIndex = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/pairsLeftIndex.csv', delimiter=',')
            TensorRightSideIndex = genfromtxt('./SiameseNeuralNetworkProject/EmbeddingOutputSpace/pairsRightIndex.csv', delimiter=',')

            print("Size of test data is: "+str(len(TestDataTensors)))
            print("Size of indeces data is: "+str(len(indices)))
            print("Size of pairsLeftTensors data is: "+str(len(pairsLeftTensors)))
            print("Size of pairsLeftTensors data is: "+str(len(pairsRightTensors)))
            print("Size of TensorLeftSideIndex data is: "+str(len(TensorLeftSideIndex)))
            print("Size of TensorRightSideIndex data is: "+str(len(TensorRightSideIndex)))
            TrainingIndexArray=[]

            for index in indices:
                TempArray=[]

                for i in range(0,(index.shape[0])):
                    TrainingTensorIndex = int(index[i])

                    #checks if right or left side of data
                    if ((TrainingTensorIndex)-(len(pairsLeftTensors)))>=0:
                        #right side index
                        #1875272-937636=937636
                        TrainingDataIndex = TensorRightSideIndex[(TrainingTensorIndex-(len(pairsLeftTensors)))]
                        TempArray.append(TrainingDataIndex)
                    else:
                        #left side index
                        TrainingDataIndex = TensorLeftSideIndex[TrainingTensorIndex]
                        TempArray.append(TrainingDataIndex)
                TrainingIndexArray.append(TempArray)


            TrainingIndexArray = np.asarray(TrainingIndexArray)
            print("Length of final indexed array is "+str(len(TrainingIndexArray)))
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TrainingDataIndexesForTestData.csv", TrainingIndexArray, delimiter=",")




        def ConvertTestDataToRankPredictions(self,neighbours):
            TrainingIndexArray = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TrainingDataIndexesForTestData.csv", delimiter=",")
            # #next step: I have training data indexes, for each test data point. find performance of training data. Get algorithm rank.use a most often and mean vote. record answer
            # # record algorithm choice and how many times each was voted


            algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/Utils/AbsoluteErrorPerformanceRawData.csv",header=None)
            print(algorithmPerformance.head())


            algorithmPerformance["indexes"] = algorithmPerformance.index

            algorithmPerformanceWithIndex=algorithmPerformance.values

            #TrainingIndexArray holds indexes for each performance value
            TestIndex=0
            counter=0
            BestPerformingAlgo = []
            MeanBestPerformingAlgo=[]
            for index in TrainingIndexArray:
                counter+=1
                if counter%10000==0:
                    print("Done: "+str(counter))
                TestPointRankedPerformance=[]
                performanceMetric=[]
                for i in range(0,(index.shape[0])):
                    TrainingIndex = int(index[i])
                    performanceMetric=algorithmPerformanceWithIndex[np.where(algorithmPerformanceWithIndex[:,8] == TrainingIndex)]
                    performanceMetric=np.delete(performanceMetric, 8, axis=1)
                    TestPointRankedPerformance.append(performanceMetric)


                TestPointRankedPerformance = np.asarray(TestPointRankedPerformance)

                rankedPerformanceForClosestTensors=np.argsort(TestPointRankedPerformance.argsort())

                uniqueRankLabels = np.unique(rankedPerformanceForClosestTensors)
                loopCount=0
                for label in uniqueRankLabels:
                    loopCount=loopCount+1
                    labelCount=[]
                    labelMean=[]

                    # transpose to get a column
                    for column in rankedPerformanceForClosestTensors.T:
                        labelCount.append(np.count_nonzero(column==label))
                        # divide by k for knn
                        labelMean.append((np.sum(column)/neighbours))
                    maxCountNum=np.amax(labelCount)
                    MinNumber=np.amin(labelMean)

                    # mean list calculation
                    if label==0:
                        if labelMean.count(MinNumber) > 1:
                            row=[]
                            indexesOfAlgos=np.where(labelMean==MinNumber)
                            indexesOfAlgos=np.asarray(indexesOfAlgos)
                            row.append(TestIndex)
                            row.append(labelMean.count(MinNumber))
                            for y in range(0,labelMean.count(MinNumber)):
                                row.append(int(indexesOfAlgos[:,y]))
                            for x in range(0,8-labelMean.count(MinNumber)):
                                row.append(6)

                            MeanBestPerformingAlgo.append(row)

                        else:
                            # the 6 is used to signify an empty point.
                            MeanBestPerformingAlgo.append([TestIndex,0,labelMean.index(MinNumber),6,6,6,6,6,6,6])

                    if labelCount.count(maxCountNum) > 1:
                        if label==7:
                            #this means two algorithms performed the exact same on the test data record the first one on index
                            BestPerformingAlgo.append([TestIndex,loopCount,maxCountNum,labelCount.index(maxCountNum)])
                        else:
                            continue
                    else:
                        #record best performing algorithm and by how much
                        BestPerformingAlgo.append([TestIndex,loopCount,maxCountNum,labelCount.index(maxCountNum)])
                        break
                TestIndex=TestIndex+1

            BestPerformingAlgo=np.asarray(BestPerformingAlgo)
            MeanBestPerformingAlgo=np.asarray(MeanBestPerformingAlgo)
            print(len(BestPerformingAlgo))
            print(len(MeanBestPerformingAlgo))
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataEstimatedBestPerformingAlgo.csv", BestPerformingAlgo, delimiter=",")
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataEstimatedBestPerformingAlgoMeanCal.csv", MeanBestPerformingAlgo, delimiter=",")

        def getActualPerformanceRank(self):
            #actualPerformance = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TrainingDataIndexesForTestData.csv", delimiter=",")
            labelsTestData = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TestDataTargetColumn.csv",delimiter=",")
            SGDRegressionPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictionsTestData.csv",delimiter=",")
            CatBoostPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictionsTestData.csv",delimiter=",")
            MLPRegressorPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictionsTestData.csv",delimiter=",")
            RandomForestRegressorPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictionsTestData.csv",delimiter=",")
            lassoCVRegressionPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictionsTestData.csv",delimiter=",")
            adaBoostRegressor = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictionsTestData.csv",delimiter=",")
            RANSACRegressor = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictionsTestData.csv",delimiter=",")
            gradientBoostingRegressor = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictionsTestData.csv",delimiter=",")

            predictionArray=[lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressor,RANSACRegressor,gradientBoostingRegressor]

            # unnormalize the data
            for index in range(0,len(predictionArray)):
                predictionArray[index]=(predictionArray[index]*(30.99-5.31))+5.31

            labelsTestData=(labelsTestData*(30.99-5.31))+5.31


            result = np.vstack((predictionArray[0], predictionArray[1]))
            for i in range(2,(len(predictionArray))):
                result = np.vstack((result, predictionArray[i]))
            print(result[:5,:])
            result=result.T
            print(result[:5,:])
            for i in range(0,(len(predictionArray))):
                result[:,i]=abs(np.subtract(result[:,i],labelsTestData))

            print(result.shape)
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/errorOnPerInstanceBasis.csv", result, delimiter=",")

            TestIndex=0
            BestPerformingAlgo=[]

            rankedPerformanceForActualTestData = np.argsort(result.argsort())
            print(rankedPerformanceForActualTestData)

            uniqueRankLabels=np.unique(rankedPerformanceForActualTestData)
            for row in rankedPerformanceForActualTestData:
                index=np.where(row==0)
                index=index[0]
                BestPerformingAlgo.append([TestIndex,index])
                TestIndex=TestIndex+1

            BestPerformingAlgo=np.asarray(BestPerformingAlgo)
            print(len(BestPerformingAlgo))
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataActualBestPerformingAlgo.csv", BestPerformingAlgo, delimiter=",")


        def getPerInstanceScore(self):
            errorOnPerInstanceBasis = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/errorOnPerInstanceBasis.csv", delimiter=",")
            TestDataActualBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataActualBestPerformingAlgo.csv", delimiter=",")
            TestDataEstimatedBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataEstimatedBestPerformingAlgo.csv", delimiter=",")

            for i in range(0,errorOnPerInstanceBasis.shape[1]):
                SummedValue=np.sum(errorOnPerInstanceBasis[:,i])
                meanValue=SummedValue/len(errorOnPerInstanceBasis[:,i])
                print("ALGORITHM "+self.AlgorithmNames[i]+" MAE ERROR IS: "+str(meanValue))

            for i in range(0,errorOnPerInstanceBasis.shape[1]):
                SquaredMeanValue = errorOnPerInstanceBasis[:,i] ** 2
                SummedValue=np.sum(errorOnPerInstanceBasis[:,i])
                meanValue=SummedValue/len(errorOnPerInstanceBasis[:,i])
                print("ALGORITHM "+self.AlgorithmNames[i]+" RSME ERROR IS: "+str(math.sqrt(meanValue)))

            TheoricallyBestPerformer=0
            SiamesePerformance=0
            for i in range(0,len(TestDataEstimatedBestPerformingAlgo)-1):
                SiameseBestPerformingIndex=int(TestDataEstimatedBestPerformingAlgo[i,3])
                ActualBestPerformingIndex=int(TestDataActualBestPerformingAlgo[i,1])

                TheoricallyBestPerformer=TheoricallyBestPerformer+errorOnPerInstanceBasis[i,ActualBestPerformingIndex]
                SiamesePerformance=SiamesePerformance+errorOnPerInstanceBasis[i,SiameseBestPerformingIndex]

                TheoricallyBestPerformerRSME=TheoricallyBestPerformer+(errorOnPerInstanceBasis[i,ActualBestPerformingIndex])**2
                SiamesePerformanceRSME=SiamesePerformance+(errorOnPerInstanceBasis[i,SiameseBestPerformingIndex])**2


            meanValueTher=TheoricallyBestPerformer/len(TestDataEstimatedBestPerformingAlgo)
            meanValueTherRSME = TheoricallyBestPerformerRSME / len(TestDataEstimatedBestPerformingAlgo)

            meanValueSiamese = SiamesePerformance/len(TestDataEstimatedBestPerformingAlgo)
            meanValueSiameseRSME = SiamesePerformance / len(TestDataEstimatedBestPerformingAlgo)

            print("MAE ERROR FOR PERFECT META LEARNER IS: "+str(meanValueTher))
            print("MAE ERROR FOR SIAMESE LEARNED NETWORK IS: "+str(meanValueSiamese))
            print("RSME ERROR FOR PERFECT META LEARNER IS: "+str(math.sqrt(meanValueTherRSME)))
            print("RSME ERROR FOR SIAMESE LEARNED NETWORK IS: "+str(math.sqrt(meanValueSiameseRSME)))



        def getPeformanceScore(self):
            # # compare the data. Check accuracy. check if confidence level affects results

            TestDataActualBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataActualBestPerformingAlgo.csv", delimiter=",")
            TestDataEstimatedBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataEstimatedBestPerformingAlgo.csv", delimiter=",")
            MeanEstimatedBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TestDataEstimatedBestPerformingAlgoMeanCal.csv", delimiter=",")
            plotRankEstimated=[]
            plotRankActual=[]
            right=0
            wrong=0
            meanRight=0
            meanWrong=0
            IntraClassCorrectCount = []
            IntraClassWrongCount = []
            ConfidenceLevel=[]

            for i in range(0,len(TestDataEstimatedBestPerformingAlgo)):
                plotRankEstimated.append(TestDataEstimatedBestPerformingAlgo[i,3])
                plotRankActual.append(TestDataActualBestPerformingAlgo[i,1])
                if TestDataEstimatedBestPerformingAlgo[i,3]==TestDataActualBestPerformingAlgo[i,1]:
                    right=right+1
                    ConfidenceLevel.append(TestDataEstimatedBestPerformingAlgo[i,1])
                    IntraClassCorrectCount.append(self.AlgorithmNames[int(TestDataActualBestPerformingAlgo[i,1])])
                else:
                    wrong=wrong+1
                    IntraClassWrongCount.append(self.AlgorithmNames[int(TestDataActualBestPerformingAlgo[i,1])])

                if MeanEstimatedBestPerformingAlgo[i,1]>0:
                    correct=False
                    for x in range(0,int(MeanEstimatedBestPerformingAlgo[i,1])):
                        if MeanEstimatedBestPerformingAlgo[i,x+2]==TestDataActualBestPerformingAlgo[i,1]:
                            meanRight=meanRight+1
                            correct=True
                            break
                    if correct==False:
                        meanWrong=meanWrong+1
                else:
                    if MeanEstimatedBestPerformingAlgo[i,2]==TestDataActualBestPerformingAlgo[i,1]:
                        meanRight=meanRight+1
                    else:
                        meanWrong=meanWrong+1




            IntraClassCorrectCount=np.asarray(IntraClassCorrectCount)
            IntraClassWrongCount = np.asarray(IntraClassWrongCount)
            (unique, counts) = np.unique(IntraClassCorrectCount, return_counts=True)
            (uniqueWrong, countsWrong) = np.unique(IntraClassWrongCount, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            frequenciesWrong = np.asarray((uniqueWrong, countsWrong)).T
            print("Right"+str(right))
            print("wrong"+str(wrong))
            print("Intra Class Accuracy/True Positives")
            print(frequencies)
            print("Intra Class Accuracy/True Positives Wrong predictions")
            print(frequenciesWrong)
            print(len(frequenciesWrong))
            print("True positive and false positive rate")
            for i in range(0,len(frequencies)):
                print("Algorithm: "+ str(frequencies[i,0]))
                print("True positive rate: "+str((int(frequencies[i,1])/plotRankActual.count(self.AlgorithmNames.index(frequencies[i,0])))*100))
                print("False positive rate: "+str((1-(int(frequencies[i,1])/plotRankActual.count(self.AlgorithmNames.index(frequencies[i,0]))))*100))

            print("Accuracy of result is: "+str((right/len(TestDataEstimatedBestPerformingAlgo))*100))
            print("Confidence in accuracy")
            uniqueLevels=np.unique(ConfidenceLevel)
            for level in uniqueLevels:
                count=ConfidenceLevel.count(level)
                print("Confidence level of "+str(int(level))+" time is "+str((count/len(ConfidenceLevel))*100))
            print("RightMean: "+str(meanRight))
            print("wrongMean: "+str(meanWrong))
            print("Mean Accuracy of result is: "+str((meanRight/len(TestDataEstimatedBestPerformingAlgo))*100))

            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/IntraClassValues.csv",frequencies,delimiter=",",fmt='%s')
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/plotRankEstimated.csv", plotRankEstimated, delimiter=",")
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/plotRankActual.csv", plotRankActual, delimiter=",")


        def plotRanks(self,):
            plotRankSiameseNetwork = pd.read_csv("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/plotRankEstimated.csv", header=None)
            plotRankActual =  pd.read_csv("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/plotRankActual.csv", header=None)
            print(plotRankSiameseNetwork.iloc[:, 0].value_counts())
            values = plotRankSiameseNetwork.iloc[:, 0].value_counts()
            values = values.sort_index()
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/plotRankEstimatedGrouped.csv", values, delimiter=",")
            print("Plotting rank graph ")
            ax = values.plot(kind='bar',title="Plotting rank graph for siamese network test data",)
            ax.set_xlabel("Rank (0 = Best)")
            ax.set_ylabel("Amount")
            plt.show()


            print(plotRankActual.iloc[:, 0].value_counts())
            values = plotRankActual.iloc[:, 0].value_counts()
            values = values.sort_index()
            np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/plotRankActualGrouped.csv", values, delimiter=",")
            print("Plotting rank graph ")
            ax = values.plot(kind='bar',title="Plotting rank graph for actual test data",)
            ax.set_xlabel("Rank of algorithms")
            ax.set_ylabel("Amount")
            plt.show()

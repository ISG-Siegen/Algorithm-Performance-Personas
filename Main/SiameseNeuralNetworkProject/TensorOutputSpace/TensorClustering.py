import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt
from matplotlib import pyplot as plt
import time
import math

class TensorClustering:
        AlgorithmNames = ["SGD Regression","Lasso Regression","CatBoost","MLP Regressor","Random Forest Regression","ADA Boost","RANSC Regression","Gradient Boosting"]

        def ClusterTensorsWithLabels(self,neighbours=12):
            pairsLeftTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputLeftSide.csv', delimiter=',')
            pairsRightTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputRightSide.csv', delimiter=',')
            TestDataTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/OverallTestDataTensors.csv', delimiter=',')
            TensorLeftSideIndex = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/pairsLeftIndex.csv', delimiter=',')
            TensorRightSideIndex = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/pairsRightIndex.csv', delimiter=',')
            TensorTrainingData = np.concatenate((pairsLeftTensors, pairsRightTensors), axis=0)
            TensorTrainingDataIndexes = np.concatenate((TensorLeftSideIndex, TensorRightSideIndex), axis=0)
            # check the merge happened correctly
            print(len(TensorTrainingData))
            print(TensorTrainingData[10,:])
            print(len(TensorTrainingDataIndexes))
            print(TensorTrainingDataIndexes)

            algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/Utils/InverseErrorPerformanceRawData.csv",header=None)
            #print(algorithmPerformance.head())

            algorithmPerformance["indexes"] = algorithmPerformance.index
            print(algorithmPerformance.head())
            print(algorithmPerformance.tail())
            print("Length of algorithmPerformance is "+str(len(algorithmPerformance)))
            algorithmPerformanceWithIndex=algorithmPerformance.values
            print("Length of algorithmPerformanceWithIndex is"+str(len(algorithmPerformanceWithIndex)))

            LabelsForEmbeddings=[]
            if len(TensorTrainingDataIndexes) == len(TensorTrainingData):
                print("here")
                for index,instance in list(zip(TensorTrainingDataIndexes,TensorTrainingData)):
                    TrainingIndex = int(index)
                    performanceMetric=algorithmPerformanceWithIndex[np.where(algorithmPerformanceWithIndex[:,8] == TrainingIndex)]
                    performanceMetric=np.delete(performanceMetric, 8, axis=1)
                    print(performanceMetric)
                    performanceMetric = np.argsort(np.argsort(performanceMetric))
                    print(np.where(performanceMetric==0))
                    LabelsForEmbeddings.append(performanceMetric)


            else:
                print("Invalid size exiting the system")
                exit()


            knn = KNeighborsClassifier(n_neighbors=neighbours, algorithm='ball_tree',weights="distance",n_jobs=9)
            knn.fit(TensorTrainingData,LabelsForEmbeddings)
            TestDataLabels=knn.predict(TestDataTensors)
            print(TestDataLabels)
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataLabelled.csv", TestDataLabels, delimiter=",")


        #takes 4 hours 30 minuts
        def ClusterTensors(self,neighbours=12):
            pairsLeftTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputLeftSide.csv', delimiter=',')
            pairsRightTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputRightSide.csv', delimiter=',')
            TestDataTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/OverallTestDataTensors.csv', delimiter=',')
            TensorLeftSideIndex = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/pairsLeftIndex.csv', delimiter=',')
            TensorRightSideIndex = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/pairsRightIndex.csv', delimiter=',')
            TensorTrainingData = np.concatenate((pairsLeftTensors, pairsRightTensors), axis=0)
            # check the merge happened correctly
            print(len(TensorTrainingData))
            print(TensorTrainingData[10,:])
            print("here")
            nbrs = NearestNeighbors(n_neighbors=neighbours, algorithm='ball_tree',n_jobs=9).fit(TensorTrainingData)
            distances, indices = nbrs.kneighbors(TestDataTensors)
            print(indices)
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/FinalIndicesOfTrainingTensorsTestData.csv", indices, delimiter=",")

        def ConvertNearestTensorsToTrainingDataIndexes(self):
            indices = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/FinalIndicesOfTrainingTensorsTestData.csv", delimiter=",")
            pairsLeftTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputLeftSide.csv', delimiter=',')
            pairsRightTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputRightSide.csv', delimiter=',')
            TestDataTensors = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/OverallTestDataTensors.csv', delimiter=',')
            TensorLeftSideIndex = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/pairsLeftIndex.csv', delimiter=',')
            TensorRightSideIndex = genfromtxt('./SiameseNeuralNetworkProject/TensorOutputSpace/pairsRightIndex.csv', delimiter=',')
            print("Size of test data is: "+str(len(TestDataTensors)))
            print("Size of indeces data is: "+str(len(indices)))
            print("Size of pairsLeftTensors data is: "+str(len(pairsLeftTensors)))
            print("Size of pairsLeftTensors data is: "+str(len(pairsRightTensors)))
            print("Size of TensorLeftSideIndex data is: "+str(len(TensorLeftSideIndex)))
            print("Size of TensorRightSideIndex data is: "+str(len(TensorRightSideIndex)))
            TrainingIndexArray=[]
            #check that after printing out the length using this number as an index i dont need to minus 1
            # print("Length of left side: "+str(TensorLeftSideIndex.shape))
            # print("Length of right side: "+str(TensorRightSideIndex.shape))
            x = 0
            for index in indices:
                x=x+1
                #if x%1000==0:
                    #print(x)
                TempArray=[]
                # print(index.shape[0])
                # print(index)
                for i in range(0,(index.shape[0])):
                    #print("Index i is: "+str(int(index[i])))
                    TrainingTensorIndex = int(index[i])
                    #checks if right or left side
                    if ((TrainingTensorIndex)-(len(pairsLeftTensors)))>=0:
                        #right side index

                        #1875272-937636=937636
                        TrainingDataIndex = TensorRightSideIndex[(TrainingTensorIndex-(len(pairsLeftTensors)))]
                        TempArray.append(TrainingDataIndex)
                    else:
                        #left side index
                        #print(TensorLeftSideIndex.shape)
                        TrainingDataIndex = TensorLeftSideIndex[TrainingTensorIndex]#TrainingTensorIndex
                        #print("Result index left side is: "+str(TrainingDataIndex))
                        TempArray.append(TrainingDataIndex)
                #print(TempArray)
                TrainingIndexArray.append(TempArray)
                # if x==20:
                #     break

            #print(TrainingIndexArray)

            TrainingIndexArray = np.asarray(TrainingIndexArray)
            print("Length of array is "+str(len(TrainingIndexArray)))
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TrainingDataIndexesForTestData.csv", TrainingIndexArray, delimiter=",")




        def ConvertTestDataToRankPredictions(self,neighbours):
            TrainingIndexArray = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TrainingDataIndexesForTestData.csv", delimiter=",")
            # #next step: I have training data indexes, for each test data point. find performance of training data. Get algorithm rank.use a most often vote. record answer
            # # record algorithm choice and how many times each was voted

            # interesting idea: test on the converted performance metrics
            #algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetric.csv")
            #algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetric.csv")


            algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/Utils/InverseErrorPerformanceRawData.csv",header=None)
            print(algorithmPerformance.head())

            # print(len(TraingingData))
            # print(len(algorithmPerformance))
            algorithmPerformance["indexes"] = algorithmPerformance.index
            print(algorithmPerformance.head())
            print(algorithmPerformance.tail())
            print("Length of testIndexResults is "+str(len(TrainingIndexArray)))
            print("Length of algorithmPerformance is "+str(len(algorithmPerformance)))
            algorithmPerformanceWithIndex=algorithmPerformance.values
            print("Length of algorithmPerformanceWithIndex is"+str(len(algorithmPerformanceWithIndex)))
            #algorithmPerformanceWithIndex = pd.concat([algorithmPerformance, TraingingData["indexes"]], axis=1)
            #TrainingIndexArray holds indexes for each performance value
            TestIndex=0
            time1=0
            time2=0
            time3=0
            time4=0
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
                    #print("Index i is: "+str(int(index[i])))
                    TrainingIndex = int(index[i])
                    performanceMetric=algorithmPerformanceWithIndex[np.where(algorithmPerformanceWithIndex[:,8] == TrainingIndex)]
                    performanceMetric=np.delete(performanceMetric, 8, axis=1)
                    TestPointRankedPerformance.append(performanceMetric)#1-
                    #print(performanceMetric)

                #if TestIndex%1000==0:
                    #print(TestIndex)
                    #print(len(BestPerformingAlgo))
                    #print(len(MeanBestPerformingAlgo))
                    # print(MeanBestPerformingAlgo)
                    #print(BestPerformingAlgo)
                    #exit()

                TestPointRankedPerformance = np.asarray(TestPointRankedPerformance)

                #start = time.time()
                #print(TestPointRankedPerformance)
                rankedPerformanceForClosestTensors=np.argsort(TestPointRankedPerformance.argsort())

                uniqueRankLabels = np.unique(rankedPerformanceForClosestTensors)
                loopCount=0
                for label in uniqueRankLabels:
                    loopCount=loopCount+1
                    # if TestIndex==3000:
                    #     exit()
                    labelCount=[]
                    labelMean=[]
                    # transpose to get a column
                    for column in rankedPerformanceForClosestTensors.T:
                        labelCount.append(np.count_nonzero(column==label))
                        # divide by 6 for knn
                        labelMean.append((np.sum(column)/neighbours))
                    maxCountNum=np.amax(labelCount)
                    MinNumber=np.amin(labelMean)

                    # mean list calculation
                    if label==0:
                        if labelMean.count(MinNumber) > 1:
                            row=[]
                            indexesOfAlgos=np.where(labelMean==MinNumber)
                            indexesOfAlgos=np.asarray(indexesOfAlgos)
                            #print("indexes of algos: "+str(indexesOfAlgos))
                            row.append(TestIndex)
                            row.append(labelMean.count(MinNumber))
                            # print(labelMean.count(MinNumber))
                            # for i in range(0,labelMean.count(MinNumber)):
                            #     value=indexesOfAlgos[i]
                            #     row.append(value)
                            # print("min number: "+str(MinNumber))
                            # print("row data: "+str(labelMean))
                            for y in range(0,labelMean.count(MinNumber)):
                                row.append(int(indexesOfAlgos[:,y]))
                            for x in range(0,8-labelMean.count(MinNumber)):
                                row.append(6)
                            #print("value count was: "+str(labelMean.count(MinNumber)))
                            #print(row)
                            MeanBestPerformingAlgo.append(row)
                            #print(len(row))
                            #print(MeanBestPerformingAlgo[-1])
                            #MeanBestPerformingAlgo.append([TestIndex,0,labelMean.index(MinNumber),6,6,6,6])
                        else:
                            MeanBestPerformingAlgo.append([TestIndex,0,labelMean.index(MinNumber),6,6,6,6,6,6,6])
                    #print(maxCountNum)
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
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataEstimatedBestPerformingAlgo.csv", BestPerformingAlgo, delimiter=",")
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataEstimatedBestPerformingAlgoMeanCal.csv", MeanBestPerformingAlgo, delimiter=",")

        def getActualPerformanceRank(self):
            #actualPerformance = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TrainingDataIndexesForTestData.csv", delimiter=",")
            labelsTestData = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/TestDataTargetColumn.csv",delimiter=",")
            SGDRegressionPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/SGDRegressionPredictionsTestData.csv",delimiter=",")
            CatBoostPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictionsTestData.csv",delimiter=",")
            MLPRegressorPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictionsTestData.csv",delimiter=",")
            RandomForestRegressorPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictionsTestData.csv",delimiter=",")
            lassoCVRegressionPredictions = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictionsTestData.csv",delimiter=",")
            adaBoostRegressor = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/AdaBoostRegressorPredictionsTestData.csv",delimiter=",")
            RANSACRegressor = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RANSACRegressorPredictionsTestData.csv",delimiter=",")
            gradientBoostingRegressor = genfromtxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/GradientBoostingRegressorPredictionsTestData.csv",delimiter=",")
            # print("length is: "+str(len(labelsTestData)))
            # print("length is: "+str(len(lassoCVRegressionPredictions)))
            # print("length is: "+str(len(SGDRegressionPredictions)))
            # print("length is: "+str(len(CatBoostPredictions)))
            # print("length is: "+str(len(MLPRegressorPredictions)))
            predictionArray=[lassoCVRegressionPredictions,SGDRegressionPredictions,CatBoostPredictions,MLPRegressorPredictions,RandomForestRegressorPredictions,adaBoostRegressor,RANSACRegressor,gradientBoostingRegressor]


            print("before")
            print(predictionArray[0])

            # unnormalize the data
            for index in range(0,len(predictionArray)):
                predictionArray[index]=(predictionArray[index]*(30.99-5.31))+5.31
                #print(column)

            print("after")
            print(predictionArray[0])
            labelsTestData=(labelsTestData*(30.99-5.31))+5.31



            result = np.vstack((predictionArray[0], predictionArray[1]))
            for i in range(2,(len(predictionArray))):
                result = np.vstack((result, predictionArray[i]))
            print(result[:5,:])
            result=result.T
            print(result[:5,:])
            for i in range(0,(len(predictionArray))):
                result[:,i]=abs(np.subtract(result[:,i],labelsTestData))
            #print(result)
            print(result.shape)
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/errorOnPerInstanceBasis.csv", result, delimiter=",")

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
                # if TestIndex==10:
                #     exit()
                # if TestIndex%10000==0:
                #     print(TestIndex)
            BestPerformingAlgo=np.asarray(BestPerformingAlgo)
            print(len(BestPerformingAlgo))
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataActualBestPerformingAlgo.csv", BestPerformingAlgo, delimiter=",")


        def getPerInstanceScore(self):
            errorOnPerInstanceBasis = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/errorOnPerInstanceBasis.csv", delimiter=",")
            TestDataActualBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataActualBestPerformingAlgo.csv", delimiter=",")
            TestDataEstimatedBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataEstimatedBestPerformingAlgo.csv", delimiter=",")

            # print(len(errorOnPerInstanceBasis))
            # print(len(TestDataActualBestPerformingAlgo))
            # print(len(TestDataEstimatedBestPerformingAlgo))


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


            #SummedValueTher=np.sum(errorOnPerInstanceBasis[:,i])
            meanValueTher=TheoricallyBestPerformer/len(TestDataEstimatedBestPerformingAlgo)
            meanValueTherRSME = TheoricallyBestPerformerRSME / len(TestDataEstimatedBestPerformingAlgo)

            #SummedValueSiamese=np.sum(errorOnPerInstanceBasis[:,i]
            meanValueSiamese = SiamesePerformance/len(TestDataEstimatedBestPerformingAlgo)
            meanValueSiameseRSME = SiamesePerformance / len(TestDataEstimatedBestPerformingAlgo)

            print("MAE ERROR FOR PERFECT META LEARNER IS: "+str(meanValueTher))
            print("MAE ERROR FOR SIAMESE LEARNED NETWORK IS: "+str(meanValueSiamese))
            print("RSME ERROR FOR PERFECT META LEARNER IS: "+str(math.sqrt(meanValueTherRSME)))
            print("RSME ERROR FOR SIAMESE LEARNED NETWORK IS: "+str(math.sqrt(meanValueSiameseRSME)))



        def getPeformanceScore(self):
            # # compare the data. Check accuracy. check if confidence level affects results

            TestDataActualBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataActualBestPerformingAlgo.csv", delimiter=",")
            TestDataEstimatedBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataEstimatedBestPerformingAlgo.csv", delimiter=",")
            MeanEstimatedBestPerformingAlgo = genfromtxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TestDataEstimatedBestPerformingAlgoMeanCal.csv", delimiter=",")
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
                # print(TestDataEstimatedBestPerformingAlgo[i,3])
                # print(TestDataActualBestPerformingAlgo[i,1])
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

            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/IntraClassValues.csv",frequencies,delimiter=",",fmt='%s')
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/plotRankEstimated.csv", plotRankEstimated, delimiter=",")
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/plotRankActual.csv", plotRankActual, delimiter=",")


        def plotRanks(self,):
            plotRankSiameseNetwork = pd.read_csv("./SiameseNeuralNetworkProject/TensorOutputSpace/plotRankEstimated.csv", header=None)
            plotRankActual =  pd.read_csv("./SiameseNeuralNetworkProject/TensorOutputSpace/plotRankActual.csv", header=None)
            print(plotRankSiameseNetwork.iloc[:, 0].value_counts())
            values = plotRankSiameseNetwork.iloc[:, 0].value_counts()
            values = values.sort_index()
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/plotRankEstimatedGrouped.csv", values, delimiter=",")
            print("Plotting rank graph ")
            ax = values.plot(kind='bar',title="Plotting rank graph for siamese network test data",)
            ax.set_xlabel("Rank (0 = Best)")
            ax.set_ylabel("Amount")
            plt.show()


            print(plotRankActual.iloc[:, 0].value_counts())
            values = plotRankActual.iloc[:, 0].value_counts()
            values = values.sort_index()
            np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/plotRankActualGrouped.csv", values, delimiter=",")
            print("Plotting rank graph ")
            ax = values.plot(kind='bar',title="Plotting rank graph for actual test data",)
            ax.set_xlabel("Rank of algorithms")
            ax.set_ylabel("Amount")
            plt.show()

import random
import numpy as np
import pandas as pd

class TrainingPairingSystem:

    def compute_dist(self,a,b):
        return np.sum(np.square(a-b))


    def estimatePositiveANdNegitiveMargin(self):
            TrainingData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
            algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetricSecondApproach.csv")
            distanceArrayPerformance=[]
            distanceArrayFeature=[]
            # .5 confidence interval(margin of error) 95% confidence level
            TrainingData = TrainingData.values
            algorithmPerformance = algorithmPerformance.values

            sample = 1000000

            # create a 200,000 random indexes from above traiing size
            print(str(len(TrainingData)-1))
            print(str(len(algorithmPerformance)-1))

            marginIndexes = random.sample(range(0, len(TrainingData)-1), 1000000)
            print(marginIndexes[:10])
            marginIndexesArray = np.asarray(marginIndexes)
            print(marginIndexesArray.shape)
            TrainingData1 = TrainingData[marginIndexesArray[:500000]]
            algorithmPerformance1 = algorithmPerformance[marginIndexesArray[:500000]]

            TrainingData2 = TrainingData[marginIndexesArray[500000:]]
            algorithmPerformance2 = algorithmPerformance[marginIndexesArray[500000:]]

            print(algorithmPerformance1[:10,:])
            print(TrainingData1.shape)
            print(algorithmPerformance1.shape)
            print(TrainingData2.shape)
            print(algorithmPerformance2.shape)

            averageDistanceFeatureSpace=0.0
            averageDistancePerformanceSpace=0.0
            maxDistanceFeatureSpace=0.0
            maxDistancePerformanceSpace=0.0
            minDistanceFeatureSpace=9999999999.0
            minDistancePerformanceSpace=9999999999.0
            changeAverage=0.0
            changeminimum=9999999999.0
            changeMaximum=0.0

            for index in range(0,len(algorithmPerformance1)-1):

                currentInstanceFeatrue = TrainingData1[index]
                currentInstancePerformance = algorithmPerformance1[index]

                currentInstanceFeatrue1 = TrainingData2[index]
                currentInstancePerformance1 = algorithmPerformance2[index]

                distanceFeature = self.compute_dist(currentInstanceFeatrue1,currentInstanceFeatrue)
                distancePerformance = self.compute_dist(currentInstancePerformance1,currentInstancePerformance)
                distanceArrayPerformance.append(distancePerformance)
                distanceArrayFeature.append(distanceFeature)

                averageDistanceFeatureSpace += distanceFeature
                averageDistancePerformanceSpace += distancePerformance

                if distanceFeature > maxDistanceFeatureSpace:
                    maxDistanceFeatureSpace = distanceFeature

                if distanceFeature < minDistanceFeatureSpace:
                    minDistanceFeatureSpace = distanceFeature

                if distancePerformance > maxDistancePerformanceSpace:
                    maxDistancePerformanceSpace = distancePerformance

                if distancePerformance < minDistancePerformanceSpace:
                    minDistancePerformanceSpace = distancePerformance

                changeDistance = abs(distanceFeature-distancePerformance)
                changeAverage += changeDistance

                if changeDistance > changeMaximum:
                    changeMaximum = changeDistance

                if changeDistance < changeminimum:
                    changeminimum = changeDistance




            print("Feature space max distance: "+str(maxDistanceFeatureSpace))
            print("Feature space min distance: "+str(minDistanceFeatureSpace))
            print("Feature space average distance: "+str((averageDistanceFeatureSpace/(sample/2))))
            print("median of Feature space : ", np.median(np.asarray(distanceArrayFeature)))

            print("performance space max distance: "+str(maxDistancePerformanceSpace))
            print("performance space min distance: "+str(minDistancePerformanceSpace))
            print("performance space average distance: "+str((averageDistancePerformanceSpace/(sample/2))))
            print("median of performance space : ", np.median(np.asarray(distanceArrayPerformance)))

            print("max change: "+str(changeMaximum))
            print("min change: "+str(changeminimum))
            print("average change: "+str((changeAverage/(sample/2))))

            outputDict = dict({"maxDistanceFeatureSpace": maxDistanceFeatureSpace, "minDistanceFeatureSpace":minDistanceFeatureSpace ,"averageDistanceFeatureSpace": (averageDistanceFeatureSpace/(sample/2)),
            "maxDistancePerformanceSpace": maxDistancePerformanceSpace, "minDistancePerformanceSpace": minDistancePerformanceSpace,"averageDistancePerformanceSpace": (averageDistanceFeatureSpace/(sample/2)),
            "changeMaximum": changeMaximum, "changeminimum":changeminimum,"changeAverage":(changeAverage/(sample/2)) })

            (pd.DataFrame.from_dict(data=outputDict, orient='index').to_csv('./SiameseNeuralNetworkProject/TrainingPairingSystem/margins.csv', header=False))


    def createPairs(self,featureSpaceClose=2.5,featureSpaceFar=4.5,performanceSpaceClose=0.2,performanceSpaceFar=1.8):
        # these two values used to classify instances as close
        # featureSpaceMargin = 0.0
        # performanceSpaceMargin = 0.0

        featureSpaceClose = featureSpaceClose   #2.5 6  result ..64015    .42195
        featureSpaceFar = featureSpaceFar

        performanceSpaceClose = performanceSpaceClose#0.9    .2 1.8 best result .6296 .47345 2.4 4.5
        performanceSpaceFar = performanceSpaceFar#1.5

        # numberOfInstances = 4000000
        # easyPositiveThreshold = int(500000)
        # hardPositiveThreshold = int(500000)
        # easyNegitiveThreshold = int(500000)
        # hardNegitiveThreshold = int(500000)
        numberOfInstances = 4000000
        easyPositiveThreshold = int(500000)
        hardPositiveThreshold = int(500000)
        easyNegitiveThreshold = int(500000)
        hardNegitiveThreshold = int(500000)


        TrainingData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
        algorithmPerformance = pd.read_csv("./SiameseNeuralNetworkProject/PerformanceMetric/FinalMetricSecondApproach.csv")
        TrainingData = TrainingData.values
        algorithmPerformance = algorithmPerformance.values

        print(TrainingData.shape)
        print(algorithmPerformance.shape)

        print("Threshold: "+str(easyPositiveThreshold))

        pairs1=[]
        pairs2=[]
        pairs1Positive=[]
        pairs2Positive=[]
        pairs1IndexPositive=[]
        pairs2IndexPositive=[]
        pairs1IndexTotal=[]
        pairs2IndexTotal=[]
        labels=[]

        hardPositive=0
        easyPositive=0
        hardNegitive=0
        easyNegitive=0

        skip=0
        counter=0
        while((len(pairs1)*2)<numberOfInstances-20):
            counter+=1
            # if counter >= 100000:
            #     break

            index1=random.randrange(0, len(TrainingData)-1, 1)
            index2=random.randrange(0, len(TrainingData)-1, 1)

            if index1==index2:
                continue

            instance1FeatureSpace = TrainingData[index1]
            instance2FeatureSpace = TrainingData[index2]

            instance1PerformanceSpace = algorithmPerformance[index1]
            instance2PerformanceSpace = algorithmPerformance[index2]

            # calculate distance in each space
            distanceFeatureSpace = self.compute_dist(instance1FeatureSpace,instance2FeatureSpace)
            distancePerformanceSpace = self.compute_dist(instance1PerformanceSpace,instance2PerformanceSpace)


            if len(pairs1)%100000==0:
                print("length is: "+str(len(pairs1)))
                skip+=1

            if counter%100000==0:
                print("feature space: "+str(distanceFeatureSpace))
                print("performance space: "+str(distancePerformanceSpace))
                skip+=1

            # hard positive: far in feature space close in performance space
            if (distanceFeatureSpace > featureSpaceFar and distancePerformanceSpace < performanceSpaceClose):
                if(hardPositive <= hardPositiveThreshold):
                    pairs1.append(instance1FeatureSpace)
                    pairs2.append(instance2FeatureSpace)
                    pairs1Positive.append(instance1FeatureSpace)
                    pairs2Positive.append(instance1FeatureSpace)
                    labels.append(0)
                    pairs1IndexPositive.append(index1)
                    pairs2IndexPositive.append(index2)
                    pairs1IndexTotal.append(index1)
                    pairs2IndexTotal.append(index2)
                    hardPositive+=1
                    continue
                else:
                    skip+=1
                    continue

            # easy positive: close in feature space close in performance space
            if (distanceFeatureSpace < featureSpaceClose and distancePerformanceSpace < performanceSpaceClose):
                if(easyPositive <= easyPositiveThreshold):
                    pairs1.append(instance1FeatureSpace)
                    pairs2.append(instance2FeatureSpace)
                    pairs1Positive.append(instance1FeatureSpace)
                    pairs2Positive.append(instance1FeatureSpace)
                    labels.append(0)
                    easyPositive+=1
                    pairs1IndexPositive.append(index1)
                    pairs2IndexPositive.append(index2)
                    pairs1IndexTotal.append(index1)
                    pairs2IndexTotal.append(index2)
                    continue
                else:
                    skip+=1
                    continue

            # hard negitive: close in feature space far in performance space
            if (distanceFeatureSpace < featureSpaceClose and distancePerformanceSpace > performanceSpaceFar):
                if(hardNegitive <= hardNegitiveThreshold):
                    pairs1.append(instance1FeatureSpace)
                    pairs2.append(instance2FeatureSpace)
                    labels.append(1)
                    hardNegitive+=1
                    pairs1IndexTotal.append(index1)
                    pairs2IndexTotal.append(index2)
                    continue
                else:
                    skip+=1
                    continue

            # easy negitive: far in feature space far in performance space
            if (distanceFeatureSpace > featureSpaceFar and distancePerformanceSpace > performanceSpaceFar):
                if(easyNegitive <= easyNegitiveThreshold):
                    pairs1.append(instance1FeatureSpace)
                    pairs2.append(instance2FeatureSpace)
                    labels.append(1)
                    easyNegitive+=1
                    pairs1IndexTotal.append(index1)
                    pairs2IndexTotal.append(index2)
                    continue
                else:
                    skip+=1
                    continue

        pairs1=np.asarray(pairs1)
        pairs2=np.asarray(pairs2)
        pairs1Positive=np.asarray(pairs1Positive)
        pairs2Positive=np.asarray(pairs2Positive)
        labels=np.asarray(labels)
        pairs1IndexPositive=np.asarray(pairs1IndexPositive)
        pairs2IndexPositive=np.asarray(pairs2IndexPositive)
        pairs1IndexTotal=np.asarray(pairs1IndexTotal)
        pairs2IndexTotal=np.asarray(pairs2IndexTotal)
        print("pair shape index 0 final: "+str(pairs1.shape))
        print("pair shape index 1 final"+str(pairs2.shape))
        print("labels shape index 1 final"+str(labels.shape))
        print("Hard positive : "+str(hardPositive))
        print("easy positive: "+str(easyPositive))
        print("Hard negitive: "+str(hardNegitive))
        print("easy negitive: "+str(easyNegitive))


        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsLeftChosenByDistance.csv", pairs1Positive, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsRightChosenByDistance.csv", pairs2Positive, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsLeftChosenByDistance.csv", pairs1, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsRightChosenByDistance.csv", pairs2, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/labelsChosenByDistance.csv", labels, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsLeftIndex.csv", pairs1IndexPositive, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsRightIndex.csv", pairs2IndexPositive, delimiter=",")
        #np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsLeftIndex.csv", pairs1IndexTotal, delimiter=",")
        #np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsRightIndex.csv", pairs2IndexTotal, delimiter=",")




def testIndexMechanism(self):
        TrainingData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")

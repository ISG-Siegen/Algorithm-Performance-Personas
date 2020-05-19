from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from SiameseNeuralNetworkProject.MarginEstimation.MarginEstimation import MarginEstimation
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
#from tensorflow.losses.contrastive.contrastive_loss import contrastive_loss
import random
import time


from os import listdir
from os.path import isfile, join

#pairs=[]

class SiameseNeuralNetwork:

    def compute_dist(self,a,b):
        return np.sum(np.square(a-b))

    def euclidean_distance(self,vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    def eucl_dist_output_shape(self,shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


    def setUpModel(self,input_shape):
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(1, activation='relu')(x)
        return Model(input, x)

    def compute_accuracy(self,y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)

    def R_squared(self,y, y_pred):
        residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
        total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
        r2 = tf.subtract(1.0, tf.div(residual, total))
        return r2


    def accuracy(self,y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


    # sets up network with only positive pairs
    def setUpNetwork(self):
        pairs1=[]
        pairs2=[]
        # pairs1WithNegitive=[]
        # pairs2WithNegitive=[]
        labels=[]
        # labelsWithNegitive=[]
        PairsIndex=[]
        LabelsIndex=[]
        array1 = []
        array2 = []
        fileNames = [f for f in listdir("./SiameseNeuralNetworkProject/SiameseTrainingData/") if isfile(join("./SiameseNeuralNetworkProject/SiameseTrainingData/", f))]
        IndexFileNames = [f for f in listdir("./SiameseNeuralNetworkProject/SiameseTrainingDataIndex/") if isfile(join("./SiameseNeuralNetworkProject/SiameseTrainingDataIndex/", f))]
        # print(fileNames)
        i = 0
        for fileName in fileNames:
            i=i+1
            clusterData = pd.read_csv("./SiameseNeuralNetworkProject/SiameseTrainingData/"+str(fileName))

            print("Cluster is: "+str(i))
            if len(clusterData)%2!=0:
                clusterData.drop(clusterData.tail(1).index,inplace=True)
            if len(clusterData)==0:
                continue
            clusterDataArr=clusterData.values
            clusterDataArray=[]
            # not sure if this line is needed and will hamper resutls (test)
            #np.random.shuffle(clusterDataArray)

            clusterDataArray.append(clusterDataArr[:int((len(clusterDataArr)/2)),:])
            clusterDataArray.append(clusterDataArr[int((len(clusterDataArr)/2)):,:])

            pairs1.append(clusterDataArray[0][:,:])
            pairs2.append(clusterDataArray[1][:,:])
            # end of for Loop

        pairs=[]
        pairs.append(np.concatenate(pairs1,axis=0))
        pairs.append(np.concatenate( pairs2, axis=0 ))
        print(len(pairs))
        print(type(pairs[0]))
        print(clusterDataArray[0].shape)
        print(clusterDataArray[0].shape[1])
        labels.append(pairs[0][:,pairs[0].shape[1]-2])
        labels.append(pairs[1][:,pairs[1].shape[1]-2])
        PairsIndex.append(pairs[0][:,pairs[0].shape[1]-1])
        PairsIndex.append(pairs[1][:,pairs[1].shape[1]-1])

        # delete the target column from the training data
        pairs[0] = np.delete(pairs[0], pairs[0].shape[1]-1, 1)
        pairs[1] = np.delete(pairs[1], pairs[1].shape[1]-1, 1)
        # delete the index column from the training data
        pairs[0] = np.delete(pairs[0], pairs[0].shape[1]-1, 1)
        pairs[1] = np.delete(pairs[1], pairs[1].shape[1]-1, 1)

        print(PairsIndex[0])
        print(PairsIndex[1])


        print("pair shape index 0 final: "+str(pairs[0].shape))
        print("pair shape index 1 final"+str(pairs[1].shape))
        print("label shape index 0 final"+str(labels[0].shape))
        print("label shape index 1 final"+str(labels[1].shape))
        print("index shape index 0 final"+str(PairsIndex[0].shape))
        print("index shape index 1 final"+str(PairsIndex[1].shape))
        print(pairs[0])
        print(pairs[1])

        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsLeft.csv", pairs[0], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/labelsLeft.csv", labels[0], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsRight.csv", pairs[1], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/labelsRight.csv", labels[1], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsLeftIndex.csv", PairsIndex[0], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsRightIndex.csv", PairsIndex[1], delimiter=",")

    def SetUpNetworkWithNegitivePairs(self):
        pairs=[]
        labels=[]
        FileArray=[]
        fileNames = [f for f in listdir("./SiameseNeuralNetworkProject/SiameseTrainingData/") if isfile(join("./SiameseNeuralNetworkProject/SiameseTrainingData/", f))]
        tempPairsLeft=pd.read_csv("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsLeft.csv")
        tempPairsRight=pd.read_csv("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsRight.csv")
        pairs.append(tempPairsLeft.values)
        pairs.append(tempPairsRight.values)
        index=0
        print(pairs[0])
        print(len(pairs[0]))
        print(len(pairs[1]))
        labels.append(np.ones(len(pairs[0])))
        print(len(labels[0]))
        print(pairs[0].shape)
        print(pairs[1].shape)

        result=np.concatenate((pairs[0], pairs[1]), axis=1)
        print(result.shape)
        labels=labels[0].reshape(len(labels[0]),1)
        joinedArray = np.concatenate((result, labels), axis=1)
        print(joinedArray)
        print(joinedArray.shape)

        q=0
        print("starting number of files are: "+str(len(fileNames)))
        for file in fileNames:
            file = pd.read_csv("./SiameseNeuralNetworkProject/SiameseTrainingData/"+str(file))
            file = file.values
            file = file[:,:74]
            if len(file)>=2:
                FileArray.append(file)
                q=q+1
                if q%500==0:
                    print(q)
        print("final number of files are: "+str(len(FileArray)))
        print("here")
        q=0
        time1=0
        time2=0
        time3=0
        time4=0
        NegitiveArray=[]
        for index in range(0,len(joinedArray)-1):
            q=q+1
            if q%500==0:
                print(q)

            index1Negitive=random.randrange(0, len(FileArray)-1, 1)
            index2Negitive=random.randrange(0, len(FileArray)-1, 1)
            if index1Negitive!=index2Negitive:

                if len(FileArray[index1Negitive])>1:
                    indexFile1=random.randrange(0, len(FileArray[index1Negitive]), 1)
                else:
                    print("here index = 1")
                    indexFile1=0
                if len(FileArray[index2Negitive])>1:
                    indexFile2=random.randrange(0, len(FileArray[index2Negitive]), 1)
                else:
                    print("here index = 1")
                    indexFile2=0
                start = time.time()
                newRow=FileArray[int(index1Negitive)][int(indexFile1),:]
                newRow=np.concatenate((newRow,FileArray[int(index2Negitive)][int(indexFile2),:]), axis=0)

                newRow=np.append(newRow,[0,])
                NegitiveArray.append(newRow)

        NegitiveArray=np.asarray(NegitiveArray)
        print(len(joinedArray))
        print(joinedArray.shape)
        print(len(NegitiveArray))
        print(NegitiveArray.shape)
        joinedArray=np.concatenate((joinedArray, NegitiveArray), axis=0)
        np.random.shuffle(joinedArray)
        np.random.shuffle(joinedArray)
        np.random.shuffle(joinedArray)
        print(len(joinedArray))
        print(joinedArray.shape)
        print(len(NegitiveArray))
        print(NegitiveArray.shape)


        leftPairs=joinedArray[:,:74]
        rightPairs=joinedArray[:,74:148]
        labels=joinedArray[:,148:]

        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/LeftSidePostiveAndNegitivePairs.csv", leftPairs, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/RightSidePostiveAndNegitivePairs.csv", rightPairs, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/LabelsPostiveAndNegitivePairs.csv", labels, delimiter=",")


    def SetUpNetworkWithPositivePairsDistanceSelection(self):
        margin=2
        pairs1=[]
        pairs2=[]
        labels=[]
        PairsIndex=[]
        LabelsIndex=[]
        array1 = []
        array2 = []
        fileNames = [f for f in listdir("./SiameseNeuralNetworkProject/SiameseTrainingData/") if isfile(join("./SiameseNeuralNetworkProject/SiameseTrainingData/", f))]
        IndexFileNames = [f for f in listdir("./SiameseNeuralNetworkProject/SiameseTrainingDataIndex/") if isfile(join("./SiameseNeuralNetworkProject/SiameseTrainingDataIndex/", f))]
        # print(fileNames)
        i = 0
        for fileName in fileNames:
            i=i+1
            SkipCount=0
            clusterData = pd.read_csv("./SiameseNeuralNetworkProject/SiameseTrainingData/"+str(fileName))

            print("Cluster is: "+str(i))
            if len(clusterData)%2!=0:
                clusterData.drop(clusterData.tail(1).index,inplace=True)
            if len(clusterData)==0:
                continue
            clusterDataArr=clusterData.values
            PairedClusterDataArrayLeft=[]
            PairedClusterDataArrayRight=[]
            # not sure if this line is needed and will hamper resutls (test)
            #np.random.shuffle(clusterDataArray)

            #continue in this loop while the cluster size has not been met
            while(len(PairedClusterDataArrayLeft)<len(clusterDataArr)/2):#/2
                index1=random.randint(0,len(clusterDataArr)-1)
                index2=random.randint(0,len(clusterDataArr)-1)
                item1 = clusterDataArr[index1,:74]
                item2 = clusterDataArr[index2,:74]
                #checks if items are the same can happen for small clusters alot
                if index1==index2:
                    continue
                distance = self.compute_dist(item1,item2)
                # if distance larger then margin add the points to the list
                if distance > margin:
                    PairedClusterDataArrayLeft.append(clusterDataArr[index1,:])
                    PairedClusterDataArrayRight.append(clusterDataArr[index2,:])
                else:
                    #distance was to small
                    if SkipCount > len(clusterDataArr)+10000:
                        break
                    else:
                        SkipCount+=1
            PairedClusterDataArrayLeft=np.asarray(PairedClusterDataArrayLeft)
            PairedClusterDataArrayRight=np.asarray(PairedClusterDataArrayRight)
            print(PairedClusterDataArrayLeft.shape)
            print(PairedClusterDataArrayRight.shape)
            if len(PairedClusterDataArrayRight)>0:
                pairs1.append(PairedClusterDataArrayLeft[:,:])
                pairs2.append(PairedClusterDataArrayRight[:,:])
            print("pair shape left current: "+str(len(pairs1)))
            print("pair shape right current: "+str(len(pairs2)))
            print("skips: "+str(SkipCount))
            # end of for Loop

        print("pair shape left final: "+str(len(np.concatenate(pairs1,axis=0))))
        print("pair shape right final: "+str(len(np.concatenate(pairs2,axis=0))))
        pairs=[]
        pairs.append(np.concatenate(pairs1,axis=0))
        pairs.append(np.concatenate(pairs2,axis=0))
        print(len(pairs))
        print(type(pairs[0]))
        labels.append(pairs[0][:,pairs[0].shape[1]-2])
        labels.append(pairs[1][:,pairs[1].shape[1]-2])
        PairsIndex.append(pairs[0][:,pairs[0].shape[1]-1])
        PairsIndex.append(pairs[1][:,pairs[1].shape[1]-1])

        # delete the target column from the training data
        pairs[0] = np.delete(pairs[0], pairs[0].shape[1]-1, 1)
        pairs[1] = np.delete(pairs[1], pairs[1].shape[1]-1, 1)
        # delete the index column from the training data
        pairs[0] = np.delete(pairs[0], pairs[0].shape[1]-1, 1)
        pairs[1] = np.delete(pairs[1], pairs[1].shape[1]-1, 1)



        print("pair shape index 0 final: "+str(pairs[0].shape))
        print("pair shape index 1 final"+str(pairs[1].shape))

        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsLeftChosenByDistance.csv", pairs[0], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositiveLabelsLeftChosenByDistance.csv", labels[0], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsRightChosenByDistance.csv", pairs[1], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositiveLabelsRightChosenByDistance.csv", labels[1], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsLeftIndex.csv", PairsIndex[0], delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/pairsRightIndex.csv", PairsIndex[1], delimiter=",")


    def SetUpNetworkWithNegitivePairsDistanceSelection(self,cushion):
            pairs1=[]
            pairs2=[]
            marginEstimation = MarginEstimation()
            tempPairsLeft = genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsLeftChosenByDistance.csv",delimiter=",")
            tempPairsRight = genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsRightChosenByDistance.csv",delimiter=",")
            clusterMaxDistances = genfromtxt("./SiameseNeuralNetworkProject/TempFolder/MaxPerformanceClusterDistance.csv",delimiter=",")
            clusterCentersCOOrdinates = genfromtxt("./SiameseNeuralNetworkProject/TempFolder/clusterCenters.csv",delimiter=",")
            maxClusterInstanceDistance = genfromtxt("./SiameseNeuralNetworkProject/TempFolder/MaxPerformanceClusterDistance.csv",delimiter=",")
            # pairs.append(tempPairsLeft.values)
            # pairs.append(tempPairsRight.values)

            #labels=np.ones(len(tempPairsRight))
            labels=np.zeros(len(tempPairsRight))

            clusterData=[]
            # read in all the clusters data
            fileNames = [f for f in listdir("./SiameseNeuralNetworkProject/SiameseTrainingData/") if isfile(join("./SiameseNeuralNetworkProject/SiameseTrainingData/", f))]
            for fileName in fileNames:
                currentCluster = pd.read_csv("./SiameseNeuralNetworkProject/SiameseTrainingData/"+str(fileName))
                # gets the current data for the cluster this is a positive pair
                currentCluster = currentCluster.values
                #convert to numpy array
                currentCluster = currentCluster[:,:74]
                # add it to the list
                clusterData.append(currentCluster)


            # for each cluster create the same amount of negitive pairs as positive
            for index in range(0,len(clusterCentersCOOrdinates)-1):
                # if index>200:
                #     break
                counter=0
                currentClusterDataArrayLeft=[]
                currentClusterDataArrayRight=[]
                currentClusterCentersCOOrdinate = clusterCentersCOOrdinates[index,:]
                currentMaxClusterInstanceDistance = maxClusterInstanceDistance[index]
                # gets the current data for the cluster this is a positive pair
                CurrentClusterData = clusterData[index]

                closestCenters = marginEstimation.getClusterDistancesToClusterCenter(clusterCentersCOOrdinates,currentClusterCentersCOOrdinate)
                # get 5 closest centers
                closestCentersIndexes = closestCenters[:5]
                skip=0
                # fill up the data with negitive pairs
                while(counter < len(CurrentClusterData)/2):
                    randomPositiveIndex = random.randint(0,len(CurrentClusterData)-1)
                    # pick a random cluster with in the 5 options
                    randomClusterIndex = random.randint(0,len(closestCentersIndexes)-1)
                    closeCluster=closestCentersIndexes[randomClusterIndex]
                    randomNegitiveInstanceIndex = random.randint(0,len(clusterData[int(closeCluster)])-1)



                    # print(randomPositiveIndex)
                    # print(closeCluster)
                    # print(randomNegitiveInstanceIndex)
                    posNegDistance = self.compute_dist(CurrentClusterData[randomPositiveIndex,:],clusterData[int(closeCluster)][randomNegitiveInstanceIndex,:])

                    if posNegDistance < currentMaxClusterInstanceDistance+cushion:
                        currentClusterDataArrayLeft.append(clusterData[int(closeCluster)][randomNegitiveInstanceIndex,:])
                        currentClusterDataArrayRight.append(CurrentClusterData[randomPositiveIndex,:])
                        counter+=1
                    else:
                        skip+=1
                        if skip > len(CurrentClusterData)+40000:
                            break
                        # labels.append([0,])

                # fill the other half up with random negitives
                while(counter < len(CurrentClusterData)):
                        randomPositiveIndex = random.randint(0,len(CurrentClusterData)-1)

                        # pick a random cluster with in the 5 options
                        randomClusterIndex = random.randint(0,len(closestCentersIndexes)-1)
                        closeCluster=closestCentersIndexes[randomClusterIndex]
                        randomNegitiveInstanceIndex = random.randint(0,len(clusterData[int(closeCluster)])-1)

                        # append the random choice to the array
                        currentClusterDataArrayLeft.append(clusterData[int(closeCluster)][randomNegitiveInstanceIndex,:])
                        currentClusterDataArrayRight.append(CurrentClusterData[randomPositiveIndex,:])
                        counter+=1
                        # labels.append([0,])

                currentClusterDataArrayLeft=np.asarray(currentClusterDataArrayLeft)
                currentClusterDataArrayRight=np.asarray(currentClusterDataArrayRight)
                print(currentClusterDataArrayLeft.shape)
                print(currentClusterDataArrayRight.shape)
                pairs1.append(currentClusterDataArrayLeft[:,:])
                pairs2.append(currentClusterDataArrayRight[:,:])
                print("pair shape left current: "+str(len(pairs1)))
                print("pair shape right current"+str(len(pairs2)))
                print("skips: "+str(skip))


            pairs1=np.concatenate(pairs1,axis=0)
            pairs2=np.concatenate(pairs2,axis=0)
            # add negitive labels to label array 0s
            labels=np.concatenate((labels, np.ones(len(pairs1))), axis=0)
            # join the negitive array to the positive array
            print(tempPairsLeft.shape)
            print(pairs1.shape)
            tempPairsLeft = np.concatenate((tempPairsLeft, pairs1), axis=0)
            tempPairsRight = np.concatenate((tempPairsRight, pairs2), axis=0)
            print(tempPairsLeft.shape)

            pairs=[]
            pairs.append(tempPairsLeft)
            pairs.append(tempPairsRight)
            print(len(pairs))
            print(type(pairs[0]))

            print(pairs[0].shape)
            print(pairs[1].shape)
            joinedArray=np.concatenate((pairs[0], pairs[1]), axis=1)
            print(joinedArray.shape)
            print(len(labels))
            joinedArray=np.concatenate((joinedArray, labels.reshape(-1,1)), axis=1)
            np.random.shuffle(joinedArray)
            np.random.shuffle(joinedArray)
            np.random.shuffle(joinedArray)
            print(len(joinedArray))
            print(joinedArray.shape)
            print(len(tempPairsLeft))
            print(tempPairsLeft.shape)

            leftPairs=joinedArray[:,:74]
            rightPairs=joinedArray[:,74:148]
            labels=joinedArray[:,148:]

            print("pair shape index 0 final: "+str(leftPairs.shape))
            print("pair shape index 1 final"+str(rightPairs.shape))
            print("labels"+str(labels.shape))

            np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsLeftChosenByDistance.csv", leftPairs, delimiter=",")
            np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsRightChosenByDistance.csv", rightPairs, delimiter=",")
            np.savetxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/labelsChosenByDistance.csv", labels, delimiter=",")
















    def runNetwork(self):
        OverallTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv")



        # pairsLeft = genfromtxt('./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsLeft.csv', delimiter=',')
        # pairsRight = genfromtxt('./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsRight.csv', delimiter=',')

        # LeftPairsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/LeftSidePostiveAndNegitivePairs.csv", delimiter=",")
        # RightPairsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/RightSidePostiveAndNegitivePairs.csv", delimiter=",")
        # LabelsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/LabelsPostiveAndNegitivePairs.csv", delimiter=",")


        pairsLeft = genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsLeftChosenByDistance.csv",delimiter=",")
        pairsRight = genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsRightChosenByDistance.csv",delimiter=",")

        LeftPairsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsLeftChosenByDistance.csv", delimiter=",")
        RightPairsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsRightChosenByDistance.csv", delimiter=",")
        LabelsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/labelsChosenByDistance.csv", delimiter=",")

        # testPairsRight = genfromtxt('./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/testPairsRight.csv', delimiter=',')
        # testLabelsRight = genfromtxt('./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/testLabelsRight.csv', delimiter=',')
        pairs=[]
        testPairs=[]
        labels=[]
        testLabels=[]
        CleanPairs=[]
        # pairs.append(pairsLeft)
        # pairs.append(pairsRight)
        # labels.append(labelsLeft)
        # labels.append(labelsRight)

        CleanPairs.append(pairsLeft)
        CleanPairs.append(pairsRight)

        print(len(RightPairsPosAndNeg))
        pairs.append(LeftPairsPosAndNeg[:(len(RightPairsPosAndNeg)-20000),:])
        pairs.append(RightPairsPosAndNeg[:(len(RightPairsPosAndNeg)-20000),:])
        labels.append(LabelsPosAndNeg[:(len(RightPairsPosAndNeg)-20000)])

        testPairs.append(LeftPairsPosAndNeg[(len(RightPairsPosAndNeg)-20000):,:])
        testPairs.append(RightPairsPosAndNeg[(len(RightPairsPosAndNeg)-20000):,:])
        testLabels.append(LabelsPosAndNeg[(len(RightPairsPosAndNeg)-20000):])


        # pairs.append(LeftPairsPosAndNeg[:(len(RightPairsPosAndNeg)-800000),:])
        # pairs.append(RightPairsPosAndNeg[:(len(RightPairsPosAndNeg)-800000),:])
        # labels.append(LabelsPosAndNeg[:(len(RightPairsPosAndNeg)-800000)])
        #
        # testPairs.append(LeftPairsPosAndNeg[(len(RightPairsPosAndNeg)-800000):900000,:])
        # testPairs.append(RightPairsPosAndNeg[(len(RightPairsPosAndNeg)-800000):900000,:])
        # testLabels.append(LabelsPosAndNeg[(len(RightPairsPosAndNeg)-800000):900000])


        print("pair shape index 0 final: "+str(pairs[0].shape))
        print("pair shape index 1 final"+str(pairs[1].shape))
        print("label shape index 0 final"+str(labels[0].shape))
        print("test pair shape index 0 final: "+str(testPairs[0].shape))
        print("test pair shape index 1 final"+str(testPairs[1].shape))
        print("test label shape index 0 final"+str(testLabels[0].shape))

        #print("label shape index 1 final"+str(labels[1].shape))
        # print("test pair shape index 0 final: "+str(testPairs[0].shape))
        # print("test pair shape index 1 final"+str(testPairs[1].shape))
        # print("test label shape index 0 final"+str(testLabels[0].shape))
        # print("test label shape index 1 final"+str(testLabels[1].shape))
        # # remove labels
        y = labels
        #create shape


        input_shape = pairs[0].shape[1:]

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `model`,
        # the weights of the network
        # will be shared across the two branches
        # model = keras.Sequential([
        # layers.Dense(64, activation='relu', input_shape=input_shape),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(32, activation='relu')
        # ])
        model = keras.Sequential()
        model.add(Dense(40,activation='relu', input_shape=input_shape, name="FullyConnected1"))#64 activation='relu',
        model.add(Dense(20,activation='relu', name="FullyConnected2"))#40
        model.add(Dense(20,activation='relu', name="FullyConnected3"))#30
        outputLayer = model.add(Dense(8,name="FullyConnected4"))#none,activation='relu' 32


        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = model(input_a)
        processed_b = model(input_b)
        print("Processed B")
        print(processed_b)

        sess = tf.compat.v1.Session()

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        siamese_net  = Model([input_a, input_b],distance)

        # train

        siamese_net.compile(loss=self.contrastive_loss, optimizer=Adam(), metrics=[self.accuracy])#self.contrastive_loss,"mse","mae"
        history = siamese_net.fit([pairs[0], pairs[1]], labels[0],
                      batch_size=128,
                      epochs=801,
                      callbacks=self.get_callbacks(),
                      validation_data=([testPairs[0], testPairs[1]], testLabels[0]),
                      verbose=1)



        # compute final accuracy on training and test sets
        y_pred = siamese_net.predict([pairs[0], pairs[1]])
        tr_acc = self.compute_accuracy(labels[0], y_pred)
        y_pred = siamese_net.predict([testPairs[0], testPairs[1]])
        te_acc = self.compute_accuracy(testLabels[0], y_pred)

        print(" Accuracy on training set:"+str(tr_acc))
        print(" Accuracy on test set:"+str(te_acc))

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv("./SiameseNeuralNetworkProject/SiameseNeuralNetwork/LogFile.csv",index=False)

        # gets every fifth entry for validation accuracy to give a smoother curve
        val_accuracy_dict=dict()
        for i in range(0,len(history.history['val_accuracy'])):
            if i%1==0:
                val_accuracy_dict[i]=history.history['val_accuracy'][i]

        plt.plot(list(val_accuracy_dict.keys()),list(val_accuracy_dict.values()))
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        plt.savefig('./SiameseNeuralNetworkProject/SiameseNeuralNetwork/LearningCurveSiameseNeuralNetwork.png')

        # plt.scatter(labels[0], y_pred)
        # plt.xlabel('True Values')
        # plt.ylabel('Predictions')
        # plt.axis('equal')
        # plt.axis('square')
        # plt.xlim([0,plt.xlim()[1]])
        # plt.ylim([0,plt.ylim()[1]])
        # plt.plot([-100, 100], [-100, 100])
        # plt.show()

        # feed_dict_test  = {outputLayer: testPairs[0]}
        # #with tf.Session() as sess:
        # FC1 = tf.get_default_graph().get_tensor_by_name('sequential_1/FullyConnected1/Relu:0')
        # FC2 = tf.get_default_graph().get_tensor_by_name('sequential_1/FullyConnected2/Relu:0')
        # FC3 = tf.get_default_graph().get_tensor_by_name('sequential_1/FullyConnected3/Relu:0')
        # FC4 = tf.get_default_graph().get_tensor_by_name('sequential_1/FullyConnected4/Relu:0')
        # FC1_values = sess.run(FC1,feed_dict={input_a: pairs[0]})#, feed_dict={x: pairs[0]}#feed_dict={FC1: pairs[0]}
        # FC2_values = sess.run(FC2,feed_dict={FC2: FC1_values})
        # FC3_values = sess.run(FC3,feed_dict={FC3: FC2_values})
        # FC4_values = sess.run(FC4,feed_dict={FC4: FC3_values})
        # print(FC4_values)


        get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])

        layer_output_Pairs_Left = get_3rd_layer_output([CleanPairs[0]])[0]
        layer_output_Pairs_Right = get_3rd_layer_output([CleanPairs[1]])[0]

        # layer_output_testPairs = get_3rd_layer_output([testPairs[0]])[0]

        # layer_output_Pairs_again = get_3rd_layer_output([pairs[0]])[0]

        overallTestDataTensors = get_3rd_layer_output([OverallTestData])[0]




        # print(layer_output_Pairs)
        # print(layer_output_Pairs.shape)
        # print(layer_output_testPairs)
        # print(layer_output_testPairs.shape)
        # print(layer_output_Pairs_again)

        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputLeftSide.csv", layer_output_Pairs_Left, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/TensorOutputRightSide.csv", layer_output_Pairs_Right, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/TensorOutputSpace/OverallTestDataTensors.csv", overallTestDataTensors, delimiter=",")
        # layer_output_PairsDF = pd.DataFrame(layer_output_Pairs)
        # layer_output_PairsDF.to_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/Tensor/PerformanceData.csv", index=False)
        # error = y_pred - testLabels[0]
        # plt.hist(error, bins = 25)
        # plt.xlabel("Prediction Error")
        # plt.ylabel("Count")
        # plt.show()

    def get_callbacks(self):
      return [
        tfdocs.modeling.EpochDots(),
      ]

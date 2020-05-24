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
from os import listdir
from os.path import isfile, join





class SiameseNeuralNetwork:

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



    def runNetwork(self):

        OverallTestData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv")

        pairsLeft = genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsLeftChosenByDistance.csv",delimiter=",")
        pairsRight = genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/PositivePairsRightChosenByDistance.csv",delimiter=",")

        LeftPairsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsLeftChosenByDistance.csv", delimiter=",")
        RightPairsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/pairsRightChosenByDistance.csv", delimiter=",")
        LabelsPosAndNeg=genfromtxt("./SiameseNeuralNetworkProject/SiameseTrainingDataPaired/labelsChosenByDistance.csv", delimiter=",")


        pairs=[]
        testPairs=[]
        labels=[]
        testLabels=[]
        CleanPairs=[]

        CleanPairs.append(pairsLeft)
        CleanPairs.append(pairsRight)

        print(len(RightPairsPosAndNeg))

        # very small batch of pairs taken as a validations et for SNN
        pairs.append(LeftPairsPosAndNeg[:(len(RightPairsPosAndNeg)-20000),:])
        pairs.append(RightPairsPosAndNeg[:(len(RightPairsPosAndNeg)-20000),:])
        labels.append(LabelsPosAndNeg[:(len(RightPairsPosAndNeg)-20000)])

        testPairs.append(LeftPairsPosAndNeg[(len(RightPairsPosAndNeg)-20000):,:])
        testPairs.append(RightPairsPosAndNeg[(len(RightPairsPosAndNeg)-20000):,:])
        testLabels.append(LabelsPosAndNeg[(len(RightPairsPosAndNeg)-20000):])



        print("pair shape index 0 final: "+str(pairs[0].shape))
        print("pair shape index 1 final"+str(pairs[1].shape))
        print("label shape index 0 final"+str(labels[0].shape))
        print("test pair shape index 0 final: "+str(testPairs[0].shape))
        print("test pair shape index 1 final"+str(testPairs[1].shape))
        print("test label shape index 0 final"+str(testLabels[0].shape))


        y = labels


        #create shape
        input_shape = pairs[0].shape[1:]

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)


        model = keras.Sequential()
        model.add(Dense(40,activation='relu', input_shape=input_shape, name="FullyConnected1"))
        model.add(Dense(20,activation='relu', name="FullyConnected2"))
        model.add(Dense(20,activation='relu', name="FullyConnected3"))
        outputLayer = model.add(Dense(8,name="FullyConnected4"))


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
        siamese_net.compile(loss=self.contrastive_loss, optimizer=Adam(), metrics=[self.accuracy])
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

        # currently set to 1 but can be changed to allow for steps of learning curve to be plotted e.g. 5, 10, 20
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


        get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])

        layer_output_Pairs_Left = get_3rd_layer_output([CleanPairs[0]])[0]
        layer_output_Pairs_Right = get_3rd_layer_output([CleanPairs[1]])[0]

        overallTestDataTensors = get_3rd_layer_output([OverallTestData])[0]


        np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TensorOutputLeftSide.csv", layer_output_Pairs_Left, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/TensorOutputRightSide.csv", layer_output_Pairs_Right, delimiter=",")
        np.savetxt("./SiameseNeuralNetworkProject/EmbeddingOutputSpace/OverallTestDataTensors.csv", overallTestDataTensors, delimiter=",")


    def get_callbacks(self):
      return [
        tfdocs.modeling.EpochDots(),
      ]

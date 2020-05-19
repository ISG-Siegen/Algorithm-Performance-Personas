from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import r2_score

# Display progress prints single dot each go over the data
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

class MLPRegressor:
    def getName(self):
        return "MLP Regressor"

    def build_model(self,training_dataset):
        # Sequential model used with two dense layers of 64 nodes
      model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(training_dataset.keys())]),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # output layer
      ])
      optimizer = tf.keras.optimizers.RMSprop(0.0005)
      model.compile(loss='mse',
                    optimizer='adam',
                    metrics=['mae', 'mse'])
      return model

    def printResults(self,test_targets, test_predictions):
        print('Mean Absolute Error:', metrics.mean_absolute_error(test_targets, test_predictions))
        print('Mean Squared Error:', metrics.mean_squared_error(test_targets, test_predictions))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_targets, test_predictions)))


    def plot_history(self,history):
      hist = pd.DataFrame(history.history)
      hist['epoch'] = history.epoch

      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Mean Abs Error')
      plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
      plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Validation Error')
      plt.legend()

      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Mean Square Error')
      plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
      plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Validation Error')
      plt.legend()
      plt.show()

      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('loss')
      plt.plot(hist['epoch'], hist['loss'], label='Train Error')
      plt.plot(hist['epoch'], hist['val_loss'], label = 'Validation Error')
      plt.legend()
      plt.show()


    def run(self,trainingDasaset,plotting):
        dataset = trainingDasaset
        accuracy = 0
        train = dataset.copy()
        y = train['int_rate']
        train = train.drop(columns=['int_rate',])
        #split data 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            train, y, test_size=0.2)
        # number of itterations over the dataset
        EPOCHS = 200
        if plotting == True:
            # Train the model
            model = self.build_model(X_train)

            # The patience parameter will check for loss and stop if loss becomes stagnant
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

            # fit the model
            history = model.fit(X_train, y_train, epochs=EPOCHS,
                                validation_split = 0.2, verbose=1,batch_size=70, callbacks=[early_stop, PrintDot()])

            # print the history of the model
            #self.plot_history(history)

            loss, mae, mse = model.evaluate(X_test, y_test, verbose=1)

            print("Testing set Mean Abs Error:"+str(mae))

            test_predictions = model.predict(X_test).flatten()

            # create a scatter plot of the data
            # plt.scatter(y_test, test_predictions)
            # plt.xlabel('True Values')
            # plt.ylabel('Predictions')
            # plt.axis('equal')
            # plt.axis('square')
            # plt.xlim([0,plt.xlim()[1]])
            # plt.ylim([0,plt.ylim()[1]])
            # plt.plot([-100, 100], [-100, 100])
            #
            # error = test_predictions - y_test
            # plt.hist(error, bins = 25)
            # plt.xlabel("Prediction Error")
            # plt.ylabel("Count")
            # plt.show()

            # show the results of predicted vs actual labels
            print("###################################MLPRegressor#############################")
            self.printResults(y_test, test_predictions)
            accuracy=r2_score(y_test, test_predictions)
            #accuracy = np.sqrt(metrics.mean_squared_error(y_test, test_predictions))
        else:
            # Train the model
            model = self.build_model(train)

            # The patience parameter will check for loss and stop if loss becomes stagnant
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

            # fit the model
            history = model.fit(train, y, epochs=EPOCHS,
                                validation_split = 0.2, verbose=1,batch_size=70, callbacks=[early_stop, PrintDot()])

            # show the results of predicted vs actual labels
            print("###################################MLPRegressor#############################")
            #predict on the test data
            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
            predictions = model.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictions.csv", predictions, delimiter=",")

            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv")
            predictions = model.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/MLPRegressorPredictionsTestData.csv", predictions, delimiter=",")

        return accuracy

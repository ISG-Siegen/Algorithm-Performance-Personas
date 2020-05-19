import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn import metrics

class LassoCVRegression:

    def getName(self):
        return "Lasso Regression"

    def run(self,trainingDasaset,plotting):
        dataset = trainingDasaset
        accuracy = 0
        y = dataset['int_rate']
        X = dataset.drop(columns=['int_rate',])
        if plotting==True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            lassoreg = LassoCV(cv=5, random_state=42)
            lassoreg.fit(X_train,y_train)
            print("###################################LassoRegression#############################")
            accuracy=lassoreg.score(X_test, y_test)
            pred = lassoreg.predict(X_test)
            #accuracy = np.sqrt(metrics.mean_squared_error( y_test,pred))
            print("score:"+str(accuracy))
        else:
            lassoreg = LassoCV(cv=5, random_state=42)
            lassoreg.fit(X,y)
            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
            predictions = lassoreg.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictions.csv", predictions, delimiter=",")

            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv")
            predictions = lassoreg.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/LassoCVRegressionPredictionsTestData.csv", predictions, delimiter=",")

        return accuracy

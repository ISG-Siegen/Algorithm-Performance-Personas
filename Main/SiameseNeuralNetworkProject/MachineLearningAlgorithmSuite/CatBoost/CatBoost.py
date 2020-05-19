import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat
from sklearn.metrics import r2_score
from sklearn import metrics
import math

class CatBoost:

    def getName(self):
        return "CatBoost"

    def gridSearch(self,regressor,X_train, y_train):
        parameters = {'depth'         : sp_randInt(6,10),#
                      'learning_rate' : sp_randFloat(),
                      'iterations'    : sp_randInt(600,900)#
                     }

        randm = RandomizedSearchCV(estimator=regressor, param_distributions = parameters,
                                  cv = 3, n_iter = 4, n_jobs=8)#
        randm.fit(X_train, y_train)

        #Results from Random Search
        print("\n========================================================")
        print(" Results from Random Search ")
        print("========================================================")

        print("\n s:\n", randm.best_estimator_)

        print("\n The best score across ALL searched params:\n", randm.best_score_)

        print("\n The best parameters across ALL searched params:\n",randm.best_params_)

        #new catboost model using best parameters
        regressor = CatBoostRegressor(iterations=randm.best_params_['iterations'],
                                      learning_rate=randm.best_params_['learning_rate'],
                                      depth=randm.best_params_['depth'],
                                      od_type='IncToDec')
        return regressor,randm.best_params_

    def run(self,trainingDasaset,plotting):
        dataset = trainingDasaset
        accuracy = 0
        train = dataset.copy()
        y = train['int_rate']
        train = train.drop(columns=['int_rate',])
        regressor = CatBoostRegressor(od_type='IncToDec')
        #split data 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            train, y, test_size=0.2)

        #grid search for optimal parameters
        print ('Fitting')
        parameters = {'depth'         : sp_randInt(6,10),
                      'learning_rate' : sp_randFloat(),
                      'iterations'    : sp_randInt(600, 1000)
                     }

        regressor = CatBoostRegressor(iterations=643,
                                      learning_rate=0.9600690303599169,
                                      depth=6,
                                      od_type='IncToDec')

        bestParams = None
        #regressor,bestParams = self.gridSearch(regressor,X_train, y_train)

        if plotting == True:
            print ('Fitting Test Data')
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)
            print("###################################CatBoost#############################")
            print('MAE is: {}'.format(mean_absolute_error(np.exp(y_test), np.exp(y_pred))))
            accuracy=r2_score(y_test, y_pred)
            if bestParams != None:
                print(bestParams)
            #accuracy = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        #predict test data
        else:
            regressor.fit(train, y)
            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
            predictions = regressor.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictions.csv", predictions, delimiter=",")

            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv")
            predictions = regressor.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/CatBoostPredictionsTestData.csv", predictions, delimiter=",")

        return accuracy

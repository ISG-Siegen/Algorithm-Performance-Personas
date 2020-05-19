import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV,train_test_split
from sklearn import metrics

class RandomForestRegression:

    def getName(self):
        return "Random Forest Regression"

    def gridSearch(self,X_train, y_train):
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'max_depth': range(3,8),#
                'n_estimators': (80,100,150,200),#
            },
            cv=5, scoring='neg_mean_squared_error', verbose=2,n_jobs=5)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        #Results from Random Search
        print("\n========================================================")
        print(" Results from grid Search ")
        print("========================================================")
        print(best_params)

        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)
        # Perform K-Fold CV
        scores = cross_val_score(rfr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error',verbose=2,n_jobs=5)
        print(scores)
        return rfr

    def run(self,trainingDasaset,plotting):
        #create a dataframe with all training data except the target column
        dataset = trainingDasaset
        accuracy = 0
        X = dataset.drop(columns=['int_rate',])
        #separate target values
        y = dataset['int_rate']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        regr = RandomForestRegressor(max_depth=7,n_estimators=80, random_state=0, verbose = 2, n_jobs = 5)

        #regr = self.gridSearch(X_train, y_train)

        if plotting==True:
            regr.fit(X_train, y_train)
            print("###################################RandomForestRegression#############################")
            accuracy=regr.score(X_test, y_test)
            #pred = regr.predict(X_test)
            #accuracy = np.sqrt(metrics.mean_squared_error(y_test,pred))
            print("score:"+str(accuracy))

        else:
            regr.fit(X, y)
            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/SiameseTrainingData.csv")
            predictions = regr.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictions.csv", predictions, delimiter=",")

            testData = pd.read_csv("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/CleanedData/OverallTestingData.csv")
            predictions = regr.predict(testData)
            np.savetxt("./SiameseNeuralNetworkProject/MachineLearningAlgorithmSuite/OutputFiles/RandomForestRegressorPredictionsTestData.csv", predictions, delimiter=",")

        return accuracy

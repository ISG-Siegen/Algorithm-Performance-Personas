from sklearn import preprocessing
import pandas as pd
import numpy as np

class LoanDatasetCleaning:

    def dropColumns(self,dataframe,columnNamesList):
        if type(columnNamesList)=="str":
            dataframe.drop(columns = [columnNamesList])
        else:
            listOfColumnNames=""
            for columnName in columnNamesList:
                listOfColumnNames=listOfColumnNames+columnName+","
            dataframe.drop(columnNamesList, axis=1, inplace=True)
            return dataframe

    def dropColumnsAboveMissingPercentage(self,dataframe,columnNamesList,percentageOfMissingData):
        for column in columnNamesList:
            if dataframe[column].count()<int((len(dataframe.index)*(1-percentageOfMissingData))):
                dataframe=dataframe.drop(columns = [column])
        return dataframe

    def dropMissingDataRows(self,dataframe):
        dataframe=dataframe.dropna()
        return dataframe

    def replaceMissingValues(self,dataframe,columnNameList,searchForList):
        for column in columnNameList:
            for item in searchForList:
                dataframe[column] = dataframe[column].replace(item, np.nan)
        return dataframe

    def backFillColumns(self,dataframe,columnNamesList):
        for columnName in columnNamesList:
            dataframe[columnName] = dataframe[columnName].fillna(method='ffill')
        return dataframe

    def meanFillColumns(self,dataframe,columnNamesList):
        for column in columnNamesList:
            columnMean=dataframe[column].mean()
            dataframe[column] = dataframe[column].replace(np.nan,columnMean)
        return dataframe

    def mostOftenFillColumns(self,dataframe,columnNamesList):
        for column in columnNamesList:
            columnMO=dataframe[column].mode()
            if len(columnMO)>1:
                columnMONum=int(columnMO.get(1))
            else:
                columnMONum=int(columnMO.get(0))
            dataframe[column] = dataframe[column].replace(np.nan,columnMONum)
        return dataframe

    def zeroFillColumns(self,dataframe,columnNamesList):
        for column in columnNamesList:
            dataframe[column] = dataframe[column].replace(np.nan,0)
        return dataframe

    def labelEncoder(self,dataframe,ColumnHeaders,columnNamesList):
        for column in columnNamesList:
            le = preprocessing.LabelEncoder()
            le.fit(dataframe[column].unique())
            dataframe[column] = le.transform(dataframe[column])
        return dataframe

    def oneHotEncoder(self,dataframe,columnNamesList):
        for column in columnNamesList:
            one_hot = pd.get_dummies(dataframe[column])
            data = dataframe.drop(column,axis = 'columns')
            dataframe=pd.concat([data,one_hot],axis='columns')
        return dataframe

    def normalizeDataset(self,dataframe):
        normalized_df=(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
        return normalized_df

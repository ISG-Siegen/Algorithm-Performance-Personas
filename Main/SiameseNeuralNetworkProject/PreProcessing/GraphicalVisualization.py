
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt

class GraphicalVisualization:

    '''
        Reads csv file into pandas dataframe
        numberOfRows: 0 for all rows
                      X for a selsct number of rows

    '''
    def read_CSV_file(self,filename,numberOfRows):
        if ".csv" in filename and numberOfRows==0:
            return pd.read_csv(filename)
        elif ".csv" in filename and numberOfRows!=0:
            return pd.read_csv(filename+".csv",nrows=numberOfRows)
        elif numberOfRows==0:
            return pd.read_csv(filename+".csv")
        else:
            return pd.read_csv(filename+".csv",nrows=numberOfRows)


    '''
        Profiles data using pandas profiling library
        numberOfRows: 0 = all rows profiled
                      X = number of rows to be profiled
    '''
    def pandasProfileRows(self,dataframe,numberOfRows):
        if len(dataframe.index) >= numberOfRows and numberOfRows!=0:
            dataframeSample=dataframe.sample(n=9000)
            # Use pandas profiling to view a gui based plot of the preprocessed data
            dataframeSample.profile_report(title="initialData",check_recoded = False).to_file("initialData"+str(numberOfRows)+".html")
        elif numberOfRows==0:
                # profile all rows
                dataframe.profile_report(title="initialData",check_recoded = False).to_file("initialData.html")
        else:
                print("invalid parameters passed in")

    '''
        Profiles data using pandas profiling library
        SampleNumber: 0 = all rows in correlation
                      X = number of rows to be correlation
    '''
    def correlationMatrix(self,dataframe,SampleNumber):
            dataframe.corr()
            plt.show()

    '''
        print the head of the dataframe
        numberOfRows: X = number of rows to be shown
    '''
    def printDataframeHead(self,dataframe,numberOfRows):
            dataframe.head(numberOfRows)

    '''
        plots a scatter plot for a 2 axis graph
        xAxisName: name of the x axis which should be row column name
        yAxisName: name of the y axis which should be row column name
    '''
    def createScatterPlot(self,dataframe,xAxisName,yAxisName):
        # create a figure and axis
        fig, ax = plt.subplots()
        # scatter the x axis against the y axis
        ax.scatter(dataframe[xAxisName], dataframe[yAxisName])
        # set a title and labels
        ax.set_title(str(xAxisName)+" vs "+str(yAxisName)+" scatterplot")
        ax.set_xlabel(xAxisName)
        ax.set_ylabel(yAxisName)
        plt.show()



    '''
        plots a Histogram plot
        colunmn: name of the column to be plotted
        value: what quantity to plot the histogram on
    '''
    def createHistogramPlot(self,dataframe,colunmn):
        # create figure and axis
        fig, ax = plt.subplots()
        # plot histogram
        ax.hist(dataframe[colunmn])
        # set title and labels
        ax.set_title(colunmn+" Histogram")
        ax.set_xlabel(colunmn)
        ax.set_ylabel("Frequeny")
        plt.show()

    '''
        plots a barchart
        colunmn: name of the column to be plotted
    '''
    def createBarChart(self,dataframe,column):
        # create a figure and axis
        fig, ax = plt.subplots()
        # count the occurrence of each class
        data = dataframe[column].value_counts()
        # get x and y data
        points = data.index
        frequency = data.values
        # create bar chart
        ax.bar(points, frequency)
        # set title and labels
        ax.set_title(column+" BarChart")
        ax.set_xlabel('Points')
        ax.set_ylabel('Frequency')
        plt.show()

    '''
        plots a boxPlot
        colunmn: name of the column to be plotted
    '''
    def createBoxPlot(self,dataframe,colunmn):
         boxplot = dataframe.boxplot(column=[colunmn])
         plt.show()

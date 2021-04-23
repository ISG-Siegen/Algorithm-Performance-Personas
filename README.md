
# ‘Algorithm-Performance Personas’ for Siamese Meta-Learning and Automated Algorithm Selection

This repository is the official implementation of [‘Algorithm-Performance Personas’ for Siamese Meta-Learning and Automated Algorithm Selection].


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Training

The Siamese selection model in this paper can be trained in one sitting or in segments. A command line interface was implemented to allow for easier training of the pipeline.

To train the pipeline in steps:

```train
python .\RunFile.py runSuiteOfAlgorithms
python .\RunFile.py CalculatePeformanceMetricSecondMetric
python .\RunFile.py createPerformancePairs
python .\RunFile.py runNetwork
python .\RunFile.py calculateKNN
python .\RunFile.py getTrainingTensorIndexes
```

To train the entire pipeline in one command with default parameters (WARNING: Current pipeline takes several hours from start to finish):

```train
python .\RunFile.py runpipeline
```


## Evaluation

To evaluate the model on the Kaggle loan dataset, run:

Please note: The data from the paper has been uploaded to this repository. By running the pipeline again, this data will be overwritten.

The current results the SNN approach with 128 Neighbors can be seen by running the command below.

```eval
python .\RunFile.py performance
```

## Results

Our model achieves the following performance on :

### Siamese Algorithm Selection on Kaggle Loan Dataset

| Model name         | MAE  | Selection Accuracy |
| ------------------ |---------------- | -------------- |
| MLP Regressor   |     0.212         |      34.7%       |
| SNN with 128 Neighbours   |     0.180         |      43.2%       |
| Random Forest Meta Learner   |     0.176         |      50%       |
| Oracle   |     0.088         |      100%       |



### Data

Please Note: Github only allows files sizes of 100mb maximum, to get all datasets used in this experiment please download from here:
https://1drv.ms/u/s!AtfAgPR4VDcEu5Ila5QKzb5SmDSAhg?e=y7ismj

Original Dataset: Kaggle loan dataset filename: loan.csv

cleaned Dataset: Kaggle loan dataset filename: CleanedData/outputCleanedAndNormalized.csv

Algorithm suite Training Data, filename: training.csv

SNN Training Data & SNN Test Data, folder: SiameseTrainingDataPaired/

Final Embeddings: Embeddings/



### Models

Both these models can be created by the code. To create the Siamese Algorithm Selection model, download the dataset and run the pipeline listed above.
To run the RF baseline download the augmented dataset and run the baseline script in the rf baseline folder.

Siamese Algorithm Selection

Random Forest Meta Learner

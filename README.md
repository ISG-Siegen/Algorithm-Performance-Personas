> 📋A template README.md for code accompanying a Machine Learning paper

# ‘Algorithm-Performance Personas’ for Siamese Meta-Learning and Automated Algorithm Selection

This repository is the official implementation of [‘Algorithm-Performance Personas’ for Siamese Meta-Learning and Automated Algorithm Selection](https://arxiv.org/abs/2030.12345).

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

The Siamese selection model in this paper can be trained in one sitting or in segments. A command line interface was implimented to allow for easier training of the pipeline.

To train the pipeline in steps:

```train
python .\RunFile.py runSuiteOfAlgorithms()
python .\RunFile.py CalculatePeformanceMetricSecondMetric()
python .\RunFile.py createPerformancePairs()
python .\RunFile.py runNetwork()
python .\RunFile.py calculateKNN()
python .\RunFile.py getTrainingTensorIndexes()
```

To train the entire pipelie in one command with default parapeters (WARNING: Current pipeline takes several hours from start to finish):

```train
python .\RunFile.py runpipeline
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate the model on the Kaggle loan dataset, run:

Please note: The data from the paper has been uploaded to this repsoitory. By running the pipeline again, this data will be overwritten.

The current results the SNN approach with 128 Neighbours can be seen by running the command below.

```eval
python .\RunFile.py getTrainingTensorIndexes()
```

## Results

Our model achieves the following performance on :

### [Siamese Algorithm Selection on Kaggle Loan Dataset](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | MAE  | Selection Accuracy |
| ------------------ |---------------- | -------------- |
| MLP Regressor   |     0.212         |      34.7%       |
| SNN with 128 Neighbours   |     0.180         |      43.2%       |
| Random Forest Meta Learner   |     0.176         |      50%       |
| Oracle   |     0.088         |      100%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

> 📋Include a link to the augmented dataset D', and, if available, the Excel sheet containing additional calculations and visualizations.

## List of Files

### Data
Original Dataset:

Algorithm suite Training Data:

SNN Training Data:

SNN Test Data:



### Models
Siamese Algorithm Selection

Random Forest Meta Learner


### Publication
Final Publication

Pre-Print


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository.

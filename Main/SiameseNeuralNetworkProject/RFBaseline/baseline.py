import pandas
import pickle
import argparse
import numpy as np
import itertools

from os.path import join, exists, dirname

from sklearn.ensemble import RandomForestRegressor

np.random.seed(1729)

datapath = join('data', 'AugmentedTestLoanDataset.h5')
model_path = join('data', 'models', 'rfmodel.pkl')
embeddings_path = join('data', 'OverallTestDataTensors.csv')

ds = pandas.read_hdf(datapath)

i_features = range(0, 74)
i_label = 74
i_predictions = range(75, 83)
i_errors = range(83, 91)
i_ranks = range(91, 99)

n_algos = 8
algos = ['Lasso', 'SGD', 'CatBoost', 'MLP', 'Random Forest', 'Adaboost',
         'RANSAC', 'Gradient Boosting']


# To train or not to train
model = {}
if exists(model_path):
    print(f'Using model at {model_path}')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

else:
    print(f'Trainig new model, no model at {model_path}')
    params = {'n_estimators': 10, 'random_state':42, 'verbose':True}
    model = RandomForestRegressor(**params)

    train_data = pandas.read_hdf(datapath)

    x = train_data.iloc[:, i_features]
    y = train_data.iloc[:, i_errors]
    model.fit(x, y)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

test_data = pandas.read_hdf(datapath)

features = test_data.iloc[:, i_features].to_numpy()
errors = test_data.iloc[:, i_errors].to_numpy()

# mae of predictions
ypred = model.predict(features)
i_pred = ypred.argmin(axis=1)
pred_selection = np.fromiter((row[i] for row, i in zip(errors, i_pred)), dtype=float)

mae_pred = np.mean(np.abs(pred_selection))

# mae of perfect selection
i_perfect = errors.argmin(axis=1)
perfect_selection = np.fromiter((row[i] for row, i in zip(errors, i_perfect)), dtype=float)

mae_perfect = np.mean(np.abs(perfect_selection))

# Single algorithm
i_best = np.argmin(np.sum(errors, axis=0))
best_selection = errors[:, i_best]

mae_best= np.mean(np.abs(best_selection))

# Ratio of correct selection to amount of instances
best_accuracy = sum(i_best == j for j in i_perfect) / len(i_perfect)
pred_accuracy = sum(i == j for i, j in zip(i_pred, i_perfect)) / len(i_perfect)

# Individual accuracies
# Create a bin for each algorithm, for each prediction, add to that
# algorithm whether it was the correct or wrong choice
bins = [[] for i in range(n_algos)]
for i, j in zip(i_pred, i_perfect):
    is_correct = (i == j)
    bins[i].append(is_correct)

accuracies = [sum(b)/len(b) for b in bins]

# Free upsome memort
del test_data

# Get distances for algo embeddings
embeddings = pandas.read_csv(embeddings_path, names=[f'd{i}' for i in range(32)])
mean_distances = {
    name: None
    for name in algos
}
for algo_idx, name in enumerate(algos):

    print('Sampling for {name}')
    # Get instances where 'algo_idx' is the same as in i_perfect
    indices = [(algo_idx == i) for i in i_perfect]
    distances = embeddings.iloc[indices].to_numpy()

    n_sample = 1000 if len(distances) > 1000 else len(distances)

    random_sampling = np.random.choice(len(distances), n_sample, replace=False)
    subset = distances[random_sampling]

    # Every possible pair of instances
    all_pairs = itertools.product(subset, subset)
    total_distance = sum(np.linalg.norm(a - b) for (a, b) in all_pairs)
    mean_distances[name] = total_distance / float(n_sample**2)

results = {
    'mae_perfect' : mae_perfect,
    'mae_best' : mae_best,
    'mae_pred' : mae_pred,
    'best_accuracy' : best_accuracy,
    'pred_accuracy' : pred_accuracy,
    'individual accuracies' : {
        name : accuracy
        for name, accuracy in zip(algos, accuracies)
    },
    'mean_persona_distance' : mean_distances
}
print(results)

import numpy as np
import sklearn.model_selection
from numpy import genfromtxt
import sklearn.tree
import scipy.io
from scipy import stats
from collections import Counter

from save_csv import results_to_csv

eps = 1e-5  # a small number

def preprocess(data, fill_mode=True, min_freq=4, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            #print(term)
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.

    if fill_mode:
        for i in range(data.shape[-1]):
            
            mode = stats.mode(data[data[:, i] != -1][:, i]).mode[0]
            np.put(data[:, i], np.where(data[:, i] == -1), mode)

            # print("feature {} has mode {}".format(i, mode))
            # mode = stats.mode(data[((data[:, i] < -1 - eps) +
            #                         (data[:, i] > -1 + eps))][:, i]).mode[0]
            # data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def generateKFold(X, y, k=5):
    full = np.concatenate((X,y), axis=1)
    np.random.shuffle(full)
    return np.array_split(full, 5, axis=0)

def split_xy(data):
    return data[:, :-1], data[:, -1:]

def pickKFold(kfold, index):
    val = kfold[index]
    
    toTrain = kfold[:index] + kfold[index+1:]
    train = np.concatenate(toTrain, axis=0)

    train_x, train_y = split_xy(train)

    val_x, val_y = split_xy(val)

    return train_x, train_y, val_x, val_y

def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)

def getDataset(dataset):
    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)

        data = np.delete(data, 6 ,axis=1) ##deleting ticket number
        data[1:, 7] = data[1:, 7].astype('<U1').astype('|S18') ##changing cabin to first letter
        np.put(data[:, 7], np.where(data[:, 7] == b''), stats.mode(data[data[:, 7]!=b''][:, 7]).mode[0]) 

        test_data = np.delete(test_data, 5 ,axis=1) ##deleting ticket number
        test_data[1:, 6] = test_data[1:, 6].astype('<U1').astype('|S18') ##changing cabin to first letter
        np.put(test_data[:, 6], np.where(test_data[:, 6] == b''), stats.mode(test_data[test_data[:, 6]!=b''][:, 6]).mode[0]) 
        #print(data[1:, 7])
        #print(stats.mode(data[data[:, 7]!=b''][:, 7]).mode[0])

        
        
        #print(data[1:, 7])
        #print(data[0, :])

        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0] ##finds indices where label is given
        y = np.array(y[labeled_idx], dtype=np.int) #converts from byte strings into integers

        # #print("\n\nPart (b): preprocessing the titanic dataset")

        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 6, 7])
        X = X[labeled_idx, :]

        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 6, 7])

        features = list(data[0, 1:]) + onehot_features
        # print(features, len(features))
        # print(X.shape, Z.shape)
        assert X.shape[1] == Z.shape[1]
        

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    return X, y, Z, features, class_names

def measure_accuracy(prediction, labels):
    return np.sum(prediction == labels) / len(labels)

def evaluateModel(X, y, split, model, filename=None, Z=None):
    full = np.concatenate((X,y.reshape(-1,1)), axis=1)
    np.random.shuffle(full)
    train = full[:split, :]
    val = full[split:, :]

    train_x, train_y = split_xy(train)
    train_y = train_y.reshape(-1,)

    val_x, val_y = split_xy(val)
    val_y = val_y.reshape(-1)

    model.fit(train_x, train_y)

    prediction_train = model.predict(train_x)
    prediction_val = model.predict(val_x)

    if filename:
        prediction_test = model.predict(Z)
        results_to_csv(prediction_test, filename)
        print("predictions saved")

    return measure_accuracy(prediction_train, train_y), measure_accuracy(prediction_val, val_y)
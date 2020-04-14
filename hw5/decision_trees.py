# You may want to install "gprof2dot"
import io
from train_utils import *

import numpy as np
import sklearn.model_selection
import sklearn.tree
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt

import pydot

eps = 1e-5  # a small number

class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None, m=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

        self.m = m

    @staticmethod
    def calculate_entropy(y):
        base_probabilities = []
        for class_label in np.unique(y):
            count = len(y[np.where(y==class_label)])
            base_probabilities.append(float(count / len(y)))

        H_S = -1* sum([p_c * np.log2(p_c) for p_c in base_probabilities])
        return H_S

    @staticmethod
    def information_gain(X, y, thresh):
        #Base entropy calculations
        H_S = DecisionTree.calculate_entropy(y)
        

        #Split entropy calculations
        S_l_y = y[np.where(X < thresh)]
        S_r_y = y[np.where(X >= thresh)]

        H_Sl = DecisionTree.calculate_entropy(S_l_y)
        H_Sr = DecisionTree.calculate_entropy(S_r_y)

        H_after = ( len(S_l_y)*H_Sl + len(S_r_y)*H_Sr ) / len(y)

        return H_S - H_after

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []

            original_data = X
            if self.m:
                attribute_bag = np.random.choice(list(range(len(self.features))), size=self.m, replace=False)
                X = original_data[:, attribute_bag]
            else:
                attribute_bag = None
                X = original_data

            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.

            # print("original", original_data.shape)
            # print("attr_bag", X.shape)

            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))

            #print("thresh", thresh.shape)
            #print("gainns", gains.shape)

            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
           
            #print(gains)
            #print("split/thresh indx", self.split_idx, thresh_idx)

            self.thresh = thresh[self.split_idx, thresh_idx]

            #print("thresh ", self.thresh)

            if self.m:
                self.split_idx = attribute_bag[self.split_idx]
                #print("new index", self.split_idx)

            X0, y0, X1, y1 = self.split(original_data, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = original_data, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X, verbose=False):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            if (verbose and X.shape[0] != 0):
                print("feature", self.features[self.split_idx], "value", X[0, self.split_idx], ">/<", self.thresh)
            
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0, verbose=verbose)
            yhat[idx1] = self.right.predict(X1, verbose=verbose)
            return yhat


class BaggedTrees:
    def __init__(self, maxdepth=3, n=25, features=None, sample_size=None):
        self.n = n
        self.sample_size = sample_size
        self.decision_trees = [
            DecisionTree(max_depth=maxdepth, feature_labels=features)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        assert self.sample_size <= len(y), "Sample size cannot be greater or equal to input size"

        full = np.concatenate((X,y.reshape(-1,1)), axis=1)

        for tree in self.decision_trees:
            bagged_samples = np.random.choice(list(range(len(full))), size=self.sample_size, replace=True)

            train_data = full[bagged_samples, :]
            train_data_x = train_data[:, :-1]
            train_data_y = train_data[:, -1:]

            tree.fit(train_data_x, train_data_y)

    def predict(self, X):
        predictions = []
        for tree in self.decision_trees:
            predictions.append(tree.predict(X))

        total_pred = np.vstack(predictions)
        mode_prediction = stats.mode(total_pred).mode[0]

        return mode_prediction

class RandomForest(BaggedTrees):
    def __init__(self, maxdepth=7, n=25, features=None, sample_size=None, m=1):
        self.n = n
        self.sample_size = sample_size
        self.decision_trees = [
            DecisionTree(max_depth=maxdepth, feature_labels=features, m=m)
            for i in range(self.n)
        ]

def crossValRF(X, y, features, m, sample_size=500):
    for num_trees in [25, 40, 55, 70, 85, 100]:
    #for depth in [3,4,5,6,7,8,9,10]:
        kfold = generateKFold(X,y.reshape(-1, 1))
        
        #print("using depth size {}".format(depth))
        print("Using Num Tree Size {}".format(num_trees))
        accuracies = []

        for i in range(len(kfold)):
            train_x, train_y, val_x, val_y = pickKFold(kfold, i)

            dt = RandomForest(maxdepth=5, n=num_trees, features=features, m=m, sample_size=sample_size)
            dt.fit(train_x, train_y)

            val_predict = dt.predict(val_x)
            val_y = val_y.reshape(-1,)

            accuracies.append(np.sum(val_predict == val_y) / len(val_y))
            print("KFold {} Val acc: ".format(i), accuracies[-1])

        accuracies = np.array(accuracies)
        print("Average validation accuracy: ", np.mean(accuracies))

        print() 

def crossValBaseDT(X, y, features):
    for depth in [3,4,5,6,7,8,9,10]:
        kfold = generateKFold(X,y.reshape(-1, 1))
        
        print("Using Depth {}".format(depth))
        accuracies = []

        for i in range(len(kfold)):
            train_x, train_y, val_x, val_y = pickKFold(kfold, i)
            dt = DecisionTree(max_depth=depth, feature_labels=features)
            dt.fit(train_x, train_y)

            val_predict = dt.predict(val_x)
            val_y = val_y.reshape(-1,)

            accuracies.append(np.sum(val_predict == val_y) / len(val_y))
            print("KFold {} Val acc: ".format(i), accuracies[-1])

        accuracies = np.array(accuracies)
        print("Average validation accuracy: ", np.mean(accuracies))

        print()   

def crossValBaggedDT(X, y, features, sample_size=500):
    #for num_trees in [25, 40, 55, 70, 85, 100]:
    for depth in [3,4,5,6,7,8,9,10]:
        kfold = generateKFold(X,y.reshape(-1, 1))
        
        print("using depth size {}".format(depth))
        #print("Using Num Tree Size {}".format(num_trees))
        accuracies = []

        for i in range(len(kfold)):
            train_x, train_y, val_x, val_y = pickKFold(kfold, i)

            dt = BaggedTrees(maxdepth=depth, n=25, features=features, sample_size=sample_size)
            dt.fit(train_x, train_y)

            val_predict = dt.predict(val_x)
            val_y = val_y.reshape(-1,)

            accuracies.append(np.sum(val_predict == val_y) / len(val_y))
            print("KFold {} Val acc: ".format(i), accuracies[-1])

        accuracies = np.array(accuracies)
        print("Average validation accuracy: ", np.mean(accuracies))

        print()           
from train_utils import *
from decision_trees import *
import numpy as np

if __name__ == "__main__":

    ##TITANIC DATASET
    np.random.seed(102)

    dataset = "titanic"

    X, y, Z, features, class_names = getDataset(dataset)

    print("Features:", features)
    # print("Train/test size:", X.shape, Z.shape)
    # print()

    """
        Training Base Decision Tree
    """

    #crossValBaseDT(X, y, features)
    # #Depth 6 indicated best Base Decision Tree Classifier (Titanic)
    base_dt = DecisionTree(max_depth=6, feature_labels=features)
    base_train_acc, base_val_acc = evaluateModel(X, y, 700, base_dt)
    print("Base DT Train: {}; Val: {}".format(base_train_acc, base_val_acc))

    # Base DT Train: 0.8357142857142857; Val: 0.782608695652174



    """
        Training shallow decision tree for writeup
    """
    # base_dt = DecisionTree(max_depth=3, feature_labels=features)
    # base_train_acc, base_val_acc = evaluateModel(X, y, 700, base_dt)
    # print() #Debugging stopping point to visualize trained tree. Tried using graphviz and rcviz but neither worked. 

    """
        Training Bagged Decision Tree
    """

    # crossValBaggedDT(X, y, features)
    # #Depth 7 indicated best Bagged DT Classifier (Titanic)
    # number trees, 40
    bagged_dt = BaggedTrees(maxdepth=7, n=40, features=features, sample_size=700)
    bagged_train_acc, bagged_val_acc = evaluateModel(X, y, 700, bagged_dt)
    print("Bagged DT Train: {}; Val: {}".format(bagged_train_acc, bagged_val_acc))

    # #Bagged DT Train: 0.88; Val: 0.7759197324414716


    """
        Training Random Forest
    """

    #crossValRF(X, y, features, 4)
    # #Depth 6, N=80 trees works best
    randforest = RandomForest(maxdepth=6, n=80, features=features, sample_size=700, m=4)
    rand_train_acc, rand_val_acc = evaluateModel(X, y, 700, randforest)
    print("Random Forest Train: {}; Val: {}".format(rand_train_acc, rand_val_acc))

    #Random Forest Train: 0.8171428571428572; Val: 0.7926421404682275

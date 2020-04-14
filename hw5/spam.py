from train_utils import *
from decision_trees import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ##TITANIC DATASET
    np.random.seed(230)

    dataset = "spam"

    X, y, Z, features, class_names = getDataset(dataset)

    """
        Visualizing accuracies vs. depth. 
    """
    # train_accuracies = []
    # valid_accuracies = []
    # depths = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70]

    # for depth in depths:
    #     base_dt = DecisionTree(max_depth=depth, feature_labels=features)
    #     base_train_acc, base_val_acc = evaluateModel(X, y, 4137, base_dt) #, filename="spam_base.csv", Z=Z)
    #     train_accuracies.append(base_train_acc)
    #     valid_accuracies.append(base_val_acc)

    #     print("Depth {} -- Base DT Train: {}; Val: {}".format(depth, base_train_acc, base_val_acc))

    # fig, axes = plt.subplots(1, 1, figsize=(7, 7)) 
    # axes.plot(depths, train_accuracies, '.r-') 
    # axes.plot(depths, valid_accuracies, '.b-') 
    # axes.legend(['Training', 'Validation'], loc='upper right') 
    # axes.set_title("SPAM: Accuracies vs. Depth of Decision Tree") 
    # axes.set_xlabel("Depth of Tree")
    # axes.set_ylabel("Accuracy")
    # plt.ylim([0.7, .9])
    # plt.show()
    # plt.waitforbuttonpress()

    """
        Training Decision Tree
    """

    base_dt = DecisionTree(max_depth=30, feature_labels=features)
    base_train_acc, base_val_acc = evaluateModel(X, y, 4137, base_dt)
    print("SPAM Base DT Train: {}; Val: {}".format(base_train_acc, base_val_acc))
    
    #SPAM Base DT Train: 0.8868745467730239; Val: 0.8106280193236715

    spam_sample = X[y==1, :][0,:].reshape(1, 32)
    ham_sample = X[y==0, :][4, :].reshape(1, 32)

    """
        Predictions for spam sample
    """

    base_dt.predict(spam_sample, verbose=True)
    # feature exclamation value 1.0 >/< 1e-05
    # feature meter value 0.0 >/< 1e-05
    # feature ampersand value 0.0 >/< 1e-05
    # feature money value 0.0 >/< 1e-05
    # feature prescription value 0.0 >/< 1e-05
    # feature volumes value 0.0 >/< 1e-05
    # feature dollar value 0.0 >/< 1e-05
    # feature message value 0.0 >/< 1e-05
    # feature semicolon value 0.0 >/< 1e-05
    # feature pain value 0.0 >/< 1e-05
    # feature path value 0.0 >/< 1e-05
    # feature drug value 0.0 >/< 1e-05
    # feature parenthesis value 0.0 >/< 1e-05
    # feature business value 0.0 >/< 1e-05
    # feature spam value 0.0 >/< 1e-05
    # feature other value 0.0 >/< 1e-05
    # feature sharp value 0.0 >/< 1e-05
    # feature out value 0.0 >/< 1.0000033333333334
    # feature square_bracket value 0.0 >/< 1e-05
    # feature energy value 2.0 >/< 1e-05

    """
        Predictions for ham sample (surprisingly short!)
    """

    base_dt.predict(ham_sample, verbose=True)
    # feature exclamation value 0.0 >/< 1e-05
    # feature meter value 3.0 >/< 1e-05

    """
        Training Bagged Decision Tree
    """

    # #crossValBaggedDT(X, y, features, sample_size=3000)
    bagged_dt = BaggedTrees(maxdepth=9, n=25, features=features, sample_size=3000)
    bagged_train_acc, bagged_val_acc = evaluateModel(X, y, 4137, bagged_dt)
    print("SPAM Bagged DT Train: {}; Val: {}".format(bagged_train_acc, bagged_val_acc))

    #SPAM Bagged DT Train: 0.8334541938602852; Val: 0.8289855072463768

    """
        Training Random Forest
    """

    # #crossValRF(X, y, features, 5)
    randforest = RandomForest(maxdepth=9, n=25, features=features, sample_size=3000, m=6)
    rand_train_acc, rand_val_acc = evaluateModel(X, y, 4137, randforest, filename="spam_rf.csv", Z=Z)
    print("Random Forest Train: {}; Val: {}".format(rand_train_acc, rand_val_acc))

    #Random Forest Train: 0.8274111675126904; Val: 0.8328502415458937

    


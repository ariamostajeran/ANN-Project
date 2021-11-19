import numpy as np
import random
import pickle


def get_trainset():
    f = open("Datasets/new_train_set_features.pkl", "rb")
    train_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=train_set_features2, axis=0)
    train_set_features = train_set_features2[:, features_STDs > 87]

    # changing the range of data between 0 and 1
    train_set_features = np.divide(train_set_features, train_set_features.max())

    # loading training set labels
    f = open("Datasets/new_train_set_labels.pkl", "rb")
    train_set_labels = pickle.load(f)
    f.close()


    train_set = []

    for i in range(len(train_set_features)):
        label = np.array([0, 0, 0, 0, 0, 0, 0])
        label[int(train_set_labels[i])] = 1
        label = label.reshape(7, 1)
        train_set.append((train_set_features[i].reshape(60, 1), label))

    random.shuffle(train_set)

    return train_set

def get_testset():

    # loading test set features
    f = open("Datasets/test_set_features.pkl", "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > 80.1]

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading test set labels
    f = open("Datasets/test_set_labels.pkl", "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # ------------
    test_set = []

    for i in range(len(test_set_features)):
        label = np.array([0, 0, 0, 0, 0, 0, 0])
        label[int(test_set_labels[i])] = 1
        label = label.reshape(7, 1)
        test_set.append((test_set_features[i].reshape(60, 1), label))

    # shuffle
    random.shuffle(test_set)

    # print(len(test_set)) #662
    return test_set

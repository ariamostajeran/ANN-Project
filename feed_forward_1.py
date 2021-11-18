import numpy as np
from ANN_Project_Assets.Loading_Datasets import get_trainset


def sigmoid(x):
   return 1 / (1 + np.exp(-x))

pics_count = 491# 100 * 100

train_set = get_trainset()

feature_count = len(train_set[0][0])

first_layer_neurons = 150
second_layer_neurons = 60
output_neurons = len(train_set[0][1])

W_1 = np.random.randn(first_layer_neurons * feature_count).reshape(first_layer_neurons, feature_count)
b_1 = np.zeros((first_layer_neurons, 1))

W_2 = np.random.randn(second_layer_neurons * first_layer_neurons).reshape(second_layer_neurons, first_layer_neurons)
b_2 = np.zeros((second_layer_neurons, 1))

W_3 = np.random.randn(output_neurons * second_layer_neurons).reshape(output_neurons, second_layer_neurons)
b_3 = np.zeros((output_neurons, 1))


# print(len(test_set[0][0]))
X = [i[0] for i in train_set[:200]]
Y = [i[1] for i in train_set[:200]]
# print(len(Y[0]))
corrects = 0
for i, x in enumerate(X):
    # x = np.array(x)
    # print(x.shape)
    # print(W_1.shape)
    A1 = sigmoid(W_1 @ x + b_1)
    A2 = sigmoid(W_2 @ A1 + b_2)
    out = sigmoid(W_3 @ A2 + b_3)
    answer = list(out).index(max(out))
    # print(answer)
    # print(out)
    if answer == list(Y[i]).index(1):
        corrects += 1
    # print(out)

accuracy = (corrects / 200) * 100
print(accuracy)
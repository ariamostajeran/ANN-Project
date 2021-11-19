import math
import numpy as np
import matplotlib.pyplot as plt
from ANN_Project_Assets.Loading_Datasets import get_trainset, get_testset


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_pr(x):
    s = sigmoid(x)
    return s * (1 - s)


def backpropagation(W_1, W_2, W_3, b_1, b_2, b_3, A1, A2, out, Y, X):
    dout = 2 * (out - Y)

    dout_dw3 = sig_pr(W_3 @ A2 + b_3) @ A2.transpose()

    d_w3 = np.array(dout * dout_dw3)

    d_b3 = dout * sig_pr(W_3 @ A2 + b_3)

    dout_da2 = W_3.transpose() @ sig_pr(W_3 @ A2 + b_3)
    da2_dw2 = sig_pr(W_2 @ A1 + b_2) @ A1.transpose()
    da2 = W_3.transpose() @ (dout * sig_pr(W_3 @ A2 + b_3))

    d_w2 = da2 * da2_dw2
    d_b2 = da2 * sig_pr(W_2 @ A1 + b_2)

    da2_da1 = W_2.transpose() @ sig_pr(W_2 @ A1 + b_2)
    da1_dw1 = sig_pr(W_1 @ X + b_1) @ X.transpose()

    da1 = W_2.transpose() @ (da2 * sig_pr(W_2 @ A1 + b_2))
    d_w1 = da1 * da1_dw1
    d_b1 = da1 * sig_pr(W_1 @ X + b_1)

    return d_w1, d_w2, d_w3, d_b1, d_b2, d_b3


mean = 0
test_set = get_testset()
train_set = get_trainset()

feature_count = len(train_set[0][0])

first_layer_neurons = 100
second_layer_neurons = 50
output_neurons = len(train_set[0][1])

W_1 = np.random.randn(first_layer_neurons * feature_count).reshape(first_layer_neurons, feature_count)
b_1 = np.zeros((first_layer_neurons, 1))

W_2 = np.random.randn(second_layer_neurons * first_layer_neurons).reshape(second_layer_neurons, first_layer_neurons)
b_2 = np.zeros((second_layer_neurons, 1))

W_3 = np.random.randn(output_neurons * second_layer_neurons).reshape(output_neurons, second_layer_neurons)
b_3 = np.zeros((output_neurons, 1))

X = [i[0] for i in train_set]
Y = [i[1] for i in train_set]

Xt = [i[0] for i in test_set]
Yt = [i[1] for i in test_set]

batch_size = 8
epochs = 15
learning_rate = 0.05
total_costs = []
optima = 0.9
test_accs = []
train_accs = []

for epoch in range(epochs):
    batches = []
    print("EPOCHS : ", epoch)

    for x in range(0, len(X), batch_size):
        batches.append(train_set[x:x + batch_size])
    VdW3 = 0
    VdW2 = 0
    VdW1 = 0
    Vdb3 = 0
    Vdb2 = 0
    Vdb1 = 0

    for i, batch in enumerate(batches):
        # print(f"number of batch {i}")
        grad_w1 = np.zeros((first_layer_neurons, feature_count))
        grad_w2 = np.zeros((second_layer_neurons, first_layer_neurons))
        grad_w3 = np.zeros((output_neurons, second_layer_neurons))

        grad_b1 = np.zeros((first_layer_neurons, 1))
        grad_b2 = np.zeros((second_layer_neurons, 1))
        grad_b3 = np.zeros((output_neurons, 1))

        for x, y in batch:
            A1 = sigmoid(W_1 @ x + b_1)
            A2 = sigmoid(W_2 @ A1 + b_2)
            out = sigmoid(W_3 @ A2 + b_3)

            gw1, gw2, gw3, gb1, gb2, gb3 = backpropagation(W_1, W_2, W_3, b_1, b_2, b_3, A1, A2, out
                                                           , y, x)
            grad_w1 += gw1
            grad_w2 += gw2
            grad_w3 += gw3

            grad_b1 += gb1
            grad_b2 += gb2
            grad_b3 += gb3

        VdW3 = VdW3 * optima + (1 - optima) * (grad_w3 / batch_size)
        W_3 = W_3 - (learning_rate * VdW3)
        VdW2 = VdW2 * optima + (1 - optima) * (grad_w2 / batch_size)
        W_2 = W_2 - (learning_rate * VdW2)
        VdW1 = VdW1 * optima + (1 - optima) * (grad_w1 / batch_size)
        W_1 = W_1 - (learning_rate * VdW1)

        Vdb3 = Vdb3 * optima + (1 - optima) * (grad_b3 / batch_size)
        b_3 = b_3 - (learning_rate * Vdb3)
        Vdb2 = Vdb2 * optima + (1 - optima) * (grad_b2 / batch_size)
        b_2 = b_2 - (learning_rate * Vdb2)
        Vdb1 = Vdb1 * optima + (1 - optima) * (grad_b1 / batch_size)
        b_1 = b_1 - (learning_rate * Vdb1)

    cost = 0
    for train_data in train_set:
        a0 = train_data[0]
        a1 = sigmoid(W_1 @ a0 + b_1)
        a2 = sigmoid(W_2 @ a1 + b_2)
        a3 = softmax(W_3 @ a2 + b_3)

        for j in range(output_neurons):
            cost += np.power((a3[j, 0] - train_data[1][j, 0]), 2)

    print(cost)
    cost /= 100
    total_costs.append(cost)
    corrects = 0
    for i, x in enumerate(X):
        # print(x.shape)
        A1 = sigmoid(W_1 @ x + b_1)
        A2 = sigmoid(W_2 @ A1 + b_2)
        out = softmax(W_3 @ A2 + b_3)
        # print("i")
        # print(f"TEST : {Yt[i]}, PREDICT : {out}")
        predicted_number = list(out).index(max(out))
        real_number = list(Y[i]).index(1)

        if predicted_number == real_number:
            corrects += 1
    accuracy = corrects / len(X)
    print("TRAIN Accuracy = ", accuracy)
    train_accs.append(accuracy)
    corrects = 0
    for i, x in enumerate(Xt):
        # print(x.shape)
        A1 = sigmoid(W_1 @ x + b_1)
        A2 = sigmoid(W_2 @ A1 + b_2)
        out = softmax(W_3 @ A2 + b_3)
        # print("i")
        # print(f"TEST : {Yt[i]}, PREDICT : {out}")
        predicted_number = list(out).index(max(out))
        real_number = list(Yt[i]).index(1)

        if predicted_number == real_number:
            corrects += 1
    accuracy = corrects / len(Xt)
    mean += accuracy
    # print("TEST Accuracy = ", accuracy)
    test_accs.append(accuracy)

epoch_size = [x for x in range(epochs)]
plt.plot(epoch_size, train_accs)
plt.plot(epoch_size, test_accs)
# plt.savefig(f'{i}.png')
plt.show()

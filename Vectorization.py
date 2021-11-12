import math
import numpy as np
import matplotlib.pyplot as plt
from ANN_Project_Assets.Loading_Datasets import trainset, testset


def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def sig_pr(x):
    s = sigmoid(x)
    return s * (1 - s)


def backpropagation(W_1, W_2, W_3, b_1, b_2, b_3, A1, A2, out, Y, X):
    dout = 2 * (out - Y)
    # print(A2.shape)
    dout_dw3 = sig_pr(W_3 @ A2 + b_3) @ A2.transpose()
    # print(dout_dw3.shape)
    d_w3 = np.array(dout * dout_dw3)
    # print(d_w3.shape)
    d_b3 = dout * sig_pr(W_3 @ A2 + b_3)

    # print(W_3.transpose().shape)
    dout_da2 = W_3.transpose() @ sig_pr(W_3 @ A2 + b_3)
    da2_dw2 = sig_pr(W_2 @ A1 + b_2) @ A1.transpose()
    da2 = W_3.transpose() @ (dout * sig_pr(W_3 @ A2 + b_3))
    # print(dout_da2.shape)
    # print(da2_dw2.shape)

    d_w2 = da2 * da2_dw2
    d_b2 = da2 * sig_pr(W_2 @ A1 + b_2)

    da2_da1 = W_2.transpose() @ sig_pr(W_2 @ A1 + b_2)
    da1_dw1 = sig_pr(W_1 @ X + b_1) @ X.transpose()
    # print(da2.shape)
    # print(A2.shape)
    # print(W_2.shape)
    da1 = W_2.transpose() @ (da2 * sig_pr(W_2 @ A1 + b_2))
    d_w1 = da1 * da1_dw1
    d_b1 = da1 * sig_pr(W_1 @ X + b_1)

    return d_w1, d_w2, d_w3, d_b1, d_b2, d_b3



pics_count = 491# 100 * 100

W_1 = np.random.randn(150 * 102).reshape(150, 102)
b_1 = np.zeros((150, 1))
# print(W_1)
# print(b_1)
W_2 = np.random.randn(60 * 150).reshape(60, 150)
b_2 = np.zeros((60, 1))

W_3 = np.random.randn(4 * 60).reshape(4, 60)
b_3 = np.zeros((4, 1))

test_set = testset()
train_set = trainset()
# print(len(test_set[0][0]))
X = [i[0] for i in train_set]
Y = [i[1] for i in train_set]

Xt = [i[0] for i in test_set]
Yt = [i[1] for i in test_set]

feature_count = len(X[0])

print(len(X))

batch_size = 10
epochs = 10
learning_rate = 1
total_costs = []

for epoch in range(epochs):
    batches = []
    print("EPOCHS : ", epoch)


    for x in range(0, len(X), batch_size):
        batches.append(train_set[x:x+batch_size])
    for i, batch in enumerate(batches):
        # print(f"number of batch {i}")
        grad_w1 = np.zeros((150, feature_count))
        grad_w2 = np.zeros((60, 150))
        grad_w3 = np.zeros((4, 60))

        grad_b1 = np.zeros((150, 1))
        grad_b2 = np.zeros((60, 1))
        grad_b3 = np.zeros((4, 1))


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

        W_3 = W_3 - (learning_rate * (grad_w3 / batch_size))
        W_2 = W_2 - (learning_rate * (grad_w2 / batch_size))
        W_1 = W_1 - (learning_rate * (grad_w1 / batch_size))

        b_3 = b_3 - (learning_rate * (grad_b3 / batch_size))
        b_2 = b_2 - (learning_rate * (grad_b2 / batch_size))
        b_1 = b_1 - (learning_rate * (grad_b1 / batch_size))

    cost = 0
    for train_data in train_set:
        a0 = train_data[0]
        a1 = sigmoid(W_1 @ a0 + b_1)
        a2 = sigmoid(W_2 @ a1 + b_2)
        a3 = sigmoid(W_3 @ a2 + b_3)

        for j in range(4):
            cost += np.power((a3[j, 0] - train_data[1][j, 0]), 2)

    print(cost)
    cost /= 100
    total_costs.append(cost)


epoch_size = [x for x in range(epochs)]
plt.plot(epoch_size, total_costs)
plt.savefig('20_epoch_Vectorization.png')
plt.show()

corrects = 0



for i, x in enumerate(Xt):
    A1 = sigmoid(W_1 @ x + b_1)
    A2 = sigmoid(W_2 @ A1 + b_2)
    out = sigmoid(W_3 @ A2 + b_3)
    # print("i")
    print(f"TEST : {Yt[i]}, PREDICT : {out}")
    predicted_number = list(out).index(max(out))
    real_number = list(Yt[i]).index(1)

    if predicted_number == real_number:
        corrects += 1
accuracy = corrects / len(Xt)
print("TEST Accuracy = ", accuracy)
corrects = 0

# for test_data in train_set:
#     a0 = test_data[0]
#     a1 = sigmoid(W_1 @ a0 + b_1)
#     a2 = sigmoid(W_2 @ a1 + b_2)
#     a3 = sigmoid(W_3 @ a2 + b_3)
#
#     predicted_number = list(a3).index(max(a3))
#     real_number = list(test_data[1]).index(max(test_data[1]))
#
#     if predicted_number == real_number:
#         corrects += 1
# accuracy = corrects / len(X)
# print("TRAIN Accuracy = ", accuracy)
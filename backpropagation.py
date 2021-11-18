import math
import numpy as np
import matplotlib.pyplot as plt
from ANN_Project_Assets.Loading_Datasets import get_trainset, get_testset


def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def sig_pr(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)



train_set = get_trainset()
test_set = get_testset()
Xt = [i[0] for i in test_set]
Yt = [i[1] for i in test_set]

pics_count = 491
train_set = train_set[:200]
X = [i[0] for i in train_set]
Y = [i[1] for i in train_set]
feature_count = len(X[0])

W_1 = np.random.randn(150 * feature_count).reshape(150, feature_count)
b_1 = np.zeros((150, 1))

W_2 = np.random.randn(60 * 150).reshape(60, 150)
b_2 = np.zeros((60, 1))

W_3 = np.random.randn(4 * 60).reshape(4, 60)
b_3 = np.zeros((4, 1))


corrects = 0
batch_size = 10
epochs = 5
learning_rate = 0.3
total_costs = []

for epoch in range(epochs):
    batches = []
    print("EPOCHS : ", epoch)
    for x in range(0, len(X), batch_size):
        batches.append(train_set[x:x+batch_size])
    for i, batch in enumerate(batches):
        print(f"number of batch {i}")
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

            for j in range(grad_w3.shape[0]):
                for k in range(grad_w3.shape[1]):
                    grad_w3[j, k] += 2 * (out[j, 0] - y[j, 0]) * out[j, 0] * (1 - out[j, 0]) * A2[k, 0]

            for j in range(grad_w3.shape[0]):
                grad_b3[j, 0] += 2 * (out[j, 0] - y[j, 0]) * out[j, 0] * (1 - out[j, 0])

            delta_3 = np.zeros((grad_w3.shape[1], 1))
            for k in range(grad_w3.shape[1]):
                for j in range(grad_w3.shape[0]):
                    delta_3[k, 0] += 2 * (out[j, 0] - y[j, 0]) * out[j, 0] * (1 - out[j, 0]) * W_3[j, k]

            for k in range(grad_w2.shape[0]):
                for m in range(grad_w2.shape[1]):
                    # print(type(delta_3))
                    grad_w2[k, m] += delta_3[k, 0] * A2[k, 0] * (1 - A2[k, 0]) * A1[m, 0]

            for k in range(grad_w2.shape[0]):
                grad_b2[k, 0] += delta_3[k, 0] * A2[k, 0] * (1- A2[k, 0])

            delta_2 = np.zeros((grad_w2.shape[1], 1))
            for m in range(grad_w2.shape[1]):
                for k in range(grad_w2.shape[0]):
                    delta_2[m, 0] += delta_3[k, 0] * A2[k, 0] * (1 - A2[k, 0]) * W_2[k, m]

            for m in range(grad_w1.shape[0]):
                for v in range(grad_w1.shape[1]):
                    grad_w1[m, v] += delta_2[m, 0] * A1[m, 0] * (1 - A1[m, 0]) * x[v, 0]

            for m in range(grad_w1.shape[0]):
                grad_b1[m, 0] += delta_2[m, 0] * A1[m, 0] * (1 - A1[m, 0])

        # print(grad_w3)
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
plt.savefig('5_epoch_backpropagation.png')
plt.show()
corrects = 0

for i, test_data in enumerate(Xt):
    a0 = test_data
    a1 = sigmoid(W_1 @ a0 + b_1)
    a2 = sigmoid(W_2 @ a1 + b_2)
    a3 = sigmoid(W_3 @ a2 + b_3)
    print(f"TEST : {Yt[i]}, PREDICT : {a3}")

    predicted_number = list(a3).index(max(a3))
    real_number = list(Yt[i]).index(1)

    if predicted_number == real_number:
        corrects += 1
accuracy = corrects / len(Xt)
print("Accuracy = ", accuracy)

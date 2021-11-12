import numpy as np
from ANN_Project_Assets.Loading_Datasets import testset

def sigmoid(x):
   return 1 / (1 + np.exp(-x))


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
# print(len(test_set[0][0]))
X = [i[0] for i in test_set[:200]]
Y = [i[1] for i in test_set[:200]]
# print(len(Y[0]))
corrects = 0
for i, x in enumerate(X):
    # x = np.array(x)
    # print(x.shape)
    # print("X : ", x[0])
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
import numpy as np
import operator
'''
    trainData - 训练集  N
    testData - 测试   1
    labels - 训练集标签
'''
# train_data =np.arange(12).reshape(3,4)
# test_data = train_data[1]
# print(train_data)
# print(test_data)
# print(train_data-test_data)
# a =train_data-test_data
# print(a**2)
# a =np.matmul(train_data-test_data,(train_data-test_data).T)
# print(a)
# a =np.dot(test_data,test_data)
# print(a)
def knn(trainData, testData, labels, k):
    # distance = np.sum((trainData-testData)**2)
    distance_matrix = trainData - testData
    D =[]
    for distance in distance_matrix:
        D.append(np.sqrt(np.dot(distance,distance)))
    # distance = np.sqrt((trainData-testData)**2)
    D=np.array(D)
    index =D.argsort()
    count = {}
    for i in range(k):
        vote = labels[index[i]]
        # print(vote)
        count[vote] = count.get(vote, 0) + 1
    # 对类别出现的频数从高到低进行排序
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortCount[0][0]
    # print(index)
    # index =index[:k].tolist()
    # label = labels[index]
    # key = set(label)
    # res = {}
    # for k in key:
    #     res[k]=0
    #
    # for L in label:
    #     res[L] += 1
    # temp = 0
    # for k,v in res.items():
    #     if v>temp:
    #         temp = v
    #         result = k
    # return result


file_data = 'iris.data'

# 数据读取
data = np.loadtxt(file_data, dtype=float, delimiter=',', usecols=(0, 1, 2, 3))
lab = np.loadtxt(file_data, dtype=str, delimiter=',', usecols=(4))

# 分为训练集和测试集和
N = 150
N_train = 100
N_test = 50

perm = np.random.permutation(N)

index_train = perm[:N_train]
index_test = perm[N_train:]

data_train = data[index_train, :]
lab_train = lab[index_train]

data_test = data[index_test, :]
lab_test = lab[index_test]

# 参数设定
k = 5
n_right = 0
for i in range(N_test):
    test = data_test[i, :]

    det = knn(data_train, test, lab_train, k)

    if det == lab_test[i]:
        n_right = n_right + 1

    print('Sample %d  lab_ture = %s  lab_det = %s' % (i, lab_test[i], det))

# 结果分析
print('Accuracy = %.2f %%' % (n_right * 100 / N_test))


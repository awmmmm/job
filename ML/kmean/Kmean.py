import numpy as np
import matplotlib.pyplot as plt
# 构造聚类中心
# dataset [N,D]
# K 聚类中心的数目 [K,D]
def creat_centers(dataset,K):
    val_max = np.max(dataset,axis=0)
    val_min = np.min(dataset, axis=0)
    return np.linspace(val_min,val_max,K+2)[1:-1,:]
# index = np.random.permutation(N)随机重新排列

# keams 绘图
# dataset (N,D)
# lab (N,)
# dic_colors K 种颜色
# centers (K,D)
def draw_kmeans(dataset,lab,centers,dic_colors=None,name="0.jpg"):
    plt.cla()

    vals_lab = set(lab.tolist())

    for i, val in enumerate(vals_lab):
        print(np.where(lab == val))
        index = np.where(lab == val)[0]
        sub_dataset = dataset[index, :]
        plt.scatter(sub_dataset[:, 0], sub_dataset[:, 1], s=16., color=dic_colors[i])

    for i in range(np.shape(centers)[0]):
        plt.scatter(centers[i, 0], centers[i, 1], color="k", marker="+", s=200.)

    plt.savefig(name)
# a =1
# a= float(a)
def run_kmeans(dataset, K, m=20,dic_colors=None,b_draw=False):
    N,D = dataset.shape
    center = creat_centers(dataset,K)
    labs = np.zeros(N)
    if b_draw:
        draw_kmeans(dataset,labs,center,dic_colors, name="int.jpg")
    for it in range(m):
        #根据中心算出标签，并根据标签更新中心
        k_distance = np.empty((N,K))
        for k in range(K):
            distance = ((dataset - center[k,:])**2).sum(axis=1)
            k_distance[:,k] = distance
        error = np.sum(np.min(k_distance, axis=1)) / N
        print("第 %d 次聚类 距离误差 %.2f" % (it, error))
        labs_new = np.argmin(k_distance,axis=1)
        # 绘图
        if b_draw:
            draw_kmeans(dataset, labs_new, center,
                        dic_colors, name=str(it) + "_oldcenter.jpg")
        centers = np.empty((K,D))
        for k in range(K):
            index = np.where(labs_new==k)[0]
            centers[k,:] = np.mean(dataset[index,:],axis=0)
        if b_draw:
            draw_kmeans(dataset, labs_new, centers,
                        dic_colors, name=str(it) + "_newcenter.jpg")

        # 如果聚类结果和上次相同，退出
        if np.sum(labs_new - labs) == 0:
            return labs_new
        else:
            labs = labs_new


    return labs
# def run_kmeans(dataset, K, m=20, dic_colors=None, b_draw=False):
#     N, D = np.shape(dataset)
#     # print(N,D)
#     # 确定初始化聚类中心
#     centers = creat_centers(dataset, K)
#     lab = np.zeros(N)
#     if b_draw:
#         draw_kmeans(dataset, lab, centers, dic_colors, name="int.jpg")
#
#     # 进行m轮迭代
#     labs = np.zeros(N)  # 初始聚类结果
#     for it in range(m):
#         # 计算每个点距离中心的距离
#         distance = np.zeros([N, K])
#         for k in range(K):
#             center = centers[k, :]
#
#             # 计算欧式距离
#             diff = np.tile(center, (N, 1)) - dataset
#             sqrDiff = diff ** 2
#             sqrDiffSum = sqrDiff.sum(axis=1)
#             distance[:, k] = sqrDiffSum
#
#         # 距离排序，进行聚类
#         labs_new = np.argmin(distance, axis=1)
#         error = np.sum(np.min(distance, axis=1)) / N
#         print("第 %d 次聚类 距离误差 %.2f" % (it, error))
#
#         # 绘图
#         if b_draw:
#             draw_kmeans(dataset, labs_new, centers,
#                         dic_colors, name=str(it) + "_oldcenter.jpg")
#
#         # 计算新的聚类中心
#         for k in range(K):
#             index = np.where(labs_new == k)[0]
#             centers[k, :] = np.mean(dataset[index, :], axis=0)
#
#         # 绘图
#         if b_draw:
#             draw_kmeans(dataset, labs_new, centers,
#                         dic_colors, name=str(it) + "_newcenter.jpg")
#
#         # 如果聚类结果和上次相同，退出
#         if np.sum(labs_new - labs) == 0:
#             return labs_new
#         else:
#             labs = labs_new
#
#     return labs


if __name__=="__main__":
    a = np.random.multivariate_normal([2,2], [[.5,0],[0,.5]], 100)
    # print(a)
    b = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], 100)
    dataset = np.r_[a,b]
    lab_ture = np.r_[np.zeros(100),np.ones(100)].astype(int)
    # print(lab_ture)
    # print(dataset.shape)
    # print(lab_ture.shape)
    # np.stack()
    # arrays = [np.random.randn(3, 4) for _ in range(10)]
    # np.stack(arrays, axis=0).shape
    # np.concatenate()
    # dataset =
    # dic_colors={0:(0.,0.5,0.),1:(0.8,0,0)}
    # labs = run_kmeans(dataset,K=2, m = 20,dic_colors=dic_colors,b_draw=True)
    dic_colors = {0: (0., 0.5, 0.), 1: (0.8, 0, 0)}
    labs = run_kmeans(dataset, K=2, m=20, dic_colors=dic_colors, b_draw=True)
    # k = np.zeros((2,3))
    # print(np.asarray(k==0).nonzero())
    # print(np.where(k==0))
    # plt.scatter([5,7],[5,8] , color=(0.5, 1.0, 1.0), marker="+", s=200.)
    # plt.show()
import numpy as np
# 一维高斯分布的概率密度函数
def gaussian(x,mu,sigma):
    return np.exp(-np.square(x-mu)/(2*(sigma)))/(np.sqrt(2.0*np.pi*sigma))
def EM(dataset,k,itertimes):
    alphas = [1.0/k for i in range(k)]
    mus = [i for i in range(k)]
    sigmas = [1 for i in range(k)]
    N=len(dataset)
    for i in range(itertimes):
        # E步
        gamma = []
        for i in range(len(dataset)):
            gamma.append([alpha * gaussian(dataset[i], mu, sigma) for alpha, mu, sigma in zip(alphas, mus, sigmas)])
        gamma = np.array(gamma)
        gamma = gamma / np.sum(gamma, axis=1).reshape(-1, 1)
        # M步
        mus = np.sum(gamma * (dataset.reshape(-1, 1)), axis=0) / np.sum(gamma, axis=0)
        sigmas = np.sum(gamma * np.square(dataset.reshape(-1, 1) - mus), axis=0) / np.sum(gamma, axis=0)
        alphas = np.sum(gamma, axis=0) / N
    # 学习到的三个一维高斯分布的均值、方差和权重
    print(mus,sigmas,alphas)
if __name__ == "__main__":
    # 构造数据集：均值分别是19,1,10；标准差分别是1,4,5；权重分别是50%，20%，30%
    dataset1 = np.random.normal(19,1,50)
    dataset2 = np.random.normal(1,4,20)
    dataset3 = np.random.normal(10,5,30)
    dataset = np.concatenate((dataset1,dataset2,dataset3),axis=0)
    EM(dataset,k=3,itertimes=500)
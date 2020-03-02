import numpy as np
import matplotlib.pyplot as plt


def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item==0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d:d[1], reverse=True)
    return count_dict[0][0]


def sk_pca(X, k):
    from sklearn.decomposition import PCA
    pca = PCA(k)
    pca.fit(X)
    vec = pca.components_
    #print(vec.shape)
    return vec

def fld(x1, x2):
    x1, x2 = np.mat(x1), np.mat(x2)
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    k = x1.shape[1]

    m1 = np.mean(x1, axis=0)
    m2 = np.mean(x2, axis=0)
    m = np.mean(np.concatenate((x1, x2), axis=0), axis=0)
    print(x1.shape, m1.shape)


    c1 = np.cov(x1.T)
    s1 = c1*(n1-1)
    c2 = np.cov(x2.T)
    s2 = c2*(n2-1)
    Sw = s1/n1 + s2/n2
    print(Sw.shape)
    W = np.dot(np.linalg.inv(Sw), (m1-m2).T)
    print(W.shape)
    W = W / np.linalg.norm(W, 2)
    return np.mean(np.dot(x1, W)), np.mean(np.dot(x2, W)), W

def pca(X, k):
    n, m = X.shape
    mean = np.mean(X, 0)
    #print(mean.shape)
    temp = X - mean
    conv = np.cov(X.T)
    #print(conv.shape)
    conv1 = np.cov(temp.T)
    #print(conv-conv1)

    w, v = np.linalg.eig(conv)
    #print(w.shape)
    #print(v.shape)
    index = np.argsort(-w)
    vec = np.matrix(v.T[index[:k]])
    #print(vec.shape)

    recon = (temp * vec.T)*vec+mean

    #print(X-recon)
    return vec








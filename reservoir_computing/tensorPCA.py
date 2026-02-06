import numpy as np
import numpy.linalg as linalg


class tensorPCA:
    r"""
    对表示为3维张量的多变量时间序列数据集计算PCA，
    并将第三维的大小从 ``[N, T, V]`` 降为 ``[N, T, D]``，其中 ``D <= V``。

    输入数据集必须是3维张量，其中第一维 ``N`` 表示观测数量，
    第二维 ``T`` 表示时间序列中的时间步数，
    第三维 ``V`` 表示时间序列中的变量数量。

    参数
    ----------
    n_components : int
        降维后保留的主成分数量。这决定了输出张量中第三维 ``D`` 的大小。
    """

    def __init__(self, n_components):
        self.n_components=n_components
        self.first_eigs = None
        
    def fit(self, X):
        r"""
        将tensorPCA模型拟合到输入数据集 ``X``。
        
        参数:
        ------------
        X : np.ndarray
            时间序列，形状为 ``[N,T,V]`` 的3维数组，其中 ``N`` 是时间序列的数量，
            ``T`` 是每个时间序列的长度，``V`` 是每个时间序列中的变量数量。

        返回:
        ------------
        None
        """
        if len(X.shape) != 3:
            raise RuntimeError('Input must be a 3d tensor')
        
        Xt = np.swapaxes(X,1,2)  # [N,T,V] --> [N,V,T]
        Xm = np.expand_dims(np.mean(X, axis=0), axis=0) # 平均样本
        Xmt = np.swapaxes(Xm,1,2)
        
        C = np.tensordot(X-Xm,Xt-Xmt,axes=([1,0],[2,0])) / (X.shape[0]-1) # 0模式切片的协方差
        
        # 对协方差矩阵的特征值进行排序
        eigenValues, eigenVectors = linalg.eig(C)
        idx = eigenValues.argsort()[::-1]   
        eigenVectors = eigenVectors[:,idx]
        
        self.first_eigs = eigenVectors[:,:self.n_components]
        
    def transform(self, X):
        r"""
        使用tensorPCA模型转换输入数据集X。

        参数:
        ------------
        X : np.ndarray
            时间序列，形状为 ``[N,T,V]`` 的3维数组，其中 ``N`` 是时间序列的数量，
            ``T`` 是每个时间序列的长度，``V`` 是每个时间序列中的变量数量。

        返回:
        ------------
        Xpca : np.ndarray
            转换后的时间序列，形状为 ``[N,T,D]`` 的3维数组，其中 ``N`` 是时间序列的数量，
            ``T`` 是每个时间序列的长度，``D`` 是主成分的数量。
        """
        return np.einsum('klj,ji->kli',X,self.first_eigs)
    
    def fit_transform(self, X):
        r"""
        将tensorPCA模型拟合到输入数据集 ``X`` 并转换它。

        参数:
        ------------
        X : np.ndarray
            时间序列，形状为 ``[N,T,V]`` 的3维数组，其中 ``N`` 是时间序列的数量，
            ``T`` 是每个时间序列的长度，``V`` 是每个时间序列中的变量数量。

        返回:
        ------------
        Xpca : np.ndarray
            转换后的时间序列，形状为 ``[N,T,D]`` 的3维数组，其中 ``N`` 是时间序列的数量，
            ``T`` 是每个时间序列的长度，``D`` 是主成分的数量。
        """
        self.fit(X)
        return self.transform(X)
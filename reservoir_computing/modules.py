import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import pdist, cdist, squareform

from .reservoir import Reservoir
from .tensorPCA import tensorPCA

            
class RC_model(object):
    r"""构建并评估基于储备池计算的时间序列分类或聚类模型。

    训练和测试的多变量时间序列（MTS）是形状为 ``[N,T,V]`` 的多维数组，其中 ``N`` 是样本数量，``T`` 是每个样本的时间步数，``V`` 是每个样本的变量数量。
    
    训练和测试标签的形状为 ``[N,C]``，其中 ``C`` 是类别数量。
    
    数据集由训练数据和相应标签 ``(X, Y)`` 以及测试数据和相应标签 ``(Xte, Yte)`` 组成。
    
    **储备池参数:**
    
    :param reservoir: ``Reservoir`` 类的对象 (默认 ``None``) 
        预计算的储备池。如果为 ``None``，必须指定以下结构超参数。
    :param n_internal_units: int (默认 ``100``) 
        储备池中的处理单元数量。
    :param spectral_radius: float (默认 ``0.99``) 
        储备池连接权重矩阵的最大特征值。
        为确保回声状态属性，设置 ``spectral_radius <= leak <= 1``)
    :param leak: float (默认 ``None``) 
        储备池状态更新中的泄漏量。
        如果为 ``None`` 或 ``1.0``，则不使用泄漏。
    :param connectivity: float (默认 ``0.3``) 
        非零连接权重的百分比。
        在圆形储备池中不使用。
    :param input_scaling: float (默认 ``0.2``)
        输入连接权重的缩放因子。
        注意输入权重是从 ``{-1,1}`` 中随机抽取的。
    :param noise_level: float (默认 ``0.0``) 
        在状态更新中注入的高斯噪声的标准差。
    :param n_drop: int (默认 ``0``)
        要丢弃的瞬态状态数量。
    :param bidir: bool (默认 ``False``)
        使用双向储备池（``True``）或标准储备池（``False``）。
    :param circle: bool (默认 ``False``)
        生成具有圆形拓扑的确定性储备池，其中每个连接具有相同的权重。
    
    **降维参数:**
    
    :param dimred_method: str (默认 ``None``)
        用于减少储备池状态序列中特征数量的过程。
        可能的选项有：``None``（不进行降维）、``'pca'``（标准PCA）、
        或 ``'tenpca'``（用于多变量时间序列数据的张量PCA）。
    :param n_dim: int (默认 ``None``) 
        降维过程后的结果维度数量。
    
    **表示参数:**
    
    :param mts_rep: str (默认 ``None``) 
        MTS表示的类型。
        可以是 ``'last'``（最后状态）、``'mean'``（所有状态的均值）、
        ``'output'``（输出模型空间）或 ``'reservoir'``（储备池模型空间）。
    :param w_ridge_embedding: float (默认 ``1.0``) 
        输出模型空间和储备池模型空间表示中岭回归的正则化参数；如果 ``mts_rep == None`` 则忽略。
    
    **读出参数:**
    
    :param readout_type: str (默认 ``'lin'``) 
        用于分类的读出类型。可以是 ``'lin'``（岭回归）、
        ``'mlp'``（多层感知器）、``'svm'``（支持向量机）、
        或 ``None``。
        如果为 ``None``，输入表示将保存在 ``.input_repr`` 属性中：
        这对于聚类和可视化很有用。
        另外，如果为 ``None``，其他读出超参数可以留空。
    :param w_ridge: float (默认 ``1.0``) 
        岭回归读出的正则化参数（仅用于 ``readout_type=='lin'``）。
    :param mlp_layout: tuple (默认 ``None``) 
        包含MLP层大小的元组，例如，``(20, 10)`` 定义了一个具有2层的MLP，分别有20和10个单元。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param num_epochs: int (默认 ``None``) 
        优化过程中的迭代次数。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param w_l2: float (默认 ``None``) 
        L2正则化的权重。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param nonlinearity: str (默认 ``None``) 
        激活函数的类型 ``{'relu', 'tanh', 'logistic', 'identity'}``。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param svm_gamma: float (默认 ``1.0``) 
        RBF核的带宽。
        仅在 ``readout_type=='svm'`` 时使用。
    :param svm_C: float (默认 ``1.0``) 
        SVM超平面的正则化参数。
        仅在 ``readout_type=='svm'`` 时使用。
    """
    
    def __init__(self,
              # reservoir
              reservoir=None,     
              n_internal_units=100,
              spectral_radius=0.99,
              leak=None,
              connectivity=0.3,
              input_scaling=0.2,
              noise_level=0.0,
              n_drop=0,
              bidir=False,
              circle=False,
              # dim red
              dimred_method=None, 
              n_dim=None,
              # representation
              mts_rep='mean',
              w_ridge_embedding=1.0,
              # readout
              readout_type='lin',               
              w_ridge=1.0,              
              mlp_layout=None,
              num_epochs=None,
              w_l2=None,
              nonlinearity=None, 
              svm_gamma=1.0,
              svm_C=1.0):

        self.n_drop=n_drop
        self.bidir=bidir
        self.dimred_method=dimred_method
        self.mts_rep=mts_rep
        self.readout_type=readout_type
        self.svm_gamma=svm_gamma
                        
        # 初始化储备池
        if reservoir is None:
            self._reservoir = Reservoir(n_internal_units=n_internal_units,
                                  spectral_radius=spectral_radius,
                                  leak=leak,
                                  connectivity=connectivity,
                                  input_scaling=input_scaling,
                                  noise_level=noise_level,
                                  circle=circle)
        else:
            self._reservoir = reservoir
                
        # 初始化降维方法
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            elif dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components=n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID')
                
        # 初始化岭回归模型
        if mts_rep=='output' or mts_rep=='reservoir':
            self._ridge_embedding = Ridge(alpha=w_ridge_embedding, fit_intercept=True)
                        
        # 初始化读出类型            
        if self.readout_type is not None:
            
            if self.readout_type == 'lin': # 岭回归
                self.readout = Ridge(alpha=w_ridge)        
            elif self.readout_type == 'svm': # SVM读出
                self.readout = SVC(C=svm_C, kernel='precomputed')          
            elif readout_type == 'mlp': # MLP（深度读出）  
                # pass
                self.readout = MLPClassifier(
                    hidden_layer_sizes=mlp_layout, 
                    activation=nonlinearity, 
                    alpha=w_l2,
                    batch_size=32, 
                    learning_rate='adaptive', # 'constant' 或 'adaptive'
                    learning_rate_init=0.001, 
                    max_iter=num_epochs, 
                    early_stopping=False, # 如果为True，设置validation_fraction > 0
                    validation_fraction=0.0 # 用于早停
                    )
            else:
                raise RuntimeError('Invalid readout type')  
        
        
    def fit(self, X, Y=None, verbose=True):
        r"""训练RC模型。

        参数:
        -----------
        X : np.ndarray 
            形状为 ``[N, T, V]`` 的数组，表示训练数据。

        Y : np.ndarray 
            形状为 ``[N, C]`` 的数组，表示目标值。

        verbose : bool
            如果为 ``True``，打印训练时间。

        返回:
        --------
        None
        """
                
        time_start = time.time()
        
        # ============ 计算储备池状态 ============ 
        res_states = self._reservoir.get_states(X, n_drop=self.n_drop, bidir=self.bidir)
        
        # ============ 储备池状态的降维 ============  
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # 矩阵化
                N_samples = res_states.shape[0]
                res_states = res_states.reshape(-1, res_states.shape[2])                   
                # ..变换..
                red_states = self._dim_red.fit_transform(res_states)          
                # ..并转换回张量形式
                red_states = red_states.reshape(N_samples,-1,red_states.shape[1])          
            elif self.dimred_method.lower() == 'tenpca':
                red_states = self._dim_red.fit_transform(res_states)       
        else: # 跳过降维
            red_states = res_states

        # ============ 生成MTS的表示 ============
        coeff_tr = []
        biases_tr = []   
        
        # 输出模型空间表示
        if self.mts_rep=='output':
            if self.bidir:
                X = np.concatenate((X,X[:, ::-1, :]),axis=2)                
                
            for i in range(X.shape[0]):
                self._ridge_embedding.fit(red_states[i, 0:-1, :], X[i, self.n_drop+1:, :])
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            
        # 储备池模型空间表示
        elif self.mts_rep=='reservoir':
            for i in range(X.shape[0]):
                self._ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        
        # 最后状态表示        
        elif self.mts_rep=='last':
            input_repr = red_states[:, -1, :]
            
        # 平均状态表示        
        elif self.mts_rep=='mean':
            input_repr = np.mean(red_states, axis=1)
            
        else:
            raise RuntimeError('Invalid representation ID')            
            
        # ============ 训练读出 ============
        if self.readout_type == None: # 仅存储输入表示
            self.input_repr = input_repr
            
        elif self.readout_type == 'lin': # 岭回归
            self.readout.fit(input_repr, Y)          
            
        elif self.readout_type == 'svm': # SVM读出
            Ktr = squareform(pdist(input_repr, metric='sqeuclidean')) 
            Ktr = np.exp(-self.svm_gamma*Ktr)
            self.readout.fit(Ktr, np.argmax(Y,axis=1))
            self.input_repr_tr = input_repr # 存储它们以构建测试核
            
        elif self.readout_type == 'mlp': # MLP（深度读出）
            self.readout.fit(input_repr, Y)
                        
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")

            
    def predict(self, Xte):
        r"""计算样本外（测试）数据的预测。

        参数:
        -----------
        Xte : np.ndarray
            形状为 ``[N, T, V]`` 的数组，表示测试数据。

        返回:
        --------
        pred_class : np.ndarray
            形状为 ``[N]`` 的数组，表示预测的类别。
        """

        # ============ 计算储备池状态 ============
        res_states_te = self._reservoir.get_states(Xte, n_drop=self.n_drop, bidir=self.bidir) 
        
        # ============ 储备池状态的降维 ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # 矩阵化
                N_samples_te = res_states_te.shape[0]
                res_states_te = res_states_te.reshape(-1, res_states_te.shape[2])                    
                # ..变换..
                red_states_te = self._dim_red.transform(res_states_te)            
                # ..并转换回张量形式
                red_states_te = red_states_te.reshape(N_samples_te,-1,red_states_te.shape[1])            
            elif self.dimred_method.lower() == 'tenpca':
                red_states_te = self._dim_red.transform(res_states_te)        
        else: # 跳过降维
            red_states_te = res_states_te             
        
        # ============ 生成MTS的表示 ============
        coeff_te = []
        biases_te = []   
        
        # 输出模型空间表示
        if self.mts_rep=='output':
            if self.bidir:
                Xte = np.concatenate((Xte,Xte[:, ::-1, :]),axis=2)  
                    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], Xte[i, self.n_drop+1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
        
        # 储备池模型空间表示
        elif self.mts_rep=='reservoir':    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
    
        # 最后状态表示        
        elif self.mts_rep=='last':
            input_repr_te = red_states_te[:, -1, :]
            
        # 平均状态表示        
        elif self.mts_rep=='mean':
            input_repr_te = np.mean(red_states_te, axis=1)
            
        else:
            raise RuntimeError('Invalid representation ID')   
            
        # ============ 应用读出 ============
        if self.readout_type == 'lin': # 岭回归        
            logits = self.readout.predict(input_repr_te)
            pred_class = np.argmax(logits, axis=1)
            
        elif self.readout_type == 'svm': # SVM读出
            Kte = cdist(input_repr_te, self.input_repr_tr, metric='sqeuclidean')
            Kte = np.exp(-self.svm_gamma*Kte)
            pred_class = self.readout.predict(Kte)
            
        elif self.readout_type == 'mlp': # MLP（深度读出）
            pred_class = self.readout.predict(input_repr_te)
            pred_class = np.argmax(pred_class, axis=1)
            
        return pred_class
    

class RC_forecaster(object):
    r"""使用RC进行时间序列预测的类。

    训练和测试数据是形状为 ``[T,V]`` 的多维数组，其中

    - ``T`` = 每个样本的时间步数，
    - ``V`` = 每个样本的变量数量。

    给定时间序列 ``X``，训练数据应该如下：
    
        ``Xtr, Ytr = X[0:-forecast_horizon,:], X[forecast_horizon:,:]``

    训练完成后，模型可用于计算提前 ``forecast_horizon`` 步的预测：
        
            ``Yhat[t,:] = Xte[t+forecast_horizon,:]``

    **储备池参数:**

    :param reservoir: ``Reservoir`` 类的对象 (默认 ``None``)
        预计算的储备池。如果为 ``None``，必须指定以下结构超参数。
    :param n_internal_units: int (默认 ``100``) 
        储备池中的处理单元数量。
    :param spectral_radius: float (默认 ``0.99``) 
        储备池连接权重矩阵的最大特征值。
        为确保回声状态属性，设置 ``spectral_radius <= leak <= 1``)
    :param leak: float (默认 ``None``) 
        储备池状态更新中的泄漏量。
    :param connectivity: float (默认 ``0.3``)
        非零连接权重的百分比。
    :param input_scaling: float (默认 ``0.2``) 
        输入连接权重的缩放因子。
        注意输入权重是从 ``{-1,1}`` 中随机抽取的。
    :param noise_level: float (默认 ``0.0``)
        在状态更新中注入的高斯噪声的标准差。
    :param n_drop: int (默认 ``0``)
        要丢弃的瞬态状态数量。
    :param circle: bool (默认 ``False``)
        生成具有圆形拓扑的确定性储备池，其中每个连接具有相同的权重。

    **降维参数:**

    :param dimred_method: str (默认 ``None``)
        用于减少储备池状态序列中特征数量的过程。
        可能的选项有：``None``（不进行降维）、``'pca'``（标准PCA）、
        或 ``'tenpca'``（用于多变量时间序列数据的张量PCA）。
    :param n_dim: int (默认 ``None``) 
        降维过程后的结果维度数量。

    **读出参数:**

    :param w_ridge: float (默认 ``1.0``) 
        岭回归读出的正则化参数。
    """
    
    def __init__(self,
                # reservoir
                reservoir=None,     
                n_internal_units=100,
                spectral_radius=0.99,
                leak=None,
                connectivity=0.3,
                input_scaling=0.2,
                noise_level=0.0,
                n_drop=0,
                circle=False,
                # dim red
                dimred_method=None, 
                n_dim=None,
                # readout              
                w_ridge=1.0):
        self.n_drop=n_drop
        self.dimred_method=dimred_method  
                        
        # 初始化储备池
        if reservoir is None:
            self._reservoir = Reservoir(n_internal_units=n_internal_units,
                                        spectral_radius=spectral_radius,
                                        leak=leak,
                                        connectivity=connectivity,
                                        input_scaling=input_scaling,
                                        noise_level=noise_level,
                                        circle=circle)
        else:
            self._reservoir = reservoir
                
        # 初始化降维方法
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            else:
                raise RuntimeError('Invalid dimred method ID')
            
        # 初始化读出
        self.readout = Ridge(alpha=w_ridge)


    def fit(self, X, Y, verbose=True):
        r"""训练用于预测的RC模型。

        参数:
        -----------
        X : np.ndarray 
            形状为 ``[T, V]`` 的数组，表示训练数据。

        Y : np.ndarray
            形状为 ``[T, V]`` 的数组，表示目标值。

        verbose : bool
            如果为 ``True``，打印训练时间。

        返回:
        --------
        red_states : np.ndarray
            形状为 ``[T, n_dim]`` 的数组，表示用于训练的时间步的储备池状态。
        """
        
        time_start = time.time()
        
        # ============ 计算储备池状态 ============ 
        res_states = self._reservoir.get_states(X[None,:,:], n_drop=self.n_drop, bidir=False)
        
        # ============ 储备池状态的降维 ============  
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                red_states = self._dim_red.fit_transform(res_states[0])          
        else: # 跳过降维
            red_states = res_states[0]

        self._fitted_states = red_states

        # ============ 训练读出 ============
        self.readout.fit(red_states, Y[self.n_drop:,:])          
            
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")

        return red_states

    def predict(self, Xte, return_states=False):
        r"""计算样本外（测试）数据的预测。

        参数:
        -----------
        Xte : np.ndarray
            形状为 ``[T, V]`` 的数组，表示测试数据。
        
        return_states : bool
            如果为 ``True``，返回预测的状态。

        返回:
        --------
        Yhat : np.ndarray
            形状为 ``[T, V]`` 的数组，表示预测值。
        
        red_states_te : np.ndarray
            形状为 ``[T, n_dim]`` 的数组，表示新时间步的储备池状态。
        """

        # ============ 计算储备池状态 ============
        res_states_te = self._reservoir.get_states(Xte[None,:,:], n_drop=self.n_drop, bidir=False) 
        
        # ============ 储备池状态的降维 ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                red_states_te = self._dim_red.transform(res_states_te[0])                          
        else: # 跳过降维
            red_states_te = res_states_te[0]        

        self._predicted_states = red_states_te

        # ============ 应用读出 ============
        Yhat = self.readout.predict(red_states_te)

        if return_states:
            return Yhat, red_states_te
        return Yhat

    def get_fitted_states(self):
        r"""返回拟合的储备池状态。

        返回:
        --------
        fitted_states : np.ndarray
            形状为 ``[T, n_dim]`` 的数组，表示拟合的储备池状态。
        """
        return self._fitted_states

    def get_predicted_states(self):
        r"""返回预测的储备池状态。

        返回:
        --------
        predicted_states : np.ndarray
            形状为 ``[T, n_dim]`` 的数组，表示预测的储备池状态。
        """
        return self._predicted_states


class StackedRC_model(object):
    r"""构建并评估基于层叠储备池计算的时间序列分类或聚类模型。

    通过多个Reservoir串联，逐层提取抽象特征，提升模型表达能力。
    每层的输出状态序列作为下一层的输入，最后一层的输出用于生成表示。

    训练和测试的多变量时间序列（MTS）是形状为 ``[N,T,V]`` 的多维数组，其中 ``N`` 是样本数量，``T`` 是每个样本的时间步数，``V`` 是每个样本的变量数量。
    
    训练和测试标签的形状为 ``[N,C]``，其中 ``C`` 是类别数量。
    
    **层叠储备池参数:**
    
    :param n_layers: int (默认 ``2``)
        层叠Reservoir的层数。
    :param reservoir_configs: dict 或 list of dict (默认 ``None``)
        每层Reservoir的配置。
        - 如果为单个字典，所有层使用相同配置
        - 如果为字典列表，每层使用对应配置（列表长度应等于n_layers）
        - 如果为None，使用默认渐进式配置
        字典可包含的键：``n_internal_units``, ``spectral_radius``, ``leak``, 
        ``connectivity``, ``input_scaling``, ``noise_level``, ``circle``
    :param n_drop: int (默认 ``0``)
        要丢弃的瞬态状态数量（每层相同）。
    :param bidir: bool (默认 ``False``)
        使用双向储备池（``True``）或标准储备池（``False``，每层相同）。
    
    **降维参数:**
    
    :param dimred_method: str (默认 ``None``)
        用于减少最后一层储备池状态序列中特征数量的过程。
        可能的选项有：``None``（不进行降维）、``'pca'``（标准PCA）、
        或 ``'tenpca'``（用于多变量时间序列数据的张量PCA）。
    :param n_dim: int (默认 ``None``) 
        降维过程后的结果维度数量。
    
    **表示参数:**
    
    :param mts_rep: str (默认 ``'mean'``) 
        MTS表示的类型。
        可以是 ``'last'``（最后状态）、``'mean'``（所有状态的均值）、
        ``'output'``（输出模型空间）或 ``'reservoir'``（储备池模型空间）。
    :param w_ridge_embedding: float (默认 ``1.0``) 
        输出模型空间和储备池模型空间表示中岭回归的正则化参数；如果 ``mts_rep == None`` 则忽略。
    
    **读出参数:**
    
    :param readout_type: str (默认 ``'lin'``) 
        用于分类的读出类型。可以是 ``'lin'``（岭回归）、
        ``'mlp'``（多层感知器）、``'svm'``（支持向量机）、
        或 ``None``。
        如果为 ``None``，输入表示将保存在 ``.input_repr`` 属性中：
        这对于聚类和可视化很有用。
    :param w_ridge: float (默认 ``1.0``) 
        岭回归读出的正则化参数（仅用于 ``readout_type=='lin'``）。
    :param mlp_layout: tuple (默认 ``None``) 
        包含MLP层大小的元组，例如，``(20, 10)`` 定义了一个具有2层的MLP，分别有20和10个单元。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param num_epochs: int (默认 ``None``) 
        优化过程中的迭代次数。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param w_l2: float (默认 ``None``) 
        L2正则化的权重。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param nonlinearity: str (默认 ``None``) 
        激活函数的类型 ``{'relu', 'tanh', 'logistic', 'identity'}``。
        仅在 ``readout_type=='mlp'`` 时使用。
    :param svm_gamma: float (默认 ``1.0``) 
        RBF核的带宽。
        仅在 ``readout_type=='svm'`` 时使用。
    :param svm_C: float (默认 ``1.0``) 
        SVM超平面的正则化参数。
        仅在 ``readout_type=='svm'`` 时使用。
    """
    
    def __init__(self,
              # stacked reservoir
              n_layers=2,
              reservoir_configs=None,
              n_drop=0,
              bidir=False,
              # dim red
              dimred_method=None, 
              n_dim=None,
              # representation
              mts_rep='mean',
              w_ridge_embedding=1.0,
              # readout
              readout_type='lin',               
              w_ridge=1.0,              
              mlp_layout=None,
              num_epochs=None,
              w_l2=None,
              nonlinearity=None, 
              svm_gamma=1.0,
              svm_C=1.0):

        self.n_layers = n_layers
        self.n_drop = n_drop
        self.bidir = bidir
        self.dimred_method = dimred_method
        self.mts_rep = mts_rep
        self.readout_type = readout_type
        self.svm_gamma = svm_gamma
        
        # 处理储备池配置
        if reservoir_configs is None:
            # 默认渐进式配置：层数越多，单元数递减
            if n_layers == 2:
                default_units = [200, 150]
            elif n_layers == 3:
                default_units = [300, 200, 150]
            elif n_layers == 4:
                default_units = [400, 300, 200, 150]
            elif n_layers == 5:
                default_units = [400, 350, 300, 200, 150]
            elif n_layers == 6:
                default_units = [400, 350, 300, 250, 200, 150]
            else:
                # 对于更多层，使用线性递减
                base_units = 400
                default_units = [max(base_units - i * 50, 100) for i in range(n_layers)]  # 最少100个单元
            
            reservoir_configs = [
                {
                    'n_internal_units': units,
                    'spectral_radius': 0.99,
                    'leak': None,
                    'connectivity': 0.3,
                    'input_scaling': 0.2,
                    'noise_level': 0.0,
                    'circle': False
                }
                for units in default_units
            ]
        elif isinstance(reservoir_configs, dict):
            # 单个字典，所有层使用相同配置
            reservoir_configs = [reservoir_configs.copy() for _ in range(n_layers)]
        elif isinstance(reservoir_configs, list):
            # 字典列表，每层使用对应配置
            if len(reservoir_configs) != n_layers:
                raise ValueError(f"reservoir_configs列表长度({len(reservoir_configs)})必须等于n_layers({n_layers})")
        else:
            raise ValueError("reservoir_configs必须是dict、list of dict或None")
        
        # 初始化多个储备池
        self._reservoirs = []
        for layer_idx, config in enumerate(reservoir_configs):
            reservoir = Reservoir(
                n_internal_units=config.get('n_internal_units', 100),
                spectral_radius=config.get('spectral_radius', 0.99),
                leak=config.get('leak', None),
                connectivity=config.get('connectivity', 0.3),
                input_scaling=config.get('input_scaling', 0.2),
                noise_level=config.get('noise_level', 0.0),
                circle=config.get('circle', False)
            )
            self._reservoirs.append(reservoir)
                
        # 初始化降维方法（仅用于最后一层）
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            elif dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components=n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID')
                
        # 初始化岭回归模型
        if mts_rep=='output' or mts_rep=='reservoir':
            self._ridge_embedding = Ridge(alpha=w_ridge_embedding, fit_intercept=True)
                        
        # 初始化读出类型            
        if self.readout_type is not None:
            
            if self.readout_type == 'lin': # 岭回归
                self.readout = Ridge(alpha=w_ridge)        
            elif self.readout_type == 'svm': # SVM读出
                self.readout = SVC(C=svm_C, kernel='precomputed')          
            elif readout_type == 'mlp': # MLP（深度读出）  
                self.readout = MLPClassifier(
                    hidden_layer_sizes=mlp_layout, 
                    activation=nonlinearity, 
                    alpha=w_l2,
                    batch_size=32, 
                    learning_rate='adaptive',
                    learning_rate_init=0.001, 
                    max_iter=num_epochs, 
                    early_stopping=False,
                    validation_fraction=0.0
                    )
            else:
                raise RuntimeError('Invalid readout type')  
        
        
    def fit(self, X, Y=None, verbose=True):
        r"""训练层叠RC模型。

        参数:
        -----------
        X : np.ndarray 
            形状为 ``[N, T, V]`` 的数组，表示训练数据。

        Y : np.ndarray 
            形状为 ``[N, C]`` 的数组，表示目标值。

        verbose : bool
            如果为 ``True``，打印训练时间。

        返回:
        --------
        None
        """
                
        time_start = time.time()
        
        # ============ 逐层计算储备池状态 ============ 
        current_input = X  # 第一层输入：原始数据 [N, T, V]
        
        for layer_idx, reservoir in enumerate(self._reservoirs):
            # 计算当前层的状态
            res_states = reservoir.get_states(current_input, n_drop=self.n_drop, bidir=self.bidir)
            
            # 如果不是最后一层，将状态序列作为下一层的输入
            if layer_idx < len(self._reservoirs) - 1:
                # 状态序列 [N, T, H] 作为下一层的输入
                current_input = res_states
            else:
                # 最后一层的状态用于生成表示
                final_states = res_states
        
        # ============ 最后一层状态的降维 ============  
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # 矩阵化
                N_samples = final_states.shape[0]
                final_states = final_states.reshape(-1, final_states.shape[2])                   
                # ..变换..
                red_states = self._dim_red.fit_transform(final_states)          
                # ..并转换回张量形式
                red_states = red_states.reshape(N_samples,-1,red_states.shape[1])          
            elif self.dimred_method.lower() == 'tenpca':
                red_states = self._dim_red.fit_transform(final_states)       
        else: # 跳过降维
            red_states = final_states

        # ============ 生成MTS的表示 ============
        coeff_tr = []
        biases_tr = []   
        
        # 输出模型空间表示
        if self.mts_rep=='output':
            if self.bidir:
                X = np.concatenate((X,X[:, ::-1, :]),axis=2)                
                
            for i in range(X.shape[0]):
                self._ridge_embedding.fit(red_states[i, 0:-1, :], X[i, self.n_drop+1:, :])
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            
        # 储备池模型空间表示
        elif self.mts_rep=='reservoir':
            for i in range(X.shape[0]):
                self._ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        
        # 最后状态表示        
        elif self.mts_rep=='last':
            input_repr = red_states[:, -1, :]
            
        # 平均状态表示        
        elif self.mts_rep=='mean':
            input_repr = np.mean(red_states, axis=1)
            
        else:
            raise RuntimeError('Invalid representation ID')            
            
        # ============ 训练读出 ============
        if self.readout_type == None: # 仅存储输入表示
            self.input_repr = input_repr
            
        elif self.readout_type == 'lin': # 岭回归
            self.readout.fit(input_repr, Y)          
            
        elif self.readout_type == 'svm': # SVM读出
            Ktr = squareform(pdist(input_repr, metric='sqeuclidean')) 
            Ktr = np.exp(-self.svm_gamma*Ktr)
            self.readout.fit(Ktr, np.argmax(Y,axis=1))
            self.input_repr_tr = input_repr # 存储它们以构建测试核
            
        elif self.readout_type == 'mlp': # MLP（深度读出）
            self.readout.fit(input_repr, Y)
                        
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")

            
    def predict(self, Xte):
        r"""计算样本外（测试）数据的预测。

        参数:
        -----------
        Xte : np.ndarray
            形状为 ``[N, T, V]`` 的数组，表示测试数据。

        返回:
        --------
        pred_class : np.ndarray
            形状为 ``[N]`` 的数组，表示预测的类别。
        """

        # ============ 逐层计算储备池状态 ============
        current_input = Xte  # 第一层输入：原始测试数据 [N, T, V]
        
        for layer_idx, reservoir in enumerate(self._reservoirs):
            # 计算当前层的状态
            res_states_te = reservoir.get_states(current_input, n_drop=self.n_drop, bidir=self.bidir)
            
            # 如果不是最后一层，将状态序列作为下一层的输入
            if layer_idx < len(self._reservoirs) - 1:
                current_input = res_states_te
            else:
                final_states_te = res_states_te
        
        # ============ 最后一层状态的降维 ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # 矩阵化
                N_samples_te = final_states_te.shape[0]
                final_states_te = final_states_te.reshape(-1, final_states_te.shape[2])                    
                # ..变换..
                red_states_te = self._dim_red.transform(final_states_te)            
                # ..并转换回张量形式
                red_states_te = red_states_te.reshape(N_samples_te,-1,red_states_te.shape[1])            
            elif self.dimred_method.lower() == 'tenpca':
                red_states_te = self._dim_red.transform(final_states_te)        
        else: # 跳过降维
            red_states_te = final_states_te             
        
        # ============ 生成MTS的表示 ============
        coeff_te = []
        biases_te = []   
        
        # 输出模型空间表示
        if self.mts_rep=='output':
            if self.bidir:
                Xte = np.concatenate((Xte,Xte[:, ::-1, :]),axis=2)  
                    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], Xte[i, self.n_drop+1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
        
        # 储备池模型空间表示
        elif self.mts_rep=='reservoir':    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
    
        # 最后状态表示        
        elif self.mts_rep=='last':
            input_repr_te = red_states_te[:, -1, :]
            
        # 平均状态表示        
        elif self.mts_rep=='mean':
            input_repr_te = np.mean(red_states_te, axis=1)
            
        else:
            raise RuntimeError('Invalid representation ID')   
            
        # ============ 应用读出 ============
        if self.readout_type == 'lin': # 岭回归        
            logits = self.readout.predict(input_repr_te)
            pred_class = np.argmax(logits, axis=1)
            
        elif self.readout_type == 'svm': # SVM读出
            Kte = cdist(input_repr_te, self.input_repr_tr, metric='sqeuclidean')
            Kte = np.exp(-self.svm_gamma*Kte)
            pred_class = self.readout.predict(Kte)
            
        elif self.readout_type == 'mlp': # MLP（深度读出）
            pred_class = self.readout.predict(input_repr_te)
            pred_class = np.argmax(pred_class, axis=1)
            
        return pred_class


class MultiExpertStackedRC_model(object):
    r"""多专家 + 残差式层叠储备池 RC 模型。

    相比于 ``StackedRC_model`` 仅做简单串联，本模型在每一层内部引入多个并行的 Reservoir 专家，
    并在读出阶段将各层的表示进行级联融合，从“多专家集成 + 多层残差”的角度提升表达能力，
    缓解深层结构可能出现的性能退化。

    设计要点：

    - 每层包含 ``n_experts`` 个并行 Reservoir，输入相同但随机权重不同；
    - 同一层内各专家的状态序列在特征维上拼接，作为下一层的输入；
    - 每一层都会根据 ``mts_rep`` 生成整体表示（例如时间维 mean 或 last），并在读出前拼接所有层的表示，
      得到最终的多尺度/残差式特征表示；
    - 支持的表示方式：``'mean'``、``'last'``、``'output'``、``'reservoir'``。

    参数大体与 ``StackedRC_model`` 保持一致，额外增加：

    :param n_experts: int (默认 ``3``)
        每一层的并行 Reservoir 专家个数。
    
    **降维参数:**
    
    :param dimred_method: str (默认 ``None``)
        用于减少最后一层储备池状态序列中特征数量的过程。
        可能的选项有：``None``（不进行降维）、``'pca'``（标准PCA）、
        或 ``'tenpca'``（用于多变量时间序列数据的张量PCA）。
    :param n_dim: int (默认 ``None``) 
        降维过程后的结果维度数量。
    """

    def __init__(self,
                 # stacked reservoir
                 n_layers=2,
                 n_experts=3,
                 reservoir_configs=None,
                 n_drop=0,
                 bidir=False,
                 # dim red（当前版本不实现，可保持 None）
                 dimred_method=None,
                 n_dim=None,
                 # representation
                 mts_rep='mean',
                 w_ridge_embedding=1.0,  # 占位参数，为接口一致暂不使用
                 # readout
                 readout_type='lin',
                 w_ridge=1.0,
                 mlp_layout=None,
                 num_epochs=None,
                 w_l2=None,
                 nonlinearity=None,
                 svm_gamma=1.0,
                 svm_C=1.0):

        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_drop = n_drop
        self.bidir = bidir
        self.dimred_method = dimred_method
        self.mts_rep = mts_rep
        self.readout_type = readout_type
        self.svm_gamma = svm_gamma

        # 初始化降维方法（用于最后一层）
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            elif dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components=n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID')
        
        # 初始化岭回归模型（用于 output 和 reservoir 表示）
        if mts_rep == 'output' or mts_rep == 'reservoir':
            self._ridge_embedding = Ridge(alpha=w_ridge_embedding, fit_intercept=True)

        # 处理储备池配置，与 StackedRC_model 保持一致
        if reservoir_configs is None:
            if n_layers == 2:
                default_units = [200, 150]
            elif n_layers == 3:
                default_units = [300, 200, 150]
            elif n_layers == 4:
                default_units = [400, 300, 200, 150]
            elif n_layers == 5:
                default_units = [400, 350, 300, 200, 150]
            elif n_layers == 6:
                default_units = [400, 350, 300, 250, 200, 150]
            else:
                base_units = 400
                default_units = [max(base_units - i * 50, 100) for i in range(n_layers)]

            reservoir_configs = [
                {
                    'n_internal_units': units,
                    'spectral_radius': 0.99,
                    'leak': None,
                    'connectivity': 0.3,
                    'input_scaling': 0.2,
                    'noise_level': 0.0,
                    'circle': False
                }
                for units in default_units
            ]
        elif isinstance(reservoir_configs, dict):
            reservoir_configs = [reservoir_configs.copy() for _ in range(n_layers)]
        elif isinstance(reservoir_configs, list):
            if len(reservoir_configs) != n_layers:
                raise ValueError(f"reservoir_configs 列表长度({len(reservoir_configs)})必须等于 n_layers({n_layers})")
        else:
            raise ValueError("reservoir_configs 必须是 dict、list[dict] 或 None")

        # 初始化多层多专家储备池：self._reservoirs[layer_idx][expert_idx]
        self._reservoirs = []
        for layer_idx, config in enumerate(reservoir_configs):
            layer_reservoirs = []
            for _ in range(self.n_experts):
                reservoir = Reservoir(
                    n_internal_units=config.get('n_internal_units', 100),
                    spectral_radius=config.get('spectral_radius', 0.99),
                    leak=config.get('leak', None),
                    connectivity=config.get('connectivity', 0.3),
                    input_scaling=config.get('input_scaling', 0.2),
                    noise_level=config.get('noise_level', 0.0),
                    circle=config.get('circle', False)
                )
                layer_reservoirs.append(reservoir)
            self._reservoirs.append(layer_reservoirs)

        # 初始化读出
        if self.readout_type is not None:
            if self.readout_type == 'lin':  # 岭回归
                self.readout = Ridge(alpha=w_ridge)
            elif self.readout_type == 'svm':  # SVM 读出
                self.readout = SVC(C=svm_C, kernel='precomputed')
            elif self.readout_type == 'mlp':  # MLP（深度读出）
                self.readout = MLPClassifier(
                    hidden_layer_sizes=mlp_layout,
                    activation=nonlinearity,
                    alpha=w_l2,
                    batch_size=32,
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    max_iter=num_epochs,
                    early_stopping=False,
                    validation_fraction=0.0
                )
            else:
                raise RuntimeError('Invalid readout type')

    def _forward_layers(self, X, X_original=None):
        """内部函数：多层多专家前向传播，返回各层的序列状态和层级表示。

        参数:
            X: 当前输入 [N, T, V] 或 [N, T, H]
            X_original: 原始输入 [N, T, V]，用于 'output' 表示方法

        返回:
            layer_states_list: list，每个元素是该层拼接后的状态序列 [N, T, H_total]
            layer_repr_list: list，每个元素是该层的整体表示 [N, H_total]
        """
        if X_original is None:
            X_original = X  # 保存原始输入用于 'output' 表示
        
        current_input = X  # [N, T, V] 或上一层的 [N, T, H_total]
        layer_states_list = []
        layer_repr_list = []

        for layer_idx, layer_reservoirs in enumerate(self._reservoirs):
            expert_states = []
            for reservoir in layer_reservoirs:
                states = reservoir.get_states(current_input, n_drop=self.n_drop, bidir=self.bidir)
                expert_states.append(states)

            # 在特征维拼接各专家状态: [N, T, sum(H_e)]
            layer_states = np.concatenate(expert_states, axis=2)

            # 计算该层的整体表示（残差节点）
            if self.mts_rep == 'mean':
                layer_repr = np.mean(layer_states, axis=1)  # [N, H_total]
            elif self.mts_rep == 'last':
                layer_repr = layer_states[:, -1, :]  # [N, H_total]
            elif self.mts_rep == 'output':
                # 输出模型空间表示：用岭回归拟合从状态到原始输入的映射
                # 对于第一层，使用原始输入 X_original；对于后续层，使用前一层状态
                if layer_idx == 0:
                    target_input = X_original
                else:
                    target_input = layer_states_list[layer_idx - 1]  # 使用前一层状态
                
                if self.bidir:
                    target_input = np.concatenate((target_input, target_input[:, ::-1, :]), axis=2)
                
                coeff_list = []
                biases_list = []
                for i in range(layer_states.shape[0]):
                    self._ridge_embedding.fit(
                        layer_states[i, 0:-1, :], 
                        target_input[i, self.n_drop+1:, :]
                    )
                    coeff_list.append(self._ridge_embedding.coef_.ravel())
                    biases_list.append(self._ridge_embedding.intercept_.ravel())
                layer_repr = np.concatenate((np.vstack(coeff_list), np.vstack(biases_list)), axis=1)
            elif self.mts_rep == 'reservoir':
                # 储备池模型空间表示：用岭回归拟合状态序列的自回归映射
                coeff_list = []
                biases_list = []
                for i in range(layer_states.shape[0]):
                    self._ridge_embedding.fit(
                        layer_states[i, 0:-1, :], 
                        layer_states[i, 1:, :]
                    )
                    coeff_list.append(self._ridge_embedding.coef_.ravel())
                    biases_list.append(self._ridge_embedding.intercept_.ravel())
                layer_repr = np.concatenate((np.vstack(coeff_list), np.vstack(biases_list)), axis=1)
            else:
                raise RuntimeError(f"Invalid representation ID: {self.mts_rep}")

            layer_states_list.append(layer_states)
            layer_repr_list.append(layer_repr)

            # 下一层输入为当前层的序列状态
            current_input = layer_states

        return layer_states_list, layer_repr_list

    def fit(self, X, Y=None, verbose=True):
        r"""训练多专家层叠 RC 模型。

        参数:
        -----------
        X : np.ndarray
            形状为 ``[N, T, V]`` 的数组，表示训练数据。

        Y : np.ndarray
            形状为 ``[N, C]`` 的数组，表示目标值。

        verbose : bool
            如果为 ``True``，打印训练时间。
        """
        time_start = time.time()

        # 逐层前向，获得所有层的状态序列和表示
        layer_states_list, layer_repr_list = self._forward_layers(X, X_original=X)

        # 对最后一层状态进行降维（如果启用）
        if self.dimred_method is not None:
            final_states = layer_states_list[-1]  # [N, T, H_total]
            if self.dimred_method.lower() == 'pca':
                # 矩阵化
                N_samples = final_states.shape[0]
                final_states = final_states.reshape(-1, final_states.shape[2])
                # 变换
                red_states = self._dim_red.fit_transform(final_states)
                # 转换回张量形式
                red_states = red_states.reshape(N_samples, -1, red_states.shape[1])
            elif self.dimred_method.lower() == 'tenpca':
                red_states = self._dim_red.fit_transform(final_states)
            else:
                red_states = final_states
            
            # 基于降维后的状态重新计算最后一层的表示
            if self.mts_rep == 'mean':
                layer_repr_list[-1] = np.mean(red_states, axis=1)
            elif self.mts_rep == 'last':
                layer_repr_list[-1] = red_states[:, -1, :]
            elif self.mts_rep == 'output':
                # 对于 output，使用降维后的状态预测目标
                layer_idx = len(layer_states_list) - 1
                if layer_idx == 0:
                    target_input = X
                else:
                    target_input = layer_states_list[layer_idx - 1]
                
                if self.bidir:
                    target_input = np.concatenate((target_input, target_input[:, ::-1, :]), axis=2)
                
                coeff_list = []
                biases_list = []
                for i in range(red_states.shape[0]):
                    self._ridge_embedding.fit(
                        red_states[i, 0:-1, :], 
                        target_input[i, self.n_drop+1:, :]
                    )
                    coeff_list.append(self._ridge_embedding.coef_.ravel())
                    biases_list.append(self._ridge_embedding.intercept_.ravel())
                layer_repr_list[-1] = np.concatenate((np.vstack(coeff_list), np.vstack(biases_list)), axis=1)
            elif self.mts_rep == 'reservoir':
                # 对于 reservoir，使用降维后的状态计算自回归映射
                coeff_list = []
                biases_list = []
                for i in range(red_states.shape[0]):
                    self._ridge_embedding.fit(
                        red_states[i, 0:-1, :], 
                        red_states[i, 1:, :]
                    )
                    coeff_list.append(self._ridge_embedding.coef_.ravel())
                    biases_list.append(self._ridge_embedding.intercept_.ravel())
                layer_repr_list[-1] = np.concatenate((np.vstack(coeff_list), np.vstack(biases_list)), axis=1)

        # 多层残差式融合：将所有层的表示在特征维拼接
        input_repr = np.concatenate(layer_repr_list, axis=1)  # [N, sum_layers(H_total)]
        self.input_repr = input_repr

        # 训练读出
        if self.readout_type is None:
            # 仅存储表示，用于聚类或可视化
            pass
        elif self.readout_type == 'lin':
            self.readout.fit(input_repr, Y)
        elif self.readout_type == 'svm':
            Ktr = squareform(pdist(input_repr, metric='sqeuclidean'))
            Ktr = np.exp(-self.svm_gamma * Ktr)
            self.readout.fit(Ktr, np.argmax(Y, axis=1))
            self.input_repr_tr = input_repr
        elif self.readout_type == 'mlp':
            self.readout.fit(input_repr, Y)
        else:
            raise RuntimeError('Invalid readout type')

        if verbose:
            tot_time = (time.time() - time_start) / 60
            print(f"Training completed in {tot_time:.2f} min")

    def predict(self, Xte):
        r"""计算样本外（测试）数据的预测。

        参数:
        -----------
        Xte : np.ndarray
            形状为 ``[N, T, V]`` 的数组，表示测试数据。

        返回:
        --------
        pred_class : np.ndarray
            形状为 ``[N]`` 的数组，表示预测的类别。
        """
        # 前向传播，获取各层状态序列和表示
        layer_states_list_te, layer_repr_list_te = self._forward_layers(Xte, X_original=Xte)

        # 对最后一层状态进行降维（如果启用）
        if self.dimred_method is not None:
            final_states_te = layer_states_list_te[-1]  # [N, T, H_total]
            if self.dimred_method.lower() == 'pca':
                # 矩阵化
                N_samples_te = final_states_te.shape[0]
                final_states_te = final_states_te.reshape(-1, final_states_te.shape[2])
                # 变换
                red_states_te = self._dim_red.transform(final_states_te)
                # 转换回张量形式
                red_states_te = red_states_te.reshape(N_samples_te, -1, red_states_te.shape[1])
            elif self.dimred_method.lower() == 'tenpca':
                red_states_te = self._dim_red.transform(final_states_te)
            else:
                red_states_te = final_states_te
            
            # 基于降维后的状态重新计算最后一层的表示
            if self.mts_rep == 'mean':
                layer_repr_list_te[-1] = np.mean(red_states_te, axis=1)
            elif self.mts_rep == 'last':
                layer_repr_list_te[-1] = red_states_te[:, -1, :]
            elif self.mts_rep == 'output':
                # 对于 output，使用降维后的状态预测目标
                layer_idx = len(layer_states_list_te) - 1
                if layer_idx == 0:
                    target_input = Xte
                else:
                    target_input = layer_states_list_te[layer_idx - 1]
                
                if self.bidir:
                    target_input = np.concatenate((target_input, target_input[:, ::-1, :]), axis=2)
                
                coeff_list = []
                biases_list = []
                for i in range(red_states_te.shape[0]):
                    self._ridge_embedding.fit(
                        red_states_te[i, 0:-1, :], 
                        target_input[i, self.n_drop+1:, :]
                    )
                    coeff_list.append(self._ridge_embedding.coef_.ravel())
                    biases_list.append(self._ridge_embedding.intercept_.ravel())
                layer_repr_list_te[-1] = np.concatenate((np.vstack(coeff_list), np.vstack(biases_list)), axis=1)
            elif self.mts_rep == 'reservoir':
                # 对于 reservoir，使用降维后的状态计算自回归映射
                coeff_list = []
                biases_list = []
                for i in range(red_states_te.shape[0]):
                    self._ridge_embedding.fit(
                        red_states_te[i, 0:-1, :], 
                        red_states_te[i, 1:, :]
                    )
                    coeff_list.append(self._ridge_embedding.coef_.ravel())
                    biases_list.append(self._ridge_embedding.intercept_.ravel())
                layer_repr_list_te[-1] = np.concatenate((np.vstack(coeff_list), np.vstack(biases_list)), axis=1)

        # 多层残差式融合：将所有层的表示在特征维拼接
        input_repr_te = np.concatenate(layer_repr_list_te, axis=1)

        if self.readout_type == 'lin':
            logits = self.readout.predict(input_repr_te)
            pred_class = np.argmax(logits, axis=1)
        elif self.readout_type == 'svm':
            Kte = cdist(input_repr_te, self.input_repr_tr, metric='sqeuclidean')
            Kte = np.exp(-self.svm_gamma * Kte)
            pred_class = self.readout.predict(Kte)
        elif self.readout_type == 'mlp':
            pred_class = self.readout.predict(input_repr_te)
            pred_class = np.argmax(pred_class, axis=1)
        else:
            raise RuntimeError('Invalid readout type')

        return pred_class


class MoEStackedRC_model(object):
    r"""混合专家（MoE）层叠储备池 RC 模型。

    在 ``MultiExpertStackedRC_model`` 的基础上，进一步引入双层门控机制（Mixture of Experts, MoE）：

    **双层门控设计：**

    1. **层内门控（Intra-layer Gate）**：每一层拥有一个独立的门控网络，
       对该层内各 Reservoir 专家的表示向量进行自适应加权混合，
       输出该层的混合表示（而非简单拼接）。门控权重由输入数据经过
       线性变换 + Softmax 计算得到，实现"软路由"。

    2. **层间门控（Inter-layer Gate）**：在所有层的混合表示都计算完毕后，
       一个全局门控网络对各层的混合表示进行再次加权融合，
       生成最终送入读出层的特征向量，实现多尺度/多层次信息的自适应整合。

    **前向流程：**

    .. code-block:: text

        输入 X [N, T, V]
            │
            ▼
        Layer 0: E 个专家 Reservoir → E 个状态序列
            │   → 层内门控（基于当前层输入的时间均值）→ 加权混合 → 层0表示 r_0 [N, H]
            │   → 各专家状态序列在特征维拼接，作为下一层输入
            ▼
        Layer 1: E 个专家 Reservoir → E 个状态序列
            │   → 层内门控 → r_1 [N, H]
            ...
            ▼
        Layer L-1: → r_{L-1} [N, H]
            │
            ▼
        层间门控（基于所有层表示的拼接）→ 加权融合 → 最终表示 [N, H]
            │
            ▼
        读出层（lin / svm / mlp）

    **训练策略（两阶段）：**

    1. **预热阶段**：以均匀门控权重跑一遍前向传播，用得到的表示训练读出层，
       获得初始读出权重；
    2. **门控精调阶段**：利用读出权重作为近似梯度信号（Adam 优化），
       监督训练各层内门控和层间门控，再次重新训练读出层。

    **注意事项：**

    - 门控网络为线性层（轻量级 numpy 实现），权重在 ``fit`` 阶段通过 Adam 学习；
    - 若 ``readout_type=None``，模型仅保存最终表示到 ``.input_repr``，
      适合聚类或可视化；
    - 门控网络的输入维度在第一次 ``fit`` 时自动推断，无需手动指定；
    - 可通过 ``get_gate_weights()`` 查看各门控权重，用于可解释性分析。

    参数:
    -----
    **层叠储备池参数:**

    :param n_layers: int (默认 ``2``)
        层叠 Reservoir 的层数。
    :param n_experts: int (默认 ``3``)
        每一层内的并行 Reservoir 专家个数。
    :param reservoir_configs: dict 或 list of dict (默认 ``None``)
        每层 Reservoir 的配置，格式与 ``StackedRC_model`` 一致。
        字典可包含：``n_internal_units``, ``spectral_radius``, ``leak``,
        ``connectivity``, ``input_scaling``, ``noise_level``, ``circle``。
    :param n_drop: int (默认 ``0``)
        每层丢弃的瞬态状态数量。
    :param bidir: bool (默认 ``False``)
        是否使用双向储备池。

    **MoE 门控参数:**

    :param gate_lr: float (默认 ``0.01``)
        门控网络的学习率（Adam 优化器）。
    :param gate_epochs: int (默认 ``200``)
        门控网络的训练轮数。
    :param gate_reg: float (默认 ``1e-4``)
        门控网络权重的 L2 正则化系数。
    :param intra_gate_input: str (默认 ``'mean'``)
        层内门控的输入特征构造方式：

        - ``'mean'``：对当前层输入的时间维取均值 ``[N, V_in]``；
        - ``'last'``：取最后一个时间步 ``[N, V_in]``。

    **降维参数:**

    :param dimred_method: str (默认 ``None``)
        对最后一层专家状态降维（``None`` / ``'pca'`` / ``'tenpca'``）。
    :param n_dim: int (默认 ``None``)
        降维后的维度数。

    **表示参数:**

    :param mts_rep: str (默认 ``'mean'``)
        单个专家的 MTS 表示方式（``'mean'`` / ``'last'`` / ``'output'`` / ``'reservoir'``）。
    :param w_ridge_embedding: float (默认 ``1.0``)
        ``'output'`` / ``'reservoir'`` 表示中岭回归的正则化系数。

    **读出参数:**

    :param readout_type: str (默认 ``'lin'``)
        读出类型（``'lin'`` / ``'svm'`` / ``'mlp'`` / ``None``）。
    :param w_ridge: float (默认 ``1.0``)
        岭回归读出的正则化参数。
    :param mlp_layout: tuple (默认 ``None``)
        MLP 各隐藏层尺寸，例如 ``(64, 32)``。
    :param num_epochs: int (默认 ``None``)
        MLP 训练轮数。
    :param w_l2: float (默认 ``None``)
        MLP L2 正则化权重。
    :param nonlinearity: str (默认 ``None``)
        MLP 激活函数 ``{'relu', 'tanh', 'logistic', 'identity'}``。
    :param svm_gamma: float (默认 ``1.0``)
        SVM RBF 核带宽。
    :param svm_C: float (默认 ``1.0``)
        SVM 正则化参数。
    """

    # ------------------------------------------------------------------
    # 内部辅助类：轻量级线性门控网络（numpy 实现，Adam 优化）
    # ------------------------------------------------------------------
    class _LinearGate:
        """线性 Softmax 门控网络（numpy 实现）。

        将形状为 ``[N, D_in]`` 的输入映射为 ``[N, K]`` 的概率分布（经 softmax），
        用于对 K 个专家/层的表示进行加权混合。
        """

        def __init__(self, d_in, k, lr=0.01, epochs=200, reg=1e-4):
            self.d_in = d_in
            self.k = k
            self.lr = lr
            self.epochs = epochs
            self.reg = reg
            # Xavier 初始化
            scale = np.sqrt(2.0 / (d_in + k))
            self.W = np.random.randn(d_in, k) * scale  # [D_in, K]
            self.b = np.zeros(k)                        # [K]
            # Adam 状态
            self._mW = np.zeros_like(self.W)
            self._vW = np.zeros_like(self.W)
            self._mb = np.zeros_like(self.b)
            self._vb = np.zeros_like(self.b)
            self._t = 0

        @staticmethod
        def _softmax(z):
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)

        def _forward(self, X):
            """返回 logits 和 softmax 门控权重。"""
            logits = X @ self.W + self.b   # [N, K]
            weights = self._softmax(logits)
            return logits, weights

        def _adam_step(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
            self._t += 1
            self._mW = beta1 * self._mW + (1 - beta1) * gW
            self._vW = beta2 * self._vW + (1 - beta2) * gW ** 2
            self._mb = beta1 * self._mb + (1 - beta1) * gb
            self._vb = beta2 * self._vb + (1 - beta2) * gb ** 2
            mW_hat = self._mW / (1 - beta1 ** self._t)
            vW_hat = self._vW / (1 - beta2 ** self._t)
            mb_hat = self._mb / (1 - beta1 ** self._t)
            vb_hat = self._vb / (1 - beta2 ** self._t)
            self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

        def fit(self, X, expert_reprs, Y_onehot=None, readout_W=None,
                concat_mode=False):
            """训练门控网络。

            当提供 ``Y_onehot`` 和 ``readout_W`` 时，以监督方式训练（通过下游分类损失的近似梯度）；
            当仅提供 ``Y_onehot`` 时，使用熵正则化（鼓励权重多样性）；
            否则退化为均匀混合，不训练。

            参数:
                X : [N, D_in] 门控输入特征
                expert_reprs : list of [N, H_k]，K 个专家/层的表示（各 H_k 可不同）
                Y_onehot : [N, C] one-hot 标签（可为 None）
                readout_W : [H_total, C] 下游线性读出权重矩阵（可为 None）
                    - ``concat_mode=False`` 时：H_total = H（所有专家维度相同），
                      混合表示为加权求和 [N, H]；
                    - ``concat_mode=True`` 时：H_total = sum(H_k)，
                      混合表示为加权拼接 [N, sum(H_k)]（层间门控使用）。
                concat_mode : bool (默认 False)
                    是否使用加权拼接（而非加权求和）混合专家表示。
                    层间门控应设为 True，层内门控设为 False。
            """
            if Y_onehot is None:
                # 无监督：不训练
                return

            K = len(expert_reprs)
            N = X.shape[0]
            dims = [e.shape[1] for e in expert_reprs]

            if not concat_mode:
                # 层内门控：所有专家维度相同，加权求和混合
                # 堆叠为 [N, K, H]，使用 einsum 高效计算
                E = np.stack(expert_reprs, axis=1)  # [N, K, H]

                for _ in range(self.epochs):
                    logits, gate_w = self._forward(X)   # gate_w: [N, K]
                    mixed = np.einsum('nk,nkh->nh', gate_w, E)  # [N, H]

                    if readout_W is not None:
                        logits_out = mixed @ readout_W
                        probs_out = self._softmax(logits_out)
                        d_mixed = (probs_out - Y_onehot) / N   # [N, C]
                        d_gate_w = np.einsum('nc,hc,nkh->nk', d_mixed, readout_W, E)
                    else:
                        d_gate_w = gate_w - 1.0 / K

                    s = gate_w
                    d_logits = s * (d_gate_w - (d_gate_w * s).sum(axis=1, keepdims=True))
                    gW = X.T @ d_logits + self.reg * self.W
                    gb = d_logits.sum(axis=0)
                    self._adam_step(gW, gb)
            else:
                # 层间门控：各层维度可不同，加权拼接混合
                # 混合表示 = concat(w_k * r_k for k)，维度 = sum(H_k)
                for _ in range(self.epochs):
                    logits, gate_w = self._forward(X)   # gate_w: [N, K]

                    # 计算加权拼接的混合表示 [N, sum(H_k)]
                    mixed = np.concatenate([
                        gate_w[:, k:k+1] * expert_reprs[k]
                        for k in range(K)
                    ], axis=1)

                    if readout_W is not None:
                        logits_out = mixed @ readout_W
                        probs_out = self._softmax(logits_out)
                        # d_loss/d_final_repr: [N, sum(H_k)]
                        d_final_repr = (probs_out - Y_onehot) / N @ readout_W.T

                        # 对每个专家 k，梯度 d_gate_w[:,k] = sum_h d_final_repr 对应块 * r_k[:,h]
                        d_gate_w = np.zeros((N, K))
                        offset = 0
                        for k in range(K):
                            h_k = dims[k]
                            d_gate_w[:, k] = (d_final_repr[:, offset:offset+h_k] * expert_reprs[k]).sum(axis=1)
                            offset += h_k
                    else:
                        d_gate_w = gate_w - 1.0 / K

                    s = gate_w
                    d_logits = s * (d_gate_w - (d_gate_w * s).sum(axis=1, keepdims=True))
                    gW = X.T @ d_logits + self.reg * self.W
                    gb = d_logits.sum(axis=0)
                    self._adam_step(gW, gb)

        def predict_weights(self, X):
            """返回门控权重 [N, K]。"""
            _, weights = self._forward(X)
            return weights

    # ------------------------------------------------------------------
    # 构造函数
    # ------------------------------------------------------------------
    def __init__(self,
                 # stacked reservoir
                 n_layers=2,
                 n_experts=3,
                 reservoir_configs=None,
                 n_drop=0,
                 bidir=False,
                 # MoE gate
                 gate_lr=0.01,
                 gate_epochs=200,
                 gate_reg=1e-4,
                 intra_gate_input='mean',
                 # dim red
                 dimred_method=None,
                 n_dim=None,
                 # representation
                 mts_rep='mean',
                 w_ridge_embedding=1.0,
                 # readout
                 readout_type='lin',
                 w_ridge=1.0,
                 mlp_layout=None,
                 num_epochs=None,
                 w_l2=None,
                 nonlinearity=None,
                 svm_gamma=1.0,
                 svm_C=1.0):

        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_drop = n_drop
        self.bidir = bidir
        self.gate_lr = gate_lr
        self.gate_epochs = gate_epochs
        self.gate_reg = gate_reg
        self.intra_gate_input = intra_gate_input
        self.dimred_method = dimred_method
        self.mts_rep = mts_rep
        self.readout_type = readout_type
        self.svm_gamma = svm_gamma

        # 初始化降维方法
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)
            elif dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components=n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID')

        # 初始化岭回归嵌入（用于 output / reservoir 表示）
        if mts_rep == 'output' or mts_rep == 'reservoir':
            self._ridge_embedding = Ridge(alpha=w_ridge_embedding, fit_intercept=True)

        # 处理储备池配置
        if reservoir_configs is None:
            if n_layers == 2:
                default_units = [200, 150]
            elif n_layers == 3:
                default_units = [300, 200, 150]
            elif n_layers == 4:
                default_units = [400, 300, 200, 150]
            elif n_layers == 5:
                default_units = [400, 350, 300, 200, 150]
            elif n_layers == 6:
                default_units = [400, 350, 300, 250, 200, 150]
            else:
                base_units = 400
                default_units = [max(base_units - i * 50, 100) for i in range(n_layers)]
            reservoir_configs = [
                {
                    'n_internal_units': units,
                    'spectral_radius': 0.99,
                    'leak': None,
                    'connectivity': 0.3,
                    'input_scaling': 0.2,
                    'noise_level': 0.0,
                    'circle': False
                }
                for units in default_units
            ]
        elif isinstance(reservoir_configs, dict):
            reservoir_configs = [reservoir_configs.copy() for _ in range(n_layers)]
        elif isinstance(reservoir_configs, list):
            if len(reservoir_configs) != n_layers:
                raise ValueError(
                    f"reservoir_configs 列表长度({len(reservoir_configs)})必须等于 n_layers({n_layers})")
        else:
            raise ValueError("reservoir_configs 必须是 dict、list[dict] 或 None")

        # 初始化多层多专家储备池：self._reservoirs[layer_idx][expert_idx]
        self._reservoirs = []
        for config in reservoir_configs:
            layer_reservoirs = []
            for _ in range(n_experts):
                res = Reservoir(
                    n_internal_units=config.get('n_internal_units', 100),
                    spectral_radius=config.get('spectral_radius', 0.99),
                    leak=config.get('leak', None),
                    connectivity=config.get('connectivity', 0.3),
                    input_scaling=config.get('input_scaling', 0.2),
                    noise_level=config.get('noise_level', 0.0),
                    circle=config.get('circle', False)
                )
                layer_reservoirs.append(res)
            self._reservoirs.append(layer_reservoirs)

        # 门控网络占位（维度在 fit 时自动推断）
        self._intra_gates = None   # list[_LinearGate]，长度 = n_layers
        self._inter_gate = None    # _LinearGate

        # 初始化读出层
        if self.readout_type is not None:
            if self.readout_type == 'lin':
                self.readout = Ridge(alpha=w_ridge)
            elif self.readout_type == 'svm':
                self.readout = SVC(C=svm_C, kernel='precomputed')
            elif self.readout_type == 'mlp':
                self.readout = MLPClassifier(
                    hidden_layer_sizes=mlp_layout,
                    activation=nonlinearity,
                    alpha=w_l2,
                    batch_size=32,
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    max_iter=num_epochs,
                    early_stopping=False,
                    validation_fraction=0.0
                )
            else:
                raise RuntimeError('Invalid readout type')

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _build_gate_input(self, current_input):
        """根据 intra_gate_input 构造层内门控的输入特征 [N, D_in]。"""
        if self.intra_gate_input == 'last':
            return current_input[:, -1, :]   # [N, V_in]
        else:  # 默认 'mean'
            return np.mean(current_input, axis=1)  # [N, V_in]

    def _compute_expert_repr(self, states, layer_input):
        """由单个专家的状态序列 [N, T, H] 计算表示向量 [N, repr_dim]。"""
        if self.mts_rep == 'mean':
            return np.mean(states, axis=1)
        elif self.mts_rep == 'last':
            return states[:, -1, :]
        elif self.mts_rep == 'output':
            target = layer_input
            if self.bidir:
                target = np.concatenate((target, target[:, ::-1, :]), axis=2)
            coeff_list, biases_list = [], []
            for i in range(states.shape[0]):
                self._ridge_embedding.fit(
                    states[i, 0:-1, :],
                    target[i, self.n_drop + 1:, :]
                )
                coeff_list.append(self._ridge_embedding.coef_.ravel())
                biases_list.append(self._ridge_embedding.intercept_.ravel())
            return np.concatenate(
                (np.vstack(coeff_list), np.vstack(biases_list)), axis=1)
        elif self.mts_rep == 'reservoir':
            coeff_list, biases_list = [], []
            for i in range(states.shape[0]):
                self._ridge_embedding.fit(
                    states[i, 0:-1, :],
                    states[i, 1:, :]
                )
                coeff_list.append(self._ridge_embedding.coef_.ravel())
                biases_list.append(self._ridge_embedding.intercept_.ravel())
            return np.concatenate(
                (np.vstack(coeff_list), np.vstack(biases_list)), axis=1)
        else:
            raise RuntimeError(f'Invalid representation ID: {self.mts_rep}')

    def _run_layers(self, X, fit_dimred=False):
        """运行所有层的专家，返回各层专家表示列表和拼接状态。

        参数:
            X : [N, T, V] 输入
            fit_dimred : 是否拟合降维器（仅训练时对最后一层使用）

        返回:
            layer_expert_reprs : list of list，[n_layers][n_experts] 每个元素 [N, H]
            layer_all_states   : list of [N, T, H_total]
            layer_gate_inputs  : list of [N, D_in]，层内门控输入特征
        """
        current_input = X
        layer_expert_reprs = []
        layer_all_states = []
        layer_gate_inputs = []

        for layer_idx, layer_reservoirs in enumerate(self._reservoirs):
            is_last = (layer_idx == self.n_layers - 1)
            gate_input = self._build_gate_input(current_input)
            layer_gate_inputs.append(gate_input)

            expert_states_list = []
            expert_repr_list = []

            for reservoir in layer_reservoirs:
                states = reservoir.get_states(
                    current_input, n_drop=self.n_drop, bidir=self.bidir)

                # 可选降维（仅最后一层）
                if self.dimred_method is not None and is_last:
                    if self.dimred_method.lower() == 'pca':
                        N_s = states.shape[0]
                        states_flat = states.reshape(-1, states.shape[2])
                        if fit_dimred:
                            states_flat = self._dim_red.fit_transform(states_flat)
                            fit_dimred = False  # 只对第一个专家 fit，后续 transform
                        else:
                            states_flat = self._dim_red.transform(states_flat)
                        states = states_flat.reshape(N_s, -1, states_flat.shape[1])
                    elif self.dimred_method.lower() == 'tenpca':
                        if fit_dimred:
                            states = self._dim_red.fit_transform(states)
                            fit_dimred = False
                        else:
                            states = self._dim_red.transform(states)

                expert_states_list.append(states)
                expert_repr_list.append(
                    self._compute_expert_repr(states, current_input))

            layer_expert_reprs.append(expert_repr_list)
            all_states = np.concatenate(expert_states_list, axis=2)
            layer_all_states.append(all_states)
            current_input = all_states

        return layer_expert_reprs, layer_all_states, layer_gate_inputs

    def _mix_with_gates(self, layer_expert_reprs, layer_gate_inputs,
                        Y_onehot=None, train_gates=False):
        """使用层内和层间门控混合各专家/层的表示。

        参数:
            layer_expert_reprs : list of list，[n_layers][n_experts] 每个元素 [N, H]
            layer_gate_inputs  : list of [N, D_in]
            Y_onehot           : [N, C]，one-hot 标签（可为 None）
            train_gates        : 是否训练门控网络

        返回:
            final_repr : [N, H] 最终融合表示
            layer_mixed_reprs : list of [N, H]，各层混合表示
        """
        N = layer_gate_inputs[0].shape[0]
        layer_mixed_reprs = []

        for layer_idx in range(self.n_layers):
            gate_input = layer_gate_inputs[layer_idx]
            expert_repr_list = layer_expert_reprs[layer_idx]

            # 懒初始化层内门控
            if self._intra_gates[layer_idx] is None:
                self._intra_gates[layer_idx] = self._LinearGate(
                    d_in=gate_input.shape[1],
                    k=self.n_experts,
                    lr=self.gate_lr,
                    epochs=self.gate_epochs,
                    reg=self.gate_reg
                )

            if train_gates:
                # 层内门控：仅用熵正则（鼓励多样性），不使用下游读出权重
                # 因为层内表示维度与最终融合表示维度不匹配
                self._intra_gates[layer_idx].fit(
                    gate_input, expert_repr_list, Y_onehot, readout_W=None)

            intra_w = self._intra_gates[layer_idx].predict_weights(gate_input)  # [N, E]
            E_reprs = np.stack(expert_repr_list, axis=1)                         # [N, E, H]
            mixed_repr = np.einsum('ne,neh->nh', intra_w, E_reprs)              # [N, H]
            layer_mixed_reprs.append(mixed_repr)

        # 层间门控
        inter_gate_input = np.concatenate(layer_mixed_reprs, axis=1)  # [N, L*H]

        if self._inter_gate is None:
            self._inter_gate = self._LinearGate(
                d_in=inter_gate_input.shape[1],
                k=self.n_layers,
                lr=self.gate_lr,
                epochs=self.gate_epochs,
                reg=self.gate_reg
            )

        if train_gates:
            readout_W = None
            if (self.readout_type == 'lin'
                    and hasattr(self, 'readout')
                    and hasattr(self.readout, 'coef_')):
                readout_W = self.readout.coef_.T
            self._inter_gate.fit(
                inter_gate_input, layer_mixed_reprs, Y_onehot, readout_W,
                concat_mode=True)

        inter_w = self._inter_gate.predict_weights(inter_gate_input)  # [N, L]

        # 各层维度可能不同，采用"加权拼接"：每层表示乘以对应标量权重后拼接
        # 门控权重控制各层的信噪比，最终表示维度 = sum(H_l for l in layers)
        weighted_reprs = [
            inter_w[:, l:l+1] * layer_mixed_reprs[l]   # [N, 1] * [N, H_l] = [N, H_l]
            for l in range(self.n_layers)
        ]
        final_repr = np.concatenate(weighted_reprs, axis=1)  # [N, sum(H_l)]

        return final_repr, layer_mixed_reprs

    def _fit_readout(self, repr_vec, Y):
        """用给定表示训练读出层。"""
        if self.readout_type == 'lin':
            self.readout.fit(repr_vec, Y)
        elif self.readout_type == 'svm':
            Ktr = squareform(pdist(repr_vec, metric='sqeuclidean'))
            Ktr = np.exp(-self.svm_gamma * Ktr)
            self.readout.fit(Ktr, np.argmax(Y, axis=1))
            self.input_repr_tr = repr_vec
        elif self.readout_type == 'mlp':
            self.readout.fit(repr_vec, Y)

    # ------------------------------------------------------------------
    # 公共接口：fit / predict / get_gate_weights
    # ------------------------------------------------------------------
    def fit(self, X, Y=None, verbose=True):
        r"""训练 MoE 层叠 RC 模型。

        采用两阶段训练策略：

        1. **预热阶段**：以均匀门控权重跑一遍前向传播，用得到的表示训练读出层，
           获得初始读出权重作为后续门控训练的监督信号；
        2. **门控精调阶段**：利用初始读出权重作为近似梯度信号，
           以 Adam 优化训练各层内门控和层间门控，
           再次用精调后的最终表示重新训练读出层。

        参数:
        -----------
        X : np.ndarray
            形状为 ``[N, T, V]`` 的数组，表示训练数据。
        Y : np.ndarray
            形状为 ``[N, C]`` 的数组，表示目标值（one-hot 编码）。
        verbose : bool
            如果为 ``True``，打印训练时间。
        """
        time_start = time.time()

        # 初始化门控网络列表（占位）
        self._intra_gates = [None] * self.n_layers
        self._inter_gate = None

        # ======== 阶段 1：预热（均匀门控）========
        layer_expert_reprs, _, layer_gate_inputs = self._run_layers(
            X, fit_dimred=(self.dimred_method is not None))

        # 使用均匀门控生成预热表示
        N = X.shape[0]
        layer_mixed_reprs_warm = []
        for layer_idx in range(self.n_layers):
            E_reprs = np.stack(layer_expert_reprs[layer_idx], axis=1)
            uniform_w = np.ones((N, self.n_experts)) / self.n_experts
            layer_mixed_reprs_warm.append(
                np.einsum('ne,neh->nh', uniform_w, E_reprs))

        inter_gate_input_warm = np.concatenate(layer_mixed_reprs_warm, axis=1)
        uniform_inter = np.ones((N, self.n_layers)) / self.n_layers
        # 各层维度可能不同，采用加权拼接（均匀权重）
        warm_repr = np.concatenate([
            uniform_inter[:, l:l+1] * layer_mixed_reprs_warm[l]
            for l in range(self.n_layers)
        ], axis=1)

        # 初始化所有门控网络（确保 _intra_gates 和 _inter_gate 不为 None）
        for layer_idx in range(self.n_layers):
            gate_input = layer_gate_inputs[layer_idx]
            self._intra_gates[layer_idx] = self._LinearGate(
                d_in=gate_input.shape[1],
                k=self.n_experts,
                lr=self.gate_lr,
                epochs=self.gate_epochs,
                reg=self.gate_reg
            )
        self._inter_gate = self._LinearGate(
            d_in=inter_gate_input_warm.shape[1],
            k=self.n_layers,
            lr=self.gate_lr,
            epochs=self.gate_epochs,
            reg=self.gate_reg
        )

        # 用预热表示训练读出层，获得初始权重
        if self.readout_type is not None and Y is not None:
            self._fit_readout(warm_repr, Y)

        # ======== 阶段 2：门控精调 ========
        # 使用读出权重作为近似梯度，监督训练门控
        _, _ = self._mix_with_gates(
            layer_expert_reprs, layer_gate_inputs,
            Y_onehot=Y, train_gates=True)

        # 用精调后的门控生成最终表示，重新训练读出层
        final_repr, _ = self._mix_with_gates(
            layer_expert_reprs, layer_gate_inputs,
            Y_onehot=None, train_gates=False)

        self.input_repr = final_repr

        if self.readout_type is not None and Y is not None:
            self._fit_readout(final_repr, Y)

        if verbose:
            tot_time = (time.time() - time_start) / 60
            print(f"Training completed in {tot_time:.2f} min")

    def predict(self, Xte):
        r"""计算测试数据的预测类别。

        参数:
        -----------
        Xte : np.ndarray
            形状为 ``[N, T, V]`` 的数组，表示测试数据。

        返回:
        --------
        pred_class : np.ndarray
            形状为 ``[N]`` 的数组，表示预测的类别。
        """
        if self._intra_gates is None or self._inter_gate is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        layer_expert_reprs, _, layer_gate_inputs = self._run_layers(
            Xte, fit_dimred=False)

        input_repr_te, _ = self._mix_with_gates(
            layer_expert_reprs, layer_gate_inputs,
            Y_onehot=None, train_gates=False)

        if self.readout_type == 'lin':
            logits = self.readout.predict(input_repr_te)
            pred_class = np.argmax(logits, axis=1)
        elif self.readout_type == 'svm':
            Kte = cdist(input_repr_te, self.input_repr_tr, metric='sqeuclidean')
            Kte = np.exp(-self.svm_gamma * Kte)
            pred_class = self.readout.predict(Kte)
        elif self.readout_type == 'mlp':
            pred_class = self.readout.predict(input_repr_te)
            pred_class = np.argmax(pred_class, axis=1)
        else:
            raise RuntimeError("readout_type 为 None，请使用 .input_repr 进行聚类。")

        return pred_class

    def get_gate_weights(self, X):
        r"""返回给定输入样本的门控权重，用于可解释性分析。

        参数:
        -----------
        X : np.ndarray
            形状为 ``[N, T, V]`` 的数组。

        返回:
        --------
        intra_weights : list of np.ndarray
            长度为 ``n_layers``，每个元素形状为 ``[N, n_experts]``，
            表示各层的层内专家门控权重（各行归一化到 1）。
        inter_weights : np.ndarray
            形状为 ``[N, n_layers]``，表示层间门控权重（各行归一化到 1）。

        示例:
        --------
        >>> intra_w, inter_w = model.get_gate_weights(X_test)
        >>> print("层0专家权重（前3个样本）：", intra_w[0][:3])
        >>> print("层间权重（前3个样本）：", inter_w[:3])
        """
        if self._intra_gates is None or self._inter_gate is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        layer_expert_reprs, _, layer_gate_inputs = self._run_layers(
            X, fit_dimred=False)

        intra_weights = []
        layer_mixed_reprs = []

        for layer_idx in range(self.n_layers):
            gate_input = layer_gate_inputs[layer_idx]
            intra_w = self._intra_gates[layer_idx].predict_weights(gate_input)
            intra_weights.append(intra_w)

            E_reprs = np.stack(layer_expert_reprs[layer_idx], axis=1)
            mixed_repr = np.einsum('ne,neh->nh', intra_w, E_reprs)
            layer_mixed_reprs.append(mixed_repr)

        inter_gate_input = np.concatenate(layer_mixed_reprs, axis=1)
        inter_weights = self._inter_gate.predict_weights(inter_gate_input)

        return intra_weights, inter_weights

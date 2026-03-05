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
    - 目前支持的表示方式：``'mean'``、``'last'``；暂不支持 ``'output'`` 与 ``'reservoir'``。

    参数大体与 ``StackedRC_model`` 保持一致，额外增加：

    :param n_experts: int (默认 ``3``)
        每一层的并行 Reservoir 专家个数。
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

        if self.dimred_method is not None:
            raise RuntimeError("当前版本的 MultiExpertStackedRC_model 尚未实现降维（dimred_method 必须为 None）")

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

    def _forward_layers(self, X):
        """内部函数：多层多专家前向传播，返回各层的序列状态和层级表示。

        返回:
            layer_states_list: list，每个元素是该层拼接后的状态序列 [N, T, H_total]
            layer_repr_list: list，每个元素是该层的整体表示 [N, H_total]
        """
        current_input = X  # [N, T, V] 或上一层的 [N, T, H_total]
        layer_states_list = []
        layer_repr_list = []

        for layer_reservoirs in self._reservoirs:
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
            else:
                raise RuntimeError("MultiExpertStackedRC_model 当前仅支持 mts_rep 为 'mean' 或 'last'")

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

        # 逐层前向，获得所有层的表示
        _, layer_repr_list = self._forward_layers(X)

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
        # 前向传播，获取各层表示
        _, layer_repr_list_te = self._forward_layers(Xte)
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

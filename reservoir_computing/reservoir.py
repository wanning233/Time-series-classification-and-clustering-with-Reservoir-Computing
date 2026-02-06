import numpy as np
from scipy import sparse

class Reservoir(object):    
    r"""
        构建储备池并计算内部状态序列。
        
        参数:
        ------------
        n_internal_units : int (默认 ``100``)
            储备池中的处理单元数量。
        spectral_radius : float (默认 ``0.99``)
            储备池连接权重矩阵的最大特征值。
            为确保回声状态属性，设置 ``spectral_radius <= leak <= 1``)
        leak : float (默认 ``None``)
            储备池状态更新中的泄漏量。
            如果为 ``None`` 或 ``1.0``，则不使用泄漏。
        connectivity : float (默认 ``0.3``)
            非零连接权重的百分比。
            在圆形储备池中不使用。
        input_scaling : float (默认 ``0.2``)
            输入连接权重的缩放因子。
            注意输入权重是从 ``{-1,1}`` 中随机抽取的。
        noise_level : float (默认 ``0.0``)
            在状态更新中注入的高斯噪声的标准差。
        circle : bool (默认 ``False``)
            生成具有圆形拓扑的确定性储备池，其中每个连接具有相同的权重。
        """

    def __init__(self, 
                 n_internal_units=100, 
                 spectral_radius=0.99, 
                 leak=None,
                 connectivity=0.3, 
                 input_scaling=0.2, 
                 noise_level=0.0, 
                 circle=False):
       
        # 初始化超参数
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak

        # 输入权重取决于输入大小：在提供数据时设置
        self._input_weights = None

        # 生成内部权重
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                    n_internal_units,
                    spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units,
                connectivity,
                spectral_radius)


    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        """生成具有圆形拓扑的内部权重。
        """
        
        # 构建具有圆形拓扑的储备池
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = 1.0
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = 1.0
            
        # 调整谱半径
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius 
                
        return internal_weights
    
    
    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        """生成具有稀疏、均匀随机拓扑的内部权重。
        """

        # 生成稀疏、均匀分布的权重
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # 确保非零值在 [-0.5, 0.5] 范围内均匀分布
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # 调整谱半径
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius       

        return internal_weights


    def _compute_state_matrix(self, X, n_drop=0, previous_state=None):
        """计算输入数据 X 上的储备池状态。
        """

        N, T, _ = X.shape
        if previous_state is None:
            previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # 存储
        if T - n_drop > 0:
            window_size = T - n_drop
        else:
            window_size = T
        state_matrix = np.empty((N, window_size, self._n_internal_units), dtype=float)

        for t in range(T):
            current_input = X[:, t, :]

            # 计算状态
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)

            # 添加噪声
            state_before_tanh += np.random.rand(self._n_internal_units, N)*self._noise_level

            # 应用非线性函数和泄漏（可选）
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak)*previous_state + np.tanh(state_before_tanh).T

            # 存储丢弃期之后的所有状态
            if T - n_drop > 0 and t > n_drop - 1:
                state_matrix[:, t - n_drop, :] = previous_state
            elif T - n_drop <= 0:
                state_matrix[:, t, :] = previous_state

        return state_matrix


    def get_states(self, X, n_drop=0, bidir=True, initial_state=None):
        r"""
        计算储备池状态并返回它们。

        参数:
        ------------
        X : np.ndarray
            时间序列，形状为 ``[N,T,V]`` 的3维数组，其中 ``N`` 是时间序列的数量，
            ``T`` 是每个时间序列的长度，``V`` 是每个时间点的变量数量。
        n_drop : int (默认为 ``0``)
            冲刷期，即由于瞬态阶段而丢弃的初始样本数量。
        bidir : bool (默认为 ``True``)
            如果为 ``True``，使用双向储备池
        initial_state : np.ndarray (默认为 ``None``)
            将储备池的第一个状态初始化为给定值。
            如果为 ``None``，初始状态为零向量。 

        返回:
        ------------
        states : np.ndarray
            储备池状态，形状为 ``[N,T,n_internal_units]`` 的3维数组，其中 ``N`` 是时间序列的数量，
            ``T`` 是每个时间序列的长度，``n_internal_units`` 是储备池中处理单元的数量。
        """

        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = (2.0*np.random.binomial(1, 0.5 , [self._n_internal_units, V]) - 1.0)*self._input_scaling

        # 计算储备池状态序列
        states = self._compute_state_matrix(X, n_drop, previous_state=initial_state)
    
        # 时间反转输入上的储备池状态
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states
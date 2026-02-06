import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def compute_test_scores(pred_class, Yte):
    """
    计算分类准确率和F1分数的包装函数

    参数:
    -----------
    pred_class : np.ndarray
        预测的类别标签
    Yte : np.ndarray
        真实的类别标签

    返回:
    --------
    accuracy : float
        分类准确率
    f1 : float
        F1分数
    """
    
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)
    if Yte.shape[1] > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')

    return accuracy, f1


def make_forecasting_dataset(X,
        horizon,
        test_percent = 0.15, 
        val_percent = 0.0, 
        scaler = None):
    r"""
    此函数执行以下操作：

    1. 将数据集分割为训练集、验证集和测试集
    2. 将目标数据移动'horizon'步以创建预测问题
    3. 标准化数据

    参数:
    -----------
    X : np.ndarray
        输入数据
    horizon : int
        预测范围
    test_percent : float
        用于测试的数据百分比
    val_percent : float
        用于验证的数据百分比
        如果为0，则不创建验证集
    scaler : sklearn.preprocessing中的缩放器对象
        用于标准化数据的缩放器对象
        如果为None，则创建StandardScaler

    返回:
    --------
    Xtr : np.ndarray
        训练输入数据
    Ytr : np.ndarray
        训练目标数据
    Xte : np.ndarray 
        测试输入数据
    Yte : np.ndarray
        测试目标数据
    scaler : sklearn.preprocessing中的缩放器对象
        用于标准化数据的缩放器对象
    Xval : np.ndarray (可选)
        验证输入数据
    Yval : np.ndarray (可选)
        验证目标数据
    """
    n_data, _ = X.shape

    n_te = np.ceil(test_percent*n_data).astype(int)
    n_val = np.ceil(val_percent*n_data).astype(int)
    n_tr = n_data - n_te - n_val

    # 将数据集分割为训练集、验证集和测试集
    tr = X[:n_tr, :]
    te = X[-n_te:, :]
    if n_val > 0:
        val = X[n_tr:-n_te, :]

    # 移动目标数据以创建预测问题
    Xtr = tr[:-horizon,:]
    Ytr = tr[horizon:,:]
    Xte = te[:-horizon,:]
    Yte = te[horizon:,:]
    if n_val > 0:
        Xval = val[:-horizon,:]
        Yval = val[horizon:,:]

    # 如果未提供缩放器，则定义缩放器
    if scaler is None:
        scaler = StandardScaler()

    # 在训练集上拟合缩放器
    Xtr = scaler.fit_transform(Xtr)

    # 转换其余数据
    Ytr = scaler.transform(Ytr)
    Xte = scaler.transform(Xte)
    if n_val > 0:
        Xval = scaler.transform(Xval)
    
    # 添加常数输入
    Xtr = np.concatenate((Xtr,np.ones((Xtr.shape[0],1))),axis=1)
    Xte = np.concatenate((Xte,np.ones((Xte.shape[0],1))),axis=1)
    if n_val > 0:
        Xval = np.concatenate((Xval,np.ones((Xval.shape[0],1))),axis=1)

    if n_val > 0:
        return Xtr, Ytr, Xte, Yte, Xval, Yval, scaler
    else:
        return Xtr, Ytr, Xte, Yte, scaler
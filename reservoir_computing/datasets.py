import requests
import numpy as np
from io import BytesIO
import warnings
import collections
from scipy.integrate import solve_ivp

class ClfLoader:
    r"""
    用于下载和加载时间序列分类数据集的类。
    """
    def __init__(self) -> None:
        self.datasets = {
            'AtrialFibrillation': ('https://zenodo.org/records/10852712/files/AF.npz?download=1', 'Multivariate time series classification.\nSamples: 5008 (4823 training, 185 test)\nFeatures: 2\nClasses: 3\nTime series length: 45'),
            'ArabicDigits': ('https://zenodo.org/records/10852747/files/ARAB.npz?download=1', 'Multivariate time series classification.\nSamples: 8800 (6600 training, 2200 test)\nFeatures: 13\nClasses: 10\nTime series length: 93'),
            'Auslan': ('https://zenodo.org/records/10839959/files/Auslan.npz?download=1', 'Multivariate time series classification.\nSamples: 2565 (1140 training, 1425 test)\nFeatures: 22\nClasses: 95\nTime series length: 136'),
            'CharacterTrajectories': ('https://zenodo.org/records/10852786/files/CHAR.npz?download=1', 'Multivariate time series classification.\nSamples: 2858 (300 training, 2558 test)\nFeatures: 3\nClasses: 20\nTime series length: 205'),
            'CMUsubject16': ('https://zenodo.org/records/10852831/files/CMU.npz?download=1', 'Multivariate time series classification.\nSamples: 58 (29 training, 29 test)\nFeatures: 62\nClasses: 2\nTime series length: 580'),
            'ECG2D': ('https://zenodo.org/records/10839881/files/ECG_2D.npz?download=1', 'Multivariate time series classification.\nSamples: 200 (100 training, 100 test)\nFeatures: 2\nClasses: 2\nTime series length: 152'),
            'Japanese_Vowels': ('https://zenodo.org/records/10837602/files/Japanese_Vowels.npz?download=1', 'Multivariate time series classification.\nSamples: 640 (270 training, 370 test)\nFeatures: 12\nClasses: 9\nTime series length: 29'),
            'KickvsPunch': ('https://zenodo.org/records/10852865/files/KickvsPunch.npz?download=1', 'Multivariate time series classification.\nSamples: 26 (16 training, 10 test)\nFeatures: 62\nClasses: 2\nTime series length: 841'),
            'Libras': ('https://zenodo.org/records/10852531/files/LIB.npz?download=1', 'Multivariate time series classification.\nSamples: 360 (180 training, 180 test)\nFeatures: 2\nClasses: 15\nTime series length: 45'),
            'NetFlow': ('https://zenodo.org/records/10840246/files/NET.npz?download=1', 'Multivariate time series classification.\nSamples: 1337 (803 training, 534 test)\nFeatures: 4\nClasses: 2\nTime series length: 997'),
            'RobotArm': ('https://zenodo.org/records/10852893/files/Robot.npz?download=1', 'Multivariate time series classification.\nSamples: 164 (100 training, 64 test)\nFeatures: 6\nClasses: 5\nTime series length: 15'),
            'UWAVE': ('https://zenodo.org/records/10852667/files/UWAVE.npz?download=1', 'Multivariate time series classification.\nSamples: 628 (200 training, 428 test)\nFeatures: 3\nClasses: 8\nTime series length: 315'),
            'Wafer': ('https://zenodo.org/records/10839966/files/Wafer.npz?download=1', 'Multivariate time series classification.\nSamples: 1194 (298 training, 896 test)\nFeatures: 6\nClasses: 2\nTime series length: 198'),
            'Chlorine': ('https://zenodo.org/records/10840284/files/CHLO.npz?download=1', 'Univariate time series classification.\nSamples: 4307 (467 training, 3840 test)\nFeatures: 1\nClasses: 3\nTime series length: 166'), 
            'Phalanx': ('https://zenodo.org/records/10852613/files/PHAL.npz?download=1', 'Univariate time series classification.\nSamples: 539 (400 training, 139 test)\nFeatures: 1\nClasses: 3\nTime series length: 80'),
            'SwedishLeaf': ('https://zenodo.org/records/10840000/files/SwedishLeaf.npz?download=1', 'Univariate time series classification.\nSamples: 1125 (500 training, 625 test)\nFeatures: 1\nClasses: 15\nTime series length: 128'),
        }

    def available_datasets(self, details=False):
        r"""
        打印可用的数据集。

        参数:
        -----------
        details : bool
            如果为True，打印数据集的描述。

        返回:
        --------
        None
        """
        print("Available datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias):
        r"""
        下载并加载数据集。

        参数:
        -----------
        alias : str
            要下载的数据集的别名。

        返回:
        --------
        Xtr : np.ndarray
            训练数据
        Ytr : np.ndarray
            训练标签
        Xte : np.ndarray
            测试数据
        Yte : np.ndarray
            测试标签
        """

        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        url, _ = self.datasets[alias]
        response = requests.get(url)
        if response.status_code == 200:

            data = np.load(BytesIO(response.content))
            Xtr = data['Xtr']  # 形状为 [N,T,V]
            if len(Xtr.shape) < 3:
                Xtr = np.atleast_3d(Xtr)
            Ytr = data['Ytr']  # 形状为 [N,1]
            Xte = data['Xte']
            if len(Xte.shape) < 3:
                Xte = np.atleast_3d(Xte)
            Yte = data['Yte']
            n_classes_tr = len(np.unique(Ytr))
            n_classes_te = len(np.unique(Yte))
            if n_classes_tr != n_classes_te:
                warnings.warn(f"Number of classes in training and test sets do not match for {alias} dataset.")
            print(f"Loaded {alias} dataset.\nNumber of classes: {n_classes_tr}\nData shapes:\n  Xtr: {Xtr.shape}\n  Ytr: {Ytr.shape}\n  Xte: {Xte.shape}\n  Yte: {Yte.shape}")

            return (Xtr, Ytr, Xte, Yte)
        else:
            print(f"Failed to download {alias} dataset.")
            return None


class PredLoader():
    """
    用于下载和加载时间序列预测数据集的类。
    """
    def __init__(self) -> None:
        self.datasets = {
            'ElecRome': ('https://zenodo.org/records/10910985/files/Elec_Rome.npz?download=1', 'Univariate time series forecasting.\nLength: 137376\nFeatures: 1'),
            'CDR': ('https://zenodo.org/records/10911142/files/CDR.npz?download=1', 'Multivariate time series forecasting.\nLength: 3336\nFeatures: 8'),
        }

    def available_datasets(self, details=False):
        """
        打印可用的数据集。

        参数:
        -----------
        details : bool
            如果为True，打印数据集的描述。

        返回:
        --------
        None
        """
        print("Available datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias) -> np.ndarray:
        """
        下载并加载数据集。

        参数:
        -----------
        alias : str
            要下载的数据集的别名。

        返回:
        --------
        X : np.ndarray
            时间序列数据
        """
        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        url, _ = self.datasets[alias]
        response = requests.get(url)
        if response.status_code == 200:

            data = np.load(BytesIO(response.content))
            X = data['X']
            print(f"Loaded {alias} dataset.\nData shape:\n  X: {X.shape}")

            return X

        else:
            print(f"Failed to download {alias} dataset.")
            return None


def mackey_glass(sample_len=1000, tau=17, delta_t=1, seed=None, n_samples = 1):
    r"""生成Mackey Glass时间序列。 

        参数:
        -----------
        sample_len : int (默认 ``1000``)
            时间序列的长度（时间步数）。
        tau : int (默认 ``17``)
            MG系统的延迟。常用值为tau=17（轻度混沌）
            和tau=30（中度混沌）。
        delta_t : int (默认 ``1``)
            仿真的时间步长。
        seed : int 或 None (默认 ``None``)
            随机数生成器的种子。可用于每次生成相同的时间序列。
        n_samples : int (默认 ``1``)
            要生成的样本数量。

        返回:
        --------
        np.ndarray | list
            生成的Mackey-Glass时间序列。
            如果n_samples为1，返回单个数组。否则返回列表。
    """
    np.random.seed(seed)
    history_len = tau * delta_t 
    
    # 初始条件
    timeseries = 1.2
    
    samples = []
    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # 为时间序列预分配数组
        inp = np.zeros((sample_len,1))
        
        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries
        
        # 通过tanh压缩时间序列
        inp = np.tanh(inp - 1)
        samples.append(inp)

    if n_samples == 1:
        return samples[0]
    else:
        return samples


def mso(T=1000, N=10, seed=None, freq=0.5):
    r"""通过组合具有不可通约周期的正弦波生成多重正弦波振荡器（MSO）时间序列。
    要组合的正弦波是随机选择的。

    参数:
    -----------
    T : int (默认 ``1000``)
        时间步数。
    N : int (默认 ``10``)
        要组合的正弦波的最大数量。
    seed : int 或 None (默认 ``None``)
        随机数生成器的种子。
    freq : float (默认 ``0.5``)
        正弦波的频率。

    返回:
    --------
    np.ndarray
        MSO时间序列。
    """
    np.random.seed(seed)

    t = np.arange(T * freq, step=freq)
    print(f"MSO - signal frequencies:")
    print(f"  min period: {2 * np.pi * (1 / freq):.2f}")
    print(f"  max period: {np.exp((N - 1) / N) * 2 * np.pi * (1 / freq):.2f}")
    x_t = np.arange(N)
    base_sinusoids = np.sin(1 / np.exp(x_t / N)[:, None] @ t[None])
    
    mixer = np.random.choice([0, 1], size=(1,N), p=[0.5, 0.5])
    np.random.seed(None)
    X = mixer@base_sinusoids

    return X.T


def _lorenz_system(t, y, sigma, rho, beta):
    """Lorenz微分方程组。
    """
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def lorenz(sigma=10, rho=28, beta=8/3, y0=[0, -0.01, 9.0], t_span=[0, 100], dt=1e-3):
    r"""生成Lorenz吸引子时间序列。
    
    参数:
    -----------
    sigma : float (默认 ``10``)
        Lorenz系统的第1个参数。
    rho : float (默认 ``28``)
        Lorenz系统的第2个参数。
    beta : float (默认 ``8/3``)
        Lorenz系统的第3个参数。
    y0 : list (默认 ``[0, -0.01, 9.0]``)
        Lorenz系统的初始条件。
    t_span : list (默认 ``[0, 100]``)
        仿真的时间跨度。
    dt : float (默认 ``1e-3``)
        仿真的时间步长。

    返回:
    --------
    np.ndarray
        Lorenz时间序列。
    """
    t = np.linspace(t_span[0], t_span[1], int(1/dt))
    solution = solve_ivp(_lorenz_system, t_span, y0, args=(sigma, rho, beta), t_eval=t)
    return solution.y.T


def _rossler_system(t, y, a, b, c):
    """Rossler微分方程组。
    """
    x, y, z = y
    dxdt = -y - z
    dydt = x + a*y
    dzdt = b + z*(x - c)
    return [dxdt, dydt, dzdt]


def rossler(a=0.2, b=0.2, c=5.7, y0=[0.5, 0.5, 0.5], t_span=[0, 200], dt=1e-3):
    r"""生成Rossler吸引子时间序列。
    
    参数:
    -----------
    a : float (默认 ``0.2``)
        Rossler系统的第1个参数。
    b : float (默认 ``0.2``)
        Rossler系统的第2个参数。
    c : float (默认 ``5.7``)
        Rossler系统的第3个参数。
    y0 : list (默认 ``[0, 0.1, 0]``)
        Rossler系统的初始条件。
    t_span : list (默认 ``[0, 100]``)
        仿真的时间跨度。
    dt : float (默认 ``1e-3``)
        仿真的时间步长。

    返回:
    --------
    np.ndarray
        Rossler时间序列。
    """
    t = np.linspace(t_span[0], t_span[1], int(1/dt))
    solution = solve_ivp(_rossler_system, t_span, y0, args=(a, b, c), t_eval=t)
    return solution.y.T


class SynthLoader:
    """
    用于生成合成时间序列的类。
    """

    def __init__(self) -> None:
        self.datasets = {
            'MG': (mackey_glass, 'Generate the Mackey Glass time-series'),
            'MSO': (mso, 'Generate the Multiple Superimposed Oscillator time-series'),
            'Lorenz': (lorenz, 'Generate the Lorenz attractor time-series'),
            'Rossler': (rossler, 'Generate the Rossler attractor time-series'),
        }

    def available_datasets(self, details=False):
        """
        打印可用的合成数据集。

        返回:
        --------
        None
        """
        print("Available synthetic datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias, **kwargs):
        """
        生成合成时间序列。

        参数:
        -----------
        alias : str
            要生成的合成数据集的别名。
        kwargs : dict
            合成数据集的附加参数。

        返回:
        --------
        np.ndarray
            合成时间序列。
        """
        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        generator, _ = self.datasets[alias]
        X = generator(**kwargs)
        print(f"Generated {alias} dataset.\nData shape:\n  X: {X.shape}")

        return X


if __name__ == '__main__':
    # 示例用法（分类）
    downloader = ClfLoader()
    downloader.available_datasets(details=False)  # 打印可用数据集
    Xtr, Ytr, Xte, Yte = downloader.get_data('Libras')  # 下载数据集并返回数据

    # 示例用法（预测）
    downloader = PredLoader()
    downloader.available_datasets(details=False)  # 打印可用数据集
    X = downloader.get_data('CDR')  # 下载数据集并返回数据

    # 示例用法（合成）
    synth = SynthLoader()
    synth.available_datasets()  # 打印可用数据集
    Xs = synth.get_data('Lorenz')  # 生成合成时间序列
# import sys
# sys.path.insert(0, '/home/manh/Downloads/Core')

import numpy as np
from utils.utility import euclidean_cdist


class Dcmeans():

    def __init__(self, n_clusters: int, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000):
        if m <= 1:
            raise RuntimeError('m>1')
        self.__n_clusters = n_clusters
        self.__m = m
        self.__epsilon = epsilon
        self.__max_iter = max_iter

    def __numpy_one(self, n_samples: int) -> np.ndarray:
        return np.ones((n_samples, self.__n_clusters))

    # Khởi tạo ma trận thành viên
    def __init_membership(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if seed > 0:
            np.random.seed(seed=seed)
        if self.__m == 1:
            return self.__numpy_one(n_samples=n_samples)
        U0 = np.random.rand(n_samples, self.__n_clusters)
        # U0 = self.__numpy_one(n_samples=n_samples)
        return U0 / U0.sum(axis=1)[:, None]

    # Cập nhật ma trận tâm cụm
    def __update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        # Nhân ma trận X với từng độ thuộc của nó
        # Tổng kết quả cho toàn bộ các điểm của mỗi tâm cụm
        _umT = (membership ** self.__m).T
        V = ((_umT[:, :, None]) * data).sum(axis=1)
        # Chia mỗi tâm cụm cho tổng giá trị độ thuộc của các điểm thuộc trọng tâm
        return V / ((_umT).sum(axis=1)[:, None])

    # Cập nhật ma trận độ thuộc
    def calculate_membership(self, distances: np.ndarray) -> np.ndarray:
        if self.__m == 1:
            return self.__numpy_one(n_samples=len(distances.shape))
        power = 2 / (self.__m - 1)
        U = distances[:, :, None] * ((1 / distances)[:, None, :])
        U = (U ** power).sum(axis=2)
        return 1 / U

    def update_membership(self, data: np.ndarray, centroids: np.ndarray):
        sdistances = euclidean_cdist(data, centroids)  # Khoảng cách Euclidean giữa data và centroids
        return self.calculate_membership(sdistances)

    def fit(self, data: np.ndarray, seed: int = 0) -> tuple:
        # Initialize membership matrix
        u = self.__init_membership(n_samples=len(data), seed=seed)
        # Initialize centroids
        # _n, n_features = data.shape
        # v = np.random.rand(self._n_clusters, n_features)
        for step in range(self.__max_iter):
            old_u = u.copy()
            v = self.__update_centroids(data, old_u)
            u = self.update_membership(data, v)
            # if np.linalg.norm(u - old_u) < self._epsilon:
            #     break
            if (np.abs(u - old_u)).max(axis=(0, 1)) < self.__epsilon:
                break
        return u, v, step + 1

    # Ma trận độ thuộc ra nhãn (giải mờ)
    @staticmethod
    def membership2labels(membership: np.ndarray) -> np.ndarray:
        return np.argmax(membership, axis=1)

    # Dự đoán 1 điểm mới thuộc nhãn nào
    def predict_label_index(self, new_point: np.ndarray, centroids: np.ndarray):
        _X = np.array([new_point])
        _U = self.update_membership(_X, centroids)
        # print('_U', _U)
        return self.membership2labels(_U)[0]

# FCM cục bộ tại mỗi datasite và thực hiện bán giám sát, cộng tác các datasite
# import sys
# sys.path.insert(0, '/home/manh/Downloads/Core/fuzzy')
import numpy as np
from numpy import ndarray
from .cmeans import Dcmeans
from utils.utility import euclidean_cdist

class SSFCM:
    def __init__(self, n_clusters: int, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000):
        self.__n_clusters = n_clusters
        self.__m = m
        self.__local_data = None
        self.__fcm = Dcmeans(n_clusters=n_clusters, m=m, epsilon=epsilon, max_iter=max_iter)
        self.__membership = None
        self.__centroids = None
        self.__membership_bar = None

    # Khởi tạo điểm dữ liệu có nhãn
    def __init_membership_bar(self, n_samples: int, labels: np.ndarray = None):
        u_bar = np.zeros((n_samples, self.__n_clusters))
        if labels is not None:
            for i, label in enumerate(labels):
                if label is not None:
                    u_bar[i, label] = 1
        # print('U_bar:',u_bar)
        return u_bar

    @property
    def membership_bar(self) -> np.ndarray:
        return self.__membership_bar

    @property
    def membership(self) -> np.ndarray:
        return self.__membership

    @property
    def centroids(self) -> np.ndarray:
        return self.__centroids

    @centroids.setter
    def centroids(self, value: np.ndarray):
        self.__centroids = value
    
    @membership.setter
    def membership(self, value: np.ndarray):
        self.__membership = value

    @property
    def local_data(self) -> np.ndarray:
        return self.__local_data
    @property
    def n_clusters(self) -> int:
        return self.__n_clusters

    def calculate_membership(self, distances: np.ndarray) -> np.ndarray:
        return self.__fcm.calculate_membership(distances=distances)

    def fit(self, data: np.ndarray, labels: np.ndarray = None, seed: int = 0):
        self.__local_data = data
        self.__membership_bar = self.__init_membership_bar(n_samples=len(data), labels=labels)
        self.__membership, self.__centroids, __step = self.__fcm.fit(data=data, seed=seed)

    def predict(self, new_data: np.ndarray) -> np.ndarray:
        _u = self.__fcm.update_membership(new_data, self.__centroids)
        return self.__fcm.membership2labels(_u)


class CollaborativeSCFCM:
    def __init__(self, n_clusters, m, epsilon, max_iter, beta, n_sites):
        self.__n_clusters = n_clusters
        self.__n_sites = n_sites
        self.__m = m
        self.beta = beta
        self.__epsilon = epsilon
        self.__max_iter = max_iter
        self.__datasites = [SSFCM(n_clusters=n_clusters, m=m, epsilon=epsilon, max_iter=max_iter) for _ in range(n_sites)]
        self.betas = [beta] * n_sites  # Khởi tạo beta với giá trị ban đầu


    def fit(self, data: np.ndarray, labeled_data: np.ndarray = None, seed: int = 0) -> tuple:
        datas = np.array_split(data, self.__n_sites)
        # Giai đoạn 1: Phân cụm cục bộ trên từng data site
        for i, datasite in enumerate(self.__datasites):
            datasite.fit(data=datas[i], labels=labeled_data[i] if labeled_data is not None else None, seed=seed)
        
        # Giai đoạn 2: Cộng tác giữa các data site
        for step in range(self.__max_iter):
            old_centroids = [datasite.centroids.copy() for datasite in self.__datasites]
            # old_betas = self.betas.copy()


            # Kết nối các tâm cụm, tính V ngã
            all_centroids = np.array([datasite.centroids for datasite in self.__datasites])            
            v_tilde = np.mean(all_centroids, axis=0)

            # all_centroids = np.concatenate([datasite.centroids for datasite in self.__datasites], axis=0)
            # fcm = Dcmeans(self.__n_clusters)
            # _, v_tilde, _ = fcm.fit(all_centroids)

            for i, datasite in enumerate(self.__datasites):
                self.__update_datasite(i, datasite, datas[i], v_tilde)

            # Thỏa mãn điều kiện dừng trên mọi data site
            if all(np.linalg.norm(datasite.centroids - old) < self.__epsilon for datasite, old in zip(self.__datasites, old_centroids)):
                break
        # ---------------------------
        Uds, Vds = [], []
        for model in self.__datasites:
            Uds.append(model.membership)
            Vds.append(model.centroids)
        return Uds, Vds, step, datas

    def result_all_uvs(self, data: np.ndarray, centroids: list) -> tuple:
        _vds = np.concatenate(centroids, axis=0)
        print('vds', _vds.shape, _vds[0])
        _fcm = Dcmeans(n_clusters=self.__n_clusters, m=self.__m, epsilon=self.__epsilon, max_iter=self.__max_iter)
        _, Vs, _ = _fcm.fit(data=_vds)
        Us = _fcm.update_membership(data, Vs)
        return Us, Vs

    def __update_datasite(self, site_index, model: SSFCM, X, v_tilde):
        N = model.membership.shape[0]
        C = model.n_clusters

        # Cập nhật ma trận phân vùng cộng tác U_tilde và tính beta
        self.betas[site_index] = self.update_beta(model=model, X=X, site_index= site_index)

        # Cập nhật ma trận thành viên U
        model.membership = self.update_membership(model=model, X=X, v_tilde=v_tilde, N=N, C=C, beta= self.betas[site_index])

        # Cập nhật ma trận tâm cụm centroids
        model.centroids = self.update_centroid(model=model, X=X, v_tilde=v_tilde, beta=self.betas[site_index])

    def update_beta(self, model: SSFCM, X: ndarray, site_index: int):
        J = np.sum(((model.membership) ** self.__m) * (euclidean_cdist(X, model.centroids) ** 2))
        beta_sum = 0
        for jj in range(self.__n_sites):
            if jj != site_index:
                U_tilde = self.compute_U_tilde(X, self.__datasites[jj].centroids)
                J_tilde = np.sum((U_tilde ** 2) * (euclidean_cdist(X, self.__datasites[jj].centroids) ** 2))
                beta_sum += min(1, J / J_tilde)
        
        beta = beta_sum / (self.__n_sites - 1)
        return beta
    
    def update_membership(self, model: SSFCM, X: ndarray, v_tilde: ndarray, N: int, C: int, beta: float):
        distances = euclidean_cdist(X, model.centroids)
        d_tilde = np.linalg.norm(model.centroids - v_tilde, axis=1)

        denominators = np.zeros((N, C))
        for j in range(C):
            denominators[:, j] = (1/(distances[:, j]**2 + beta * (d_tilde[j]**2)))**(1/(self.__m-1))
        sum_denominators = np.sum(denominators, axis=1)

        for r in range(C):
            numerator = (distances[:, r]**2 + beta * (d_tilde[r]**2))**(-1)
            model.membership[:, r] = numerator / sum_denominators
        return model.membership
    
    def update_centroid(self, model: SSFCM, X: ndarray, v_tilde: ndarray, beta: float):
        um = (model.membership - model.membership_bar) ** self.__m
        _umT = um.T
        _V = ((_umT[:, :, None]) * X).sum(axis=1)
        model.centroids = (_V + beta * np.sum(um[:, :, None] * v_tilde, axis=0)) / ((1 + beta) * np.sum(um, axis=0)[:, np.newaxis])
        return model.centroids
    
    # Tính U ngã thể hiện sự cộng tác của datasite ii và jj. Datasite jj truyền V cho datasite ii 
    def compute_U_tilde(self,  X, centroids):
        distances = euclidean_cdist(X, centroids)
        if self.__m == 1:
            return self.__numpy_one(n_samples=len(distances.shape))
        power = 2
        U = distances[:, :, None] * ((1 / distances)[:, None, :])
        U = (U ** power).sum(axis=2)
        return 1 / U    

    def predict(self, new_data: np.ndarray):
        predictions = [model.predict(new_data=new_data) for model in self.__datasites]
        return np.mean(predictions, axis=0).astype(int)



if __name__ == '__main__':
    import time
    from utils.utility import LabelEncoder
    from utils.utility import fetch_data_from_uci, TEST_CASES, round_float, data_semi_supervised_learning

    datas = {602: 'DryBean', 109: 'Wine', 53: 'Iris'}
    clustering_report = []
    data_id = 602
    if data_id in TEST_CASES:
        _start_time = time.time()
        _dt = fetch_data_from_uci(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        # ------------------------
        _size = f"{_dt['data']['num_instances']}x{_dt['data']['num_features']}"
        print(f'size={_size}')
        P = 3  # Số lượng datasite
        # Mã hóa nhãn =========================================
        dlec = LabelEncoder()
        labeled_data = data_semi_supervised_learning(dlec, Y.flatten(), n_site=P, ratio=0.3)
        C = len(dlec.classes_)
        print(f'Data site={P}, So cum={C}')

        _start_time = time.time()
        # Khởi tạo mô hình
        model = CollaborativeSCFCM(n_clusters=C, n_sites=P, m=2, epsilon=1e-5, max_iter=10000)

        # Huấn luyện mô hình
        Uds, Vds, step = model.fit(X, labeled_data)
        _end_time = time.time() - _start_time
        # In U,V của từng datasite
        print('Tổng thời gian thực hiện =', _end_time)
        print('Tổng số bước lặp của quá trình cộng tác =', step)
        for i, (Ui, Vi) in enumerate(zip(Uds, Vds)):
            print('--------------------')
            print(f'U của datasite {i} = {Ui.shape} \n{Ui[0]}')
            print(f'V của datasite {i} = {Vi.shape} \n{Vi[0]})')
        # ---------------------
        Us, Vs = model.result_all_uvs(X, centroids=Vds)
        print('U tổng hợp=', Us.shape, Us[0])
        print('V tổng hợp=', Vs.shape, Vs[0])
        # Dự đoán
        predictions = model.predict(X)

        # In kết quả
        print("Predictions:", dlec.inverse_transform(predictions))

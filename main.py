from fuzzy.cmeans import Dcmeans
from ucimlrepo import fetch_ucirepo 
import yaml
from utils.validity import * 
from utils.utility import *
from fuzzy.sscfcm import CollaborativeSCFCM
import time
from numpy import ndarray

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def compute_evaluation_index(config, X: ndarray, U: ndarray, V: ndarray, m: float=2, labels: ndarray= None, clusters: ndarray= None):
    if config['validity_indices']['dunn_index']:
        dunn = dunn_index(clusters)
        print(f"Chỉ số Dunn: {dunn:.6f}")
    if config['validity_indices']['davies_bouldin_index']:
        db = davies_bouldin_index(X, labels)
        print(f"Chỉ số DB: {db:.6f}")
    if config['validity_indices']['separation_index']:
        s = separation_index(X, U, V, m)
        print(f"Chỉ số S: {s:.6f}")
    if config['validity_indices']['calinski_harabasz_index']:
        ch = calinski_harabasz_index(X, labels)
        print(f"Chỉ số CH: {ch:.6f}")
    if config['validity_indices']['silhouette_index']:
        si = silhouette_index(X, labels)
        print(f"Chỉ số SI: {si:.6f}")
    if config['validity_indices']['partition_coefficient']:
        pc = partition_coefficient(U)
        print(f"Chỉ số PC:{pc:.6f}")
    if config['validity_indices']['classification_entropy']:
        ce = classification_entropy(U)
        print(f"Chỉ số CE:{ce:.6f}")
    if config['validity_indices']['fuzzy_hypervolume']:
        fhv = fuzzy_hypervolume(U, m)
        print(f"Chỉ số FHV: {fhv:.6f}")
    if config['validity_indices']['cs_index']:
        cs = cs_index(X, U, V, m)
        print(f"Chỉ số CS: {cs:.6f}")
if __name__ == '__main__':
    # Load file config
    config = load_config('config/sscfcm_data.yaml')

    
    # fetch dataset 
    dry_bean = fetch_ucirepo(id=602) 
    # data (as pandas dataframes) 
    X = dry_bean.data.features.to_numpy()
    Y = dry_bean.data.targets.to_numpy()
    C = 7
    seed = config['seed']
    m = config['m']
    IS_FCM = config['fcm']
    IS_SSCFCM = config['sscfcm']
    P = 3 # Số lượng datasite

    # Thực hiện đánh giá FCM
    if IS_FCM:
        X_split0 = np.array_split(X, P)[0]
        model = Dcmeans(n_clusters=C)
        U, V, _ = model.fit(data= X_split0, seed= seed)
        print('Hoàn tất phân cụm FCM!')

        labels = extract_labels(U)
        clusters = extract_clusters(X_split0, labels, C)
        for i, cluster in enumerate(clusters):
            print(f'Cluster {i}: {cluster.shape}')

        # Đánh giá chất lượng phân cụm FCM
        compute_evaluation_index(config, X_split0, U, V, m, labels, clusters)
        print('Hoàn tất đánh giá phân cụm FCM!')

        
        
    if IS_SSCFCM:
        # Mã hóa nhãn =========================================
        dlec = LabelEncoder()
        labeled_data = data_semi_supervised_learning(dlec, Y.flatten(), n_site=P, ratio=0.1)
        C = len(dlec.classes_)
        print(f'Data site={P}, So cum={C}')

        _start_time = time.time()
        # Khởi tạo mô hình
        model = CollaborativeSCFCM(n_clusters=C, n_sites=P, m=2, beta=0.5, epsilon=1e-5, max_iter=10000)

        # Huấn luyện mô hình
        Uds, Vds, step, Xi = model.fit(X, labeled_data)
        _end_time = time.time() - _start_time
        # In U,V của từng datasite
        print('Tổng thời gian thực hiện =', _end_time)
        print('Tổng số bước lặp của quá trình cộng tác =', step)
        for i, (Ui, Vi) in enumerate(zip(Uds, Vds)):
            print('--------------------')
            labels = extract_labels(Ui)
            clusters = extract_clusters(Xi[i], labels, C)
            compute_evaluation_index(config, Xi[i], Ui, Vi, m, labels, clusters)

            for ii, cluster in enumerate(clusters):
                print(f'Cluster {ii}: {cluster.shape}')
            print(f'labels datasite {i}:', labels.shape)
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

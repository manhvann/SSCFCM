import numpy as np 
from utils.utility import round_float, norm_distances

# 1.1 Chỉ số đo lường mức độ tách biệt giữa các cụm
## 1.1.1 Chỉ số Dunn (DI)
def dunn_index(clusters)->float:
    """
    Args:
    Giá trị trả về: Chỉ số Dunn, càng cao càng tốt, càng thể hiện độ tách biệt giữa các cụm.
    
    Chỉ số DI đo lường khoảng cách tối thiểu giữa các cụm so với khoảng cách tối đa trong mỗi cụm. Giá
    trị DI càng cao phản ánh chất lượng phân cụm tốt hơn, độ tách biệt cao (khoảng cách giữa các cụm lớn
    và kích thước các cụm nhỏ). Thường sử dụng DI để tìm tham số C với cùng dữ liệu, cùng thuật toán
    hoặc để so sánh các thuật toán phân cụm khác nhau. Lưu ý DI có thể nhạy cảm với nhiễu và các điểm
    ngoại lai.
    """
    n_clusters = len(clusters)
    
    # Tính khoảng cách giữa các cụm
    min_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            distances = np.linalg.norm(clusters[i][:, np.newaxis] - clusters[j], axis=2)
            min_distances[i, j] = min_distances[j, i] = np.min(distances)
    
    # Tính đường kính của mỗi cụm   
    diameters = np.array([
        np.max(np.linalg.norm(cluster[:, np.newaxis] - cluster, axis=2))
        for cluster in clusters
    ])
    
    return round_float(np.min(min_distances[min_distances > 0])  / np.max(diameters))


def davies_bouldin_index(data:np.ndarray, labels:np.ndarray) -> float:
    """
    Chỉ số DB đo lường mức độ chồng chéo giữa các cụm, giá trị càng thấp càng tốt. DB thấp phản ánh
    chất lượng phân cụm tốt hơn (các cụm được phân tách rõ ràng hơn), độ tương đồng nơi cụm thấp (các
    điểm trong cùng một cụm tập trung chặt chẽ quanh tâm cụm), độ tách biệt giữa các cụm cao (các cụm
    cách xa nhau), sự đồng nhất giữa các cụm (các cụm có kích thước và mật độ tương đối đồng đều) và số
    lượng cụm phù hợp (chọn số lượng cụm tối ưu cho bộ dữ liệu). Thường sử dụng DB để tìm tham số C
    cho thuật toán phân cụm (so sánh các kết quả phân cụm khác nhau với cùng dữ liệu, cùng thuật toán
    nhưng truyền vào số cụm khác nhau)
    """
    # C = len(np.unique(labels))
    # V = np.array([data[labels == i].mean(axis=0) for i in range(C)])

    # #  Tính độ lệch chuẩn cho mỗi cụm 
    # d = [np.mean(np.linalg.norm(data[labels==i] - V[i], axis=1)) for i in range(C)]
        
    # # Tính Davies-Bouldin’s index
    # result = 0
    # for i in range(C):
    #     max_ratio = 0
    #     for j in range(C):
    #         if i != j:
    #             ratio = (d[i] + d[j]) / np.linalg.norm(V[i] - V[j])
    #             max_ratio = max(max_ratio, ratio)
    #     result += max_ratio
    # return round_float(result / C)

    from sklearn.metrics import davies_bouldin_score
    return round_float(davies_bouldin_score(data, labels))


## 1.1.3 Chỉ số tách biệt Separation (S)
def separation_index(data:np.ndarray, membership:np.ndarray, centroids:np.ndarray, m:float=2)->float:
    """
    Chỉ số S áp dụng cho phân cụm mờ, đo lường mức độ tách biệt giữa các cụm. S ∈ (0, ∞), giá trị càng
    thấp càng tốt, cho thấy các cụm được phân tách rõ ràng hơn. Nên được sử dụng kèm với PC, CE, thường
    được sử dụng để điều chỉnh số cụm tối ưu cho bộ dữ liệu.
    """
    _N, C = membership.shape
    _ut = membership.T
    numerator = 0
    for i in range(C):
        diff = data - centroids[i]
        squared_diff = np.sum(diff**2, axis=1)
        numerator += np.sum((_ut[i] ** m) * squared_diff)
    center_dists = np.sum((centroids[:, np.newaxis] - centroids) ** 2, axis=2)
    np.fill_diagonal(center_dists, np.inf)
    min_center_dist = np.min(center_dists)
    return round_float(numerator / min_center_dist)


## 1.1.4 Chỉ số Calinski-Harabasz (CH)
def calinski_harabasz_index(data:np.ndarray,labels:np.ndarray)->float:
    """
    Chỉ số CH đo lường mức độ phân biệt giữa các cụm so với mức độ phân tán trong mỗi cụm, giá trị
    càng cao, độ hợp lệ của phân cụm càng tốt.
    """
    # N = len(data)
    # C = len(np.unique(labels))
    
    # overall_mean = np.mean(data, axis=0)
    
    # # Tính tổng phương sai
    # within_var = 0
    # for i in range(C):
    #     cluster_i = data[labels == i]
    #     cluster_mean = np.mean(cluster_i, axis=0)
    #     within_var += np.sum((cluster_i - cluster_mean) ** 2)
    
    # # Tính phương sai giữa các cụm
    # between_var = 0
    # for i in range(C):
    #     cluster_i = data[labels == i]
    #     ni = len(cluster_i)
    #     cluster_mean = np.mean(cluster_i, axis=0)
    #     between_var += ni * np.sum((cluster_mean - overall_mean) ** 2)
                         
    # # Tính phương sai trong cụm
    # if N == C or C == 1:
    #     return round_float(0)
    # return round_float((between_var / (C - 1)) / (within_var / (N - C)))
    from sklearn.metrics import calinski_harabasz_score
    return round_float(calinski_harabasz_score(data, labels))


## 1.1.5 Chỉ số Silhouette
def silhouette_index(data:np.ndarray,labels:np.ndarray)->float:
    """
    Chỉ số SI đo lường mức độ tương tự của một điểm dữ liệu với cụm nó được gán vào so với các cụm
    khác. SI ∈ [−1, 1], SI càng gần 1, độ độ hợp lệ của phân cụm càng tốt
    """
    # N = len(data)
    # silhouette_vals = np.zeros(N)
    # for i in range(N):
    #     a_i = 0
    #     b_i = np.inf
    #     for j in range(N):
    #         if i != j:
    #             distance = np.sqrt(np.sum((data[i] - data[j])**2))
    #             if labels[i] == labels[j]:
    #                 a_i += distance
    #             else:
    #                 b_i = min(b_i, distance)

    #     if np.sum(labels == labels[i]) > 1:
    #         a_i /= (np.sum(labels == labels[i]) - 1)
    #     else:
    #         a_i = 0
    #     silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
    # return round_float(np.mean(silhouette_vals))
    from sklearn.metrics import silhouette_score
    return round_float(silhouette_score(data, labels))

# 1.2 Chỉ số đo lường mức độ rõ ràng của các cụm
## 1.2.1 Hệ số phân vùng Partition Coefficient (PCI), Giống hệt chỉ số FPC
def partition_coefficient(membership:np.ndarray)->float: # Fuzzy Partition Coefficient
    """
    Chỉ số PC đo lường mức độ rõ ràng của các cụm, giá trị càng cao, độ hợp lệ của phân cụm càng tốt.
    Chỉ số PC phù hợp với thuật toán phân cụm mờ, không phản ánh sự tách biệt giữa các cụm.
    """
    return round_float(np.sum(np.square(membership)) / membership.shape[0])
    # return round_float(np.trace(np.dot(membership.T, membership)) / membership.shape[0])
    
    
## 1.2.2 Entropy phân loại Classification Entropy (CEI)
def classification_entropy(membership:np.ndarray,a:float=np.e)->float:
    """
    CE đo lường mức độ không chắc chắn trong việc gán điểm vào các cụm, giá trị càng thấp, độ hợp lệ
    của phân cụm càng tốt. CE thường kết hợp với PC, một phân cụm tốt thường có PC cao và CE thấp,
    0 ≤ 1 − P C ≤ CE
    """
    N = membership.shape[0]
    
    # Tránh log(0) bằng cách thêm một epsilon nhỏ cho tất cả các phần tử
    epsilon = np.finfo(float).eps
    membership = np.clip(membership, epsilon, 1)
    
    # Tính tỉ lệ phần trăm điểm dữ liệu thuộc về mỗi cụm
    log_u = np.log(membership) / np.log(a) # Chuyển đổi cơ số logarit
    return round_float(-np.sum(membership * log_u) / N)


# 1.3 Các chỉ số khác
## 1.3.1 Thể tích mờ Fuzzy Hypervolume (FH)
def fuzzy_hypervolume(membership:np.ndarray, m:float=2)->float:
    """
    Args:
    FHV đo lường thể tích của các cụm trong không gian mờ. FHV phù hợp với thuật toán phân cụm
    mờ, không phản ánh sự tách biệt giữa các cụm. F HV ∈ (0, 1], giá trị càng cao, độ hợp lệ của phân cụm
    càng tốt.
    """
    C = membership.shape[1]
    fhv = 0
    for i in range(C):
        cluster_u = membership[:, i]
        n_i = np.sum(cluster_u > 0)
        if n_i > 0:
            fhv += np.sum(cluster_u ** m) / n_i
    return round_float(fhv)


def cs_index(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    """
    CS kết hợp các khái niệm về khoảng cách giữa các cụm và độ lệch chuẩn của mỗi cụm. CS kết hợp
    cả độ tách biệt và độ nén nhưng có thể bị ảnh hưởng bởi các điểm ngoại lai. CS > 0, giá trị càng thấp,
    độ hợp lệ của phân cụm càng tốt.
    """
    N, C = membership.shape
    numerator = 0
    for i in range(C):
        numerator += np.sum((membership[:, i]**m)[:, np.newaxis] *
                            np.sum((data - centroids[i])**2, axis=1)[:, np.newaxis])
    min_center_dist = np.min([np.sum((centroids[i] - centroids[j])**2)
                              for i in range(C)
                              for j in range(i+1, C)])
    return round_float(numerator / (N * min_center_dist))


# For data have labls (y_true, y_pred)
# AC
# https://stackoverflow.com/questions/37842165/sklearn-calculating-accuracy-score-of-k-means-on-the-test-data-set
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # if len(y_true) != len(y_pred):
    #     raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    # total_samples = len(y_true)
    # return correct_predictions / total_samples
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


# F1
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # if len(y_true) != len(y_pred):
    #     raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # # Tính TP, FP, FN
    # tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    # fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    # fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))

    # # Tính precision và recall
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # total = precision + recall
    # return 2 * (precision * recall) / total if total > 0 else 0
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='micro')

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # Tính TP, FP, FN
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))

    # Tính precision và recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # Tính TP, FP, FN
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))

    # Tính precision và recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # if len(y_true) != len(y_pred):
    #     raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # if len(y_true) != len(y_pred):
    #     raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)


def purity_score(y_true, y_pred):
    # https://medium.com/@vincydesy96/evaluation-of-supervised-clustering-purity-from-scratch-3ce42e1491b1
    # Purity ranges from 0 to 1, where 1 indicates perfect clustering (each cluster contains only instances of a single class), 
    # and 0 indicates the worst clustering (each cluster contains instances from all classes).
    # Also if you need to compute Inverse Purity, all you need to do is replace "axis=0" by "axis=1".
    
    from sklearn import metrics
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def normalized_mutual_info_score(y_true, y_pred):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(y_true, y_pred)


def sum_of_square_error(y_true, y_pred) -> float:
    """
    https://www.statology.org/sst-ssr-sse-in-python/
    """
    return np.sum((y_pred - y_true) ** 2)
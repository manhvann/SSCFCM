import os
import re
import json
import numpy as np
import pandas as pd
from urllib import request, parse, error
import certifi
import ssl


COLORS = ['Blue', 'Orange', 'Green', 'Red', 'Cyan', 'Yellow', 'Purple', 'Pink', 'Brown', 'Black', 'Gray', 'Beige', 'Turquoise', 'Silver', 'Gold']
TEST_CASES = {
    14: {
        'name': 'BreastCancer',
        'n_cluster': 2,
        'test_points': ['30-39', 'premeno', '30-34', '0-2', 'no', 3, 'left', 'left_low', 'no']
    },
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    80: {
        'name': 'Digits',
        'n_cluster': 10,
        'test_points': [0, 1, 6, 15, 12, 1, 0, 0, 0, 7, 16, 6, 6, 10, 0, 0, 0, 8, 16, 2, 0, 11, 2, 0, 0, 5, 16, 3, 0, 5, 7, 0, 0, 7, 13, 3, 0, 8, 7, 0, 0, 4, 12, 0, 1, 13, 5, 0, 0, 0, 14, 9, 15, 9, 0, 0, 0, 0, 6, 14, 7, 1, 0, 0]
    },
    109: {
        'name': 'Wine',
        'n_cluster': 3,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    236: {
        'name': 'Seeds',
        'n_cluster': 3,
        'test_points': [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}


# Mã hóa nhãn
class LabelEncoder:
    def __init__(self):
        self.index_to_label = {}
        self.unique_labels = None

    @property
    def classes_(self) -> np.ndarray:
        return self.unique_labels

    def fit_transform(self, labels) -> np.ndarray:
        self.unique_labels = np.unique(labels)
        label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
        self.index_to_label = {index: label for label, index in label_to_index.items()}
        return np.array([label_to_index[label] for label in labels])

    def inverse_transform(self, indices) -> np.ndarray:
        return np.array([self.index_to_label[index] for index in indices])


def extract_labels(U: np.ndarray) -> np.ndarray:
    return np.argmax(U, axis=1)

def extract_clusters(data: np.ndarray, labels: np.ndarray, C: int) -> list:
    return [data[labels == i] for i in range(C)]

def data_semi_supervised_learning(dle, labels: np.ndarray, n_site: int = 3, ratio: float = 0.3) -> list:
    y_encoded = dle.fit_transform(labels)
    y_splits = np.array_split(y_encoded, n_site)
    return [np.array([label if i < int(len(y_site) * ratio) else None for i, label in enumerate(y_site)]) for y_site in y_splits]


def name_slug(text: str, delim: str = '-') -> str:
    __punct_re = re.compile(r'[\t !’"“”#@$%&~\'()*\+:;\-/<=>?\[\\\]^_`{|},.]+')
    if text:
        from unidecode import unidecode
        result = [unidecode(word) for word in __punct_re.split(text.lower()) if word]
        result = [rs if rs != delim and rs.isalnum() else '' for rs in result]
        return re.sub(r'\s+', delim, delim.join(result).strip())


# Làm tròn số
def round_float(number: float, n: int = 3) -> float:
    if n == 0:
        return int(number)
    return round(number, n)


# Chuẩn Euclidean của một vector đo lường độ dài của vector
# là căn bậc hai của tổng bình phương các phần tử của vector đó.
# d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
def norm_distances(A: np.ndarray, B: np.ndarray, axis: int = None) -> float:
    # np.sqrt(np.sum((np.asarray(A) - np.asarray(B)) ** 2))
    # np.sum(np.abs(np.array(A) - np.array(B)))
    return np.linalg.norm(A - B, axis=axis)


# Tổng bình phương của hiệu khoảng cách giữa 2 ma trận trên tất cả các chiều
def euclidean_distance(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # Hiệu giữa các điểm trong XA và XB
    differences = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
    return np.sum(differences ** 2, axis=2)


# Ma trận khoảng cách Euclide giữa các điểm trong 2 tập hợp dữ liệu
def euclidean_cdist(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # _df = euclidean_distance(XA,XB)
    # return np.sqrt(_df)
    from scipy.spatial.distance import cdist
    return cdist(XA, XB)


# lấy giá trị lớn nhất để tránh lỗi chia cho 0
def not_division_by_zero(data: np.ndarray):
    return np.fmax(data, np.finfo(np.float64).eps)


# Chuẩn hóa mỗi hàng của ma trận sao cho tổng của mỗi hàng bằng 1.
# \mathbf{x}_{norm} = \frac{\mathbf{x}}{\sum_{i=1}^m \mathbf{x}_{i,:}}
def standardize_rows(data: np.ndarray) -> np.ndarray:
    # Ma trận tổng của mỗi cột (cùng số chiều)
    _sum = np.sum(data, axis=0, keepdims=1)
    # Chia từng phần tử của ma trận cho tổng tương ứng của cột đó.
    return data / _sum


# Đếm số lần xuất hiện của từng phần tử trong 1 mảng
def count_data_array(data: np.ndarray) -> dict:
    unique, counts = np.unique(data, return_counts=True)
    return dict(zip(unique, counts))


def load_dataset(data: dict, file_csv: str = '', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    # label_name = data['data']['target_col']
    print('uci_id=', data['data']['uci_id'])  # Mã bộ dữ liệu
    print('data name=', data['data']['name'])  # Tên bộ dữ liệu
    print('data abstract=', data['data']['abstract'])  # Tên bộ dữ liệu
    print('feature types=', data['data']['feature_types'])  # Kiểu nhãn
    print('num instances=', data['data']['num_instances'])  # Số lượng điểm dữ liệu
    print('num features=', data['data']['num_features'])  # Số lượng đặc trưng
    metadata = data['data']
    # colnames = ['Area', 'Perimeter']
    df = pd.read_csv(file_csv if file_csv != '' else metadata['data_url'], header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    # print('data top', df.head())  # Hiển thị một số dòng dữ liệu
    # Trích xuất ma trận đặc trưng X (loại trừ nhãn lớp)
    return {'data': data['data'], 'X': df.iloc[:, :-1].values, 'Y': df.iloc[:, -1:].values}


# Lấy dữ liệu từ ổ cứng
def fetch_data_from_local(name_or_id=53, folder: str = 'dataset', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    if isinstance(name_or_id, str):
        name = name_or_id
    else:
        name = TEST_CASES[name_or_id]['name']
    _folder = os.path.join(folder, name)
    fileio = os.path.join(_folder, 'api.json')
    if not os.path.isfile(fileio):
        print(f'File {fileio} not found!')
    with open(fileio, 'r') as cr:
        response = cr.read()
    return load_dataset(json.loads(response),
                        file_csv=os.path.join(_folder, 'data.csv'),
                        header=header, index_col=index_col, usecols=usecols, nrows=nrows)


# Lấy dữ liệu từ ISC UCI (53: Iris, 602: DryBean, 109: Wine)
def fetch_data_from_uci(name_or_id=53, file_csv='') -> dict:
    api_url = 'https://archive.ics.uci.edu/api/dataset'
    if isinstance(name_or_id, str):
        api_url += '?name=' + parse.quote(name_or_id)
    else:
        api_url += '?id=' + str(name_or_id)
    try:
        response = request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
        dataset = load_dataset(data=json.load(response), file_csv=file_csv)
        return dataset
    except (error.URLError, error.HTTPError):
        raise ConnectionError('Error connecting to server')

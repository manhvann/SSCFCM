import numpy as np 
import os 
import rasterio
from PIL import Image


def image2data(img_input) -> tuple:
    if isinstance(img_input, str):
        # Input is a file path
        if not os.path.isfile(img_input):
            print(f'Image path {img_input} does not exist')
            return None
        try:
            with rasterio.open(img_input) as dataset:
                image_data = dataset.read()
        except rasterio.errors.RasterioIOError:
            print(f'Unable to read image at {img_input}')
            return None
    elif isinstance(img_input, np.ndarray):
        # Input is already image data
        image_data = img_input
    else:
        print(f'Invalid input type. Expected str or np.ndarray, got {type(img_input)}')
        return None

    # Normalize each band separately
    normalized_data = []
    for band in image_data:
        # band = band - np.min(band)
        # band = band / np.max(band)
        normalized_data.append(band)

    # Stack normalized bands along a new dimension
    normalized_data = np.stack(normalized_data, axis=-1)

    # Reshape the data to (number of pixels, number of bands)
    data = normalized_data.reshape((-1, image_data.shape[0]))

    return data, image_data.shape


def image_in_folder2data(folder_path: str) -> tuple:
    if not os.path.isdir(folder_path):
        print(f'Duong dan thu muc {folder_path} khong ton tai')
        return None
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    data = []
    for imgfile in image_files:
        imgpath = os.path.join(folder_path, imgfile)
        with rasterio.open(imgpath) as dataset:
            img_data = dataset.read()
        data.append(img_data)
    data = np.concatenate(data, axis=0)
    
    return data, data.shape


def data2image(labels: np.ndarray, clusters: tuple, out_shape: tuple, output_path: str) -> None:
    segmented_image = labels.reshape(out_shape[1:])
    
    # Khai báo một bảng màu cho các cụm
    color_palette = np.array([
        [0,50,199],  # River, ponds, lakes
        [149,149,149],  # Rocks, mountains, bare soil
        [101,254,50],   # fields, grasslands
        [50,199,255],  # Planted forests, low wood 
        [0,100,1],      # Perinial tree, crops 
        [0,50,0],      # Jungle
        # [34, 139, 34],    # Forest Green (Urban Vegetation)
        # [30, 144, 255],   # Dodger Blue (Shallow Water)
        # [70, 130, 180],   # Steel Blue (Water Bodies)
        # [210, 180, 140],  # Tan (Bare Soil/Earth)
        # [244, 164, 96],   # Sandy Brown (Beaches/Sandbanks)
        # [139, 69, 19],    # Saddle Brown (Bare Ground)
        # [255, 255, 255],  # White (Clouds/Bright Surfaces)
        # [105, 105, 105],  # Dim Gray (Shadows/Dark Areas)
        # [0, 255, 0],      # Lime (Bright Vegetation)
    ], dtype=np.uint8)
    
    # Nếu số cụm lớn hơn số màu trong bảng màu, lặp lại bảng màu
    if len(clusters) > len(color_palette):
        # Lặp lại bảng màu cho đến khi đủ số cụm
        color_palette = np.tile(color_palette, (1 + len(clusters) // len(color_palette), 1))[:len(clusters)]
    
    # Tạo ảnh màu từ ảnh phân đoạn
    colored_segmented_image = color_palette[segmented_image]
    
    
    # Xử lý path đầu ra
    base, ext = os.path.splitext(output_path)
    index = 1
    while os.path.exists(output_path):
        output_path = f"{base}_{index}{ext}"
        index += 1
    
    # Image.fromarray(colored_segmented_image).save(output_path, format='TIFF')
    Image.fromarray(colored_segmented_image).save(output_path)
    print(colored_segmented_image.shape)
    print(f'Image saved to {output_path}')
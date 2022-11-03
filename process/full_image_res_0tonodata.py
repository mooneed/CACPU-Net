import numpy as np
import gdal
from tqdm import tqdm

"""将tif中的0值置nan方便ENVI等商用遥感软件的可视化分析"""
image_path = "./res/full_image_res/OURS_2022_9_29/full_image.tif"
image = gdal.Open(image_path).ReadAsArray().astype(float)
for i in tqdm(range(image.shape[0])):
    for j in range(image.shape[1]):
        if image[i][j] == 0:
            image[i][j] = np.nan
image = image[np.newaxis,:,:]
res_path = "./res/full_image_res/OURS_2022_9_29/full_image_nodata.tif"
bands, r, c = image.shape
datatype = gdal.GDT_Float32
driver = gdal.GetDriverByName("GTiff")
datas = driver.Create(res_path, c, r, bands, datatype)
for i in range(bands):
    datas.GetRasterBand(i + 1).WriteArray(image[i])
del datas


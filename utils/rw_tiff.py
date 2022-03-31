import gdal
import numpy as np
import os
from numpy.lib.stride_tricks import as_strided


def writeTiff_proj(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def readTiff_proj(filename):
    tif1 = gdal.Open(filename, gdal.GA_Update)
    tif_num_1 = tif1.ReadAsArray()
    im_geotrans = tif1.GetGeoTransform()
    # print(geoTransform_flow)
    im_proj = tif1.GetProjectionRef()
    return tif_num_1, im_geotrans, im_proj




def dofunc_patch(img, result, kernel, stride, dofunc):
    n, c, h, w = img.shape
    hsum = (kernel - h) % stride
    wsum = (kernel - w) % stride
    xinp = np.pad(img, [(0,0),(0,0), (0, hsum), (0, wsum)], 'reflect')
    result = np.pad(result, [(0, 0), (0, 0), (0, hsum), (0, wsum)], 'reflect')
    ph, pw = (xinp.shape[2]-kernel)//stride + 1, (xinp.shape[3]-kernel)//stride + 1
    patches = as_strided(xinp, shape=(n,c,ph,kernel,pw,kernel),
          strides=(xinp.strides[0],xinp.strides[1],xinp.strides[2]*stride,xinp.strides[2],xinp.strides[3]*stride,xinp.strides[3]))
    patches = patches.transpose(0,1,2,4,3,5)
    weight = np.zeros(xinp.shape[-2:], dtype=np.float32)[None,None]
    idx = 0
    for x in range(patches.shape[2]):
        for y in range(patches.shape[3]):
            w_tmp  = np.ones(patches.shape[-2:], dtype=np.float32)[None,None]
            for i in range(kernel - stride):
                w_tmp[:, :, i, :] *= i/31 + 1e-6
                w_tmp[:, :, kernel-i-1, :] *= i/31 + 1e-6
            for i in range(kernel - stride):
                w_tmp[:, :, :, i] *= i/31 + 1e-6
                w_tmp[:, :, :, kernel-i-1] *= i/31 + 1e-6

            patch_rst = dofunc(patches[:, :, x, y])
            outc = patch_rst.shape[1]
            if result.shape[1] != outc:
                result = np.zeros((result.shape[0], outc, result.shape[2], result.shape[3]), dtype=np.float32)
            result[:, :, stride*x:stride*x+kernel, stride*y:stride*y+kernel] += patch_rst * w_tmp
            weight[:, :, stride*x:stride*x+kernel, stride*y:stride*y+kernel] += w_tmp

            # result[:,:,stride*x:stride*x+kernel, stride*y:stride*y+kernel] += dofunc(patches[:,:,x,y])
            # weight[:,:,stride*x:stride*x+kernel, stride*y:stride*y+kernel] += 1
            idx+=1
            # print("{}/{}".format(idx, patches.shape[2]*patches.shape[3]))
    result = result / weight
    result = result[:,:,:h, :w]
    return result




if __name__=='__main__':
    name = r'C:\Users\citianyu\Downloads\suichang_round1_test_partA_210120\000002.tif'

'''
Project    : StereoDatasetConstruction
FileName   : img_info .py
CreateTime : 2025/1/11 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import numpy as np
import rasterio
import xml

import cv2
from rasterio.windows import Window
from rasterio.plot import show
import rpcm
import xml.dom.minidom

class RSImage:
    def __init__(self, img_path, rpc_path=None, xml_path=None):
        self.img_path = img_path
        self.rpc_path = rpc_path
        self.xml_path = xml_path

        # 读取rpc模型
        if self.rpc_path:
            # rpc.txt
            if isinstance(self.rpc_path, str) and self.rpc_path.endswith('.txt'):  # path to an RPC file 如果是rpc.txt的路径文件
                self.rpc_model = rpcm.rpc_from_rpc_file(self.rpc_path)
            # 未来实现rpb
            elif isinstance(self.rpc_path, dict):  # RPC dict in 'rpcm' format
                self.rpc_model = rpcm.RPCModel(self.rpc_path, dict_format='rpcm')  # 根据RPC dict解析rpc
            else:
                raise NotImplementedError('rpc of type {} not supported'.format(type(self.rpc_path)))
        else:
            self.rpc_model = rpcm.rpc_from_geotiff(self.img_path)  # 根据tiff进行解析rpc


        # 读取img信息
        if self.img_path:
            self.img = rasterio.open(self.img_path)
            self.bands = self.img.count
            self.width = self.img.width
            self.height = self.img.height
            self.bounds = self.img.bounds
            self.geotrans = self.img.transform
            self.geoprojc = self.img.crs


        # 读取xml信息
        if self.xml_path:
            xmls = xml.dom.minidom.parse(xml_path)

            # 影像元信息
            if xmls.getElementsByTagName('SatelliteID'):
                self.satellite = xmls.getElementsByTagName('SatelliteID')[0].firstChild.data
            if xmls.getElementsByTagName('SensorID'):
                self.sensor = xmls.getElementsByTagName('SensorID')[0].firstChild.data
            if xmls.getElementsByTagName('CenterTime'):
                try:
                    self.date = xmls.getElementsByTagName('CenterTime')[0].firstChild.data
                except Exception as e:
                    self.date = xmls.getElementsByTagName('StartTime')[0].firstChild.data

            # 四个顶点的维度经度
            if xmls.getElementsByTagName('TopLeftLatitude'):
                self.TLlat = float(xmls.getElementsByTagName('TopLeftLatitude')[0].firstChild.data)
                self.TLlon = float(xmls.getElementsByTagName('TopLeftLongitude')[0].firstChild.data)
                self.TRlat = float(xmls.getElementsByTagName('TopRightLatitude')[0].firstChild.data)
                self.TRlon = float(xmls.getElementsByTagName('TopRightLongitude')[0].firstChild.data)
                self.BLlat = float(xmls.getElementsByTagName('BottomLeftLatitude')[0].firstChild.data)
                self.BLlon = float(xmls.getElementsByTagName('BottomLeftLongitude')[0].firstChild.data)
                self.BRlat = float(xmls.getElementsByTagName('BottomRightLatitude')[0].firstChild.data)
                self.BRlon = float(xmls.getElementsByTagName('BottomRightLongitude')[0].firstChild.data)

            if xmls.getElementsByTagName('UpperLeftLat'):
                self.TLlat = float(xmls.getElementsByTagName('UpperLeftLat')[0].firstChild.data)
                self.TLlon = float(xmls.getElementsByTagName('UpperLeftLong')[0].firstChild.data)
                self.TRlat = float(xmls.getElementsByTagName('UpperRightLat')[0].firstChild.data)
                self.TRlon = float(xmls.getElementsByTagName('UpperRightLong')[0].firstChild.data)
                self.BLlat = float(xmls.getElementsByTagName('LowerLeftLat')[0].firstChild.data)
                self.BLlon = float(xmls.getElementsByTagName('LowerLeftLong')[0].firstChild.data)
                self.BRlat = float(xmls.getElementsByTagName('LowerRightLat')[0].firstChild.data)
                self.BRlon = float(xmls.getElementsByTagName('LowerRightLong')[0].firstChild.data)


    def crop_image(self, col, row, width, height):
        crop_band = self.img.read(1, window=Window(col, row, width, height))
        w_, l_ = int(width // 256), int(height // 50)
        crop_band[height // 2 - w_: height // 2 + w_, width // 2 - l_: width // 2 + l_] = 0
        crop_band[height // 2 - l_: height // 2 + l_, width // 2 - w_: width // 2 + w_] = 0
        show(crop_band)


    def show_points(self, points):
        min_col, min_row = round(min(points[:, 0])), round(min(points[:, 1]))
        max_col, max_row = round(max(points[:, 0])), round(max(points[:, 1]))

        offset = 128
        width = 1
        height = 5

        crop_band = self.img.read(1, window=Window(min_col - offset, min_row - offset, max_col - min_col + 2*offset, max_row - min_row + 2*offset))

        for point in points:
            x_, y_ = round(point[0]), round(point[1])
            crop_band[y_-min_row+offset-width: y_-min_row+offset+width, x_-min_col+offset-height: x_-min_col+offset+height] = 0
            crop_band[y_-min_row+offset-height: y_-min_row+offset+height, x_-min_col+offset-width: x_-min_col+offset+width] = 0
        show(crop_band)


    def draw_lines(self, coffes, points, offset, save_path):
        width = (offset // 512) if offset >= 512 else 1
        height = 5

        k, _, b = coffes

        min_col, min_row = round(min(np.array(points)[:, 0])) - offset, round(min(np.array(points)[:, 1])) - offset
        max_col, max_row = round(max(np.array(points)[:, 0])) + offset, round(max(np.array(points)[:, 1])) + offset

        crop_band = self.img.read(1, window=Window(min_col, min_row, max_col - min_col, max_row - min_row))
        mean_value = np.nanmean(crop_band)
        std_value = np.nanstd(crop_band)
        min_value = mean_value - std_value * 2
        max_value = mean_value + std_value * 2
        crop_band_ = np.clip(crop_band, min_value, max_value)
        crop_band_ = ((crop_band_ - min_value) / (max_value - min_value) * 255).astype(np.uint8)

        for point in points:
            x_, y_ = round(point[0]) - min_col, round(point[1]) - min_row
            crop_band_[y_-width: y_+width, x_-height: x_+height] = 255
            crop_band_[y_-height: y_+height, x_-width: x_+width] = 255

        epi_y1 = min_col * k + b - min_row
        epi_y2 = max_col * k + b - min_row

        pt1 = (0, int(epi_y1))
        pt2 = (int(max_col - min_col), int(epi_y2))

        cv2.line(crop_band_, pt1, pt2, (255, 255, 255), width)

        cv2.imwrite(save_path, crop_band_)


    def get_sub_image(self, arg):
        x, y, w, h = arg
        crop_band = self.img.read(1, window=Window(x, y, w, h))
        crop_band_ = self.standardization(crop_band)

        if x < 0 or y < 0 or (x + w) > self.width or (y + h) > self.height:
            n_crop_band_ = np.zeros((h, w), dtype=np.uint8)
            x_, y_ = 0-w, 0-h
            _x, _y = w, h
            if x < 0:
                x_ = w-crop_band_.shape[1]
            if y < 0:
                y_ = h-crop_band_.shape[0]
            if (x + w) > self.width:
                _x = -(w-crop_band_.shape[1])
            if (y + h) > self.height:
                _y = -(h-crop_band_.shape[0])

            n_crop_band_[y_:_y, x_:_x] = crop_band_
            crop_band_ = n_crop_band_

        # if (x + w) > self.width or (y + h) > self.height:
        #     n_crop_band_ = np.zeros((h, w), dtype=np.uint8)
        #     x_, y_ = w, h
        #     if (x + w) > self.width:
        #         x_ = w - crop_band_.shape[1]
        #     if (y + h) > self.height:
        #         y_ = y - crop_band_.shape[0]
        #
        #     n_crop_band_[:y_, :x_] = crop_band_
        #     crop_band_ = n_crop_band_

        # show(crop_band_)
        return crop_band_


    def standardization(self, img):
        # # # # # ========== std mean 方式
        # mean_value = np.nanmean(img)
        # std_value = np.nanstd(img)
        # min_value = mean_value - std_value * 2
        # max_value = mean_value + std_value * 2
        # img_ = np.clip(img, min_value, max_value)
        # img_ = ((img_ - min_value) / (max_value - min_value) * 255).astype(np.uint8)
        # # # # ========== 2% 98% 方式
        min_value, max_value = np.percentile(img, (2, 98))
        img_ = np.clip(img, min_value, max_value)
        img_ = ((img_ - min_value) / (max_value - min_value) * 255).astype(np.uint8)

        return img_



















if __name__ == '__main__':
    pass

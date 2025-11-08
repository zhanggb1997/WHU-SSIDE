'''
Project    : RSDeploy
FileName   : wgs84_ecef .py
CreateTime : 2025/3/10 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
from osgeo import osr


def wgs84_to_ecef(lat, lon, alt):
    """
    使用GDAL和PROJ4将WGS84大地坐标 (lat, lon, alt) 转换为空间直角坐标 (X, Y, Z)
    参数：
        lat: 纬度（度）
        lon: 经度（度）
        alt: 高程（米）
    返回：
        (X, Y, Z) 空间直角坐标（米）
    """
    # 创建WGS84地理坐标系
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)  # WGS84

    # 创建WGS84地心直角坐标系（ECEF）
    target = osr.SpatialReference()
    target.ImportFromEPSG(4978)  # ECEF

    # 创建转换对象
    transform = osr.CoordinateTransformation(target, source)

    # 进行坐标转换
    x, y, z = transform.TransformPoint(lon, lat, alt)

    return x, y, z


# 示例调用
lat, lon, alt = +51.53145733, -00.13101513, 8000.0

X=3972708.852712
Y=5000011.833765
Z=-14505.183897

# x, y, z = wgs84_to_ecef(lat, lon, alt)
x, y, z = wgs84_to_ecef(Y, X, Z)
print(f"ECEF坐标: X={x:.6f}, Y={y:.6f}, Z={z:.6f}")


if __name__ == '__main__':
    pass

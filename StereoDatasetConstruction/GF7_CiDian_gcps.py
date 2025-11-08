'''
Project    : RSDeploy
FileName   : GF7_CiDian_gcps .py
CreateTime : 2025/6/19 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''

import csv
import os
import math
import shapefile  # pip install pyshp
import rpcm  # pip install rpcm
import rasterio  # pip install rasterio
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import traceback


def read_shapefile_points_with_id(shapefile_path):
    """从多点 Shapefile 中读取经纬度和编号，返回 {id: (lat, lon)} 字典。"""
    points = {}
    sf = shapefile.Reader(shapefile_path)

    # 获取字段名
    field_names = [field[0] for field in sf.fields[1:]]  # 跳过删除标记字段

    # 查找ID字段（可能的字段名）
    id_field = None
    possible_id_fields = ['id', 'ID', 'Id', 'point_id', 'POINT_ID', 'FID', 'OBJECTID']
    for field in possible_id_fields:
        if field in field_names:
            id_field = field
            break

    if id_field is None:
        raise ValueError(f"未找到ID字段。可用字段: {field_names}")

    id_index = field_names.index(id_field)

    for shape_rec in sf.shapeRecords():
        point_id = shape_rec.record[id_index]
        for point in shape_rec.shape.points:
            points[point_id] = (point[1], point[0])  # (lat, lon)

    return points


def filter_common_points(*point_dicts):
    """过滤出所有字典中都存在的点ID，返回过滤后的点字典列表。"""
    # 找到所有字典中都存在的ID
    common_ids = set(point_dicts[0].keys())
    for point_dict in point_dicts[1:]:
        common_ids = common_ids.intersection(set(point_dict.keys()))

    # 按ID排序，确保顺序一致
    common_ids = sorted(common_ids)

    # 返回过滤后的字典
    filtered_dicts = []
    for point_dict in point_dicts:
        filtered_dict = {pid: point_dict[pid] for pid in common_ids}
        filtered_dicts.append(filtered_dict)

    return filtered_dicts, common_ids


def get_height_from_dem(lat, lon, dem_path):
    """从 DEM 获取指定经纬度处的高度。"""
    with rasterio.open(dem_path) as dem:
        row, col = dem.index(lon, lat)
        height = dem.read(1)[row, col]
    return height


def write_unified_gcp_file(dom_points_dict, img1_points_dict, img2_points_dict, rpc1_model, rpc2_model, dem_path,
                           gcp_path, log_func):
    """生成统一的 GCP 文件，包含两个影像的观测数据。"""
    # 过滤出共同点
    filtered_dicts, common_ids = filter_common_points(dom_points_dict, img1_points_dict, img2_points_dict)
    dom_filtered, img1_filtered, img2_filtered = filtered_dicts

    log_func(f"共同点数量: {len(common_ids)}")
    log_func(f"共同点ID: {common_ids}")

    gcp_lines = []

    for point_id in sorted(common_ids):
        dom_lat, dom_lon = dom_filtered[point_id]
        img1_lat, img1_lon = img1_filtered[point_id]
        img2_lat, img2_lon = img2_filtered[point_id]

        try:
            # 从 DEM 获取高度
            dom_height = get_height_from_dem(dom_lat, dom_lon, dem_path)
            log_func(f"点ID {point_id} - DOM 高度: {dom_height:.4f}")
        except Exception as e:
            log_func(f"获取DEM高度失败（点ID {point_id}）：{e}")
            continue

        try:
            # 使用 RPC1 模型进行投影计算（影像1）
            img1_h = rpc1_model.alt_offset
            img1_col, img1_row = rpc1_model.projection(img1_lon, img1_lat, img1_h)

            # 使用 RPC2 模型进行投影计算（影像2）
            img2_h = rpc2_model.alt_offset
            img2_col, img2_row = rpc2_model.projection(img2_lon, img2_lat, img2_h)

        except Exception as e:
            log_func(f"RPC 投影失败（点ID {point_id}）：{e}")
            continue

        # 格式：gcp_id 观测数量 lat lon height 影像序号1 0 col1 row1 影像序号2 1 col2 row2
        gcp_line = (f"gcp_{point_id}\t3\t{dom_lat:.8f}\t{dom_lon:.8f}\t{dom_height:.4f}\t"
                    f"2\t0\t{img1_col:.2f}\t{img1_row:.2f}\t1\t{img2_col:.2f}\t{img2_row:.2f}\n")
        gcp_lines.append(gcp_line)

    try:
        with open(gcp_path, mode='w') as gcp_file:
            gcp_file.write(f"{len(gcp_lines)}\n")
            gcp_file.writelines(gcp_lines)
        log_func(f"统一GCP 文件写入成功: {gcp_path}")
    except Exception as e:
        log_func(f"GCP 文件写入失败：{e}")


def write_points_to_csv(dom_points_dict, img1_points_dict, img2_points_dict, output_csv, log_func):
    """将 DOM 和 IMG 点写入 CSV 文件，并计算经纬度差值及 RMSE。"""
    # 过滤出共同点
    filtered_dicts, common_ids = filter_common_points(dom_points_dict, img1_points_dict, img2_points_dict)
    dom_filtered, img1_filtered, img2_filtered = filtered_dicts

    diff_lat_list_img1 = []
    diff_lon_list_img1 = []
    diff_lat_list_img2 = []
    diff_lon_list_img2 = []

    try:
        with open(output_csv, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["point_id", "dom_lat", "dom_lon",
                                 "img1_lat", "img1_lon", "diff_lat_img1 (m)", "diff_lon_img1 (m)",
                                 "img2_lat", "img2_lon", "diff_lat_img2 (m)", "diff_lon_img2 (m)"])

            for point_id in sorted(common_ids):
                dom_lat, dom_lon = dom_filtered[point_id]
                img1_lat, img1_lon = img1_filtered[point_id]
                img2_lat, img2_lon = img2_filtered[point_id]

                diff_lat_img1 = (dom_lat - img1_lat) * 110000  # 米
                diff_lon_img1 = (dom_lon - img1_lon) * 110000
                diff_lat_img2 = (dom_lat - img2_lat) * 110000  # 米
                diff_lon_img2 = (dom_lon - img2_lon) * 110000

                diff_lat_list_img1.append(diff_lat_img1)
                diff_lon_list_img1.append(diff_lon_img1)
                diff_lat_list_img2.append(diff_lat_img2)
                diff_lon_list_img2.append(diff_lon_img2)

                csv_writer.writerow([point_id, dom_lat, dom_lon,
                                     img1_lat, img1_lon, diff_lat_img1, diff_lon_img1,
                                     img2_lat, img2_lon, diff_lat_img2, diff_lon_img2])

        # 计算RMSE
        def calculate_rmse(diff_list):
            mean_diff = sum(diff_list) / len(diff_list)
            squared_diff = [(d - mean_diff) ** 2 for d in diff_list]
            return math.sqrt(sum(squared_diff) / len(squared_diff))

        rmse_lat_img1 = calculate_rmse(diff_lat_list_img1)
        rmse_lon_img1 = calculate_rmse(diff_lon_list_img1)
        rmse_lat_img2 = calculate_rmse(diff_lat_list_img2)
        rmse_lon_img2 = calculate_rmse(diff_lon_list_img2)

        log_func(f"CSV 文件写入成功: {output_csv}")
        log_func(f"IMG1 - 纬度偏差 RMSE: {rmse_lat_img1:.2f} m, 经度偏差 RMSE: {rmse_lon_img1:.2f} m")
        log_func(f"IMG2 - 纬度偏差 RMSE: {rmse_lat_img2:.2f} m, 经度偏差 RMSE: {rmse_lon_img2:.2f} m")

        if rmse_lat_img1 > 100 or rmse_lon_img1 > 100:
            log_func("警告：IMG1 RMSE 超过 100 米。")
        if rmse_lat_img2 > 100 or rmse_lon_img2 > 100:
            log_func("警告：IMG2 RMSE 超过 100 米。")

    except Exception as e:
        log_func(f"CSV 写入失败：{e}")


def compute_differences_dual_image(dom_points_dict, img1_points_dict, img2_points_dict):
    """计算双影像的差值统计。"""
    # 过滤出共同点
    filtered_dicts, common_ids = filter_common_points(dom_points_dict, img1_points_dict, img2_points_dict)
    dom_filtered, img1_filtered, img2_filtered = filtered_dicts

    diff_lats_img1, diff_lons_img1, distances_img1 = [], [], []
    diff_lats_img2, diff_lons_img2, distances_img2 = [], [], []

    for point_id in sorted(common_ids):
        dom_lat, dom_lon = dom_filtered[point_id]
        img1_lat, img1_lon = img1_filtered[point_id]
        img2_lat, img2_lon = img2_filtered[point_id]

        # IMG1差值计算
        diff_lat_img1 = (dom_lat - img1_lat) * 110000
        diff_lon_img1 = (dom_lon - img1_lon) * 110000
        diff_lats_img1.append(diff_lat_img1)
        diff_lons_img1.append(diff_lon_img1)
        distances_img1.append(math.sqrt(diff_lat_img1 ** 2 + diff_lon_img1 ** 2))

        # IMG2差值计算
        diff_lat_img2 = (dom_lat - img2_lat) * 110000
        diff_lon_img2 = (dom_lon - img2_lon) * 110000
        diff_lats_img2.append(diff_lat_img2)
        diff_lons_img2.append(diff_lon_img2)
        distances_img2.append(math.sqrt(diff_lat_img2 ** 2 + diff_lon_img2 ** 2))

    # 计算平均值
    avg_offset_img1 = sum(distances_img1) / len(distances_img1)
    avg_abs_lat_img1 = sum(diff_lats_img1) / len(diff_lats_img1)
    avg_abs_lon_img1 = sum(diff_lons_img1) / len(diff_lons_img1)

    avg_offset_img2 = sum(distances_img2) / len(distances_img2)
    avg_abs_lat_img2 = sum(diff_lats_img2) / len(diff_lats_img2)
    avg_abs_lon_img2 = sum(diff_lons_img2) / len(diff_lons_img2)

    return {
        'img1': {
            'avg_offset': avg_offset_img1,
            'distances': distances_img1,
            'diff_lats': diff_lats_img1,
            'diff_lons': diff_lons_img1,
            'avg_abs_lat': avg_abs_lat_img1,
            'avg_abs_lon': avg_abs_lon_img1
        },
        'img2': {
            'avg_offset': avg_offset_img2,
            'distances': distances_img2,
            'diff_lats': diff_lats_img2,
            'diff_lons': diff_lons_img2,
            'avg_abs_lat': avg_abs_lat_img2,
            'avg_abs_lon': avg_abs_lon_img2
        },
        'common_ids': common_ids
    }


def write_anomaly_report_dual_image(dom_points_dict, img1_points_dict, img2_points_dict, stats, output_folder,
                                    log_func):
    """写入双影像异常报告。"""
    filtered_dicts, common_ids = filter_common_points(dom_points_dict, img1_points_dict, img2_points_dict)
    dom_filtered, img1_filtered, img2_filtered = filtered_dicts

    anomaly_lines = []

    # IMG1异常检测
    img1_stats = stats['img1']
    threshold_lat_img1 = 3 * abs(img1_stats['avg_abs_lat'])
    threshold_lon_img1 = 3 * abs(img1_stats['avg_abs_lon'])

    # IMG2异常检测
    img2_stats = stats['img2']
    threshold_lat_img2 = 3 * abs(img2_stats['avg_abs_lat'])
    threshold_lon_img2 = 3 * abs(img2_stats['avg_abs_lon'])

    for i, point_id in enumerate(sorted(common_ids)):
        dom_point = dom_filtered[point_id]
        img1_point = img1_filtered[point_id]
        img2_point = img2_filtered[point_id]

        d_lat_img1 = img1_stats['diff_lats'][i]
        d_lon_img1 = img1_stats['diff_lons'][i]
        d_lat_img2 = img2_stats['diff_lats'][i]
        d_lon_img2 = img2_stats['diff_lons'][i]

        # 检查IMG1异常
        if abs(d_lat_img1) > threshold_lat_img1 or abs(d_lon_img1) > threshold_lon_img1:
            if abs(d_lat_img1) > threshold_lat_img1 and abs(d_lon_img1) > threshold_lon_img1:
                anomaly_type_img1 = "Right coordinate anomaly"
            else:
                anomaly_type_img1 = "Image pixel anomaly"
            line = (f"IMG1 - 点ID {point_id}: DOM {dom_point} - IMG1 {img1_point} 的偏移量 "
                    f"diff_lat: {d_lat_img1:.2f} m, diff_lon: {d_lon_img1:.2f} m "
                    f"({anomaly_type_img1}), 阈值 (lat: {threshold_lat_img1:.2f} m, lon: {threshold_lon_img1:.2f} m)\n")
            anomaly_lines.append(line)

        # 检查IMG2异常
        if abs(d_lat_img2) > threshold_lat_img2 or abs(d_lon_img2) > threshold_lon_img2:
            if abs(d_lat_img2) > threshold_lat_img2 and abs(d_lon_img2) > threshold_lon_img2:
                anomaly_type_img2 = "Right coordinate anomaly"
            else:
                anomaly_type_img2 = "Image pixel anomaly"
            line = (f"IMG2 - 点ID {point_id}: DOM {dom_point} - IMG2 {img2_point} 的偏移量 "
                    f"diff_lat: {d_lat_img2:.2f} m, diff_lon: {d_lon_img2:.2f} m "
                    f"({anomaly_type_img2}), 阈值 (lat: {threshold_lat_img2:.2f} m, lon: {threshold_lon_img2:.2f} m)\n")
            anomaly_lines.append(line)

    if anomaly_lines:
        anomaly_file_path = os.path.join(output_folder, "anomaly_report.txt")
        try:
            with open(anomaly_file_path, 'w') as anomaly_file:
                anomaly_file.write(f"基于 diff_lat 与 diff_lon 的平均绝对偏差:\n")
                anomaly_file.write(
                    f"IMG1 - avg_abs_lat: {img1_stats['avg_abs_lat']:.2f} m, avg_abs_lon: {img1_stats['avg_abs_lon']:.2f} m\n")
                anomaly_file.write(
                    f"IMG2 - avg_abs_lat: {img2_stats['avg_abs_lat']:.2f} m, avg_abs_lon: {img2_stats['avg_abs_lon']:.2f} m\n")
                anomaly_file.write("以下点对偏移量异常:\n")
                anomaly_file.writelines(anomaly_lines)
            log_func(f"异常报告已写入: {anomaly_file_path}")
        except Exception as e:
            log_func(f"异常报告写入失败：{e}")
    else:
        log_func("未发现异常点对。")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("双影像点数据处理工具")
        self.geometry("800x650")
        self.create_widgets()

    def create_widgets(self):
        # 定义各输入项及对应的标签、文本框和按钮
        self.fields = {
            "DOM Shapefile": {"path": tk.StringVar(), "filetypes": [("Shapefile", "*.shp")]},
            "IMG1 Shapefile": {"path": tk.StringVar(), "filetypes": [("Shapefile", "*.shp")]},
            "IMG2 Shapefile": {"path": tk.StringVar(), "filetypes": [("Shapefile", "*.shp")]},
            "RPC1 File (IMG1)": {"path": tk.StringVar(), "filetypes": [("文本文件", "*.txt")]},
            "RPC2 File (IMG2)": {"path": tk.StringVar(), "filetypes": [("文本文件", "*.txt")]},
            "DEM File": {"path": tk.StringVar(), "filetypes": [("TIFF 文件", "*.tif"), ("GeoTIFF", "*.tiff")]},
            "输出文件夹": {"path": tk.StringVar()}
        }

        row = 0
        for label_text, options in self.fields.items():
            tk.Label(self, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky="w")
            entry = tk.Entry(self, textvariable=options["path"], width=50)
            entry.grid(row=row, column=1, padx=5, pady=5)
            if label_text == "输出文件夹":
                btn = tk.Button(self, text="选择文件夹", command=lambda var=options["path"]: self.select_folder(var))
            else:
                btn = tk.Button(self, text="选择文件",
                                command=lambda var=options["path"], ft=options["filetypes"]: self.select_file(var, ft))
            btn.grid(row=row, column=2, padx=5, pady=5)
            row += 1

        # 运行按钮
        self.run_btn = tk.Button(self, text="运行", command=self.run_processing, width=20, bg="lightgreen")
        self.run_btn.grid(row=row, column=0, columnspan=3, pady=10)

        # 日志输出区域
        self.log_box = scrolledtext.ScrolledText(self, width=80, height=20)
        self.log_box.grid(row=row + 1, column=0, columnspan=3, padx=5, pady=5)

    def select_file(self, var, filetypes):
        path = filedialog.askopenfilename(title="选择文件", filetypes=filetypes)
        if path:
            var.set(path)

    def select_folder(self, var):
        path = filedialog.askdirectory(title="选择文件夹")
        if path:
            var.set(path)

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def run_processing(self):
        # 获取所有路径
        dom_shp = self.fields["DOM Shapefile"]["path"].get()
        img1_shp = self.fields["IMG1 Shapefile"]["path"].get()
        img2_shp = self.fields["IMG2 Shapefile"]["path"].get()
        rpc1_file = self.fields["RPC1 File (IMG1)"]["path"].get()
        rpc2_file = self.fields["RPC2 File (IMG2)"]["path"].get()
        dem_file = self.fields["DEM File"]["path"].get()
        output_folder = self.fields["输出文件夹"]["path"].get()

        if not all([dom_shp, img1_shp, img2_shp, rpc1_file, rpc2_file, dem_file, output_folder]):
            messagebox.showerror("错误", "请确保所有输入项均已选择。")
            return

        try:
            # 读取点数据（带编号）
            self.log("正在读取Shapefile...")
            dom_points_dict = read_shapefile_points_with_id(dom_shp)
            img1_points_dict = read_shapefile_points_with_id(img1_shp)
            img2_points_dict = read_shapefile_points_with_id(img2_shp)

            self.log(f"DOM 点数: {len(dom_points_dict)}")
            self.log(f"IMG1 点数: {len(img1_points_dict)}")
            self.log(f"IMG2 点数: {len(img2_points_dict)}")

            # 加载 RPC 模型
            self.log("正在加载RPC模型...")
            rpc1_model = rpcm.rpc_from_rpc_file(rpc1_file)
            rpc2_model = rpcm.rpc_from_rpc_file(rpc2_file)

            # 定义输出文件路径
            output_csv_path = os.path.join(output_folder, "output_points.csv")
            gcp_output_path = os.path.join(output_folder, "gcps.txt")

            # 写入统一 GCP 文件
            self.log("正在生成统一GCP文件...")
            write_unified_gcp_file(dom_points_dict, img1_points_dict, img2_points_dict,
                                   rpc1_model, rpc2_model, dem_file, gcp_output_path, self.log)

            # 写入 CSV 文件
            self.log("正在生成CSV文件...")
            write_points_to_csv(dom_points_dict, img1_points_dict, img2_points_dict, output_csv_path, self.log)

            # 计算差值统计
            self.log("正在计算统计信息...")
            stats = compute_differences_dual_image(dom_points_dict, img1_points_dict, img2_points_dict)

            self.log(f"IMG1 - 平均欧氏偏移量: {stats['img1']['avg_offset']:.2f} m")
            self.log(f"IMG1 - 平均偏移量lat: {stats['img1']['avg_abs_lat']:.2f} m")
            self.log(f"IMG1 - 平均偏移量lon: {stats['img1']['avg_abs_lon']:.2f} m")

            self.log(f"IMG2 - 平均欧氏偏移量: {stats['img2']['avg_offset']:.2f} m")
            self.log(f"IMG2 - 平均偏移量lat: {stats['img2']['avg_abs_lat']:.2f} m")
            self.log(f"IMG2 - 平均偏移量lon: {stats['img2']['avg_abs_lon']:.2f} m")

            # 生成异常报告
            self.log("正在生成异常报告...")
            write_anomaly_report_dual_image(dom_points_dict, img1_points_dict, img2_points_dict,
                                            stats, output_folder, self.log)

            self.log("所有文件处理完毕！")
            messagebox.showinfo("完成", "处理完毕！")

        except Exception as e:
            error_msg = traceback.format_exc()
            self.log("处理过程中发生错误：")
            self.log(error_msg)
            messagebox.showerror("错误", f"处理过程中发生错误：{e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
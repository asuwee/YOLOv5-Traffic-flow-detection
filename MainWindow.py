import os
import sys
import time
from gc import collect
from math import acos, pi, sqrt
from typing import Any, Dict, List

import cv2
import numpy as np
import torch.cuda
import torchvision.transforms as transforms
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import (QIcon, QImage, QMouseEvent, QPixmap, QResizeEvent,
                         QStandardItem, QStandardItemModel)
from PyQt5.QtWidgets import (QAbstractItemView, QAction, QApplication,
                             QCheckBox, QComboBox, QFileDialog, QFormLayout,
                             QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QMainWindow, QMessageBox,
                             QPushButton, QSlider, QStackedWidget, QTableView,
                             QTextEdit, QToolButton, QVBoxLayout, QWidget)
from torch.nn import ModuleList

from lib.tracker import Tracker
from lib.utils import (IMG_FORMATS, VID_FORMATS, Detection, LoadImages,
                       NearestNeighborDistanceMetric, check_img_size,
                       non_max_suppression, scale_coords, select_device,
                       time_sync, xyxy2xywh)
from models.common import DetectMultiBackend
from models.osnet import osnet_x0_5, osnet_x0_25, osnet_x0_75, osnet_x1_0


# 错误代码
class ERROR_CODES:

    @staticmethod
    def E0000(exce: str = ""):
        r"""E0000: 操作已成功完成"""
        info = "E0000\n" + "操作已成功完成\n" + exce
        QMessageBox.information(None, "ERROR", info)

    @staticmethod
    def E0001(exce: str = ""):
        r"""E0001: 检测线程创建失败"""
        info = "E0001\n" + "检测线程创建失败\n" + exce
        QMessageBox.information(None, "ERROR", info)

    @staticmethod
    def E0002(exce: str = ""):
        r"""E0002: 检测参数缺失"""
        info = "E0002\n" + "检测参数缺失\n" + exce
        QMessageBox.information(None, "ERROR", info)

    @staticmethod
    def E0003(exce: str = ""):
        r"""E0003: 检测文件类型错误"""
        info = "E0003\n" + "检测文件类型错误\n" + exce
        QMessageBox.information(None, "ERROR", info)

    @staticmethod
    def E0007(exce: str = ""):
        r"""E0007: 选择检测线超出画框位置"""
        info = "E0007\n" + "选择检测线超出画框位置\n" + exce
        QMessageBox.information(None, "消息", info)

    @staticmethod
    def ERROR(exce: str = ""):
        r"发生错误"
        info = "ERROR\n" + exce
        QMessageBox.information(None, "ERROR", info)


ABOUT = {
    '版本': 'v1.0',
    '日期': '2022-04-18',
    'OS': 'Windows 10 x64',
    'Python': 'Python 3.8.10'
}


# 配置参数
YOLO_MODELS = { "低-yolov5n":"weights/yolov5n.pt", 
                "中-yolov5s":"weights/yolov5s.pt", 
                "较高-yolov5m":"weights/yolov5m.pt", 
                "高-yolov5l":"weights/yolov5l.pt", 
                "最高-yolov5x":"weights/yolov5x.pt"}
DEEPSORT_MODELS = { 'osnet_x0_25': osnet_x0_25, 'osnet_x0_5': osnet_x0_5, 'osnet_x0_75': osnet_x0_75, 'osnet_x1_0': osnet_x1_0}
IMG_SCALES  = { "低-320x320":(320, 320), "中-640x640":(640, 640), "高-960x960":(960, 960)}
TARGETS_FILE = '--TARGETS--'


# 目标分类的颜色代码库
COLORS = [  'FF3838', '00C2FF', '344593', '6473FF', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', 'FF9D97', 'FF701F', 'FFB21D', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7'] 


class Extractor(object):
    def __init__(self, model_type, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.input_width = 128
        self.input_height = 256

        self.model = DEEPSORT_MODELS[model_type](num_classes=1000, loss='softmax', pretrained=True, use_gpu=True)
        self.model.to(self.device)
        self.model.eval()

        print("Selected model type: {}".format(model_type))
        self.size = (self.input_width, self.input_height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


    def _preprocess(self, im_crops):
        """
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()


class DeepSort(object):
    def __init__(self, model_type, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):

        self.extractor = Extractor(model_type, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=True):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int16))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


class Geometry:
    @staticmethod
    def calPoint2Point(x1, y1, x2, y2) -> int:
        r"""calPoint2Point(x1, y1, x2, y2) -> int
        @brief 计算点到点的距离
        @param x1 点1的x坐标
        @param y1 点1的y坐标
        @param x2 点2的x坐标
        @param y2 点2的y坐标
        @return int 点1到点2的距离
        """
        return int(sqrt(pow(x1-x2, 2) + pow(y1-y2, 2)))

    @staticmethod
    def calPoint2Line(lx1, ly1, lx2, ly2, xx, yy, absolute: bool = True) -> int:
        r"""calPoint2Line(lx1, ly1, lx2, ly2, xx, yy, absolute: bool = True) -> int
        @brief 计算点到线的距离
        @param lx1 线点1的x坐标
        @param ly1 线点1的y坐标
        @param lx2 线点2的x坐标
        @param ly2 线点2的y坐标
        @param xx 点的x坐标
        @param yy 点的y坐标
        @return int 点到由点1与点2连线的距离
        """
        dis = ((ly1-ly2)*xx + (lx2-lx1)*yy + lx1*ly2 - ly1*lx2) / sqrt(pow(ly1-ly2, 2) + pow(lx1-lx2, 2))
        if absolute:
            return abs(dis)
        else:
            return dis

    @staticmethod
    def calAngle(x1, y1, cx, cy, x2, y2) -> int:
        r"""calAngle(x1, y1, cx, cy, x2, y2) -> int
        @brief 计算点到线的距离
        @param x1 点1的x坐标
        @param y2 点1的y坐标
        @param cx 顶点的x坐标
        @param cy 顶点的y坐标
        @param x2 点2的x坐标
        @param y2 点2的y坐标
        @return int 以(cx, cy)为顶点, 三点形成的角度
        """
        dx1 = x1 - cx
        dy1 = y1 - cy

        dx2 = x2 - cx
        dy2 = y2 - cy

        c = sqrt(dx1*dx1 + dy1*dy1) * sqrt(dx2*dx2 + dy2*dy2)
        angle = acos((dx1 * dx2 + dy1 * dy2) / c)
            
        return int(angle/pi*180)

    @staticmethod
    def calDropOfPoint2Line(lx1, ly1, lx2, ly2, xx, yy) -> Any:
        r"""calDropOfPoint2Line(lx1, ly1, lx2, ly2, xx, yy) -> Any
        @brief 计算点到两点连线的垂足
        @param lx1 线点1的x坐标
        @param ly1 线点1的y坐标
        @param lx2 线点2的x坐标
        @param ly2 线点2的y坐标
        @param xx 点的x坐标
        @param yy 点的y坐标
        @return rx, ry
        """
        dx = lx1 - lx2
        dy = ly1 - ly2
        u = (xx - lx1)*(lx1 - lx2) + (yy - ly1)*(ly1 - ly2)
        u = u / (dx*dx+dy*dy)
        rx = lx1 + u*dx
        ry = ly1 + u*dy
        return rx, ry

    @staticmethod
    def calPoint2Plane(point, plane) -> Any:
        r"""calPoint2Plane(x, y, points) -> Any
        @brief 计算点到面的距离
        @param point 点
        @param plane 面
        """
        xx, yy = point[0], point[1]
        # dis = None
        # rx, ry = xx, yy

        # # 计算面上顶点到目标的位置
        # for p in plane:
        #     x, y = p[0], p[1]
        #     _dis = calPoint2Point(xx, yy, x, y)
        #     if dis == None:
        #         rx, ry = x, y
        #         dis = _dis
        #     elif _dis < dis:
        #         dis = _dis
        #         rx, ry = x, y
        xl, yl = [], []
        for i in range(-1, len(plane)-1):
            x1, y1 = plane[i][0], plane[i][1]
            x2, y2 = plane[i+1][0], plane[i+1][1]
            xl.extend(np.linspace(x1, x2, 100))
            yl.extend(np.linspace(y1, y2, 100))
        rx, ry = xl[0], yl[0]
        for i in range(len(xl)):
            dis = Geometry.calPoint2Point(xl[i], yl[i], xx, yy)
            disr = Geometry.calPoint2Point(rx, ry, xx, yy)
            if disr > dis:
                rx = xl[i]
                ry = yl[i]

        return rx, ry

    @staticmethod
    def isPointInPlane(pt: List[int], area: List[List[int]]) -> bool:
        r"""
        @brief 判断点是否在面里
        @param pt List[int] 点
        @param area IgnoreArea 面
        """
        xx, yy = pt[0], pt[1]

        # 判断点是否在组成面的点中间
        count = 0
        for i in range(-1, len(area)-1):
            x1, y1 = area[i][0], area[i][1]
            x2, y2 = area[i+1][0], area[i+1][1]
            # https://blog.csdn.net/yessharing/article/details/60734345
            if (x1 < xx and xx < x2) or (x2 < xx and xx < x1):
                k = (y2 - y1) / (x2 - x1)
                if yy < (k*(xx-x1)+y1):
                    count += 1
        if count % 2 == 1:
            return True
        else:
            return False


class DetectionParams:
    r"""
    @brief 检测时需要使用的参数
    """
    source = None # 检测文件路径 file/dir/URL/glob, 0 for webcam
    framesCount = None # 检测文件的帧数

    weights = None # 权重文件路径 model.pt path(s)
    names = None # 可识别目标

    imgsz = None      # 推断规模 inference size (height, width)
    classes: List[int] = []      # 检测目标 filter by class: --class 0, or --class 0 2 3

    minConf = 0 # 最低置信度

    deepsort = "osnet_x0_25" # DEEPSORT模型名称

    height = None # 高度
    width = None # 宽度

    def getMissingYOLOParam(self) -> str:
        r"""getMissingYOLOParam(self) -> str
        @brief 获取缺少设置的参数
        @return str 未设置的参数
        """
        if self.weights == None:
            return "权重文件路径未设置"
        if self.source == None:
            return "检测文件路径未设置"
        if self.imgsz == None:
            return "推断规模未设置"
        if len(self.classes) == 0:
            return "检测目标未设置"
        return None


class Target:
    r"""class Target
    @brief 检测出的目标
    """
    id = None # 目标编号
    clss = None # 分类
    conf = None # 置信度
    frameId = None # 图像编号
    box = None # 位置 

    def __init__(self, frameId: int, clss: int, conf: float, box: List[int], id: int = None) -> None:
        r"""构造函数"""
        self.id = id
        self.frameId = frameId
        self.conf = conf
        self.clss = clss
        self.box = box


class TargetProcess:
    r"""
    @brief 处理目标相关事务
    """

    @staticmethod
    def getTargetListByFrameId(TargetList: List[Target], frameId: List[int]) -> List[Target]:
        r"""
        @brief 通过图像编号, 查找所有符合出现在图像中的目标
        @param TargetList List[Target] 目标列表
        @param frameId List[int] 图像编号
        @return List[Target] 符合要求的目标列表
        """
        targetList: List[Target] = []

        for target in TargetList:
            if target.frameId in frameId:
                targetList.append(target)

        return targetList

    @staticmethod
    def getTargetListByClss(TargetList: List[Target], clss: List[int]) -> List[Target]:
        r"""
        @brief 通过目标分类, 查找所有符合目标分类的目标
        @param TargetList List[Target] 目标列表
        @param clss List[int] 分类编号
        @return List[Target] 符合要求的目标列表
        """
        targetList: List[Target] = []

        for target in TargetList:
            if target.clss in clss:
                targetList.append(target)

        return targetList

    @staticmethod
    def getTargetListByTargetId(TargetList: List[Target], targetId: List[int]) -> List[Target]:
        r"""
        @brief 通过目标编号, 查找所有符合目标分类的目标
        @param TargetList List[Target] 目标列表
        @param targetId List[int] 目标编号
        @return List[Target] 符合要求的目标列表
        """
        targetList: List[Target] = []
        for target in TargetList:
            if target.id in targetId:
                targetList.append(target)
        return targetList

    @staticmethod
    def getTargetListBeforeFrameId(TargetList: List[Target], frameId: int, cover: bool = True) -> List[Target]:
        r"""
        @brief 通过图像, 查找包含图像编号和图像编号以前的目标
        @param TargetList List[Target] 目标列表
        @param frameId int 图像编号
        @param cover bool True包括frameId, False不包含frameId
        @return List[Target] 符合要求的目标列表
        """
        targetList: List[Target] = []
        for target in TargetList:
            if cover and target.frameId <= frameId:
                targetList.append(target)
                continue
            if not cover and target.frameId < frameId:
                targetList.append(target)
                continue
        return targetList

    @staticmethod
    def getTargetListByMinimumConfidence(TargetList: List[Target], minConf: float, cover: bool = True) -> List[Target]:
        r"""
        @brief 通过目标最低置信度, 查找置信度高于阈值的目标
        @param TargetList List[Target] 目标列表
        @param minConf float 目标最低置信度
        @param cover bool True包含最低, False不包含最低
        @return List[Target] 符合要求的目标列表
        """
        targetList: List[Target] = []
        for target in TargetList:
            if cover and target.conf >= minConf:
                targetList.append(target)
                continue
            if not cover and target.conf > minConf:
                targetList.append(target)
                continue
        return targetList

    @staticmethod
    def sortTargetsByTargetId(TargetList: List[Target]) -> Dict:
        r"""
        @brief 根据目标编号整理目标结果
        @param TargetList: List[Target] 需要整理的目标结果
        @return Dict[int:List[Target]] 整理完成的目标结果
        """
        targetDictById: Dict = {}
        for target in TargetList:
            if target.id not in targetDictById.keys():
                targetDictById[target.id] = []
                targetDictById[target.id].append(target)
            else:
                targetDictById[target.id].append(target)
        return targetDictById


class Line:
    r"""
    @brief 检测线
    """
    name: str = None # 检测线名称
    pt1 = None # 检测线的1号点
    pt2 = None # 检测线的2号点
    coming = None # 目标来的方向

    def __init__(self, name: str = None) -> None:
        r"""
        @brief 构造函数
        @param name str 检测线名称
        """
        self.name = name
    
    def setComingDirection(self, x: int , y: int) -> bool:
        r"""setComingDirection(self, x:int , y:int) -> bool
        @brief 设置目标去的方向
        @param x int 目标的x坐标
        @param y int 目标的y坐标
        @return true 设置成功
        @return false 目标点在线上或来去方向相同
        """
        lx1, ly1 = self.pt1[0], self.pt1[1]
        lx2, ly2 = self.pt2[0], self.pt2[1]
        tmp = (ly1-ly2)*x + (lx2-lx1)*y + lx1*ly2 - ly1*lx2
        if tmp == 0:
            return False
        if tmp > 0:
            self.coming = 1
            return True
        if tmp < 0:
            self.coming = -1
            return True
        return False


class Area:
    r"""
    @brief 检测区
    """
    name: str = None # 区域名称
    pts: List[List[int]] = None # 构成区域的点

    def __init__(self, name: str = None) -> None:
        r"""
        @brief 构造函数
        @param name str 区域名称
        """
        self.name = name
        self.pts = []


class TrafficFlow:
    @staticmethod
    def TargetsPassLine(line: Line, targetList: List[Target]) -> List[Target]:
        r"""
        @brief 统计经过检测线目标
        @param line DetectionLine 检测线
        @param targetList List[Target] 需要分辨的目标
        @return List[Target] 经过检测线的目标
        """
        # 交通流统计(监测线)
        if line.pt1 == None or line.pt2 == None:
            return []
        lx1, ly1 = line.pt1[0], line.pt1[1]
        lx2, ly2 = line.pt2[0], line.pt2[1]
        
        TargetStatistics = {} # key: 目标编号 val: List 目标距离检测线的距离
        PassLineTargetList: List[Target] = []

        for target in targetList:
            boxX, boxY= int((target.box[0]+target.box[2])/2), int((target.box[1]+target.box[3])/2)   # 目标框的中心
            # 目标与直线的关系
            angle1 = Geometry.calAngle(boxX, boxY, lx1, ly1, lx2, ly2) # 0<angle1<90
            angle2 = Geometry.calAngle(boxX, boxY, lx2, ly2, lx1, ly1) # 0<angle2<90
            if angle1 > 0 and angle1 < 90 and angle2 > 0 and angle2 < 90:
                # 目标位于直线的范围内 目标中心 - 监测线两端中一点 - 监测线两端中另一点 的夹角 应当 大于0 并且 小于 90
                # s1 计算目标到监测线的距离 带正负
                dis = Geometry.calPoint2Line(lx1, ly1, lx2, ly2, boxX , boxY, False)
                # s2 根据目标编号查询 目标距离线的距离
                if target.id not in TargetStatistics.keys():
                    TargetStatistics[target.id] = []
                    TargetStatistics[target.id].append(dis)
                else:
                    TargetStatistics[target.id].append(dis)
                # s3 目标距离线的距离是否有正负变化
                if len(TargetStatistics[target.id]) >=2 and TrafficFlow.isTargetPassLine(TargetStatistics[target.id][-2], TargetStatistics[target.id][-1], line.coming):
                    # print(f"目标编号 {target.id} 在 {target.frameId} 经过了检测线")
                    PassLineTargetList.append(target)
                    continue
        return PassLineTargetList

    @staticmethod
    def TargetsNotInArea(area: Area, targetList: List[Target]) -> List[Target]:
        r"""
        @brief 统计不在区域内的目标
        @param area Area 区域
        @param targetList List[Target] 需要分辨的目标
        @return List[Target] 不在忽略区域内的目标
        """
        NotInAreaTargetList: List[Target] = []
        for target in targetList:
            if not TrafficFlow.isTargetInArea(area, target):
                NotInAreaTargetList.append(target)
        return NotInAreaTargetList

    @staticmethod
    def TargetsNotInAreas(areaList: List[Area], targetList: List[Target]) -> List[Target]:
        r"""
        @brief 统计不再区域内的目标
        @param areaList  List[Area] 忽略区域
        @param targetList List[Target] 需要分辨的目标
        @return List[Target] 不在忽略区域内的目标
        """
        NotInAreaTargetList: List[Target] = targetList
        for area in areaList:
            NotInAreaTargetList = TrafficFlow.TargetsNotInArea(area, NotInAreaTargetList)
        return NotInAreaTargetList

    @staticmethod
    def TargetsInArea(area: Area, targetList: List[Target]) -> List[Target]:
        r"""
        @brief 获取所有进入区域的目标
        @param area Area 检测区域
        @param targetList List[Target] 需要判断的目标
        @param List[Target] 所有进入区域的目标
        """
        targetsInArea: List[Target] = []
        for target in targetList:
            if TrafficFlow.isTargetInArea(area, target):
                targetsInArea.append(target)
        return  targetsInArea

    @staticmethod
    def isTargetPassLine(dis1, dis2, coming: int = None) -> bool:
        r"""
        @brief 判断目标是否经过了线 距离存在正负
        @param dis1 int 第一帧目标到线的距离
        @param dis2 int 下一帧目标到线的距离
        @param coming 检测时固定的检测方向
        @return true: 目标经过检测线, false: 目标未经过检测线
        """
        # 没有检测方向
        if coming == None:
            if dis1 * dis2 < 0:
                return True
            if dis1 == 0 and dis2 != 0:
                return True
            if dis1 != 0 and dis2 == 0:
                return True
            return False

        # 检测反向为正
        if coming > 0:
            if dis1 >=0 and dis2 < 0:
                return True
            return False

        # 检测反向为负
        if coming < 0:
            if dis1 <=0 and dis2 > 0:
                return True
            return False

    @staticmethod
    def isTargetInArea(area: Area, target: Target) -> bool:
        r"""
        @brief 判断目标是否在区域内
        @param area Area 区域
        @param target Target 需要判断的目标
        @return bool true 目标在区域内 false 目标不再区域内
        """
        boxX, boxY= int((target.box[0]+target.box[2])/2), int((target.box[1]+target.box[3])/2)   # 目标框的中心
        if Geometry.isPointInPlane([boxX, boxY], area.pts): # 判断目标点是否在区域内
            return True
        else:
            return False


class Annotator:
    r"""YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations"""
    def __init__(self, image: np.array) -> None:
        r"""
        @brief 构造函数
        @param im np.array cv2标注使用的图像
        """
        self.__image = image

    def box_label(self, box: List[int], label: str = "", color=(128, 128, 128), txt_color=(255, 255, 255), thickness: int = 1) -> None:
        r"""box_label(self, box: List[int], label: str = "", color=(128, 128, 128), txt_color=(255, 255, 255), thickness: int = 1) -> None
        @brief 目标标注
        @param box List[int] 目标位置 xyxy 左上右下
        @param label str 目标标签
        @param color (x, x, x) 框的颜色
        @param text_color (x, x, x) 文字的颜色
        @param thickness int 框和文字的粗细
        """
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.__image, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        if len(label) > 0:   # 标签非空
            tf = max(thickness - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.__image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.__image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, thickness / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            

    def circle(self, center, radius, color=(128, 128, 128), thickness=1, lineType=None, shift=None) -> None:
        r"""circle(self, center, radius, color=(128, 128, 128), thickness=1, lineType=None, shift=None) -> None
        @brief 画圆
        @param center 圆心坐标 如 (100, 100)
        @param radius 半径, 如 10
        @param color 圆边框颜色 如 (0, 0, 255) 红色 BGR
        @param thickness 正值表示圆边框宽度. 负值表示画一个填充圆形
        @param lineType 圆边框线型, 可为 0, 4, 8
        @param shift 圆心坐标和半径的小数点位数
        """
        cv2.circle(self.__image, center, radius, color, thickness)

    def circles(self, centers, radius, color=(128, 128, 128), thickness=1, lineType=None, shift=None) -> None:
        r"""circles(self, centers, radius, color=(128, 128, 128), thickness=1, lineType=None, shift=None) -> None
        @brief 画圆
        @param centers 圆心坐标 如 (100, 100)
        @param radius 半径, 如 10
        @param color 圆边框颜色 如 (0, 0, 255) 红色 BGR
        @param thickness 正值表示圆边框宽度. 负值表示画一个填充圆形
        @param lineType 圆边框线型, 可为 0, 4, 8
        @param shift 圆心坐标和半径的小数点位数
        """
        for center in centers:
            cv2.circle(self.__image, center, radius, color, thickness) 

    def line(self, p1, p2, color=(128, 128, 128), thickness=1, lineType=cv2.LINE_AA) -> None:
        r"""line(self, xy, color=(128, 128, 128), thickness=1, lineType=None) -> None
        @brief 划线
        @param pt1, pt2 线两端的坐标
        @param color _Ink 填充颜色
        @param thickness int 线的宽度
        @param lineType 线的类型 
        """
        cv2.line(self.__image, p1, p2, color, thickness, lineType)

    def plane(self, points, color=(128, 128, 128), thickness=1, lineType=None) -> None:
        r"""plane(self, points, color=(128, 128, 128), thickness=1, lineType=None) -> None
        @brief 面
        @param points [[x1, y2], [x2, y2] ... ]
        @param color _Ink 填充颜色
        @param thickness int 线的宽度
        @param lineType 线的类型 
        """
        for p in range(len(points)-1):
            p1 = points[p]
            p2 = points[p+1]
            cv2.line(self.__image, p1, p2, color, thickness, lineType)   
        p1 = points[-1]
        p2 = points[0]
        cv2.line(self.__image, p1, p2, color, thickness, lineType)  

    def text(self, text: str, org, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=(255, 255, 255), thickness: int = 1) -> None:
        r"""text(self, text: str, org, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=(255, 255, 255), thickness: int = 1) -> None
        @brief 文本显示
        @param text 需要显示的文本
        @param org 文本框的左下角
        """
        cv2.putText(self.__image, text, org, fontFace, fontScale, color, thickness, lineType = cv2.LINE_AA)

    def arrowedLine(self, pt1, pt2, color, thickness, tiplength) -> None:
        r"""
        @brief 
        """
        cv2.arrowedLine(self.__image, pt1, pt2, color=color, thickness=thickness, tipLength=tiplength)

    def result(self) -> np.array:
        r"""result(self) -> np.array
        @brief Return annotated image as array
        """
        return np.asarray(self.__image)


class DetectionVisualization:
    r"""
    @brief 检测结果可视化
    """

    @staticmethod
    def VisualTargets(image: np.array, TargetList: List[Target], names: List[str] = None, colors: List[str] = COLORS, showId = True, showClss = True, showConf = False) -> np.array:
        r"""VisualTargets(image: np.array, TargetList: List[Target], names: List[str] = None, colors: List[str] = COLORS) -> np.array
        @brief 将所有目标标注到图像上
        @param image np.array 需要标注的图像
        @param TargetList 需要标注的目标结果
        @param names List[str] 目标分类的名称
        @param colors List[str] 分类对应的hex颜色代码
        @return np.array 标注完成的图像
        """
        # 创建绘图
        annotator = Annotator(image)

        # 将目标标注在图像上
        for target in TargetList:
            # 对检测出的目标进行标注
            xyxy = target.box  # 目标的左上和右下
            id = target.id # 目标的编号
            name = names[target.clss] if names != None else target.clss # 目标的分类 
            conf = target.conf     # 目标的置信度
            if showId and showClss and showConf:
                label = f"{str(id)} {name} {conf:.2f}"
            elif showId and showClss and not showConf:
                label = f"{str(id)} {name}"
            elif showId and not showClss and showConf:
                label = f"{str(id)} {conf:.2f}"
            elif showId and not showClss and not showConf:
                label = f"{str(id)}"
            elif not showId and showClss and showConf:
                label = f"{name} {conf:.2f}"
            elif not showId and showClss and not showConf:
                label = f"{name}"
            elif not showId and not showClss and showConf:
                label = f"{conf:.2f}"
            else:
                label = ''
                
            annotator.box_label(xyxy, label, color=DetectionVisualization.hex2rgb(colors[target.clss]), thickness=2) # 在图像上进行目标标注
        
        # 返回绘图
        return annotator.result()

    @staticmethod
    def VisualDetectionLine(image: np.array, line: Line, color: List[int] = (255, 255, 255)) -> np.array:
        r"""VisualDetectionLines(image: np.array, DetectionLineList: List[DetectionLine]) -> np.array
        @brief 将检测线绘制到图像上
        @param image np.array 需要绘制的图像
        @param line DetectionLine 需要绘制的检测线
        @param lineColor List[int] 线的颜色
        @return 绘制完成的图像
        """
        
        # 创建绘图
        annotator = Annotator(image)

        if line.pt1 == None:
            return annotator.result()
        if line.pt2 == None:
            annotator.circle(center=line.pt1, color=color, radius=5, thickness=-1)
            return annotator.result()
        # 绘制检测线
        annotator.line(line.pt1, line.pt2, color=color, thickness=2)
        # 绘制检测方向
        if line.coming == None:
            return annotator.result()
        x1, y1 = line.pt1[0], line.pt1[1]
        x2, y2 = line.pt2[0], line.pt2[1]
        a, b = int((line.pt1[0]+line.pt2[0])/2), int((line.pt1[1]+line.pt2[1])/2)
        if abs(x1 - x2) < 5: # 竖直的线
            dis = int(abs(y1-y2)/2)
            m1 = a - dis
            n1 = b
            m2 = a + dis
            n2 = b
        elif abs(y1 - y2) < 5: # 横向的线
            dis = int(abs(x1-x2)/2)
            m1 = a
            n1 = b - dis
            m2 = a
            n2 = b + dis
        else: # 已知正方形对角的两个顶点，求另外两个顶点
            m1 = int((x1-y1+x2+y2)/2)
            n1 = int((x1+y1-x2+y2)/2)
            m2 = int((x1+y1+x2-y2)/2)
            n2 = int((-x1+y1+x2+y2)/2)
        tmp = (y1-y2)*m1 + (x2-x1)*n1 + x1*y2 - y1*x2
        if tmp * line.coming > 0:
            m = int((m1+a)/2)
            n = int((n1+b)/2)
        else:
            m = int((m2+a)/2)
            n = int((n2+b)/2)
        annotator.arrowedLine((m, n), (a, b), color, 2, 0.35)
        
        # 返回绘图
        return annotator.result()

    @staticmethod
    def VisualDetectionLines(image: np.array, lineList: List[Line], color: List[int] = (255, 255, 255)) -> np.array:
        r"""VisualDetectionLines(image: np.array, lineList: List[DetectionLine]) -> np.array
        @brief 将检测线绘制到图像上
        @param image np.array 需要绘制的图像
        @param lineList List[DetectionLine] 需要绘制的检测线
        @return 绘制完成的图像
        """
        # 标注所有检测线
        for line in lineList:
            image = DetectionVisualization.VisualDetectionLine(image, line, color)
        return image
    
    @staticmethod
    def VisualRelationBetweenDetectoinLinesAndTargets(image: np.array, DetectionLineList: List[Line], TargetList: List[Target], colors: List[str] = COLORS) -> np.array:
        r"""
        @brief 将检测线与目标之间的关系进行可视化
        @param image np.array 需要绘制的图像
        @param DetectionLineList List[DetectionLine] 需要绘制在图像上的检测线
        @param TargetList: List[Target] 需要绘制在图像上的目标
        @return np.array 绘制完成后的图像
        """
        # 创建绘图
        annotator = Annotator(image)

        # 将关系绘制在图像上
        for target in TargetList:
            for line in DetectionLineList:
                if line.pt1 == None or line.pt2 == None:
                    continue
                xx, yy = int((target.box[0]+target.box[2])/2), int((target.box[1]+target.box[3])/2)   # 目标框的中心
                mx1, my1 = line.pt1[0], line.pt1[1]
                mx2, my2 = line.pt2[0], line.pt2[1]
                dis = Geometry.calPoint2Line(mx1, my1, mx2, my2, xx , yy, False) # 目标到检测线的距离
                # 只显示符合检测线来方向的目标
                if line.coming != None and dis * line.coming <= 0:
                    continue
                # 目标与直线间的关系
                angle1 = Geometry.calAngle(xx, yy, mx1, my1, mx2, my2) # 0<angle1<90
                angle2 = Geometry.calAngle(xx, yy, mx2, my2, mx1, my1) # 0<angle2<90
                if angle1 > 0 and angle1 < 90 and angle2 > 0 and angle2 < 90:
                    # 目标位于直线的范围内
                    mx, my = Geometry.calDropOfPoint2Line(mx1, my1, mx2, my2, xx, yy)    # 获得目标对于线段的垂足
                    mx, my = int(mx), int(my)
                    annotator.line([xx, yy], [mx, my], DetectionVisualization.hex2rgb(colors[target.clss]), 1)
                    annotator.text(str(int(abs(dis))), (int((xx+mx)/2), int((yy+my)/2)), fontScale=0.6)  # 标注目标到监测点的距离
        
        # 返回绘图
        return annotator.result()

    @staticmethod
    def VisualCrossTargets(image: np.array, targetList: List[Target], colors: List[str] = COLORS) -> np.array:
        r"""
        @brief 对目标框内进行打叉
        """
        # 创建绘图
        annotator = Annotator(image)

        # 标注交通流统计结果
        for target in targetList:
            # 将该目标打叉
            x1, y1 = target.box[0], target.box[1] # 左上角
            x2, y2 = target.box[2], target.box[3] # 右下角
            # m, n = int((x1+x2)/2), int((y1+y2)/2) # 中点
            x3, y3 = x1, y2 # 左下角
            x4, y4 = x2, y1 # 右上角
            annotator.line((x1, y1), (x2, y2), color=(255, 255, 255), thickness=1)
            annotator.line((x3, y3), (x4, y4), color=(255, 255, 255), thickness=1)
            # annotator.circle(center=[m, n], radius=5, color=DetectionVisualization.hex2rgb(colors[target.clss]), thickness=-1)

        return annotator.result()

    @staticmethod
    def VisualArea(image: np.array, area: Area, color: List[int] = (255, 255, 255)) -> np.array:
        r"""
        @brief 将忽略区绘制到图像上
        @param image np.array 需要绘制的图像
        @param area IngoreArea 需要绘制的忽略区
        @param color List[int] 线的颜色
        @return 绘制完成的图像
        """
        # 创建绘图
        annotator = Annotator(image)

        # 绘制忽略区
        for i in range(len(area.pts)-1):
            pt1 = area.pts[i]
            pt2 = area.pts[i+1]
            annotator.line(pt1, pt2, color, thickness=2)
        # 绘制最后的闭合线
        pt1 = area.pts[-1]
        pt2 = area.pts[0]
        annotator.line(pt1, pt2, color, thickness=2)

        # 返回绘图
        return annotator.result()

    @staticmethod
    def VisualAreas(image: np.array, areaList: List[Area], color: List[int] = (255, 255, 255)) -> np.array:
        r"""
        @brief 将忽略区绘制到图像上
        @param image np.array 需要绘制的图像
        @param areaList List[IngoreArea] 需要绘制的忽略区列表
        @param color List[int] 线的颜色
        @return 绘制完成的图像
        """
        for area in areaList:
            image = DetectionVisualization.VisualArea(image, area, color)
        
        return image

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class DetectThread(QThread):
    r"""目标识别线程"""
    signal_new_target = pyqtSignal(Target) # 信号 检测到新的目标
    signal_new_frame = pyqtSignal(int, float)  # 信号 检测完新的图像
    signal_finished = pyqtSignal()  # 信号 检测全部完成

    def __init__(self, parent=None):
        super(DetectThread, self).__init__(parent)
        self.__ThreadStatus = True


    def ready(self):
        self.__ThreadStatus = True


    def setParams(self, params: DetectionParams) -> None:
        self.__params = params


    def stop(self):
        self.__ThreadStatus = False
    
    @torch.no_grad()    # 该函数不进行梯度计算与反向传播
    def run(self):
        # YOLO 参数设置
        weights = self.__params.weights
        imgsz = self.__params.imgsz          # inference size (height, width)
        classes = None     # filter by class: --class 0, or --class 0 2 3
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=1000  # maximum detections per imaage
        device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        half=False  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference
        
        # DeepSort 参数设置
        deep_sort_model = self.__params.deepsort

        # 加载 DeepSort 配置文件
        deepsort = DeepSort(deep_sort_model)

        # Load model 获取设备并加载模型参数
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, onnx, engine = model.stride, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half 如果使用 gpu ，使用 Float16
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        # Dataloader 通过不同的输入源来设置不同的数据加载方式
        dataset = LoadImages(self.__params.source, img_size=imgsz, stride=stride, auto=pt)

        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

        frameId = 0

        for path, im, im0s, vid_cap, count in dataset:
            if self.__ThreadStatus == False:
                print("线程提前中止")
                torch.cuda.empty_cache()
                return
            
            beginTime = time_sync()
            
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference 推断
            pred = model(im, augment=augment, visualize=False)

            # NMS 非极大值抑制
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image         
                im0s

                # 对检测结果进行识别标定
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                    
                    # YOLO + DeepSort
                    xywhs = xyxy2xywh(det[:, 0:4])  # 目标位置
                    confs = det[:, 4]   # 置信度
                    clss = det[:, 5]    # 分类

                    # pass detections to deepsort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0s) # outputs: [a, b, c, d, id, label]

                    # draw boxes for visualization 根据识别结果在图片中标识
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]    # xyxy 左上右下
                            id = output[4]  # 识别编号
                            c = int(output[5])    # 分类编号

                            # 发送信号 检测到新的目标
                            self.signal_new_target.emit(Target(frameId, c, conf, bboxes, id))
                else:
                    deepsort.increment_ages()
                    print("No detections")

                endTime = time_sync()

                # 发送信号 检测完的新的图像
                self.signal_new_frame.emit(frameId, endTime-beginTime)
                frameId = frameId + 1
        
        # 检测结束
        time.sleep(0.5)
        self.__ThreadStatus = False
        torch.cuda.empty_cache()
        self.signal_finished.emit()


DetectThread = DetectThread()


class MAIN(QMainWindow):
    r"主窗口"


    def __init__(self) -> None:
        r"""初始化"""
        super(MAIN, self).__init__()
        self.setWindowTitle("DeepTraffic")

        self.MainWidget = QWidget() # 建立主窗口
        self.MainWidget.setMinimumWidth(1080)
        self.MainWidget.setMinimumHeight(720)

        self.MenuBar() # 菜单栏
        self.Console() # 控制台
        self.Visualization() # 结果展示

        # 功能选择
        self.QWidget_FunctionSelection = QWidget(self.MainWidget)
        self.QWidget_FunctionSelection.setFixedWidth(120)
        self.QWidget_FunctionSelection.move(0, 0)
        self.QVBoxLayout_Widget_FunctionSelection = QVBoxLayout()
        self.QWidget_FunctionSelection.setLayout(self.QVBoxLayout_Widget_FunctionSelection)

        # 功能显示窗口
        self.QStackedWidget_FunctionModule = QStackedWidget(self.MainWidget)
        self.QStackedWidget_FunctionModule.move(120, 0)
        self.QStackedWidget_FunctionModule.setFixedWidth(300)
        self.QStackedWidget_FunctionModule.setVisible(False)
        self.QStackedWidget_FunctionModule.setStyleSheet("background-color:white")

        self.ObjectDetectionAndTracking() # 目标检测
        self.TrafficFlow_DetectionLines() # 交通流 检测线
        self.TrafficFlow_DetectionAreas() # 交通流 检测区
        self.TrafficFlow_IgnoreAreas() # 交通流 忽略区域

        self.assignInitialValue() # 为参数赋予初值

        self.setCentralWidget(self.MainWidget)
        self.showText("初始化完成")


    def LoadConfig(self) -> None:
        r"""Load Config 加载配置文件"""
        # with open("./config.json", 'r', encoding='utf-8') as loadFile:
        #     self.config = json.load(loadFile)
        return


    def MenuBar(self) -> None:
        r"""Menu Bar 菜单栏"""

        # 文件 标签 
        self.MenuBar_File = self.menuBar().addMenu("文件")

        self.QAction_File_OpenFile = QAction("打开文件")
        self.QAction_File_OpenFile.setShortcut("Ctrl+O")
        self.QAction_File_OpenFile.triggered.connect(self.on_QAction_File_OpenFile_triggered)
        self.MenuBar_File.addAction(self.QAction_File_OpenFile)

        self.QAction_File_OpenFileFold = QAction("打开文件夹")
        self.QAction_File_OpenFileFold.setShortcut("Ctrl+P")
        self.QAction_File_OpenFileFold.triggered.connect(self.on_QAction_File_OpenFileFold_triggered)
        self.MenuBar_File.addAction(self.QAction_File_OpenFileFold)

        self.MenuBar_File.addSeparator()

        self.QAction_File_ExportDetectionResult = QAction("导出检测结果")
        self.QAction_File_ExportDetectionResult.triggered.connect(self.on_QAction_File_ExportDetectionResult_triggered)
        self.MenuBar_File.addAction(self.QAction_File_ExportDetectionResult)

        self.QAction_File_ImportDetectionResult = QAction("导入检测结果")
        self.QAction_File_ImportDetectionResult.triggered.connect(self.on_QAction_File_ImportDetectionResult_triggered)
        self.MenuBar_File.addAction(self.QAction_File_ImportDetectionResult)

        self.MenuBar_File.addSeparator()    

        self.QAction_File_ExportAsYOLO = QAction("YOLO格式导出")
        self.QAction_File_ExportAsYOLO.triggered.connect(self.on_QAction_File_ExportAsYOLO_triggered)
        self.MenuBar_File.addAction(self.QAction_File_ExportAsYOLO)

        # 运行 标签
        self.MenuBar_Run = self.menuBar().addMenu("运行")
        
        self.QAction_Run_StartDetection = QAction("启动检测", self)
        self.QAction_Run_StartDetection.setShortcut("F5")
        self.QAction_Run_StartDetection.triggered.connect(self.on_QAction_Run_StartDetection_triggered)
        self.MenuBar_Run.addAction(self.QAction_Run_StartDetection)

        self.QAction_Run_StopDetection = QAction("停止检测", self)
        self.QAction_Run_StopDetection.setShortcut("Shift+F5")
        self.QAction_Run_StopDetection.triggered.connect(self.on_QAction_Run_StopDetection_triggered)
        self.QAction_Run_StopDetection.setEnabled(False)
        self.MenuBar_Run.addAction(self.QAction_Run_StopDetection)

        self.MenuBar_Run.addSeparator()

        self.QAction_Run_QuickTest = QAction("快捷测试", self)
        self.QAction_Run_QuickTest.setShortcut("F6")
        self.QAction_Run_QuickTest.triggered.connect(self.QuickTest)
        self.MenuBar_Run.addAction(self.QAction_Run_QuickTest)

        self.MenuBar_Run.addSeparator()

        self.QAction_Run_PreviousPage = QAction('上一张', self)
        self.QAction_Run_PreviousPage.setShortcut('a')
        self.QAction_Run_PreviousPage.triggered.connect(self.on_QAction_Run_PreviousPage_triggered)
        self.MenuBar_Run.addAction(self.QAction_Run_PreviousPage)

        self.QAction_Run_NextPage = QAction('下一张', self)
        self.QAction_Run_NextPage.setShortcut('d')
        self.QAction_Run_NextPage.triggered.connect(self.on_QAction_Run_NextPage_triggered)
        self.MenuBar_Run.addAction(self.QAction_Run_NextPage)

        # 查看 标签
        self.MenuBar_View = self.menuBar().addMenu("查看")
        
        self.QAction_View_FullScreen = QAction("全屏")
        self.QAction_View_FullScreen.setShortcut("F")
        self.QAction_View_FullScreen.triggered.connect(self.on_QAction_View_FullScreen_triggered)
        self.MenuBar_View.addAction(self.QAction_View_FullScreen)

        self.MenuBar_View.addSeparator()

        self.QAction_View_TragetId = QAction("目标编号")
        self.QAction_View_TragetId.setCheckable(True)
        self.QAction_View_TragetId.setChecked(True)
        self.QAction_View_TragetId.triggered.connect(self.on_QAction_RefreshImg)
        self.MenuBar_View.addAction(self.QAction_View_TragetId)

        self.QAction_View_TargetClss = QAction('目标分类')
        self.QAction_View_TargetClss.setCheckable(True)
        self.QAction_View_TargetClss.setChecked(True)
        self.QAction_View_TargetClss.triggered.connect(self.on_QAction_RefreshImg)
        self.MenuBar_View.addAction(self.QAction_View_TargetClss)

        self.QAction_View_TargetConf = QAction('置信度')
        self.QAction_View_TargetConf.setCheckable(True)
        self.QAction_View_TargetConf.setChecked(False)
        self.QAction_View_TargetConf.triggered.connect(self.on_QAction_RefreshImg)
        self.MenuBar_View.addAction(self.QAction_View_TargetConf)

        self.MenuBar_View.addSeparator()

        self.QAction_View_RelationBetweenTargetsAndDetectionLines = QAction("目标与检测线的关系")
        self.QAction_View_RelationBetweenTargetsAndDetectionLines.setCheckable(True)
        self.QAction_View_RelationBetweenTargetsAndDetectionLines.setChecked(False)     
        self.MenuBar_View.addAction(self.QAction_View_RelationBetweenTargetsAndDetectionLines)

        # 帮助 标签
        self.MenuBar_Help = self.menuBar().addMenu("帮助")
        self.QAction_Help_Course = QAction("教程", self)
        self.QAction_Help_Course.triggered.connect(self.on_QAction_Help_Course_triggered)
        self.MenuBar_Help.addAction(self.QAction_Help_Course)

        self.QAction_Help_About = QAction("关于", self)
        self.QAction_Help_About.triggered.connect(self.on_QAction_Help_About_triggered)
        self.MenuBar_Help.addAction(self.QAction_Help_About)


    def on_QAction_RefreshImg(self) -> None:
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QAction_Run_PreviousPage_triggered(self) -> None:
        r'跳转到上一张图片'
        frameId = self.QSlider_VisualResult.value()
        self.QSlider_VisualResult.setValue(frameId-1)


    def on_QAction_Run_NextPage_triggered(self) -> None:
        r'跳转到下一张图片'
        frameId = self.QSlider_VisualResult.value()
        self.QSlider_VisualResult.setValue(frameId+1)


    def on_QAction_File_ExportAsYOLO_triggered(self) -> None:
        r'YOLO格式导出'
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        
        if len(path) == 0:
            return

        # 数据剔除
        targets = TargetProcess.getTargetListByClss(self.TargetList, self.DetectionParams.classes) # 目标分类
        targets = TargetProcess.getTargetListByMinimumConfidence(targets, self.DetectionParams.minConf) # 最低置信度
        targets = TrafficFlow.TargetsNotInAreas(self.IgnoreAreaList, targets) # 根据忽略区筛选目标 

        # 保存类别
        file = open(os.path.join(path, 'classes.txt'), 'w')
            
        for clss in self.DetectionParams.classes:
            name = self.DetectionParams.names[clss]
            file.write(f'{name}\n')
        file.close()

        # 保存检测结果
        for index in range(len(self.imgs)):
            imgName = self.imgs[index]
            file = open(os.path.join(path, f"{os.path.split(imgName)[-1].split('.')[0]}.txt"), 'w')

            targetsInImage = TargetProcess.getTargetListByFrameId(targets, [index])
            for target in targetsInImage:
                box = self.xyxy2xywh(target.box, self.DetectionParams.width, self.DetectionParams.height) 
                file.write(f'{self.DetectionParams.classes.index(target.clss)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n')
            file.close()


    def on_QAction_File_OpenFileFold_triggered(self) -> None:
        r'打开图片数据集'
        filePath = QFileDialog.getExistingDirectory(self, "打开文件夹", "./")
        # 没有选择检测文件
        if len(filePath) == 0:
            return
        
        files = os.listdir(filePath)
        
        dataset: List[str] = []
         
        # 去除非图片
        for img in files:
            imgExt = img.split('.')[-1].lower()   # 文件扩展名
            if imgExt in IMG_FORMATS: # 图片
                dataset.append(os.path.join(filePath, img))

        # 没有可用图片
        if len(dataset) < 1:
            ERROR_CODES.ERROR("文件夹下没有图片!")
            return
        
        if self.video != None:
            self.video.release()
            self.video = None
        
        self.imgs = dataset
        
        img = cv2.imread(self.imgs[0])
        framesCount = len(self.imgs)
            
        # 获取并显示图像参数
        height, width, channel = img.shape  
        self.QSlider_VisualResult.setMaximum(int(framesCount))  # 设置滑动轴的最大滑动距离
        self.QLabel_FilePath.setText(filePath)
        self.QLabel_FileHeight.setText(f"{height}")
        self.QLabel_FileWidth.setText(f"{width}")
        self.QLabel_FileType.setText(f"数据集")
        self.Visualization_InitalImg = img.copy()
        self.QLineEdit_Visualiaztion_ShowTotalVideoTime.setText(f"{framesCount}")
        self.showImg(img)
        # 整合文件信息
        self.DetectionParams.height = height
        self.DetectionParams.width = width
        self.DetectionParams.framesCount = framesCount
        self.DetectionParams.source = filePath


    def on_QAction_Help_Course_triggered(self) -> None:
        r'菜单栏-帮助-教程'
        info =  "<font size='4'><p align='center'>教程</p></font>" + \
                "<font size='3'><p align='left'>S1. 选择文件：菜单栏-文件-打开文件</p></font>" + \
                "<font size='3'><p align='left'>S2. 检测参数：工具栏-目标检测</p></font>" + \
                "<font size='3'><p align='left'>S3. 交通流分析：工具栏-检测线、检测区域和忽略区域</p></font>" + \
                "<font size='3'><p align='left'>S4. 开始检测：菜单栏-运行-开始检测</p></font>" + \
                "<font size='3'><p align='left'>S5. 导出结果：菜单栏-文件-导出检测结果</p></font>"
        QMessageBox.information(None, "DeepTraffic", info)


    def on_QAction_Help_About_triggered(self) -> None:
        r'菜单栏-帮助-关于'
        info =  "<font size='4'><p align='center'>基于深度学习的交通流视频检测系统</p></font>" + \
                f"<font size='3'><p align='left'>版本: {ABOUT['版本']}</p></font>" + \
                f"<font size='3'><p align='left'>日期: {ABOUT['日期']}</p></font>" + \
                f"<font size='3'><p align='left'>OS: {ABOUT['OS']}</p></font>" + \
                f"<font size='3'><p align='left'>Python: {ABOUT['Python']}</p></font>"
        QMessageBox.information(None, "DeepTraffic", info)


    def on_QAction_File_OpenFile_triggered(self) -> None:
        r'打开视频文件'
        filePath, fileType = QFileDialog.getOpenFileName(self, "打开检测文件", "./", "All Files (*);;MP4 Video (*.mp4)")
        # 没有选择检测文件
        if len(filePath) == 0:
            return
        
        # 通过扩展名选择文件读取方式
        fileExt = filePath.split('.')[-1].lower()   # 文件扩展名
        if fileExt in VID_FORMATS:    # 视频
            if self.video != None:
                self.video.release()
            self.video = cv2.VideoCapture(filePath)
            bot, img = self.video.read()
            framesCount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            ERROR_CODES.E0003(str(fileExt))
            return
            
        # 获取并显示图像参数
        height, width, channel = img.shape  
        self.QSlider_VisualResult.setMaximum(int(framesCount))  # 设置滑动轴的最大滑动距离
        self.QLabel_FilePath.setText(filePath)
        self.QLabel_FileHeight.setText(f"{height}")
        self.QLabel_FileWidth.setText(f"{width}")
        self.QLabel_FileType.setText(f"{fileExt}")
        self.Visualization_InitalImg = img.copy()
        self.QLineEdit_Visualiaztion_ShowTotalVideoTime.setText(f"{framesCount}")
        self.showImg(img)
        # 整合文件信息
        self.DetectionParams.height = height
        self.DetectionParams.width = width
        self.DetectionParams.framesCount = framesCount
        self.DetectionParams.source = filePath


    def on_QAction_File_ExportDetectionResult_triggered(self) -> None:
        r"""on_QAction_File_ExportDetectionResult_triggered(self) -> None
        @brief 点击 文件-导出检测结果
        """
        filePath, fileType = QFileDialog.getSaveFileName(self, "导出检测结果", "./", "TXT (*.txt)")
        
        # 没有选择
        if len(filePath) == 0:
            return
        
        # 保存结果
        file = open(filePath, "w")
        
        # 保存文件头
        file.write(TARGETS_FILE+'\n')

        for clss in range(len(self.DetectionParams.names)-1):
            file.write(f"{self.DetectionParams.names[clss]}-")

        file.write(f"{self.DetectionParams.names[-1]}")
        file.write(f"\n")

        # 数据剔除
        targets = TargetProcess.getTargetListByClss(self.TargetList, self.DetectionParams.classes) # 目标分类
        targets = TargetProcess.getTargetListByMinimumConfidence(targets, self.DetectionParams.minConf) # 最低置信度
        targets = TrafficFlow.TargetsNotInAreas(self.IgnoreAreaList, targets) # 根据忽略区筛选目标
        
        for target in targets:
            box = self.xyxy2xywh(target.box, self.DetectionParams.width, self.DetectionParams.height)    
            file.write(f"{target.frameId} {target.id if target.id != None else -1} {target.clss} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {target.conf:.6f}\n")
            # file.write(f"{target.frameId} {target.id if target.id != None else -1} {target.clss} {target.conf} {target.box[0]} {target.box[1]} {target.box[2]} {target.box[3]}\n")
        file.close()

        return


    def on_QAction_File_ImportDetectionResult_triggered(self) -> None:
        r"""on_QAction_File_ImportDetectionResult_triggered(self) -> None
        @brief 点击 文件-导入检测结果
        """
        filePath, fileType = QFileDialog.getOpenFileName(self, "导入检测结果", "./", "TXT (*.txt)")

        # 没有选择
        if len(filePath) == 0:
            return

        # 需要先导入检测文件
        if self.DetectionParams.source == None:
            ERROR_CODES.ERROR("请先选择检测文件!\n")
            return

        # 导入结果
        file = open(filePath, "r")
        headLine = file.readline()
        if headLine.replace('\n', '') != TARGETS_FILE:
            file.close()
            ERROR_CODES.ERROR("文件格式错误!\n")
            return

        self.QComboBox_YOLO_SelectModel.setCurrentIndex(0)
        self.QComboBox_YOLO_SelectImgSize.setCurrentIndex(0)

        classLine = file.readline()
        names = classLine.replace("\n", "").split("-")
        self.DetectionParams.names = names
        self.QListWidget_YOLO_SelectTargets.clear()
        self.QListWidget_YOLO_SelectTargets.addItems(self.DetectionParams.names)

        lines = file.readlines()

        self.TargetList.clear()
        for line in lines:
            target = line.replace("\n", "").split(" ")
            frameId = int(target[0])
            id = int(target[1])
            clss = int(target[2])
            xywh = [float(target[3]), float(target[4]), float(target[5]), float(target[6])]
            xyxy = self.xywh2xyxy(xywh, self.DetectionParams.width, self.DetectionParams.height)
            conf = float(target[7])
            self.TargetList.append(Target(frameId, clss, conf, xyxy, id if id != -1 else None))
        
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QAction_Run_StartDetection_triggered(self) -> None:
        r"""on_QAction_Run_StartDetection_triggered(self) -> None
        @brief 点击 QAction_Run_StartDetection 开启检测
        """
        self.startDetection()


    def on_QAction_Run_StopDetection_triggered(self) -> None:
        r"""on_QAction_Run_StopDetection_triggered(self) -> None
        @brief 点击 QAction_Run_StopDetection 关闭检测
        """
        self.stopDetection()


    def on_QAction_View_FullScreen_triggered(self) -> None:
        r"""on_menuBar_View_FullScreen_triggered(self) -> None
        @brief 点击 menuBar_View_FullScreen 全屏
        """
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()    


    def ObjectDetectionAndTracking(self) -> None:
        r"""ObjectDetectionAndTracking(self) -> None
        @brief 目标检测模块
        """
        # 功能按键
        self.QToolButton_ObjectDetectionAndTracking = QToolButton()
        self.QToolButton_ObjectDetectionAndTracking.setIcon(QIcon("icons/eye_protection.svg"))
        self.QToolButton_ObjectDetectionAndTracking.setIconSize(QSize(32, 32))
        self.QToolButton_ObjectDetectionAndTracking.setFixedSize(100, 60)
        self.QToolButton_ObjectDetectionAndTracking.setText("目标检测")
        self.QToolButton_ObjectDetectionAndTracking.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.QToolButton_ObjectDetectionAndTracking.clicked.connect(self.on_QToolButton_ObjectDetectionAndTracking_clicked)
        self.QVBoxLayout_Widget_FunctionSelection.addWidget(self.QToolButton_ObjectDetectionAndTracking)
        
        # 创建窗口
        self.QWidget_ObjectDetectionAndTracking = QWidget()
        self.QStackedWidget_FunctionModule.addWidget(self.QWidget_ObjectDetectionAndTracking)
        self.QVBoxLayout_ObjectDetectionAndTracking = QVBoxLayout(self.QWidget_ObjectDetectionAndTracking)

        # 选择YOLO检测模型
        self.QComboBox_YOLO_SelectModel = QComboBox()
        self.QComboBox_YOLO_SelectModel.addItem("选择YOLO检测模型")
        self.QComboBox_YOLO_SelectModel.addItems(YOLO_MODELS.keys())
        self.QComboBox_YOLO_SelectModel.addItem("自定义")
        self.QComboBox_YOLO_SelectModel.setStyleSheet("selection-background-color: rgb(52, 101, 164);")
        self.QComboBox_YOLO_SelectModel.currentIndexChanged.connect(self.on_QComboBox_YOLO_SelectModel_currentIndexChanged)
        self.QVBoxLayout_ObjectDetectionAndTracking.addWidget(self.QComboBox_YOLO_SelectModel)

        # 选择YOLO检测规模
        self.QComboBox_YOLO_SelectImgSize = QComboBox()
        self.QComboBox_YOLO_SelectImgSize.addItem("选择YOLO检测规模")
        self.QComboBox_YOLO_SelectImgSize.addItems(IMG_SCALES.keys())
        self.QComboBox_YOLO_SelectImgSize.addItem("自定义")
        self.QComboBox_YOLO_SelectImgSize.setStyleSheet("selection-background-color: rgb(52, 101, 164);")
        self.QComboBox_YOLO_SelectImgSize.currentIndexChanged.connect(self.on_QComboBox_YOLO_SelectImgSize_currentIndexChanged)
        self.QVBoxLayout_ObjectDetectionAndTracking.addWidget(self.QComboBox_YOLO_SelectImgSize)

        # 选择推断模型中支持的目标分类 
        self.QListWidget_YOLO_SelectTargets = QListWidget()
        self.QListWidget_YOLO_SelectTargets.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.QListWidget_YOLO_SelectTargets.clicked.connect(self.on_QListWidget_YOLO_SelectTargets_clicked)
        self.QVBoxLayout_ObjectDetectionAndTracking.addWidget(self.QListWidget_YOLO_SelectTargets)

        # 目标置信度选择
        HBoxLayout = QHBoxLayout()
        self.QCheckBox_YOLO_MinConf = QCheckBox("最低置信度")
        self.QLabel_YOLO_ShowMinConf = QLabel("0")
        self.QLabel_YOLO_ShowMinConf.setFixedWidth(30)
        self.QSlider_YOLO_SelectMinConf = QSlider(Qt.Horizontal)
        self.QSlider_YOLO_SelectMinConf.setEnabled(False)
        self.QSlider_YOLO_SelectMinConf.setMinimum(0)
        self.QSlider_YOLO_SelectMinConf.setMaximum(100)
        self.QSlider_YOLO_SelectMinConf.setSingleStep(1)
        self.QSlider_YOLO_SelectMinConf.setValue(0)
        self.QCheckBox_YOLO_MinConf.stateChanged.connect(self.on_QCheckBox_YOLO_MinConf_stateChanged)
        self.QSlider_YOLO_SelectMinConf.valueChanged.connect(self.on_QSlider_YOLO_SelectMinConf_valueChanged)
        HBoxLayout.addWidget(self.QCheckBox_YOLO_MinConf)
        HBoxLayout.addWidget(self.QLabel_YOLO_ShowMinConf)
        HBoxLayout.addWidget(self.QSlider_YOLO_SelectMinConf)
        self.QVBoxLayout_ObjectDetectionAndTracking.addLayout(HBoxLayout)
        
        # 选择DEEPSORT的权重文件
        self.QComboBox_DEEPSORT_SelectModel = QComboBox()
        self.QComboBox_DEEPSORT_SelectModel.addItems(DEEPSORT_MODELS.keys())
        self.QComboBox_DEEPSORT_SelectModel.setStyleSheet("selection-background-color: rgb(52, 101, 164);")
        self.QComboBox_DEEPSORT_SelectModel.currentIndexChanged.connect(self.on_QComboBox_DEEPSORT_SelectModel_currentIndexChanged)
        self.QVBoxLayout_ObjectDetectionAndTracking.addWidget(self.QComboBox_DEEPSORT_SelectModel)

        # 检测开关
        self.QPushButton_DetectSwitch = QPushButton("开始检测")
        self.QPushButton_DetectSwitch.setStyleSheet("background-color:" + "red")
        self.QPushButton_DetectSwitch.clicked.connect(self.on_QPushButton_DetectSwitch_clicked)
        self.QVBoxLayout_ObjectDetectionAndTracking.addWidget(self.QPushButton_DetectSwitch)


        self.QComboBox_YOLO_SelectModel.setFixedHeight(30)
        self.QComboBox_YOLO_SelectImgSize.setFixedHeight(30)


    def on_QToolButton_ObjectDetectionAndTracking_clicked(self) -> None:
        r"""on_QToolButton_ObjectDetectionAndTracking_clicked(self) -> None
        @brief 显示与隐藏 目标检测 模块
        """
        widgetHeight = 350

        # 功能模块不可见
        if not self.QStackedWidget_FunctionModule.isVisible():
            self.QStackedWidget_FunctionModule.setFixedHeight(widgetHeight)
            self.QStackedWidget_FunctionModule.setCurrentWidget(self.QWidget_ObjectDetectionAndTracking)
            self.QStackedWidget_FunctionModule.setVisible(True)
            return

        # 功能模块可见 并且 功能不是目标检测
        if self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() != self.QWidget_ObjectDetectionAndTracking:
            self.QStackedWidget_FunctionModule.setFixedHeight(widgetHeight)
            self.QStackedWidget_FunctionModule.setCurrentWidget(self.QWidget_ObjectDetectionAndTracking)
            return

        # 功能模块可见 并且 功能是目标检测
        if self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() == self.QWidget_ObjectDetectionAndTracking:
            self.QStackedWidget_FunctionModule.setVisible(False)
            return


    def on_QComboBox_YOLO_SelectModel_currentIndexChanged(self) -> None:
        r"""on_QComboBox_YOLO_SelectModel_currentIndexChanged(self) -> None:
        @brief QComboBox_YOLO_SelectModel的选项发生改变(选择了不同的推断模型)
        """
        # 获取选项
        option = self.QComboBox_YOLO_SelectModel.currentText()

        # 获取模型的路径
        try:
            if option in YOLO_MODELS.keys():    # 预设
                modelPath = YOLO_MODELS[option]
            elif option == "自定义":   # 自选
                modelPath, type = QFileDialog.getOpenFileName(self, "getOpenFileName", "./", "All Files (*);;pt Model (*.pt)")
            else:
                self.QListWidget_YOLO_SelectTargets.clear()
                return
            # 设置参数 YOLO权重路径
            self.DetectionParams.weights = modelPath

            # 获取模型可检测的类别 详见 yolov5/models/experimental.py 的 attempt_load() 方法
            model = ModuleList()
            ckpt = torch.load(self.DetectionParams.weights, map_location='cpu') # 加载模型
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
            model = model[-1]
            self.DetectionParams.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            # 将所有标记变成复选框让人进行选择
            self.QListWidget_YOLO_SelectTargets.clear()
            self.QListWidget_YOLO_SelectTargets.addItems( self.DetectionParams.names)
        except Exception as e:
            self.QComboBox_YOLO_SelectModel.setCurrentIndex(0)
            ERROR_CODES.ERROR("选择YOLO检测模型时发生错误!\n" + str(e))


    def on_QListWidget_YOLO_SelectTargets_clicked(self) -> None:
        r"""on_QListWidget_YOLO_SelectTargets_clicked(self) -> None
        @brief QListWidget_YOLO_SelectTargets被点击(选择了不同的目标)
        """
        # 获取参数
        classes = []
        names =  self.DetectionParams.names

        # 获取需要检测的目标
        for item in self.QListWidget_YOLO_SelectTargets.selectedItems():
            classes.append(names.index(item.text())) 
        
        # 从小到大进行排序
        classes = sorted(classes)

        # 设置参数 检测目标 
        self.DetectionParams.classes = classes

        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QComboBox_YOLO_SelectImgSize_currentIndexChanged(self) -> None:
        r"""on_QComboBox_YOLO_SelectImgSize_currentIndexChanged(self) -> None
        @brief QComboBox_YOLO_SelectImgSize的选项发生改变(选择了不同的推断规模)
        """
        # 获取选项
        option = self.QComboBox_YOLO_SelectImgSize.currentText()

        # 获取规模大小
        if option in IMG_SCALES.keys(): # 预设
            self.QComboBox_YOLO_SelectImgSize.setEditable(False)
            imgsz = IMG_SCALES[option]
        elif option == "自定义":
            self.QComboBox_YOLO_SelectImgSize.setEditable(True)
            return
        else:
            self.QComboBox_YOLO_SelectImgSize.setEditable(False)
            return
    
        # 设置参数 推断规模
        self.DetectionParams.imgsz = imgsz


    def on_QCheckBox_YOLO_MinConf_stateChanged(self) -> None:
        r"""
        @brief 打开或取消最低置信度
        """
        if self.QCheckBox_YOLO_MinConf.isChecked():
            self.QSlider_YOLO_SelectMinConf.setEnabled(True)
            minConf = self.QSlider_YOLO_SelectMinConf.value()/100
            self.DetectionParams.minConf = minConf
        else:
            self.QSlider_YOLO_SelectMinConf.setEnabled(False)
            self.DetectionParams.minConf = 0
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QSlider_YOLO_SelectMinConf_valueChanged(self) -> None:
        r"""
        @brief 改变目标的最低置信度
        """
        minConf = self.QSlider_YOLO_SelectMinConf.value()/100
        self.QLabel_YOLO_ShowMinConf.setText(str(minConf))
        self.DetectionParams.minConf = minConf

        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QComboBox_DEEPSORT_SelectModel_currentIndexChanged(self) -> None:
        r"选择DEEPSORT的检测模型"
        self.DetectionParams.deepsort = self.QComboBox_DEEPSORT_SelectModel.currentText()


    def on_QPushButton_DetectSwitch_clicked(self) -> None:
        r"""on_QPushButton_DetectSwitch_clicked(self) -> None
        @brief 开启或关闭检测
        """
        if self.QPushButton_DetectSwitch.text() == "开始检测":
            self.startDetection()
        else:
            self.stopDetection()


    def TrafficFlow_DetectionLines(self) -> None:
        r"""TrafficFlow_DetectionLines(self) -> None
        @brief 交通流统计
        """
        # 功能按键
        self.QToolButton_TrafficFlow_DetectionLines = QToolButton()
        self.QToolButton_TrafficFlow_DetectionLines.setIcon(QIcon("icons/instruction.svg"))
        self.QToolButton_TrafficFlow_DetectionLines.setIconSize(QSize(32, 32))
        self.QToolButton_TrafficFlow_DetectionLines.setFixedSize(100, 60)
        self.QToolButton_TrafficFlow_DetectionLines.setText("检测线")
        self.QToolButton_TrafficFlow_DetectionLines.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.QToolButton_TrafficFlow_DetectionLines.clicked.connect(self.on_QToolButton_TrafficFlow_DetectionLines_clicked)
        self.QVBoxLayout_Widget_FunctionSelection.addWidget(self.QToolButton_TrafficFlow_DetectionLines)

        # 创建窗口
        self.QWidget_TrafficFlow_DetectionLines = QWidget()
        self.QVBoxLayout_TrafficFlow_DetectionLines = QVBoxLayout(self.QWidget_TrafficFlow_DetectionLines)
        self.QStackedWidget_FunctionModule.addWidget(self.QWidget_TrafficFlow_DetectionLines)

        # 区域名称 与 创建区域 与 删除选中检测线
        HBoxLayout = QHBoxLayout()
        self.QLineEdit_TrafficFlow_DetectionLines_EditDetectionLineName = QLineEdit("检测线0")
        self.QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine = QPushButton("创建")
        self.QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine_clicked)
        self.QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine = QPushButton("删除")
        self.QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine_clicked)
        HBoxLayout.addWidget(self.QLineEdit_TrafficFlow_DetectionLines_EditDetectionLineName)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine)
        self.QVBoxLayout_TrafficFlow_DetectionLines.addLayout(HBoxLayout)

        # 显示已经创建的检测线
        self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines = QListWidget()
        self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.setFixedHeight(100)
        self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.clicked.connect(self.on_QListWidget_TrafficFlow_DetectionLines_DetectionLines_clicked)
        self.QVBoxLayout_TrafficFlow_DetectionLines.addWidget(self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines)

        # 导出统计结果
        HBoxLayout = QHBoxLayout()
        self.QPushButton_TrafficFlow_DetectionLines_ExportResult = QPushButton("导出")
        self.QPushButton_TrafficFlow_DetectionLines_ExportResult.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionLines_ExportResult_clicked)
        self.QPushButton_TrafficFlow_DetectionLines_ImportResult = QPushButton("导入")
        self.QPushButton_TrafficFlow_DetectionLines_ImportResult.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionLines_IxportResult_clicked)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionLines_ExportResult)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionLines_ImportResult)
        self.QVBoxLayout_TrafficFlow_DetectionLines.addLayout(HBoxLayout)

        # 选择需要查看的统计信息
        self.QComboBox_TrafficFlow_TrafficCount_SelectResult = QComboBox()
        self.QComboBox_TrafficFlow_TrafficCount_SelectResult.addItems(["违章", "统计"])
        self.QComboBox_TrafficFlow_TrafficCount_SelectResult.setStyleSheet("selection-background-color: rgb(52, 101, 164);")
        self.QVBoxLayout_TrafficFlow_DetectionLines.addWidget(self.QComboBox_TrafficFlow_TrafficCount_SelectResult)  

        # 根据检测线的选择 显示记录信息
        self.QTableView_TrafficFlow_TrafficCount_Result = QTableView()
        self.QTableView_TrafficFlow_TrafficCount_Result.horizontalHeader().setSectionsClickable(False)
        self.QTableView_TrafficFlow_TrafficCount_Result.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.QTableView_TrafficFlow_TrafficCount_Result.verticalHeader().hide()
        self.QTableView_TrafficFlow_TrafficCount_Result.horizontalHeader().setStretchLastSection(True)
        self.QVBoxLayout_TrafficFlow_DetectionLines.addWidget(self.QTableView_TrafficFlow_TrafficCount_Result)

        self.QStandardItemModel_TrafficFlow_TrafficCount_Result = QStandardItemModel()
        self.QStandardItemModel_TrafficFlow_TrafficCount_Result.setHorizontalHeaderLabels(["编号", "分类", "检测时间"])
        self.QTableView_TrafficFlow_TrafficCount_Result.setModel(self.QStandardItemModel_TrafficFlow_TrafficCount_Result)


    def on_QPushButton_TrafficFlow_DetectionLines_IxportResult_clicked(self) -> None:
        r"""
        @brief 交通流统计 导入结果
        """
        filePath, fileType = QFileDialog.getOpenFileName(self, "导入检测线", "./", "TXT (*.txt)")

        # 没有选择
        if len(filePath) == 0:
            return

        # self.QStandardItemModel_TrafficFlow_DetectionLines_Result.clear()
        # self.QStandardItemModel_TrafficFlow_DetectionLines_Result.setHorizontalHeaderLabels(["检测线", "目标", "时间"])
        
        # 导入结果
        file = open(filePath, "r")
        lines = file.readlines()

        self.DetectionLineList.clear()
        self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.clear()
        for line in lines:
            result = line.replace("\n", "").split(" ")
            if result[0] == "-":
                self.DetectionLineList.append(Line(result[1]))
                self.DetectionLineList[-1].pt1 = [int(result[2]), int(result[3])]
                self.DetectionLineList[-1].pt2 = [int(result[4]), int(result[5])]
                self.DetectionLineList[-1].coming = None if int(result[6]) == 0 else int(result[6]) 
                self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.addItem(self.DetectionLineList[-1].name)

        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QPushButton_TrafficFlow_DetectionLines_ExportResult_clicked(self) -> None:
        r"""on_QPushButton_TrafficFlow_DetectionLines_ExportResult_clicked(self) -> None
        @brief 交通流统计 导出结果
        """
        filePath, fileType = QFileDialog.getSaveFileName(self, "导出检测线", "./", "TXT (*.txt)")
        
        # 没有选择
        if len(filePath) == 0:
            return
        
        # 保存结果
        file = open(filePath, "w")

        for line in self.DetectionLineList:
            file.write(f"- {line.name} {line.pt1[0]} {line.pt1[1]} {line.pt2[0]} {line.pt2[1]} {line.coming if line.coming != None else 0}\n")
        
        frameId = self.QSlider_VisualResult.value()
        targetListBeforeFrame = TargetProcess.getTargetListBeforeFrameId(self.TargetList, frameId)            
        for line in self.DetectionLineList:
            targetListPassLine = TrafficFlow.TargetsPassLine(line, targetListBeforeFrame)            
            for target in targetListPassLine:
                file.write(f"{line.name} {target.frameId} {target.id if target.id != None else -1} {target.clss} {target.conf} {target.box[0]} {target.box[1]} {target.box[2]} {target.box[3]}\n")
                
        file.close()
        return


    def on_QToolButton_TrafficFlow_DetectionLines_clicked(self) -> None:
        r"""
        @brief 显示或隐藏 功能模块-交通流统计
        """
        widgetHeight = 600
        # 功能模块不可见 或 功能模块可见 并且 功能不是 目标追踪
        if not self.QStackedWidget_FunctionModule.isVisible() or (self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() != self.QWidget_TrafficFlow_DetectionLines):
            self.QStackedWidget_FunctionModule.setFixedHeight(widgetHeight)
            self.QStackedWidget_FunctionModule.setCurrentWidget(self.QWidget_TrafficFlow_DetectionLines)
            self.QStackedWidget_FunctionModule.setVisible(True)
            return

        # 功能模块可见 并且 功能是 目标追踪
        if self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() == self.QWidget_TrafficFlow_DetectionLines:
            self.QStackedWidget_FunctionModule.setVisible(False)
            return


    def on_QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine_clicked(self) -> None:
        r"""on_QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine_clicked(self) -> None
        @brief 交通流统计 创建一条检测线
        """
        # 创建检测线
        name = self.QLineEdit_TrafficFlow_DetectionLines_EditDetectionLineName.text()
        self.DetectionLineList.append(Line(name))
        self.QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine.setEnabled(False)
        self.QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine.setEnabled(False)

        # 显示已经创建的检测线
        self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.addItem(name)


    def on_QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine_clicked(self) -> None:
        r"""on_QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine_clicked(self) -> None
        @brief 交通流统计 删除选中的检测线
        """
        if len(self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.selectedIndexes()) < 1:
            return
        # 删除选中的检测线
        row = self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.selectedIndexes()[0].row()
        item = self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.takeItem(row)
        self.DetectionLineList.pop(row)
        self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.removeItemWidget(item)

        # 刷新显示界面
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QListWidget_TrafficFlow_DetectionLines_DetectionLines_clicked(self) -> None:
        r"""on_QListWidget_TrafficFlow_DetectionLines_DetectionLines_clicked(self) -> None
        @brief 选中了不同的区域
        """
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def TrafficFlow_DetectionAreas(self) -> None:
        r"""
        @brief 交通流 检测区域
        """
        # 功能按键
        self.QToolButton_TrafficFlow_DetectionAreas = QToolButton()
        self.QToolButton_TrafficFlow_DetectionAreas.setIcon(QIcon("icons/random.svg"))
        self.QToolButton_TrafficFlow_DetectionAreas.setIconSize(QSize(32, 32))
        self.QToolButton_TrafficFlow_DetectionAreas.setFixedSize(100, 60)
        self.QToolButton_TrafficFlow_DetectionAreas.setText("检测区域")
        self.QToolButton_TrafficFlow_DetectionAreas.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.QToolButton_TrafficFlow_DetectionAreas.clicked.connect(self.on_QToolButton_TrafficFlow_DetectionAreas_clicked)
        self.QVBoxLayout_Widget_FunctionSelection.addWidget(self.QToolButton_TrafficFlow_DetectionAreas)
           
        # 创建窗口
        self.QWidget_TrafficFlow_DetectionAreas = QWidget()
        self.QVBoxLayout_TrafficFlow_DetectionAreas = QVBoxLayout(self.QWidget_TrafficFlow_DetectionAreas)
        self.QStackedWidget_FunctionModule.addWidget(self.QWidget_TrafficFlow_DetectionAreas) 

        # 名称/创建/删除
        HBoxLayout = QHBoxLayout()
        self.QLineEidt_TrafficFlow_DetectionAreas_EditDetectionAreaName = QLineEdit("检测区域")
        self.QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea = QPushButton("创建")
        self.QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea_clicked)
        self.QPushButton_TrafficFlow_DetectionAreas_DeleteDetectionArea = QPushButton("删除")
        self.QPushButton_TrafficFlow_DetectionAreas_DeleteDetectionArea.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionAreas_DeleteDetectionArea_clicked)
        HBoxLayout.addWidget(self.QLineEidt_TrafficFlow_DetectionAreas_EditDetectionAreaName)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionAreas_DeleteDetectionArea)
        self.QVBoxLayout_TrafficFlow_DetectionAreas.addLayout(HBoxLayout)

        # 显示创建的检测区域
        self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas = QListWidget()
        self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.setFixedHeight(100)
        self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.clicked.connect(self.on_QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas_clicked)
        self.QVBoxLayout_TrafficFlow_DetectionAreas.addWidget(self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas)

        # 导入导出检测区域
        HBoxLayout = QHBoxLayout()
        self.QPushButton_TrafficFlow_DetectionAreas_ExportDetectionAreas = QPushButton("导出")
        self.QPushButton_TrafficFlow_DetectionAreas_ExportDetectionAreas.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionAreas_ExportDetectionAreas_clicked)
        self.QPushButton_TrafficFlow_DetectionAreas_ImportDetectionAreas = QPushButton("导入")
        self.QPushButton_TrafficFlow_DetectionAreas_ImportDetectionAreas.clicked.connect(self.on_QPushButton_TrafficFlow_DetectionAreas_ImportDetectionAreas_clicked)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionAreas_ExportDetectionAreas)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_DetectionAreas_ImportDetectionAreas)
        self.QVBoxLayout_TrafficFlow_DetectionAreas.addLayout(HBoxLayout)

        # 选择需要查看的统计信息
        self.QComboBox_TrafficFlow_DetectionAreas_SelectResult = QComboBox()
        self.QComboBox_TrafficFlow_DetectionAreas_SelectResult.addItems(["违章", "统计"])
        self.QComboBox_TrafficFlow_DetectionAreas_SelectResult.setStyleSheet("selection-background-color: rgb(52, 101, 164);")
        self.QVBoxLayout_TrafficFlow_DetectionAreas.addWidget(self.QComboBox_TrafficFlow_DetectionAreas_SelectResult)  

        # 统计结果显示
        self.QTableView_TrafficFlow_IllegalArea_Result = QTableView()
        self.QStandardItemModel_TrafficFlow_IllegalArea_Result = QStandardItemModel()
        self.QStandardItemModel_TrafficFlow_IllegalArea_Result.setHorizontalHeaderLabels(["编号", "分类", "进入时间", "离开时间"])
        self.QTableView_TrafficFlow_IllegalArea_Result.horizontalHeader().setSectionsClickable(False)
        self.QTableView_TrafficFlow_IllegalArea_Result.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.QTableView_TrafficFlow_IllegalArea_Result.verticalHeader().hide()
        self.QTableView_TrafficFlow_IllegalArea_Result.setModel(self.QStandardItemModel_TrafficFlow_IllegalArea_Result)
        self.QTableView_TrafficFlow_IllegalArea_Result.horizontalHeader().setStretchLastSection(True)
        self.QVBoxLayout_TrafficFlow_DetectionAreas.addWidget(self.QTableView_TrafficFlow_IllegalArea_Result)


    def on_QPushButton_TrafficFlow_DetectionAreas_ImportDetectionAreas_clicked(self) -> None:
        r"""
        @brief 导入检测区域
        """
        filePath, fileType = QFileDialog.getOpenFileName(self, "导入检测区域", "./", "TXT (*.txt)")

        # 没有选择
        if len(filePath) == 0:
            return

        self.DetectionAreaList.clear()
        self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.clear()

        # 导入结果
        file = open(filePath, "r")
        for line in file.readlines():
            lineInfo = line.replace("\n", "").split(" ")
            self.DetectionAreaList.append(Area(lineInfo[0]))
            self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.addItem(self.DetectionAreaList[-1].name)
            pts = lineInfo[1:]
            for i in range(int((len(pts))/2)):
                self.DetectionAreaList[-1].pts.append([int(pts[i*2]), int(pts[i*2+1])])
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)       


    def on_QPushButton_TrafficFlow_DetectionAreas_ExportDetectionAreas_clicked(self) -> None:
        r"""
        @brief 导出检测区域
        """
        filePath, fileType = QFileDialog.getSaveFileName(self, "导出检测区域", "./", "TXT (*.txt)")
        
        # 没有选择
        if len(filePath) == 0:
            return
        
        # 保存结果
        file = open(filePath, "w")

        for area in self.DetectionAreaList:
            file.write(f"{area.name}")
            for pt in area.pts:
                file.write(f" {pt[0]} {pt[1]}")
            file.write(f"\n")
        
        file.close()
        return


    def on_QPushButton_TrafficFlow_DetectionAreas_DeleteDetectionArea_clicked(self) -> None:
        r"""
        @brief 删除选中的检测区域
        """
        if len(self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.selectedIndexes()) < 1:
            return
        # 删除选中的检测区域
        row = self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.selectedIndexes()[0].row()
        item = self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.takeItem(row)
        self.DetectionAreaList.pop(row)
        self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.removeItemWidget(item)

        # 刷新显示界面
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas_clicked(self) -> None:
        r"""
        @brief 选中已创建的检测区
        """
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea_clicked(self) -> None:
        r"""
        @brief 创建检测区
        """
        # 创建实例
        self.DetectionAreaList.append(Area(self.QLineEidt_TrafficFlow_DetectionAreas_EditDetectionAreaName.text()))
        # 显示创建的实例
        self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.addItem(self.DetectionAreaList[-1].name)
        # 设置按键不可按下
        self.QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea.setEnabled(False)
        self.QPushButton_TrafficFlow_DetectionAreas_DeleteDetectionArea.setEnabled(False)


    def on_QToolButton_TrafficFlow_DetectionAreas_clicked(self) -> None:
        r"""on_QToolButton_TrafficFlow_IgnoreAreas_clicked(self) -> None
        @brief 点击 忽略区域 按钮
        """
        widgetHeight = 600
        # 功能模块不可见 或 功能模块可见 并且 功能不是 目标追踪
        if not self.QStackedWidget_FunctionModule.isVisible() or (self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() != self.QWidget_TrafficFlow_DetectionAreas):
            self.QStackedWidget_FunctionModule.setFixedHeight(widgetHeight)
            self.QStackedWidget_FunctionModule.setCurrentWidget(self.QWidget_TrafficFlow_DetectionAreas)
            self.QStackedWidget_FunctionModule.setVisible(True)
            return

        # 功能模块可见 并且 功能是 目标追踪
        if self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() == self.QWidget_TrafficFlow_DetectionAreas:
            self.QStackedWidget_FunctionModule.setVisible(False)
            return


    def TrafficFlow_IgnoreAreas(self) -> None:
        # 功能按键
        self.QToolButton_TrafficFlow_IgnoreAreas = QToolButton()
        # self.QToolButton_TrafficFlow_IgnoreAreas.setIcon(QIcon("icons/highway first.svg"))
        # self.QToolButton_TrafficFlow_IgnoreAreas.setIconSize(QSize(32, 32))
        self.QToolButton_TrafficFlow_IgnoreAreas.setFixedSize(100, 60)
        self.QToolButton_TrafficFlow_IgnoreAreas.setText("忽略区域")
        # self.QToolButton_TrafficFlow_IgnoreAreas.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.QToolButton_TrafficFlow_IgnoreAreas.clicked.connect(self.on_QToolButton_TrafficFlow_IgnoreAreas_clicked)
        self.QVBoxLayout_Widget_FunctionSelection.addWidget(self.QToolButton_TrafficFlow_IgnoreAreas)
           
        # 创建窗口
        self.QWidget_TrafficFlow_IgnoreAreas = QWidget()
        self.QVBoxLayout_TrafficFlow_IgnoreAreas = QVBoxLayout(self.QWidget_TrafficFlow_IgnoreAreas)
        self.QStackedWidget_FunctionModule.addWidget(self.QWidget_TrafficFlow_IgnoreAreas) 

        # 名称/创建/删除
        HBoxLayout = QHBoxLayout()
        self.QLineEidt_TrafficFlow_IgnoreAreas_EditIgnoreAreaName = QLineEdit("忽略区域")
        self.QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea = QPushButton("创建")
        self.QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea.clicked.connect(self.on_QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea_clicked)
        self.QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea = QPushButton("删除")
        self.QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea.clicked.connect(self.on_QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea_clicked)
        HBoxLayout.addWidget(self.QLineEidt_TrafficFlow_IgnoreAreas_EditIgnoreAreaName)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea)
        self.QVBoxLayout_TrafficFlow_IgnoreAreas.addLayout(HBoxLayout)

        # 显示创建的忽略区域
        self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas = QListWidget()
        self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.clicked.connect(self.on_QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas_clicked)
        self.QVBoxLayout_TrafficFlow_IgnoreAreas.addWidget(self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas)

        HBoxLayout = QHBoxLayout()
        self.QPushButton_TrafficFlow_IgnoreAreas_ExportIgnoreAreas = QPushButton("导出")
        self.QPushButton_TrafficFlow_IgnoreAreas_ExportIgnoreAreas.clicked.connect(self.on_QPushButton_TrafficFlow_IgnoreAreas_ExportIgnoreAreas_clicked)
        self.QPushButton_TrafficFlow_IgnoreAreas_ImportIgnoreAreas = QPushButton("导入")
        self.QPushButton_TrafficFlow_IgnoreAreas_ImportIgnoreAreas.clicked.connect(self.on_QPushButton_TrafficFlow_IgnoreAreas_ImportIgnoreAreas_clicked)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_IgnoreAreas_ExportIgnoreAreas)
        HBoxLayout.addWidget(self.QPushButton_TrafficFlow_IgnoreAreas_ImportIgnoreAreas)
        self.QVBoxLayout_TrafficFlow_IgnoreAreas.addLayout(HBoxLayout)


    def on_QPushButton_TrafficFlow_IgnoreAreas_ImportIgnoreAreas_clicked(self) -> None:
        r"""
        @brief 导入忽略区域
        """
        filePath, fileType = QFileDialog.getOpenFileName(self, "导入忽略区域", "./", "TXT (*.txt)")

        # 没有选择
        if len(filePath) == 0:
            return

        self.IgnoreAreaList.clear()
        self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.clear()

        # 导入结果
        file = open(filePath, "r")
        for line in file.readlines():
            lineInfo = line.replace("\n", "").split(" ")
            self.IgnoreAreaList.append(Area(lineInfo[0]))
            self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.addItem(self.IgnoreAreaList[-1].name)
            pts = lineInfo[1:]
            for i in range(int((len(pts))/2)):
                self.IgnoreAreaList[-1].pts.append([int(pts[i*2]), int(pts[i*2+1])])
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)       


    def on_QPushButton_TrafficFlow_IgnoreAreas_ExportIgnoreAreas_clicked(self) -> None:
        r"""
        @brief 导出忽略区域
        """
        filePath, fileType = QFileDialog.getSaveFileName(self, "导出忽略区域", "./", "TXT (*.txt)")
        
        # 没有选择
        if len(filePath) == 0:
            return
        
        # 保存结果
        file = open(filePath, "w")

        for area in self.IgnoreAreaList:
            file.write(f"{area.name}")
            for pt in area.pts:
                file.write(f" {pt[0]} {pt[1]}")
            file.write(f"\n")
        
        file.close()
        return


    def on_QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea_clicked(self) -> None:
        r"""
        @brief 删除选中的忽略区域
        """
        ''' @FIXME 
        Traceback (most recent call last):
        File "d:/毕业设计/DeepTraffic/MainWindow.py", line 2255, in on_QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea_clicked
            row = self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.selectedIndexes()[0].row()
        IndexError: list index out of range
        空表删除报错或未选删除报错，检测线与区域适用
        '''
        if len(self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.selectedIndexes()) < 1:
            return

        # 删除选中的忽略区域
        row = self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.selectedIndexes()[0].row()
        item = self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.takeItem(row)
        self.IgnoreAreaList.pop(row)
        self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.removeItemWidget(item)

        # 刷新显示界面
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas_clicked(self) -> None:
        r"""
        @brief 选中已创建的忽略区域
        """
        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def on_QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea_clicked(self) -> None:
        r"""on_QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea_clicked(self) -> None
        @brief 创建忽略区域
        """
        # 创建实例
        self.IgnoreAreaList.append(Area(self.QLineEidt_TrafficFlow_IgnoreAreas_EditIgnoreAreaName.text()))
        # 显示创建的实例
        self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.addItem(self.IgnoreAreaList[-1].name)
        # 设置按键不可按下
        self.QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea.setEnabled(False)
        self.QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea.setEnabled(False)


    def on_QToolButton_TrafficFlow_IgnoreAreas_clicked(self) -> None:
        r"""on_QToolButton_TrafficFlow_IgnoreAreas_clicked(self) -> None
        @brief 点击 忽略区域 按钮
        """
        widgetHeight = 500
        # 功能模块不可见 或 功能模块可见 并且 功能不是 目标追踪
        if not self.QStackedWidget_FunctionModule.isVisible() or (self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() != self.QWidget_TrafficFlow_IgnoreAreas):
            self.QStackedWidget_FunctionModule.setFixedHeight(widgetHeight)
            self.QStackedWidget_FunctionModule.setCurrentWidget(self.QWidget_TrafficFlow_IgnoreAreas)
            self.QStackedWidget_FunctionModule.setVisible(True)
            return

        # 功能模块可见 并且 功能是 目标追踪
        if self.QStackedWidget_FunctionModule.isVisible() and self.QStackedWidget_FunctionModule.currentWidget() == self.QWidget_TrafficFlow_IgnoreAreas:
            self.QStackedWidget_FunctionModule.setVisible(False)
            return


    def Console(self) -> None:
        r"""Console 控制台"""
        # 控制台
        self.QWidget_Console = QWidget(self.MainWidget)
        self.QWidget_Console.setFixedWidth(self.MainWidget.width())
        self.QWidget_Console.setFixedHeight(200)
        self.QWidget_Console.move(0, self.MainWidget.height() - 70)
        self.QVBoxLayout_Console = QVBoxLayout(self.QWidget_Console)

        # 检测结果可视化滑动条
        HBoxLayout = QHBoxLayout()
        self.QSlider_VisualResult = QSlider(Qt.Horizontal, self.MainWidget)
        self.QSlider_VisualResult.setFixedHeight(20)
        self.QSlider_VisualResult.setMinimum(1)
        self.QSlider_VisualResult.setMaximum(1)
        self.QSlider_VisualResult.setSingleStep(1)
        self.QSlider_VisualResult.setValue(1)
        self.QSlider_VisualResult.valueChanged.connect(self.on_QSlider_VisualResult_valueChanged)
        HBoxLayout.addWidget(self.QSlider_VisualResult)
        # 当前的视频帧数/时间
        self.QLineEdit_Visualiaztion_ShowNowVideoTime = QLineEdit("0", self.MainWidget)
        self.QLineEdit_Visualiaztion_ShowNowVideoTime.setAlignment(Qt.AlignHCenter)
        self.QLineEdit_Visualiaztion_ShowNowVideoTime.setFixedSize(40, 20)
        self.QLineEdit_Visualiaztion_ShowTotalVideoTime = QLineEdit("0", self.MainWidget)
        self.QLineEdit_Visualiaztion_ShowTotalVideoTime.setAlignment(Qt.AlignHCenter)
        self.QLineEdit_Visualiaztion_ShowTotalVideoTime.setFixedSize(40, 20)
        self.QLineEdit_Visualiaztion_ShowTotalVideoTime.setReadOnly(True)
        HBoxLayout.addWidget(self.QLineEdit_Visualiaztion_ShowNowVideoTime)
        HBoxLayout.addWidget(QLabel(":"))
        HBoxLayout.addWidget(self.QLineEdit_Visualiaztion_ShowTotalVideoTime)
        self.QVBoxLayout_Console.addLayout(HBoxLayout)

        # 控制台顶部选项卡
        self.QWidget_Console_TopBar = QWidget()
        self.QWidget_Console_TopBar.setFixedHeight(36)
        self.QVBoxLayout_Console.addWidget(self.QWidget_Console_TopBar)
        self.QHBoxLayout_Console_TopBar = QHBoxLayout(self.QWidget_Console_TopBar)

        # 控制台窗口
        self.QStackedWidget_Console_Windows = QStackedWidget()
        self.QStackedWidget_Console_Windows.setVisible(False)
        self.QVBoxLayout_Console.addWidget(self.QStackedWidget_Console_Windows)

        # 基础信息
        self.QPushButton_Console_SelectBasicInfo = QPushButton("基础信息")
        self.QPushButton_Console_SelectBasicInfo.setFixedSize(80, 24)
        self.QPushButton_Console_SelectBasicInfo.clicked.connect(self.on_QPushButton_Console_SelectBasicInfo_clicked)
        self.QHBoxLayout_Console_TopBar.addWidget(self.QPushButton_Console_SelectBasicInfo)

        self.QWidget_Console_BasicInfo = QWidget()
        self.QStackedWidget_Console_Windows.addWidget(self.QWidget_Console_BasicInfo)
        self.QStackedWidget_Console_Windows.setCurrentWidget(self.QWidget_Console_BasicInfo)
        self.QFormLayout_Console_FileInfo = QFormLayout(self.QWidget_Console_BasicInfo)
        self.QLabel_FilePath = QLabel() # 文件路径
        self.QFormLayout_Console_FileInfo.addRow("文件路径", self.QLabel_FilePath)
        self.QLabel_FileHeight = QLabel("") # 高度
        self.QFormLayout_Console_FileInfo.addRow("高度:", self.QLabel_FileHeight)
        self.QLabel_FileWidth = QLabel("") # 宽度
        self.QFormLayout_Console_FileInfo.addRow("宽度:", self.QLabel_FileWidth)
        self.QLabel_FileType = QLabel("") # 类型
        self.QFormLayout_Console_FileInfo.addRow("类型:", self.QLabel_FileType)

        # 输出
        self.QPushButton_Console_SelectOutput= QPushButton("输出")
        self.QPushButton_Console_SelectOutput.setFixedSize(80, 24)
        self.QPushButton_Console_SelectOutput.clicked.connect(self.on_QPushButton_Console_SelectOutput_clicked)
        self.QHBoxLayout_Console_TopBar.addWidget(self.QPushButton_Console_SelectOutput)

        self.QTextEdit_Output = QTextEdit()
        self.QTextEdit_Output.setReadOnly(True)
        self.QStackedWidget_Console_Windows.addWidget(self.QTextEdit_Output)

        self.QHBoxLayout_Console_TopBar.addStretch()

        # 打开与隐藏控制台
        self.QToolButton_Console = QToolButton()
        self.QToolButton_Console.setIcon(QIcon("icons/upward.svg"))
        self.QToolButton_Console.setIconSize(QSize(24, 24))
        self.QToolButton_Console.setFixedSize(32, 24)
        self.QToolButton_Console.clicked.connect(self.on_QToolButton_Console_clicked)
        self.QHBoxLayout_Console_TopBar.addWidget(self.QToolButton_Console)

        self.QVBoxLayout_Console.addStretch()


    def on_QToolButton_Console_clicked(self) -> None:
        r"""on_QToolButton_Console_clicked(self) -> None
        @brief 打开与隐藏控制台
        """
        # 控制台处于不可见
        if not self.QStackedWidget_Console_Windows.isVisible():
            self.QStackedWidget_Console_Windows.setVisible(True)
            self.QWidget_Console.move(0, self.MainWidget.height() - 200)
            self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height() - 200)
            self.QToolButton_Console.setIcon(QIcon("icons/down.svg"))
            return
        
        # 控制台处于可见
        if self.QStackedWidget_Console_Windows.isVisible():
            self.QStackedWidget_Console_Windows.setVisible(False)
            self.QWidget_Console.move(0, self.MainWidget.height() - 70)
            self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height() - 70)
            self.QToolButton_Console.setIcon(QIcon("icons/upward.svg"))
            return


    def on_QPushButton_Console_SelectBasicInfo_clicked(self) -> None:
        r"""on_QPushButton_Console_SelectBasicInfo_clicked(self) -> None
        @brief 显示或隐藏 控制台-基础信息
        """
        # 控制台处于不可见
        if not self.QStackedWidget_Console_Windows.isVisible():
            self.QStackedWidget_Console_Windows.setVisible(True)
            self.QStackedWidget_Console_Windows.setCurrentWidget(self.QWidget_Console_BasicInfo)
            self.QWidget_Console.move(0, self.MainWidget.height() - 200)
            self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height() - 200)
            self.QToolButton_Console.setIcon(QIcon("icons/down.svg"))
            return
        
        # 控制台处于可见 并且 当前并没有显示 基础信息
        if self.QStackedWidget_Console_Windows.isVisible() and self.QStackedWidget_Console_Windows.currentWidget() != self.QWidget_Console_BasicInfo:
            self.QStackedWidget_Console_Windows.setCurrentWidget(self.QWidget_Console_BasicInfo)
            return
        
        # 控制台处于可见 并且 当前显示的是基础信息
        if self.QStackedWidget_Console_Windows.isVisible() and self.QStackedWidget_Console_Windows.currentWidget() == self.QWidget_Console_BasicInfo:
            self.QStackedWidget_Console_Windows.setVisible(False)
            self.QWidget_Console.move(0, self.MainWidget.height() - 70)
            self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height() - 70)
            self.QToolButton_Console.setIcon(QIcon("icons/upward.svg"))
            return


    def on_QPushButton_Console_SelectOutput_clicked(self) -> None:
        r"""on_QPushButton_Console_SelectOutput_clicked(self) -> None
        @brief 显示或隐藏 控制台-输出
        """
        # 控制台处于不可见
        if not self.QStackedWidget_Console_Windows.isVisible():
            self.QStackedWidget_Console_Windows.setVisible(True)
            self.QStackedWidget_Console_Windows.setCurrentWidget(self.QTextEdit_Output)
            self.QWidget_Console.move(0, self.MainWidget.height() - 200)
            self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height() - 200)
            self.QToolButton_Console.setIcon(QIcon("icons/down.svg"))
            return
        
        # 控制台处于可见 并且 当前并没有显示 输出
        if self.QStackedWidget_Console_Windows.isVisible() and self.QStackedWidget_Console_Windows.currentWidget() != self.QTextEdit_Output:
            self.QStackedWidget_Console_Windows.setCurrentWidget(self.QTextEdit_Output)
            return
        
        # 控制台处于可见 并且 当前显示的是 输出
        if self.QStackedWidget_Console_Windows.isVisible() and self.QStackedWidget_Console_Windows.currentWidget() == self.QTextEdit_Output:
            self.QStackedWidget_Console_Windows.setVisible(False)
            self.QWidget_Console.move(0, self.MainWidget.height() - 70)
            self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height() - 70)
            self.QToolButton_Console.setIcon(QIcon("icons/upward.svg"))
            return


    def Visualization(self) -> None:
        r"""Visualization(self) -> None
        @brief 检测结果可视化
        """
        # 检测结果可视化
        self.QGraphicsView_VisualResult = QGraphicsView(self.MainWidget)
        self.QGraphicsView_VisualResult.move(0, 0)
        self.QGraphicsView_VisualResult.setFixedWidth(self.MainWidget.width())
        self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height()-70)

        # FPS 显示
        self.QLabel_DetectFPS = QLabel(self.MainWidget)
        self.QLabel_DetectFPS.setFixedWidth(50)
        self.QLabel_DetectFPS.move(self.MainWidget.width()-60, 20)
        self.QLabel_DetectFPS.setStyleSheet("font-size:30px; color:#28FF28")
        self.QLabel_DetectFPS.setVisible(False)


    def on_QSlider_VisualResult_valueChanged(self) -> None:
        r"""on_QSlider_VisualResult_valueChanged(self) -> None
        @brief on_QSlider_VisualResult_valueChanged数据改变(结果可视化窗口滑块被滑动了
        """
        frameId = self.QSlider_VisualResult.value()
        self.QLineEdit_Visualiaztion_ShowNowVideoTime.setText(str(frameId))
        self.refresh(frameId)


    def QuickTest(self) -> None:
        r"""快速选择检测文件, 并按照当前设置进行目标检测"""
        return


    def mousePressEvent(self, mouseEvent: QMouseEvent) -> None:
        r"""在窗口内鼠标被按下"""
        pos = mouseEvent.localPos()
    
        # 创建检测线
        if not self.QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine.isEnabled():
            # 选择检测线的1号点
            if self.DetectionLineList[-1].pt1 == None:
                # 获取鼠标相对位置
                relPos = self.QGraphicsView_VisualResult.mapToScene(int(pos.x()), int(pos.y()))
                relPosX = int(relPos.x()-self.QGraphicsView_VisualResult.x())
                relPosY = int(relPos.y()-self.QGraphicsView_VisualResult.y()) - 25
                if relPosX < 0 or relPosY < 0:
                    ERROR_CODES.E0007()
                    return
                self.DetectionLineList[-1].pt1 = [relPosX, relPosY]
                self.showText(f"设置检测线 {self.DetectionLineList[-1].name} 的pt1为({relPosX}, {relPosY})")
                # 在图像上显示
                frameId = self.QSlider_VisualResult.value()
                self.refresh(frameId)
                return
            # 选择检测线的2号点
            if self.DetectionLineList[-1].pt2 == None:
                # 获取鼠标相对位置
                relPos = self.QGraphicsView_VisualResult.mapToScene(int(pos.x()), int(pos.y()))
                relPosX = int(relPos.x()-self.QGraphicsView_VisualResult.x())
                relPosY = int(relPos.y()-self.QGraphicsView_VisualResult.y()) - 25
                if relPosX < 0 or relPosY < 0:
                    ERROR_CODES.E0007() 
                    return    
                self.DetectionLineList[-1].pt2 = [relPosX, relPosY]
                self.showText(f"设置检测线 {self.DetectionLineList[-1].name} 的pt1为({relPosX}, {relPosY})")
                # 在图像上显示
                frameId = self.QSlider_VisualResult.value()
                self.refresh(frameId)
                isSelectingDirection = QMessageBox.question(None, "Question?", "是否需要选择检测目标的轨迹方向?", QMessageBox.Yes | QMessageBox.No)
                if isSelectingDirection == QMessageBox.No:
                    self.QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine.setEnabled(True)
                    self.QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine.setEnabled(True)
                return
            # 检测线检测目标 来的方向
            if self.DetectionLineList[-1].coming == None:
                # 获取鼠标相对位置
                relPos = self.QGraphicsView_VisualResult.mapToScene(int(pos.x()), int(pos.y()))
                relPosX = int(relPos.x()-self.QGraphicsView_VisualResult.x())
                relPosY = int(relPos.y()-self.QGraphicsView_VisualResult.y()) - 25
                if relPosX < 0 or relPosY < 0:
                    ERROR_CODES.E0007()
                    return
                if not self.DetectionLineList[-1].setComingDirection(relPosX, relPosY):
                    info = "目标来向设置在了线上，请重新尝试"
                    QMessageBox.information(None, "消息", info)
                    return
                frameId = self.QSlider_VisualResult.value()
                self.refresh(frameId)
                self.QPushButton_TrafficFlow_DetectionLines_CreateDetectionLine.setEnabled(True)
                self.QPushButton_TrafficFlow_DetectionLines_DeleteDetectionLine.setEnabled(True)
                return

        # 创建忽略区
        if not self.QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea.isEnabled():
            # 获取鼠标在画幅中的相对位置
            relPos = self.QGraphicsView_VisualResult.mapToScene(int(pos.x()), int(pos.y()))
            relPosX = int(relPos.x()-self.QGraphicsView_VisualResult.x())
            relPosY = int(relPos.y()-self.QGraphicsView_VisualResult.y()) - 25
            if relPosX < 0 or relPosY < 0:
                ERROR_CODES.E0007()
                return
            self.IgnoreAreaList[-1].pts.append([relPosX, relPosY])
            # 刷新显示图像
            frameId = self.QSlider_VisualResult.value()
            self.refresh(frameId)
            # 是否需要选择其他点
            if len(self.IgnoreAreaList[-1].pts) >= 3:
                selectMorePoints = QMessageBox.question(None, "Question?", "是否需要选择更多的点?", QMessageBox.Yes | QMessageBox.No)
                if selectMorePoints == QMessageBox.No:
                    self.QPushButton_TrafficFlow_IgnoreAreas_CreateIgnoreArea.setEnabled(True)
                    self.QPushButton_TrafficFlow_IgnoreAreas_DeleteIgnoreArea.setEnabled(True)
            return

        # 创建检测区
        if not self.QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea.isEnabled():
            # 获取鼠标在画幅中的相对位置
            relPos = self.QGraphicsView_VisualResult.mapToScene(int(pos.x()), int(pos.y()))
            relPosX = int(relPos.x()-self.QGraphicsView_VisualResult.x())
            relPosY = int(relPos.y()-self.QGraphicsView_VisualResult.y()) - 25
            if relPosX < 0 or relPosY < 0:
                ERROR_CODES.E0007()
                return
            self.DetectionAreaList[-1].pts.append([relPosX, relPosY])
            # 刷新显示图像
            frameId = self.QSlider_VisualResult.value()
            self.refresh(frameId)
            # 是否需要选择其他点
            if len(self.DetectionAreaList[-1].pts) >= 3:
                selectMorePoints = QMessageBox.question(None, "Question?", "是否需要选择更多的点?", QMessageBox.Yes | QMessageBox.No)
                if selectMorePoints == QMessageBox.No:
                    self.QPushButton_TrafficFlow_DetectionAreas_CreateDetectionArea.setEnabled(True)
                    self.QPushButton_TrafficFlow_DetectionAreas_DeleteDetectionArea.setEnabled(True)
            return


    def resizeEvent(self, resizeEvent: QResizeEvent) -> None:
        r"""resizeEvent(self, resizeEvent: QResizeEvent) -> None
        @brief 窗口大小发生了改变
        """
        # 检测结果可视化
        self.QGraphicsView_VisualResult.setFixedWidth(self.MainWidget.width())
        self.QGraphicsView_VisualResult.setFixedHeight(self.MainWidget.height() - 200 if self.QStackedWidget_Console_Windows.isVisible() else self.MainWidget.height() - 70)
        self.QLabel_DetectFPS.move(self.MainWidget.width()-60, 20)

        # 控制台
        self.QWidget_Console.setFixedWidth(self.MainWidget.width())
        self.QWidget_Console.move(0, self.MainWidget.height() - 200 if self.QStackedWidget_Console_Windows.isVisible() else self.MainWidget.height() - 70)

        frameId = self.QSlider_VisualResult.value()
        self.refresh(frameId)


    def deal_DetectThread_signal_new_target(self, target: Target) -> None:
        r"""deal_DetectThread_signal_new_target(self, target: Target) -> None
        @brief 接收到信号 检测到新的目标
        @param target Target 新的目标结果
        """
        # 保存目标结果
        self.TargetList.append(target)


    def deal_DetectThread_signal_new_frame(self, frameId: int, usedTime: float) -> None:
        r"""deal_DetectThread_signal_new_frame(self, frameId: int) -> None
        @brief 检测完新的图像
        @param frameId 图像编号
        @param usedTime 检测该图像使用的时间
        """
        # 获取相关参数
        filePath = self.DetectionParams.source  # 文件路径
        framesCount = self.DetectionParams.framesCount # 需要检测的图像数量

        # 识别结果和检测进度可视化
        self.QSlider_VisualResult.setValue(frameId)
        # 控制台输出检测信息
        self.showText("{}: {}/{} Done, Use time {:.3f}s".format(filePath, frameId, framesCount, usedTime))
        # 实时显示检测帧率
        self.showFPS(1 / usedTime)


    def deal_DetectThread_signal_finished(self) -> None:
        r"""检测线程完成检测任务或提前中止"""
        self.stopDetection()


    def assignInitialValue(self) -> None:
        r"""对项目中需要使用的参数进行初始化"""
        
        self.video: cv2.VideoCapture = None # 交通流视频资源
        self.DetectionLineList:List[Line] = [] # 检测线
        self.TargetList: List[Target] = []
        self.DetectionParams = DetectionParams() # 检测参数相关
        self.IgnoreAreaList: List[Area] = []
        self.DetectionAreaList: List[Area] = []
        self.imgs: List[str] = None

        DetectThread.signal_new_target.connect(self.deal_DetectThread_signal_new_target)
        DetectThread.signal_new_frame.connect(self.deal_DetectThread_signal_new_frame)
        DetectThread.signal_finished.connect(self.deal_DetectThread_signal_finished)


    def startDetection(self) -> None:
        r"""startDetection(self) -> None
        @brief 开启检测线程
        """
        if self.QComboBox_YOLO_SelectImgSize.currentText() not in IMG_SCALES.keys():
            try:
                imgsz = int(self.QComboBox_YOLO_SelectImgSize.currentText())
                if imgsz <= 0 :
                    ERROR_CODES.ERROR("YOLO的检测规模不能小于等于0!")
                    return
                height, width, channel = self.Visualization_InitalImg.shape
                if imgsz > height or imgsz > width:
                    ERROR_CODES.ERROR("YOLO的检测规模不能大于数据集尺寸!")
                    return
            except Exception as e:
                ERROR_CODES.ERROR("获取YOLO的检测规模是发生错误!\n" + str(e))
                return
            self.DetectionParams.imgsz = (imgsz, imgsz)

        # 检查是否已经开启检测线程, 如果开启先停止之前的线程
        if DetectThread.isRunning():
            self.stopDetection()

        model = self.DetectionParams.deepsort
        if not os.path.exists(f"weights/{model}_imagenet.pth"):
            ERROR_CODES.ERROR(f"weights/{model}_imagenet.pth并不存在!\n请重新选择DEEPSORT跟踪模型")
            return

        # 检查参数是否缺失
        missingParam = self.DetectionParams.getMissingYOLOParam()
        if missingParam != None:
            ERROR_CODES.E0002(str(missingParam))
            return
        DetectThread.setParams(self.DetectionParams)

        # 设置滑动槽
        self.QSlider_VisualResult.setMaximum(int(self.DetectionParams.framesCount))
        
        try:
            # 清空之前的检测数据
            self.TargetList.clear()
            # 启动检测线程
            DetectThread.ready()
            DetectThread.start()
        except Exception as e:
            ERROR_CODES.E0001(str(e))
            return

        # 相关控件设置
        self.QPushButton_DetectSwitch.setText("关闭检测") 
        self.QPushButton_DetectSwitch.setStyleSheet("background-color:" + "green")
        self.QSlider_VisualResult.setEnabled(False)
        self.QAction_Run_StartDetection.setEnabled(False)
        self.QAction_Run_StopDetection.setEnabled(True)
        self.QLabel_DetectFPS.setVisible(True)


    def stopDetection(self) -> None:
        r"""stopDetection(self) -> None
        @brief 关闭检测线程
        """
        try:
            DetectThread.stop()
            DetectThread.quit()
            # self.DetectThread.wait()
            # del self.DetectThread
            # self.DetectThread = None
        except Exception as e:
            info = "关闭检测失败\n" + str(e)
            QMessageBox.information(None, "错误", info)
            pass

        # 检测开关
        self.QPushButton_DetectSwitch.setText("开始检测")
        self.QPushButton_DetectSwitch.setStyleSheet("background-color:" + "red")
        # 可视化结果滑动条
        self.QSlider_VisualResult.setEnabled(True)
        # 菜单栏开始和停止调试
        self.QAction_Run_StartDetection.setEnabled(True)
        self.QAction_Run_StopDetection.setEnabled(False)
        # FPS不可见
        self.QLabel_DetectFPS.setVisible(False)
        # 释放内存
        collect()


    def refresh(self, frameId: int) -> None:
        r"""
        @brief 刷新指定图像编号的显示内容
        """
        # 视频
        if self.video != None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, float(frameId-1))
            dot, image = self.video.read()
        elif self.imgs != None:
            image = cv2.imread(self.imgs[frameId-1])
        else:
            return

        # 根据要求获取目标序列
        # targetListInFrame = TargetProcess.getTargetListByFrameId(self.TargetList, [frameId])
        # targetListInFrame = TargetProcess.getTargetListByClss(targetListInFrame, self.DetectionParams.classes) # 目标分类
        # targetListInFrame = TargetProcess.getTargetListByMinimumConfidence(targetListInFrame, self.DetectionParams.minConf) # 最低置信度
        # targetListInFrame = TrafficFlow.TargetsNotInAreas(self.IgnoreAreaList, targetListInFrame) # 根据忽略区筛选目标

        targetListBeforeFrame = TargetProcess.getTargetListBeforeFrameId(self.TargetList, frameId)  
        targetListBeforeFrame = TargetProcess.getTargetListByClss(targetListBeforeFrame, self.DetectionParams.classes) # 目标分类
        targetListBeforeFrame = TargetProcess.getTargetListByMinimumConfidence(targetListBeforeFrame, self.DetectionParams.minConf) # 最低置信度
        targetListBeforeFrame = TrafficFlow.TargetsNotInAreas(self.IgnoreAreaList, targetListBeforeFrame) # 根据忽略区筛选目标
        targetListInFrame = TargetProcess.getTargetListByFrameId(targetListBeforeFrame, [frameId])

        # 交通流-检测线
        image = DetectionVisualization.VisualDetectionLines(image, self.DetectionLineList) # 可视化检测线
        if len(self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.selectedIndexes()) > 0:
            # 凸显选中的检测线
            row = self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.selectedIndexes()[0].row()
            image = DetectionVisualization.VisualDetectionLine(image, self.DetectionLineList[row], (0, 0, 255))
        if self.QAction_View_RelationBetweenTargetsAndDetectionLines.isChecked():
            # 可视化检测线与目标的关系
            image = DetectionVisualization.VisualRelationBetweenDetectoinLinesAndTargets(image, self.DetectionLineList, targetListInFrame)

        # 交通流-忽略区域
        image = DetectionVisualization.VisualAreas(image, self.IgnoreAreaList) # 可视化忽略区
        if len(self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.selectedIndexes()) > 0:
            # 凸显选中的忽略区域
            row = self.QListWidget_TrafficFlow_IgnoreAreas_ShowCreatedIgnoreAreas.selectedIndexes()[0].row()
            image = DetectionVisualization.VisualArea(image, self.IgnoreAreaList[row], (0, 0, 255))

        # 交通流-检测区域
        image = DetectionVisualization.VisualAreas(image, self.DetectionAreaList) # 可视化检测区域
        if len(self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.selectedIndexes()) > 0:
            # 凸显选中的检测区域
            row = self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.selectedIndexes()[0].row()
            image = DetectionVisualization.VisualArea(image, self.DetectionAreaList[row], (0, 0, 255))

        # 可视化符合要的目标
        image = DetectionVisualization.VisualTargets(image, targetListInFrame, self.DetectionParams.names, \
            showId=self.QAction_View_TragetId.isChecked(), showClss=self.QAction_View_TargetClss.isChecked(), showConf=self.QAction_View_TargetConf.isChecked())

        # 交通流统计和查看数据类型
        # s1 判断需要查看的检测线
        if len(self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.selectedIndexes()) > 0:
            # s2 获取检测线
            row = self.QListWidget_TrafficFlow_DetectionLines_ShowCreatedDetectionLines.selectedIndexes()[0].row() # 获取检测线编号
            line = self.DetectionLineList[row]
            # s3 获取需要显示的数据
            if self.QComboBox_TrafficFlow_TrafficCount_SelectResult.currentText() == "违章":
                self.QStandardItemModel_TrafficFlow_TrafficCount_Result.clear()
                self.QStandardItemModel_TrafficFlow_TrafficCount_Result.setHorizontalHeaderLabels(["编号", "分类", "检测时间"])
                targetListPassLine = TrafficFlow.TargetsPassLine(line, targetListBeforeFrame)  
                for target in targetListPassLine:
                    self.QStandardItemModel_TrafficFlow_TrafficCount_Result.appendRow([QStandardItem(str(target.id)), QStandardItem(self.DetectionParams.names[target.clss]), QStandardItem(str(target.frameId))])
                # s4 在当前图像上对目标进行标注
                targetListPassLineInFrame: List[Target] = []
                for target in targetListPassLine:
                    targetListPassLineInFrame.extend(TargetProcess.getTargetListByTargetId(targetListInFrame, [target.id]))
                image = DetectionVisualization.VisualCrossTargets(image, targetListPassLineInFrame)
            elif self.QComboBox_TrafficFlow_TrafficCount_SelectResult.currentText() == "统计":
                self.QStandardItemModel_TrafficFlow_TrafficCount_Result.clear()
                self.QStandardItemModel_TrafficFlow_TrafficCount_Result.setHorizontalHeaderLabels(["分类", "数量"])
                targetListPassLine = TrafficFlow.TargetsPassLine(line, targetListBeforeFrame)
                # s4 按分类统计数据
                targetCountByClass = {}
                for clss in self.DetectionParams.classes:
                    targetCountByClass[clss] = 0
                for target in targetListPassLine:
                    if target.clss in targetCountByClass.keys():
                        targetCountByClass[target.clss] = targetCountByClass[target.clss] + 1
                for clss in targetCountByClass.keys():
                    self.QStandardItemModel_TrafficFlow_TrafficCount_Result.appendRow([QStandardItem(self.DetectionParams.names[clss]), QStandardItem(str(targetCountByClass[clss]))])

                targetListPassLineInFrame: List[Target] = []
                for target in targetListPassLine:
                    targetListPassLineInFrame.extend(TargetProcess.getTargetListByTargetId(targetListInFrame, [target.id]))
                image = DetectionVisualization.VisualCrossTargets(image, targetListPassLineInFrame)

        # 交通流-检测区域: 进出目标统计
        if len(self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.selectedIndexes()) > 0:
            # 获取选中的检测区域
            row = self.QListWidget_TrafficFlow_DetectionAreas_ShowCreatedDetectionAreas.selectedIndexes()[0].row()
            area = self.DetectionAreaList[row]
            # 选择需要统计的数据
            if self.QComboBox_TrafficFlow_DetectionAreas_SelectResult.currentText() == "违章":
                self.QStandardItemModel_TrafficFlow_IllegalArea_Result.clear()
                self.QStandardItemModel_TrafficFlow_IllegalArea_Result.setHorizontalHeaderLabels(["编号", "分类", "进入时间", "离开时间"])
                # 获取之前出现在区域内的目标
                targetListInAreaBeforeFrame = TrafficFlow.TargetsInArea(area, targetListBeforeFrame)
                # 获取当前出现在区域内的目标
                targetListInAreaInFrame = TargetProcess.getTargetListByFrameId(targetListInAreaBeforeFrame, [frameId])
                # 对在区域内的目标进行可视化
                image = DetectionVisualization.VisualCrossTargets(image, targetListInAreaInFrame)
                # Dict[目标编号] = [出现记录]
                targetDictById_InArea_BeforeFrame = TargetProcess.sortTargetsByTargetId(targetListInAreaBeforeFrame)
                for targetId in targetDictById_InArea_BeforeFrame.keys():
                    # s3 判断目标进入和离开检测区的时间
                    if targetDictById_InArea_BeforeFrame[targetId][-1].frameId == frameId: # 目标未离开检测区
                        self.QStandardItemModel_TrafficFlow_IllegalArea_Result.appendRow([QStandardItem(str(targetId)), 
                                                                                        QStandardItem(self.DetectionParams.names[targetDictById_InArea_BeforeFrame[targetId][-1].clss]), 
                                                                                        QStandardItem(str(targetDictById_InArea_BeforeFrame[targetId][0].frameId)),
                                                                                        QStandardItem("未离开检测区")])
                        continue
                    if targetDictById_InArea_BeforeFrame[targetId][-1].frameId < frameId: # 目标已离开检测区
                        self.QStandardItemModel_TrafficFlow_IllegalArea_Result.appendRow([QStandardItem(str(targetId)), 
                                                                                        QStandardItem(self.DetectionParams.names[targetDictById_InArea_BeforeFrame[targetId][-1].clss]), 
                                                                                        QStandardItem(str(targetDictById_InArea_BeforeFrame[targetId][0].frameId)),
                                                                                        QStandardItem(str(targetDictById_InArea_BeforeFrame[targetId][-1].frameId))])
                        continue
            elif self.QComboBox_TrafficFlow_DetectionAreas_SelectResult.currentText() == "统计":
                self.QStandardItemModel_TrafficFlow_IllegalArea_Result.clear()
                self.QStandardItemModel_TrafficFlow_IllegalArea_Result.setHorizontalHeaderLabels(["分类", "数量"])
                # 获取之前出现在区域内的目标
                targetListInAreaBeforeFrame = TrafficFlow.TargetsInArea(area, targetListBeforeFrame)
                # 获取当前出现在区域内的目标
                targetListInAreaInFrame = TargetProcess.getTargetListByFrameId(targetListInAreaBeforeFrame, [frameId])
                # 对在区域内的目标进行可视化
                image = DetectionVisualization.VisualCrossTargets(image, targetListInAreaInFrame)
                # Dict[目标编号] = [出现记录]
                targetDictById_InArea_BeforeFrame = TargetProcess.sortTargetsByTargetId(targetListInAreaBeforeFrame)
                # Dict[分类] = 目标数量
                targetCountByClass = {}
                # 按不同分类进行目标统计
                for clss in self.DetectionParams.classes:
                    targetCountByClass[clss] = 0
                for targetId in targetDictById_InArea_BeforeFrame:
                    if targetDictById_InArea_BeforeFrame[targetId][0].clss in targetCountByClass.keys():
                        targetCountByClass[targetDictById_InArea_BeforeFrame[targetId][0].clss] = targetCountByClass[targetDictById_InArea_BeforeFrame[targetId][0].clss] + 1
                for clss in targetCountByClass.keys():
                    self.QStandardItemModel_TrafficFlow_IllegalArea_Result.appendRow([QStandardItem(self.DetectionParams.names[clss]), QStandardItem(str(targetCountByClass[clss]))])


        # 显示新的图片
        self.showImg(image)


    def showImg(self, img: np.array) -> None:
        r"""
        @brief 显示给定的图像
        @param img np.array 显示图像
        """
        height, width, channel = img.shape
        
        # wWidth = self.MainWidget.width()
        # wHeight = self.MainWidget.height() - 200 if self.QStackedWidget_Console_Windows.isVisible() else self.MainWidget.height() - 70
        
        # if wWidth > width and wHeight > height:
        #     pWidth = wWidth / width
        #     pHeight = wHeight / height
        #     scale = pWidth if pWidth < pHeight else pHeight
        #     reWidth = int(width * scale)
        #     reHeight = int(height * scale)
        #     img = cv2.resize(img, (reWidth, reHeight))
          
        height, width, channel = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qImg = QImage(img, width, height, 3*width, QImage.Format_RGB888)  # 转换图像通道
        qPix = QPixmap.fromImage(qImg)      
        qItem = QGraphicsPixmapItem(qPix)   # 创建图元
        qScene = QGraphicsScene()   # 创建场景
        qScene.addItem(qItem)
        self.QGraphicsView_VisualResult.setScene(qScene)


    def showText(self, text: str) -> None:
        r"""
        @brief 在控制台中输出文字信息
        """
        origin = self.QTextEdit_Output.toPlainText()
        nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        changed = origin + f"{nowTime}: {text}\n"
        self.QTextEdit_Output.setPlainText(changed)
        self.QTextEdit_Output.verticalScrollBar().setValue(self.QTextEdit_Output.verticalScrollBar().maximum())


    def showFPS(self, fps) -> None:
        r"""
        @brief 显示识别时的FPS
        @param fps 识别的帧率
        """
        if fps >= 1:
            self.QLabel_DetectFPS.setText(str(int(fps)))
        elif fps < 1 and fps > 0.1:
            self.QLabel_DetectFPS.setText(f"{fps:.1f}")
        else:
            self.QLabel_DetectFPS.setText("0")


    def xyxy2xywh(self, box, w, h):
        x = (box[0] + box[2]) / 2 / w
        y = (box[1] + box[3]) / 2 / h
        w = (box[2] - box[0]) / w
        h = (box[3] - box[1]) / h
        return [x, y, w, h]


    def xywh2xyxy(self, box, w, h):
        xlt = int(box[0] * w - box[2] * w / 2)
        ylt = int(box[1] * h - box[3] * h / 2)
        xrb = int(box[0] * w + box[2] * w / 2)
        yrb = int(box[1] * h + box[3] * h / 2)
        return [xlt, ylt, xrb, yrb]


def suppress_qt_warngings() -> None:
    r"""去除 Qt 构建路径报错

    Warning: QT_DEVICE_PIXEL_RATIO is deprecated. 

    Instead use:
        QT_AUTO_SCREEN_SCALE_FACTOR to enable platform plugin controlled per-screen factors.
        
        QT_SCREEN_SCALE_FACTORS to set per-screen factors.

        QT_SCALE_FACTOR to set the application global scale factor.
    """
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"


def main() -> None:
    suppress_qt_warngings()
    app = QApplication(sys.argv)
    Main = MAIN()
    Main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLOv10


class Detection:
    # 模型对象
    __model_obj: ultralytics.YOLOv10 = None
    # 图像源（使用OpenCV的图像格式）
    __image_src: np.ndarray = None
    # 图像尺寸
    __image_size: tuple = (720, 720)
    # 置信度
    __confidence: float = 0.65
    # IoU
    __iou: float = 0.45
    # GPU设备
    __device: int = 0

    # bounding box的四个值的元组构成的列表
    boxes_xywh_np: np.ndarray = None

    # 用于绘制bounding box及其标签的字体

    # 用于处理数据
    def __process_data(self):
        # 返回值的列表，每个元素是一个元组，元组的四个值分别是bounding box的x、y、w、h
        ret_val: list = []

        # 获取推理结果
        image_detection_results = self.__model_obj.predict(source=self.__image_src, imgsz=self.__image_size,
                                                           conf=self.__confidence, iou=self.__iou, device=self.__device)

        # 判断检测结果是否为空
        if len(image_detection_results) == 0:
            return None

        # 遍历处理结果对象的bounding box的相关的量
        # （实际上，如果输入是一幅图片的话，results中仅有1个成员）
        for single_result in image_detection_results:
            # 获取置信度最高的bounding box索引
            best_box_index: int = 0
            for i in range(len(single_result.boxes)):
                if single_result.boxes.conf[i] > single_result.boxes.conf[best_box_index]:
                    best_box_index = i

            # 获取所有bounding box的xywh张量
            boxes_xywh_tensor: torch.Tensor = single_result.boxes.xywh

            # 将所有bounding box的torch张量转换为numpy数组（使用CPU进行计算）
            self.boxes_xywh_np = boxes_xywh_tensor.cpu().numpy()

            # 如果bounding box的数量为0或者为None，则返回None
            if len(self.boxes_xywh_np) == 0 or self.boxes_xywh_np is None:
                return None

            box_x: int = int(self.boxes_xywh_np[best_box_index][0])
            box_y: int = int(self.boxes_xywh_np[best_box_index][1])
            box_width: int = int(self.boxes_xywh_np[best_box_index][2])
            box_height: int = int(self.boxes_xywh_np[best_box_index][3])

            ret_val.append((box_x, box_y, box_width, box_height))

        return ret_val

    # 公共函数，获取bounding box的x、y、w、h
    def get_bbox_xywh(self):
        return self.__process_data()

    # 构造函数
    def __init__(self, model_obj, img_src: np.ndarray, img_size: tuple, confidence: float, iou: float, device: int):
        self.__model_obj = model_obj
        self.__image_src = img_src
        self.__image_size = img_size
        self.__confidence = confidence
        self.__iou = iou
        self.__device = device

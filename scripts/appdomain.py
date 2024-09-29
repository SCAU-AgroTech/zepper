import sys
import os
import math

# 导入OpenCV库、numpy库、ZED SDK库、Ultralytics模块与ROS库
import cv2
import numpy as np
import pyzed.sl as sl
import torch
from ultralytics import YOLOv10
import rospy

# 导入自定义的模块
base_path = os.path.abspath(".")
sys.path.insert(0, base_path + "/src/zepper/scripts")

from object_detection_ultralytics import Detection
from error_code import ErrorCode
from zepper.msg import VisionInfo


def draw_label(input_image: np.ndarray, label: str, x1: int, y1: int, x2: int, y2: int):
    try:
        # 在bounding box的顶部绘制文字
        label_size: tuple = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        y1 = max(y1, label_size[1])

        # 左上角顶点坐标
        top_left_point: tuple = (x1, y1 - label_size[0][1])

        # 右下角顶点坐标
        bottom_right_point: tuple = (x2, y2)

        print(bottom_right_point)

        # 绘制蓝色矩形
        cv2.rectangle(input_image, top_left_point, bottom_right_point, (0, 0, 255), 2)

        # 在矩形框上绘制文字
        cv2.putText(input_image, label, (x1, y1 + label_size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 0, 0), 2)

        # 指示绘制成功
        return True
    except Exception as draw_label_exception:
        # 反馈报错信息
        print('在绘制标签时出现错误：' + str(draw_label_exception))
        return False


def main():
    # 初始化ROS节点
    rospy.init_node("zepper")

    # 创建 ROS 发布者，消息类型为VisionInfo
    publisher = rospy.Publisher('vision_info', VisionInfo, queue_size=50)

    # 设置发布频率为10Hz
    rate = rospy.Rate(30)

    # 实例化ZED相机对象
    camera = sl.Camera()

    # 实例化ZED相机参数对象
    init_params = sl.InitParameters()

    # 设置分辨率为720p，帧率为30fps、不进行图像翻转
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60

    # 设置深度模式为超高精度模式，深度单位为毫米，深度探测范围0.15-10m
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_minimum_distance = 150
    init_params.depth_maximum_distance = 5000

    # 打开相机
    camera_launch_status = camera.open(init_params)

    # 如果相机打开失败
    if camera_launch_status == ErrorCode.SUCCESS:
        print("发生错误：打开相机失败。错误码：" + ErrorCode.CAMERA_OPEN_FAILED)
        exit(ErrorCode.CAMERA_OPEN_FAILED)

    # 创建相机运行时参数对象
    camera_runtime_params = sl.RuntimeParameters()

    # 创建图像对象。注意，它们是sl.Mat类型的对象，非OpenCV的numpy数组
    left_image_sl = sl.Mat()
    right_image_sl = sl.Mat()
    depth_image_sl = sl.Mat()
    point_cloud_matrix_sl = sl.Mat()

    # 加载YOLOv10模型
    onnx_model = YOLOv10('src/zepper/models/zepper.onnx')

    while not rospy.is_shutdown():
        # 当成功获取一帧图像时
        if camera.grab(camera_runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 获取左、右图像
            camera.retrieve_image(left_image_sl, sl.VIEW.LEFT)
            camera.retrieve_image(right_image_sl, sl.VIEW.RIGHT)
            # 获取深度图像
            camera.retrieve_image(depth_image_sl, sl.VIEW.DEPTH)
            # 获取左图像的点云矩阵
            camera.retrieve_measure(point_cloud_matrix_sl, sl.MEASURE.XYZRGBA)

            # 将左、右、深度图像转换为OpenCV格式，其中左、右图像为BGR格式，深度图像为灰度图像
            left_image_cv2 = cv2.cvtColor(left_image_sl.get_data(), cv2.COLOR_RGBA2RGB)
            right_image_cv2 = cv2.cvtColor(right_image_sl.get_data(), cv2.COLOR_RGBA2RGB)
            depth_image_cv2 = cv2.cvtColor(cv2.cvtColor(depth_image_sl.get_data(), cv2.COLOR_RGBA2RGB),
                                           cv2.COLOR_RGB2GRAY)

            # 对左、右图像进行目标检测
            left_detection = Detection(onnx_model, left_image_cv2, (720, 1280), 0.3, 0.35, 0)
            right_detection = Detection(onnx_model, right_image_cv2, (720, 1280), 0.3, 0.35, 0)
            # k-means聚类准备
            pixel_values = right_image_cv2.reshape((-1, 3))
            pixel_values =  np.float32(pixel_values)


            # 获取左、右图像检测框的中心坐标集合
            left_bbox_xywh = left_detection.get_bbox_xywh()
            right_bbox_xywh = right_detection.get_bbox_xywh()

            # 检查左、右图像检测框的中心坐标集合是否为空
            if left_bbox_xywh is None or right_bbox_xywh is None:
                print("左、右图像检测框的中心坐标集合为空")

                # 显示左、右和深度图像
                cv2.imshow("Left", left_image_cv2)
                cv2.imshow("Right", right_image_cv2)
                cv2.imshow("Depth", depth_image_cv2)
                cv2.imshow("k-means", right_image_cv2)# 若无检测框，显示黑色图像

                # 发布ROS消息，全部置0
                info = VisionInfo()
                info.distance = 0
                info.cx = 0
                info.cy = 0
                info.cz = 0

                publisher.publish(info)
                rate.sleep()
                rospy.loginfo(info)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # 进行下一轮循环
                continue
            else:
                # 获取左、右图像检测框的中心坐标与宽高集合
                left_box_cx = left_bbox_xywh[0][0]
                left_box_cy = left_bbox_xywh[0][1]
                left_box_width = left_bbox_xywh[0][2]
                left_box_height = left_bbox_xywh[0][3]

                right_box_cx = right_bbox_xywh[0][0]
                right_box_cy = right_bbox_xywh[0][1]
                right_box_width = right_bbox_xywh[0][2]
                right_box_height = right_bbox_xywh[0][3]

                # k-means参数设置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.3)
                k = 7 # 聚类数(簇)
                
                # 进行k-means聚类
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 15, cv2.KMEANS_PP_CENTERS)
                centers = np.uint8(centers) # 转换为整数
                segmented_image = centers[labels.flatten()] # 重构图像, 用聚类中心替换每个像素
                segmented_image = segmented_image.reshape(right_image_cv2.shape)
                for box in right_bbox_xywh:
                    x,y,w,h = box
                    roi = segmented_image[y:y+h, x:x+w]
                    
                    roi_labels = labels.reshape(right_image_cv2.shape[:2])[y:y+h, x:x+w] # 获取roi区域的标签
                    paper_cluster = (roi_labels == 0)
                    paper_roi = np.zeros_like(roi)
                    paper_roi[paper_cluster] = roi[paper_cluster] # 获取辣椒区域



                # 获取左图bounding box的深度值
                err, left_depth_value = point_cloud_matrix_sl.get_value(left_box_cx, left_box_cy)

                # 如果左图bounding box的深度值有效
                if err == sl.ERROR_CODE.SUCCESS and math.isfinite(left_depth_value[2]):
                    distance = math.sqrt(left_depth_value[0] * left_depth_value[0] +
                                         left_depth_value[1] * left_depth_value[1] +
                                         left_depth_value[2] * left_depth_value[2])

                    print(f"相机左目的目标点({left_box_cx}, {left_box_cy})距离相机: {distance} mm.")

                    # 发布消息
                    info = VisionInfo()
                    info.distance = int(distance)
                    info.cx = int(left_depth_value[0])
                    info.cy = int(left_depth_value[1])
                    info.cz = int(left_depth_value[2])

                    publisher.publish(info)
                    rate.sleep()
                    rospy.loginfo(info)

                    # 在左、右图像上绘制bounding box
                    draw_label(left_image_cv2, 'dist: ' + str(int(distance)), int(left_box_cx - left_box_width / 2),
                               int(left_box_cy - left_box_height / 2), int(left_box_cx + left_box_width / 2),
                               int(left_box_cy + left_box_height / 2))

                    draw_label(right_image_cv2, 'dist: ' + str(int(distance)), int(right_box_cx - right_box_width / 2),
                               int(right_box_cy - right_box_height / 2), int(right_box_cx + right_box_width / 2),
                               int(right_box_cy + right_box_height / 2))
                    draw_label(segmented_image, 'dist: ' + str(int(distance)), int(right_box_cx - right_box_width / 2),
                               int(right_box_cy - right_box_height / 2), int(right_box_cx + right_box_width / 2),
                               int(right_box_cy + right_box_height / 2))

                    # 显示图像
                    paper_roi_resized = cv2.resize(paper_roi, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Left", left_image_cv2)
                    cv2.imshow("Right", right_image_cv2)
                    cv2.imshow("Depth", depth_image_cv2)
                    cv2.imshow("k-means", paper_roi_resized)
                    # cv2.imshow("k-means", segmented_image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    camera.close()
    cv2.destroyAllWindows()
    return ErrorCode.SUCCESS


if __name__ == "__main__":
    main()

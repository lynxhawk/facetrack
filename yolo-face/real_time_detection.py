import cv2
import numpy as np
import torch
from ultralytics import YOLO
from astra_camera import AstraCamera
import os
import time
from pathlib import Path

class RealTimeYOLODetection:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        初始化实时检测系统
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.camera = AstraCamera()
        
        # 创建保存目录
        self.save_dir = Path("detections")
        self.save_dir.mkdir(exist_ok=True)
        (self.save_dir / "images").mkdir(exist_ok=True)
        (self.save_dir / "results").mkdir(exist_ok=True)
        
    def initialize_camera(self):
        """初始化摄像头"""
        return self.camera.initialize() and self.camera.start_streams()
    
    def detect_objects(self, image):
        """
        使用YOLO检测物体
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果和带标注的图像
        """
        results = self.model(image, conf=self.conf_threshold)
        
        # 在图像上绘制检测结果
        annotated_image = results[0].plot()
        
        return results[0], annotated_image
    
    def save_detection_result(self, original_image, annotated_image, depth_image, results):
        """保存检测结果"""
        timestamp = int(time.time() * 1000)
        
        # 保存原始图像
        original_path = self.save_dir / "images" / f"original_{timestamp}.jpg"
        cv2.imwrite(str(original_path), original_image)
        
        # 保存检测结果图像
        result_path = self.save_dir / "results" / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(result_path), annotated_image)
        
        # 保存深度图像
        depth_path = self.save_dir / "results" / f"depth_{timestamp}.png"
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        cv2.imwrite(str(depth_path), depth_colormap)
        
        # 保存检测信息到文本文件
        info_path = self.save_dir / "results" / f"info_{timestamp}.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测到的物体数量: {len(results.boxes)}\n")
            f.write("检测结果:\n")
            
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                f.write(f"  {i+1}. {class_name}: {conf:.2f}\n")
        
        print(f"检测结果已保存到: {result_path}")
        return result_path
    
    def get_depth_at_detection(self, depth_image, box):
        """获取检测框中心点的深度信息"""
        try:
            # 获取检测框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 将彩色图像坐标映射到深度图像坐标
            # 彩色图像: 1920x1080, 深度图像: 640x480
            depth_x = int(center_x * 640 / 1920)
            depth_y = int(center_y * 480 / 1080)
            
            # 获取深度值（毫米）
            if 0 <= depth_x < 640 and 0 <= depth_y < 480:
                depth_value = depth_image[depth_y, depth_x]
                return depth_value if depth_value > 0 else None
            return None
        except:
            return None
    
    def run_detection(self, save_detections=False):
        """运行实时检测"""
        if not self.initialize_camera():
            print("摄像头初始化失败")
            return
        
        print("实时检测已启动")
        print("按键说明:")
        print("  'q' - 退出")
        print("  's' - 保存当前检测结果")
        print("  'space' - 暂停/继续")
        
        paused = False
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                if not paused:
                    # 获取摄像头数据
                    color_img, depth_img, ir_img = self.camera.get_frames()
                    
                    if color_img is not None:
                        # YOLO检测
                        results, annotated_img = self.detect_objects(color_img)
                        
                        # 在图像上添加深度信息
                        if depth_img is not None and len(results.boxes) > 0:
                            for box in results.boxes:
                                depth_value = self.get_depth_at_detection(depth_img, box)
                                if depth_value:
                                    # 在检测框上添加深度信息
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.putText(annotated_img, f"Depth: {depth_value}mm", 
                                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.5, (0, 255, 255), 1)
                        
                        # 显示检测结果
                        display_img = cv2.resize(annotated_img, (960, 540))
                        
                        # 显示FPS
                        fps_counter += 1
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 1.0:
                            fps = fps_counter / elapsed_time
                            fps_counter = 0
                            start_time = time.time()
                            
                        cv2.putText(display_img, f"FPS: {fps:.1f}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2)
                        
                        cv2.imshow('YOLO Real-time Detection', display_img)
                        
                        # 自动保存检测结果
                        if save_detections and len(results.boxes) > 0:
                            self.save_detection_result(color_img, annotated_img, depth_img, results)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if 'color_img' in locals() and color_img is not None:
                        self.save_detection_result(color_img, annotated_img, depth_img, results)
                elif key == ord(' '):
                    paused = not paused
                    print("检测已暂停" if paused else "检测已继续")
                    
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        finally:
            cv2.destroyAllWindows()
            self.camera.close()
            print("实时检测已结束")

# 使用示例
if __name__ == "__main__":
    # 使用你项目中的YOLO模型
    model_path = "yolov8n.pt"  # 或者使用你训练的模型路径
    
    detector = RealTimeYOLODetection(
        model_path=model_path,
        conf_threshold=0.5
    )
    
    # 运行实时检测
    detector.run_detection(save_detections=False)  # 设置为True可自动保存所有检测结果
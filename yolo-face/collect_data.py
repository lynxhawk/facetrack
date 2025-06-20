import cv2
import os
import time
from pathlib import Path
from astra_camera import AstraCamera

class DataCollector:
    def __init__(self, dataset_name="custom_dataset"):
        """
        数据采集器
        
        Args:
            dataset_name: 数据集名称
        """
        self.camera = AstraCamera()
        self.dataset_path = Path(dataset_name)
        self.setup_directories()
        
    def setup_directories(self):
        """创建数据集目录结构"""
        # 创建YOLO格式的目录结构
        dirs = [
            self.dataset_path / "images" / "train",
            self.dataset_path / "images" / "val",
            self.dataset_path / "labels" / "train", 
            self.dataset_path / "labels" / "val",
            self.dataset_path / "depth" / "train",
            self.dataset_path / "depth" / "val"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"数据集目录已创建: {self.dataset_path}")
    
    def collect_images(self, mode="train", interval=1.0):
        """
        采集图像数据
        
        Args:
            mode: 'train' 或 'val'
            interval: 自动采集间隔（秒）
        """
        if not self.camera.initialize() or not self.camera.start_streams():
            print("摄像头初始化失败")
            return
        
        print(f"开始采集 {mode} 数据")
        print("按键说明:")
        print("  'c' - 手动捕获当前帧")
        print("  'a' - 开启/关闭自动采集")
        print("  'q' - 退出")
        
        auto_collect = False
        last_auto_time = time.time()
        image_count = 0
        
        try:
            while True:
                color_img, depth_img, ir_img = self.camera.get_frames()
                
                if color_img is not None:
                    # 显示预览
                    display_img = cv2.resize(color_img, (960, 540))
                    
                    # 显示状态信息
                    status_text = f"Mode: {mode} | Count: {image_count} | Auto: {'ON' if auto_collect else 'OFF'}"
                    cv2.putText(display_img, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Data Collection', display_img)
                    
                    # 自动采集
                    if auto_collect and time.time() - last_auto_time >= interval:
                        self.save_sample(color_img, depth_img, mode, image_count)
                        image_count += 1
                        last_auto_time = time.time()
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # 手动采集
                    if color_img is not None:
                        self.save_sample(color_img, depth_img, mode, image_count)
                        image_count += 1
                        print(f"已保存第 {image_count} 张图像")
                elif key == ord('a'):
                    auto_collect = not auto_collect
                    print(f"自动采集: {'开启' if auto_collect else '关闭'}")
                    
        except KeyboardInterrupt:
            print("\n数据采集被中断")
        finally:
            cv2.destroyAllWindows()
            self.camera.close()
            print(f"数据采集完成，共采集 {image_count} 张图像")
    
    def save_sample(self, color_img, depth_img, mode, count):
        """保存单个样本"""
        timestamp = int(time.time() * 1000)
        filename = f"{mode}_{count:06d}_{timestamp}"
        
        # 保存彩色图像
        color_path = self.dataset_path / "images" / mode / f"{filename}.jpg"
        cv2.imwrite(str(color_path), color_img)
        
        # 保存深度图像
        if depth_img is not None:
            depth_path = self.dataset_path / "depth" / mode / f"{filename}.png"
            cv2.imwrite(str(depth_path), depth_img)
        
        # 创建空的标签文件（待后续标注）
        label_path = self.dataset_path / "labels" / mode / f"{filename}.txt"
        label_path.touch()
        
        print(f"已保存: {filename}")
    
    def create_yaml_config(self, class_names):
        """创建YOLO配置文件"""
        yaml_content = f"""# Dataset configuration for YOLO training
path: {self.dataset_path.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
        
        yaml_path = self.dataset_path / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"配置文件已创建: {yaml_path}")

# 使用示例
if __name__ == "__main__":
    collector = DataCollector("astra_dataset")
    
    # 采集训练数据
    print("开始采集训练数据...")
    collector.collect_images(mode="train", interval=2.0)
    
    # 采集验证数据
    print("开始采集验证数据...")
    collector.collect_images(mode="val", interval=2.0)
    
    # 创建配置文件
    class_names = ["person", "car", "bottle"]  # 根据你的需求修改类别
    collector.create_yaml_config(class_names)
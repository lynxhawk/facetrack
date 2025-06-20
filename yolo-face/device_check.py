import cv2
import numpy as np
from openni_config import OpenNIConfig
import time
import threading
from queue import Queue, Empty
from pathlib import Path
import traceback

# 安全的YOLO导入
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO已加载")
except ImportError as e:
    print(f"❌ YOLO导入失败: {e}")
    YOLO_AVAILABLE = False

class RobustDepthCamera:
    """鲁棒的深度摄像头"""
    
    def __init__(self):
        self.config = OpenNIConfig()
        self.openni2 = None
        self.c_api = None
        self.dev = None
        self.depth_stream = None
        
        # 线程和队列
        self.depth_thread = None
        self.depth_queue = Queue(maxsize=2)
        self.running = False
        
    def initialize(self):
        """初始化深度摄像头"""
        try:
            self.openni2 = self.config.initialize_openni2()
            from primesense import _openni2 as c_api
            self.c_api = c_api
            
            self.dev = self.openni2.Device.open_any()
            print(f"✅ 设备已打开: {self.dev.get_device_info().name}")
            
            # 设置深度流
            self.depth_stream = self.dev.create_depth_stream()
            
            depth_mode = self.c_api.OniVideoMode(
                pixelFormat=self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640,
                resolutionY=480,
                fps=30
            )
            self.depth_stream.set_video_mode(depth_mode)
            self.depth_stream.start()
            
            print("✅ 深度流启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            traceback.print_exc()
            return False
    
    def depth_reader_thread(self):
        """深度流读取线程"""
        consecutive_errors = 0
        max_errors = 5
        
        while self.running and consecutive_errors < max_errors:
            try:
                frame = self.depth_stream.read_frame()
                depth_data = frame.get_buffer_as_uint16()
                depth_img = np.frombuffer(depth_data, dtype=np.uint16)
                depth_img = depth_img.reshape(frame.height, frame.width)
                
                # 保持队列最新
                while not self.depth_queue.empty():
                    try:
                        self.depth_queue.get_nowait()
                    except Empty:
                        break
                
                self.depth_queue.put_nowait(depth_img)
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                print(f"深度读取错误 {consecutive_errors}/{max_errors}: {e}")
                time.sleep(0.1)
        
        if consecutive_errors >= max_errors:
            print("深度读取线程因连续错误过多而退出")
        else:
            print("深度读取线程正常退出")
    
    def start_stream(self):
        """启动深度流"""
        self.running = True
        self.depth_thread = threading.Thread(target=self.depth_reader_thread)
        self.depth_thread.daemon = True
        self.depth_thread.start()
        print("✅ 深度读取线程已启动")
    
    def get_depth_frame(self):
        """获取最新深度帧"""
        try:
            return self.depth_queue.get_nowait()
        except Empty:
            return None
    
    def close(self):
        """关闭摄像头"""
        print("正在关闭深度摄像头...")
        self.running = False
        
        if self.depth_thread:
            self.depth_thread.join(timeout=3)
        
        if self.depth_stream:
            try:
                self.depth_stream.stop()
                print("深度流已停止")
            except Exception as e:
                print(f"停止深度流时出错: {e}")
        
        if self.openni2:
            try:
                self.openni2.unload()
                print("OpenNI2已卸载")
            except Exception as e:
                print(f"卸载OpenNI2时出错: {e}")
        
        print("✅ 深度摄像头已关闭")

class SafeYOLODetector:
    """安全的YOLO检测器"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.last_error_time = 0
        self.error_cooldown = 5  # 5秒错误冷却时间
        
        if YOLO_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """安全加载YOLO模型"""
        try:
            print("正在加载YOLO模型...")
            self.model = YOLO("yolov8n-face.pt")
            
            # 测试模型是否正常工作
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            results = self.model(test_img, conf=0.5, verbose=False)
            
            self.model_loaded = True
            print("✅ YOLO模型加载成功并测试通过")
            
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            traceback.print_exc()
            self.model = None
            self.model_loaded = False
    
    def detect(self, image, conf=0.5):
        """安全的检测方法"""
        if not self.model_loaded or self.model is None:
            return None, image
        
        # 检查错误冷却
        current_time = time.time()
        if current_time - self.last_error_time < self.error_cooldown:
            return None, image
        
        try:
            # 检查输入图像
            if image is None or image.size == 0:
                return None, image
            
            # 确保图像格式正确
            if len(image.shape) != 3 or image.shape[2] != 3:
                print("图像格式不正确，跳过检测")
                return None, image
            
            # 进行检测
            results = self.model(image, conf=conf, verbose=False)
            
            if not results or len(results) == 0:
                return None, image
            
            result = results[0]
            
            # 手动绘制检测结果（更稳定）
            annotated_img = image.copy()
            
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        # 获取坐标和类别信息
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf_score = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # 确保坐标在图像范围内
                        h, w = image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 > x1 and y2 > y1:  # 确保检测框有效
                            class_name = self.model.names[cls_id]
                            
                            # 绘制检测框
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # 绘制标签背景
                            label = f"{class_name}: {conf_score:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_img, (x1, y1-label_size[1]-10), 
                                        (x1+label_size[0], y1), (0, 255, 0), -1)
                            
                            # 绘制标签文本
                            cv2.putText(annotated_img, label, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    except Exception as e:
                        print(f"绘制检测框时出错: {e}")
                        continue
            
            return result, annotated_img
            
        except Exception as e:
            print(f"YOLO检测错误: {e}")
            self.last_error_time = current_time
            return None, image

class RobustDepthProcessor:
    """鲁棒的深度数据处理器"""
    
    def __init__(self):
        self.camera = RobustDepthCamera()
        self.yolo_detector = SafeYOLODetector()
        self.save_dir = Path("depth_captures")
        self.save_dir.mkdir(exist_ok=True)
        
    def depth_to_colormap(self, depth_img):
        """将深度图转换为伪彩色"""
        try:
            if depth_img is None or depth_img.size == 0:
                return None
            
            # 限制深度范围（500mm到5000mm）
            depth_clipped = np.clip(depth_img, 500, 5000)
            
            # 创建有效深度掩码
            valid_mask = depth_img > 0
            
            if not np.any(valid_mask):
                # 如果没有有效深度数据，返回黑色图像
                return np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
            
            # 归一化到0-255
            depth_norm = np.zeros_like(depth_img, dtype=np.uint8)
            depth_norm[valid_mask] = ((depth_clipped[valid_mask] - 500) / 4500 * 255).astype(np.uint8)
            
            # 应用伪彩色
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            
            # 无效区域设为黑色
            depth_colored[~valid_mask] = [0, 0, 0]
            
            return depth_colored
            
        except Exception as e:
            print(f"深度彩色映射错误: {e}")
            return None
    
    def add_depth_info(self, depth_img, annotated_img, results):
        """添加深度信息到检测结果"""
        try:
            if results is None or not hasattr(results, 'boxes') or results.boxes is None:
                return annotated_img
            
            if len(results.boxes) == 0:
                return annotated_img
            
            for box in results.boxes:
                try:
                    # 获取检测框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 计算中心点深度
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if (0 <= center_x < depth_img.shape[1] and 
                        0 <= center_y < depth_img.shape[0]):
                        
                        depth_value = depth_img[center_y, center_x]
                        if depth_value > 0:
                            # 添加深度信息文本
                            depth_text = f"{depth_value}mm"
                            cv2.putText(annotated_img, depth_text, 
                                      (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (255, 255, 0), 2)
                            
                except Exception as e:
                    print(f"添加深度信息时出错: {e}")
                    continue
            
            return annotated_img
            
        except Exception as e:
            print(f"处理深度信息时出错: {e}")
            return annotated_img
    
    def run_viewer(self):
        """运行深度查看器"""
        print("正在初始化深度摄像头...")
        
        if not self.camera.initialize():
            print("❌ 摄像头初始化失败")
            return
        
        self.camera.start_stream()
        
        print("\n=== 鲁棒深度摄像头查看器 ===")
        print("功能:")
        if self.yolo_detector.model_loaded:
            print("✅ YOLO物体检测")
        else:
            print("❌ YOLO检测不可用")
        print("✅ 深度可视化")
        print("✅ 距离测量")
        
        print("\n按键说明:")
        print("  'q' - 退出")
        print("  's' - 保存图像")
        print("  '1' - 原始深度图")
        print("  '2' - 伪彩色深度图")
        if self.yolo_detector.model_loaded:
            print("  '3' - YOLO检测结果")
        print("  'r' - 重新加载YOLO模型")
        print("  鼠标点击 - 显示该点深度值")
        
        display_mode = 2  # 默认显示伪彩色
        frame_count = 0
        click_depth = None
        click_pos = None
        current_depth_img = None
        fps_counter = 0
        last_fps_time = time.time()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal click_depth, click_pos, current_depth_img
            if event == cv2.EVENT_LBUTTONDOWN and current_depth_img is not None:
                try:
                    if 0 <= y < current_depth_img.shape[0] and 0 <= x < current_depth_img.shape[1]:
                        click_depth = current_depth_img[y, x]
                        click_pos = (x, y)
                        if click_depth > 0:
                            print(f"点击位置 ({x}, {y}) 的深度: {click_depth}mm")
                        else:
                            print(f"点击位置 ({x}, {y}) 无有效深度数据")
                except Exception as e:
                    print(f"鼠标回调错误: {e}")
        
        # 创建窗口
        window_name = 'Robust Depth Viewer'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        try:
            print("开始显示深度数据... (按 'q' 退出)")
            
            while True:
                try:
                    depth_img = self.camera.get_depth_frame()
                    
                    if depth_img is not None:
                        current_depth_img = depth_img
                        display_img = None
                        title = ""
                        
                        # 准备显示图像
                        if display_mode == 1:
                            # 原始深度图
                            try:
                                display_img = cv2.convertScaleAbs(depth_img, alpha=0.05)
                                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                                title = "Raw Depth"
                            except Exception as e:
                                print(f"原始深度图处理错误: {e}")
                                continue
                        
                        elif display_mode == 2:
                            # 伪彩色深度图
                            try:
                                display_img = self.depth_to_colormap(depth_img)
                                title = "Depth Colormap"
                            except Exception as e:
                                print(f"伪彩色深度图处理错误: {e}")
                                continue
                        
                        elif display_mode == 3 and self.yolo_detector.model_loaded:
                            # YOLO检测
                            try:
                                print(f"Frame {frame_count}: 开始YOLO检测...")
                                depth_colored = self.depth_to_colormap(depth_img)
                                
                                if depth_colored is not None:
                                    results, display_img = self.yolo_detector.detect(depth_colored, conf=0.5)
                                    
                                    if display_img is not None and results is not None:
                                        display_img = self.add_depth_info(depth_img, display_img, results)
                                        print(f"Frame {frame_count}: YOLO检测完成")
                                    else:
                                        print(f"Frame {frame_count}: YOLO检测返回空结果")
                                        display_img = depth_colored
                                else:
                                    print(f"Frame {frame_count}: 深度彩色映射失败")
                                    display_img = None
                                
                                title = "YOLO Detection (Safe Mode)"
                                
                            except Exception as e:
                                print(f"YOLO检测模式错误: {e}")
                                traceback.print_exc()
                                # 发生错误时回退到伪彩色模式
                                display_img = self.depth_to_colormap(depth_img)
                                title = "Depth Colormap (YOLO Error Fallback)"
                        
                        else:
                            # 默认伪彩色
                            display_img = self.depth_to_colormap(depth_img)
                            title = "Depth Colormap"
                        
                        if display_img is not None:
                            # 计算FPS
                            fps_counter += 1
                            current_time = time.time()
                            if current_time - last_fps_time >= 1.0:
                                fps = fps_counter / (current_time - last_fps_time)
                                fps_counter = 0
                                last_fps_time = current_time
                            else:
                                fps = 0
                            
                            # 添加文本信息
                            info_y = 30
                            cv2.putText(display_img, f"{title} - Frame {frame_count}", 
                                      (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            if fps > 0:
                                info_y += 30
                                cv2.putText(display_img, f"FPS: {fps:.1f}", 
                                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # 显示YOLO状态
                            if display_mode == 3:
                                info_y += 30
                                yolo_status = "YOLO: OK" if self.yolo_detector.model_loaded else "YOLO: ERROR"
                                color = (0, 255, 0) if self.yolo_detector.model_loaded else (0, 0, 255)
                                cv2.putText(display_img, yolo_status, 
                                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # 显示点击点的深度
                            if click_depth is not None and click_pos is not None and click_depth > 0:
                                depth_text = f"Depth: {click_depth}mm"
                                cv2.putText(display_img, depth_text, 
                                          (click_pos[0], click_pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (0, 255, 255), 2)
                                cv2.circle(display_img, click_pos, 5, (0, 255, 255), 2)
                            
                            # 添加使用说明
                            cv2.putText(display_img, "Click to measure | Keys: 1,2,3,s,q,r", 
                                      (10, display_img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (255, 255, 255), 1)
                            
                            cv2.imshow(window_name, display_img)
                            frame_count += 1
                    
                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户请求退出")
                        break
                    elif key == ord('s'):
                        if depth_img is not None:
                            try:
                                timestamp = int(time.time() * 1000)
                                
                                # 保存原始深度图
                                depth_path = self.save_dir / f"depth_{timestamp}.png"
                                cv2.imwrite(str(depth_path), depth_img)
                                
                                # 保存显示图像
                                if 'display_img' in locals() and display_img is not None:
                                    display_path = self.save_dir / f"display_{timestamp}.jpg"
                                    cv2.imwrite(str(display_path), display_img)
                                
                                print(f"图像已保存: {timestamp}")
                            except Exception as e:
                                print(f"保存图像时出错: {e}")
                    
                    elif key == ord('1'):
                        display_mode = 1
                        click_depth = None
                        click_pos = None
                        print("切换到原始深度图")
                    elif key == ord('2'):
                        display_mode = 2
                        click_depth = None
                        click_pos = None
                        print("切换到伪彩色深度图")
                    elif key == ord('3'):
                        if self.yolo_detector.model_loaded:
                            display_mode = 3
                            click_depth = None
                            click_pos = None
                            print("切换到YOLO检测模式（安全模式）")
                        else:
                            print("YOLO模型未加载，无法切换到检测模式")
                    elif key == ord('r'):
                        print("重新加载YOLO模型...")
                        self.yolo_detector = SafeYOLODetector()
                        if self.yolo_detector.model_loaded:
                            print("YOLO模型重新加载成功")
                        else:
                            print("YOLO模型重新加载失败")
                
                except KeyboardInterrupt:
                    print("\n收到键盘中断信号")
                    break
                except Exception as e:
                    print(f"主循环错误: {e}")
                    traceback.print_exc()
                    time.sleep(0.1)  # 短暂休息后继续
                    
        except Exception as e:
            print(f"查看器运行错误: {e}")
            traceback.print_exc()
        finally:
            print("正在清理资源...")
            cv2.destroyAllWindows()
            self.camera.close()
            print("程序退出完成")

if __name__ == "__main__":
    # 检查依赖
    print("=== 鲁棒深度YOLO检测系统 ===")
    print(f"OpenCV: ✅")
    print(f"NumPy: ✅")
    print(f"YOLO: {'✅' if YOLO_AVAILABLE else '❌ (功能受限)'}")
    
    if not YOLO_AVAILABLE:
        print("\n如需YOLO检测功能，请安装:")
        print("pip install ultralytics")
    
    # 创建并运行鲁棒深度处理器
    processor = RobustDepthProcessor()
    processor.run_viewer()
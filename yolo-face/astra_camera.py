import cv2
import numpy as np
from openni_config import OpenNIConfig
import time

class AstraSimple:
    def __init__(self):
        self.config = OpenNIConfig()
        self.openni2 = None
        self.c_api = None
        self.dev = None
        self.depth_stream = None
        self.color_stream = None
        
    def initialize(self, use_color=True, use_depth=True):
        """初始化摄像头 - 可选择启用的流"""
        try:
            self.openni2 = self.config.initialize_openni2()
            from primesense import _openni2 as c_api
            self.c_api = c_api
            
            # 打开设备
            self.dev = self.openni2.Device.open_any()
            print(f"设备已打开: {self.dev.get_device_info().name}")
            
            success = False
            
            # 尝试启动深度流
            if use_depth:
                if self.setup_depth_stream():
                    success = True
                    print("✅ 深度流可用")
                else:
                    print("❌ 深度流不可用")
            
            # 尝试启动彩色流
            if use_color:
                if self.setup_color_stream():
                    success = True
                    print("✅ 彩色流可用")
                else:
                    print("❌ 彩色流不可用")
            
            return success
            
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def setup_depth_stream(self):
        """设置深度流"""
        try:
            if not self.dev.has_sensor(self.c_api.OniSensorType.ONI_SENSOR_DEPTH):
                return False
            
            self.depth_stream = self.dev.create_depth_stream()
            
            # 使用最基础的深度设置
            depth_mode = self.c_api.OniVideoMode(
                pixelFormat=self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640,
                resolutionY=480,
                fps=30
            )
            self.depth_stream.set_video_mode(depth_mode)
            self.depth_stream.start()
            
            return True
            
        except Exception as e:
            print(f"深度流设置失败: {e}")
            self.depth_stream = None
            return False
    
    def setup_color_stream(self):
        """设置彩色流"""
        try:
            if not self.dev.has_sensor(self.c_api.OniSensorType.ONI_SENSOR_COLOR):
                return False
            
            self.color_stream = self.dev.create_color_stream()
            
            # 尝试不同的彩色设置，从低分辨率开始
            color_configs = [
                # (width, height, fps, pixel_format)
                (640, 480, 30, self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888),
                (320, 240, 30, self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888),
                (1280, 720, 15, self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888),
                (1920, 1080, 15, self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888),
            ]
            
            for width, height, fps, pixel_format in color_configs:
                try:
                    color_mode = self.c_api.OniVideoMode(
                        pixelFormat=pixel_format,
                        resolutionX=width,
                        resolutionY=height,
                        fps=fps
                    )
                    self.color_stream.set_video_mode(color_mode)
                    self.color_stream.start()
                    print(f"彩色流设置成功: {width}x{height} @ {fps}fps")
                    return True
                    
                except Exception as e:
                    print(f"彩色配置 {width}x{height}@{fps} 失败: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"彩色流设置失败: {e}")
            self.color_stream = None
            return False
    
    def get_frames(self):
        """获取帧数据"""
        color_img = None
        depth_img = None
        
        try:
            # 获取深度帧
            if self.depth_stream:
                try:
                    depth_frame = self.depth_stream.read_frame()
                    depth_data = depth_frame.get_buffer_as_uint16()
                    depth_img = np.frombuffer(depth_data, dtype=np.uint16)
                    depth_img = depth_img.reshape(depth_frame.height, depth_frame.width)
                except Exception as e:
                    print(f"读取深度帧失败: {e}")
            
            # 获取彩色帧
            if self.color_stream:
                try:
                    color_frame = self.color_stream.read_frame()
                    color_data = color_frame.get_buffer_as_uint8()
                    color_img = np.frombuffer(color_data, dtype=np.uint8)
                    color_img = color_img.reshape(color_frame.height, color_frame.width, 3)
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"读取彩色帧失败: {e}")
            
            return color_img, depth_img
            
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None
    
    def close(self):
        """关闭摄像头"""
        try:
            if self.depth_stream:
                self.depth_stream.stop()
            if self.color_stream:
                self.color_stream.stop()
            if self.openni2:
                self.openni2.unload()
            print("摄像头已关闭")
        except Exception as e:
            print(f"关闭失败: {e}")

def test_different_modes():
    """测试不同的流组合"""
    print("=== 测试不同的流组合 ===")
    
    # 测试1: 只用深度
    print("\n1. 测试深度流...")
    camera1 = AstraSimple()
    if camera1.initialize(use_color=False, use_depth=True):
        print("深度流测试成功，采集几帧数据...")
        for i in range(5):
            color_img, depth_img = camera1.get_frames()
            if depth_img is not None:
                print(f"  深度帧 {i+1}: {depth_img.shape}, 范围: {depth_img.min()}-{depth_img.max()}")
            time.sleep(0.1)
    camera1.close()
    
    time.sleep(1)
    
    # 测试2: 只用彩色
    print("\n2. 测试彩色流...")
    camera2 = AstraSimple()
    if camera2.initialize(use_color=True, use_depth=False):
        print("彩色流测试成功，采集几帧数据...")
        for i in range(5):
            color_img, depth_img = camera2.get_frames()
            if color_img is not None:
                print(f"  彩色帧 {i+1}: {color_img.shape}")
            time.sleep(0.1)
    camera2.close()
    
    time.sleep(1)
    
    # 测试3: 同时使用（可能会失败）
    print("\n3. 测试深度+彩色流...")
    camera3 = AstraSimple()
    if camera3.initialize(use_color=True, use_depth=True):
        print("深度+彩色流测试成功！")
        
        print("开始显示图像，按 'q' 退出...")
        while True:
            color_img, depth_img = camera3.get_frames()
            
            if depth_img is not None:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                cv2.imshow('Depth', depth_colormap)
            
            if color_img is not None:
                cv2.imshow('Color', color_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    camera3.close()

if __name__ == "__main__":
    test_different_modes()
import os
import sys
from pathlib import Path

class OpenNIConfig:
    """OpenNI2 配置管理 - 适配你的SDK结构"""
    
    def __init__(self):
        self.openni2_path = None
        self.setup_openni2_path()
    
    def setup_openni2_path(self):
        """自动检测OpenNI2安装路径"""
        # 项目本地SDK路径 - 基于你的结构
        project_root = Path(__file__).parent
        local_paths = [
            project_root / "sdk" / "libs",           # 你的SDK结构
            project_root / "openni2" / "libs",       # 备用结构
            project_root / "OpenNI2" / "Redist",     # 标准结构
        ]
        
        # 系统安装路径
        system_paths = [
            r'C:\Program Files\OpenNI2\Redist',
            r'C:\Program Files (x86)\OpenNI2\Redist',
        ]
        
        possible_paths = local_paths + system_paths
        
        # 检查环境变量
        env_path = os.environ.get('OPENNI2_REDIST64') or os.environ.get('OPENNI2_REDIST')
        if env_path and Path(env_path).exists():
            self.openni2_path = env_path
            print(f"使用环境变量中的OpenNI2路径: {self.openni2_path}")
            return
        
        # 检查可能的安装路径
        for path in possible_paths:
            if path.exists() and self.verify_path(path):
                self.openni2_path = str(path)
                print(f"找到OpenNI2路径: {self.openni2_path}")
                return
        
        if not self.openni2_path:
            print("警告: 未找到OpenNI2安装路径")
            print("请确保SDK已放置在项目目录中")
    
    def verify_path(self, path):
        """验证路径是否包含必要的OpenNI2文件"""
        path = Path(path)
        
        # 检查关键文件
        required_files = ["OpenNI2.dll"]
        
        for file in required_files:
            if not (path / file).exists():
                return False
        return True
    
    def initialize_openni2(self):
        """初始化OpenNI2"""
        if not self.openni2_path:
            raise Exception("OpenNI2路径未设置")
        
        # 设置环境变量
        os.environ['OPENNI2_REDIST'] = self.openni2_path
        
        # 添加到系统路径
        if self.openni2_path not in sys.path:
            sys.path.insert(0, self.openni2_path)
        
        try:
            from primesense import openni2
            openni2.initialize(self.openni2_path)
            print("OpenNI2初始化成功")
            return openni2
        except Exception as e:
            print(f"OpenNI2初始化失败: {e}")
            print("可能的解决方案:")
            print("1. 确保已安装 primesense: pip install primesense")
            print("2. 检查SDK文件是否完整")
            print("3. 确保使用64位Python（如果SDK是64位）")
            raise
    
    def check_installation(self):
        """检查安装是否正确"""
        print("=== OpenNI2 安装检查 ===")
        
        # 检查路径
        print(f"OpenNI2路径: {self.openni2_path}")
        if self.openni2_path and Path(self.openni2_path).exists():
            print("✓ 路径存在")
        else:
            print("✗ 路径不存在")
            return False
        
        # 检查关键文件
        key_files = ['OpenNI2.dll']
        for file in key_files:
            file_path = Path(self.openni2_path) / file
            if file_path.exists():
                print(f"✓ {file} 存在")
            else:
                print(f"✗ {file} 未找到")
                return False
        
        # 检查驱动文件夹
        drivers_path = Path(self.openni2_path) / "OpenNI2"
        if drivers_path.exists():
            print("✓ OpenNI2驱动文件夹存在")
        else:
            print("? OpenNI2驱动文件夹未找到（可能不影响基本功能）")
        
        # 检查Python库
        try:
            from primesense import openni2
            print("✓ primesense库导入成功")
        except ImportError:
            print("✗ primesense库导入失败")
            print("请运行: pip install primesense")
            return False
        
        # 测试初始化
        try:
            openni2_lib = self.initialize_openni2()
            print("✓ OpenNI2初始化成功")
            openni2_lib.unload()
            return True
        except Exception as e:
            print(f"✗ OpenNI2初始化失败: {e}")
            return False
    
    def get_sdk_info(self):
        """获取SDK信息"""
        if not self.openni2_path:
            return None
        
        sdk_path = Path(self.openni2_path)
        info = {
            "path": str(sdk_path),
            "dll_exists": (sdk_path / "OpenNI2.dll").exists(),
            "lib_exists": (sdk_path / "OpenNI2.lib").exists(),
            "drivers_exist": (sdk_path / "OpenNI2").exists(),
        }
        
        return info

# 使用示例和测试
if __name__ == "__main__":
    config = OpenNIConfig()
    
    # 显示SDK信息
    info = config.get_sdk_info()
    if info:
        print("\n=== SDK信息 ===")
        for key, value in info.items():
            print(f"{key}: {value}")
    
    # 运行完整检查
    print("\n")
    config.check_installation()
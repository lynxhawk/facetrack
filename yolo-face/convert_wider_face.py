import os
import shutil
from PIL import Image
import argparse

def parse_wider_face_annotation(annotation_file):
    """解析 WIDER FACE 标注文件"""
    annotations = {}
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过空行
        if not line:
            i += 1
            continue
        
        # 第一行是图片路径（包含 / 或 .jpg 等）
        if ('/' in line or '\\' in line) and any(ext in line.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            img_path = line
            i += 1
            
            # 下一行是人脸数量
            if i < len(lines):
                try:
                    num_faces = int(lines[i].strip())
                    i += 1
                except ValueError:
                    print(f"Warning: Expected number of faces, got: {lines[i].strip()}")
                    i += 1
                    continue
                
                faces = []
                # 读取人脸边界框
                for face_idx in range(num_faces):
                    if i < len(lines):
                        bbox_line = lines[i].strip()
                        
                        # 解析边界框: x1 y1 w h blur expression illumination invalid occlusion pose
                        bbox_info = bbox_line.split()
                        if len(bbox_info) >= 4:
                            try:
                                x1, y1, w, h = map(int, bbox_info[:4])
                                # 只保留有效的边界框 (w > 0 and h > 0)
                                if w > 0 and h > 0:
                                    faces.append([x1, y1, w, h])
                            except ValueError:
                                print(f"Warning: Invalid bbox format: {bbox_line}")
                        i += 1
                
                annotations[img_path] = faces
        else:
            # 如果不是图片路径，可能是误读，尝试作为数字解析
            try:
                # 如果能解析为数字，可能是人脸数量行被误读为图片路径
                num_faces = int(line)
                print(f"Warning: Found number {num_faces} where expected image path. Skipping.")
                i += 1
                # 跳过对应数量的边界框行
                for _ in range(num_faces):
                    if i < len(lines):
                        i += 1
            except ValueError:
                # 既不是图片路径也不是数字，跳过
                print(f"Warning: Skipping unrecognized line: {line}")
                i += 1
    
    return annotations

def convert_to_yolo_format(x1, y1, w, h, img_width, img_height):
    """将 WIDER FACE 格式转换为 YOLO 格式"""
    # 计算中心点坐标
    center_x = x1 + w / 2
    center_y = y1 + h / 2
    
    # 归一化
    center_x /= img_width
    center_y /= img_height
    w /= img_width
    h /= img_height
    
    return center_x, center_y, w, h

def convert_wider_face_to_yolo(wider_face_dir, output_dir):
    """转换 WIDER FACE 数据集到 YOLO 格式"""
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)
    
    # 处理训练集、验证集和测试集
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"Processing {split} set...")
        
        # 标注文件路径
        if split == 'train':
            annotation_file = os.path.join(wider_face_dir, 'wider_face_train_bbx_gt.txt')
            image_dir = os.path.join(wider_face_dir, 'WIDER_train', 'images')
        elif split == 'val':
            annotation_file = os.path.join(wider_face_dir, 'wider_face_val_bbx_gt.txt')
            image_dir = os.path.join(wider_face_dir, 'WIDER_val', 'images')
        else:  # test
            # 测试集通常没有标注文件，只复制图片
            annotation_file = None
            image_dir = os.path.join(wider_face_dir, 'WIDER_test', 'images')
        
        # 如果是测试集且没有标注文件，只复制图片
        if split == 'test' and not os.path.exists(os.path.join(wider_face_dir, 'wider_face_test_bbx_gt.txt')):
            print(f"Processing {split} set (images only, no annotations)...")
            if os.path.exists(image_dir):
                for root, dirs, files in os.walk(image_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src_path = os.path.join(root, file)
                            # 创建相对路径作为文件名
                            rel_path = os.path.relpath(src_path, image_dir)
                            dst_filename = rel_path.replace('/', '_').replace('\\', '_')
                            dst_path = os.path.join(output_dir, 'images', split, dst_filename)
                            shutil.copy2(src_path, dst_path)
            continue
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found: {annotation_file}")
            continue
        
        # 解析标注
        annotations = parse_wider_face_annotation(annotation_file)
        
        # 转换每张图片
        for img_path, faces in annotations.items():
            # 完整图片路径
            full_img_path = os.path.join(image_dir, img_path)
            
            if not os.path.exists(full_img_path):
                print(f"Warning: Image not found: {full_img_path}")
                continue
            
            # 获取图片尺寸
            try:
                with Image.open(full_img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {full_img_path}: {e}")
                continue
            
            # 创建 YOLO 标注
            yolo_annotations = []
            for face in faces:
                x1, y1, w, h = face
                center_x, center_y, norm_w, norm_h = convert_to_yolo_format(
                    x1, y1, w, h, img_width, img_height
                )
                # 类别 0 表示人脸
                yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            # 复制图片到输出目录
            img_filename = img_path.replace('/', '_').replace('\\', '_')
            dst_img_path = os.path.join(output_dir, 'images', split, img_filename)
            shutil.copy2(full_img_path, dst_img_path)
            
            # 保存 YOLO 标注文件
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(output_dir, 'labels', split, label_filename)
            
            with open(label_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')
        
        print(f"Finished processing {split} set")

def create_yaml_config(output_dir):
    """创建 YOLO 配置文件"""
    yaml_content = f"""# WIDER FACE dataset configuration
path: {os.path.abspath(output_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path') - optional

# Classes
names:
  0: face

# Number of classes
nc: 1
"""
    
    yaml_path = os.path.join(output_dir, 'wider_face.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YAML configuration: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert WIDER FACE dataset to YOLO format')
    parser.add_argument('--wider_face_dir', type=str, required=True,
                        help='Path to WIDER FACE dataset directory')
    parser.add_argument('--output_dir', type=str, default='./datasets/wider_face',
                        help='Output directory for YOLO format dataset')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.wider_face_dir):
        print(f"Error: WIDER FACE directory not found: {args.wider_face_dir}")
        return
    
    # 转换数据集
    convert_wider_face_to_yolo(args.wider_face_dir, args.output_dir)
    
    # 创建配置文件
    create_yaml_config(args.output_dir)
    
    print("Conversion completed!")
    print(f"Dataset saved to: {args.output_dir}")
    print(f"You can now train YOLO with: python train.py --data {args.output_dir}/wider_face.yaml")

if __name__ == "__main__":
    main()
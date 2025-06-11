# from train_model import SimpleCNN  # 从train_model.py中导入SimpleCNN
import config.config as config
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from mtcnn import MTCNN
import numpy as np
import os
import sys
import io
import torchvision.models as models
import argparse
# from torchvision.models import ResNet18_Weights # 不再需要，除非你想保留注释

# 导入模型定义 (需要确保SimpleCNN类可以被访问到)
# 将项目根目录添加到sys.path以便导入 SimpleCNN
# 将当前目录（train_model.py所在目录）添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

# 导入配置文件以获取NUM_CLASSES等参数
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 定义表情类别映射 (需要与训练时的类别索引对应)
# fer2013数据集的表情类别通常是：
try:
    emotion_map = config.EMOTION_MAP
except AttributeError:
    print("错误: 配置文件 `config/config.py` 中未定义 `EMOTION_MAP`。请在配置文件中添加。")
    sys.exit(1)

# 定义图像预处理转换 (与训练时的验证/测试集转换一致)
# 注意：模型期望RGB输入并进行了ImageNet标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为Tensor，并缩放到[0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # ImageNet的均值和标准差
])

# 初始化MTCNN检测器
detector = MTCNN()
print("MTCNN检测器已初始化。")

# 初始化模型并加载训练好的权重
num_classes = config.NUM_CLASSES  # 从配置文件中获取类别数

print(f"初始化 ResNet18 模型，类别数: {num_classes}")

# --- 恢复到直接使用 models.resnet18 ---
# <--- 初始化 ResNet18 模型结构 (不加载ImageNet权重)
model = models.resnet18(weights=None)

# 修改最后一层全连接层，使其输出维度等于表情类别数
# ResNet18 的最后一层是 model.fc
num_ftrs = model.fc.in_features  # 获取倒数第二层输出的特征数量
model.fc = nn.Linear(num_ftrs, num_classes)  # <--- 替换最后一层

# 检查是否有GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载最佳模型权重
model_weight_path = 'best_model_state_dict.pth'  # 确保路径正确，相对于项目根目录
if os.path.exists(model_weight_path):
    # 加载权重，确保map_location与当前设备匹配
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print(f"成功加载模型权重: {model_weight_path}")
    model.eval()  # 设置模型为评估模式
else:
    print(
        f"错误: 未找到模型权重文件 {model_weight_path}，无法进行预测。请先运行 train_model.py 进行模型训练。")
    sys.exit(1)

# 定义一个函数来处理单张图像


def process_image(image_path, detector, model, transform, emotion_map, device):
    # 加载图像
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return None, None

    # 使用Pillow加载图像并转换为RGB (确保一致性)
    pil_image = Image.open(image_path).convert('RGB')
    # 将PIL图像转换为OpenCV格式 (MTCNN需要numpy数组)
    image_rgb_np = np.array(pil_image)

    # MTCNN需要RGB格式的图像
    results = detector.detect_faces(image_rgb_np)

    processed_image_cv2 = cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2BGR)  # 用于绘制
    detected_faces_info = []

    if results:
        for result in results:
            x, y, width, height = result['box']

            # 裁剪人脸区域，确保边界不超出图像
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(
                image_rgb_np.shape[1], x + width), min(image_rgb_np.shape[0], y + height)
            face_image = image_rgb_np[y1:y2, x1:x2]

            if face_image.shape[0] == 0 or face_image.shape[1] == 0:
                detected_faces_info.append({
                    'box': [x, y, width, height],
                    'emotion': '人脸区域无效或过小'
                })
                continue

            try:
                face_image_resized = cv2.resize(face_image, (224, 224))
            except cv2.error as e:
                detected_faces_info.append({
                    'box': [x, y, width, height],
                    'emotion': f'调整大小失败: {e}'
                })
                continue

            face_image_pil = Image.fromarray(face_image_resized)
            face_tensor = transform(face_image_pil)
            face_tensor = face_tensor.unsqueeze(0).to(device)  # 添加batch维度并移到设备

            with torch.no_grad():
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label_index = predicted.item()

            predicted_emotion = emotion_map.get(predicted_label_index, "未知表情")

            # 在原图上绘制人脸框和表情标签
            cv2.rectangle(processed_image_cv2, (x, y),
                          (x+width, y+height), (0, 255, 0), 2)
            display_text = predicted_emotion
            cv2.putText(processed_image_cv2, display_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            detected_faces_info.append({
                'box': [x, y, width, height],
                'emotion': predicted_emotion
            })

    return processed_image_cv2, detected_faces_info


# 主执行逻辑
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single Image Emotion Detector using MTCNN and ResNet18.')
    parser.add_argument(
        '-i', '--image', type=str, required=True, help='Path to the input image file.')
    args = parser.parse_args()

    processed_img, detections = process_image(
        args.image, detector, model, transform, emotion_map, device)

    if processed_img is not None:
        # 显示结果图像
        cv2.imshow('Detected Emotion', processed_img)
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()

        # 打印检测到的信息
        if detections:
            print("检测到的人脸和表情:")
            for det in detections:
                print(f"  框: {det['box']}, 表情: {det['emotion']}")
        else:
            print("未检测到人脸或表情。")

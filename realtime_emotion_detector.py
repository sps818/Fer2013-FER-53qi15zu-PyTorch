import config.config as config  # 从config.py导入配置
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from mtcnn import MTCNN
import numpy as np
import os
import sys
import time  # 可以用于计算帧率 (可选)
import torchvision.models as models  # 导入models模块
import argparse

# 导入模型定义和配置文件
# 确保相关目录在Python的搜索路径中
# 假设 realtime_emotion_recognition.py 在项目根目录
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '')))  # 添加项目根目录
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'config')))  # 添加config目录


# 定义表情类别映射 (需要与训练时的类别索引对应)
# fer2013数据集的表情类别通常是：
emotion_map = config.EMOTION_MAP

# 定义图像预处理转换 (与训练时的验证/测试集转换一致)
# 注意：模型期望RGB输入并进行了ImageNet标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 期望的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])
])

# 初始化MTCNN检测器 (全局初始化)
detector = MTCNN()
print("MTCNN检测器已初始化。")

# 初始化模型并加载训练好的权重 (全局初始化)
num_classes = config.NUM_CLASSES

print(f"初始化 ResNet18 模型，类别数: {num_classes}")
# 初始化 ResNet18 模型结构 (不需要预训练权重，我们加载自己训练的权重)
model = models.resnet18(weights=None)

# 修改最后一层全连接层，使其输出维度等于表情类别数
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# 检查是否有GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载最佳模型权重
model_weight_path = 'best_model_state_dict.pth'  # 确保路径正确，相对于项目根目录
if os.path.exists(model_weight_path):
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print(f"成功加载模型权重: {model_weight_path}")
    model.eval()  # 设置模型为评估模式
else:
    print(f"错误: 未找到模型权重文件 {model_weight_path}，无法进行实时识别。")
    exit()  # 如果没有模型权重，则退出脚本


# 定义一个函数来处理单帧图像
def process_frame(frame, detector, model, transform, emotion_map, device):
    """
    对单帧图像进行人脸检测和表情识别。
    Args:
        frame (numpy.ndarray): OpenCV格式的图像帧 (BGR)。
        detector: MTCNN检测器实例。
        model: 表情识别模型实例。
        transform: 图像预处理转换。
        emotion_map: 表情类别映射。
        device: 使用的设备 ('cpu' 或 'cuda')。
    Returns:
        list: 包含检测到的人脸框和预测表情的列表。
    """
    # 将OpenCV的BGR图像转换为RGB (MTCNN期望RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用MTCNN检测人脸
    results = detector.detect_faces(frame_rgb)

    predictions = []  # 保存当前帧的预测结果

    if results:
        for result in results:
            x, y, width, height = result['box']

            # 裁剪人脸区域
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(
                frame_rgb.shape[1], x + width), min(frame_rgb.shape[0], y + height)
            face_image = frame_rgb[y1:y2, x1:x2]

            if face_image.shape[0] == 0 or face_image.shape[1] == 0:
                continue

            face_image_resized = cv2.resize(face_image, (224, 224))

            face_image_pil = Image.fromarray(face_image_resized)
            face_tensor = transform(face_image_pil)

            face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.to(device)

            with torch.no_grad():
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label_index = predicted.item()

            predicted_emotion = emotion_map.get(predicted_label_index, "未知表情")

            predictions.append({
                "box": [int(x), int(y), int(width), int(height)],
                "emotion": predicted_emotion
            })

    return predictions


# 主执行逻辑
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real-time Emotion Detector using MTCNN and ResNet18.')
    parser.add_argument(
        '-v', '--video', type=str, help='Path to a local video file. If not provided, uses webcam.')
    parser.add_argument('-sf', '--skip_frames', type=int, default=0,
                        help='Process only every Nth frame (N > 0). Default is 0 (process all frames).')  # <--- 添加跳帧参数
    args = parser.parse_args()

    cap = None

    if args.video:
        if os.path.isfile(args.video):
            print(f"尝试打开视频文件: {args.video}")
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                print(f"错误: 无法打开视频文件 {args.video}。请检查文件路径和格式。")
                print("将尝试打开默认摄像头。")
                cap = cv2.VideoCapture(0)
        else:
            print(f"错误: 指定的视频文件不存在或不是有效文件: {args.video}")
            print("将尝试打开默认摄像头。")
            cap = cv2.VideoCapture(0)
    else:
        print("未指定视频文件，将尝试打开默认摄像头。")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("最终错误: 无法打开任何视频源 (摄像头或指定文件)。退出。")
        exit()

    print("\n按下 'q' 键退出。")

    # 用于跳帧处理的变量
    frame_count = 0  # <--- 添加帧计数
    last_predictions = []  # <--- 添加保存上一帧预测结果的列表

    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法接收帧 (视频/流结束?)。退出 ...")
            break

        # --- 跳帧处理逻辑 ---
        current_predictions = []
        if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
            # 跳过此帧，使用上一帧的预测结果
            current_predictions = last_predictions
        else:
            # 处理此帧，并更新上一帧的预测结果
            current_predictions = process_frame(
                frame, detector, model, transform, emotion_map, device)
            last_predictions = current_predictions  # 保存当前帧的结果供后续跳过的帧使用
        frame_count += 1  # <--- 更新帧计数
        # --- 跳帧处理逻辑结束 ---

        processed_frame = frame.copy()  # 在拷贝上绘制

        # 在帧上绘制当前帧的预测结果
        if current_predictions:
            for pred in current_predictions:
                x, y, w, h = pred["box"]
                emotion = pred["emotion"]

                cv2.rectangle(processed_frame, (x, y),
                              (x+w, y+h), (0, 255, 0), 2)
                text_position = (x, y - 10)
                if text_position[1] < 10:
                    text_position = (x, y + h + 20)

                # 注意：如果需要绘制中文且没有集成Pillow，这里会是问号
                cv2.putText(processed_frame, emotion, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Real-time Emotion Recognition', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出。")

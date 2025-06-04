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

# 导入模型定义 (需要确保SimpleCNN类可以被访问到)
# 将项目根目录添加到sys.path以便导入 SimpleCNN
# 将当前目录（train_model.py所在目录）添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

# 导入配置文件以获取NUM_CLASSES等参数
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 定义表情类别映射 (需要与训练时的类别索引对应)
# fer2013数据集的表情类别通常是：
emotion_map = config.EMOTION_MAP

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
model = models.resnet18(weights=None)  # <--- 初始化 ResNet18 模型结构 (不加载ImageNet权重)

# 修改最后一层全连接层，使其输出维度等于表情类别数
# ResNet18 的最后一层是 model.fc
num_ftrs = model.fc.in_features  # 获取倒数第二层输出的特征数量
model.fc = nn.Linear(num_ftrs, num_classes)  # <--- 替换最后一层

# 检查是否有GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载最佳模型权重
model_weight_path = 'best_model_state_dict.pth'  # 确保路径正确
if os.path.exists(model_weight_path):
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print(f"成功加载模型权重: {model_weight_path}")
    model.eval()  # 设置模型为评估模式
else:
    print(f"错误: 未找到模型权重文件 {model_weight_path}，请先训练模型。")
    exit()  # 如果没有模型权重，则退出脚本

# 定义预测函数


def predict_single_image(image_path):
    """
    对单张图像进行表情识别预测。
    Args:
        image_path (string): 输入图像的路径。
    Returns:
        string or None: 预测的表情类别字符串，如果没有检测到人脸则返回None。
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像文件: {image_path}")
        return None

    # 将OpenCV的BGR图像转换为RGB (MTCNN期望RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用MTCNN检测人脸
    results = detector.detect_faces(image_rgb)

    # 处理检测结果
    if results:
        # 通常只处理检测到的第一个人脸
        x, y, width, height = results[0]['box']

        # 裁剪人脸区域
        # 注意：MTCNN可能检测到超出图像边界的框，需要进行边界检查
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_rgb.shape[1], x +
                     width), min(image_rgb.shape[0], y + height)
        face_image = image_rgb[y1:y2, x1:x2]

        # --- 关键：将裁剪后的人脸图像调整到模型期望的尺寸 (224x224) ---
        if face_image.shape[0] == 0 or face_image.shape[1] == 0:  # 检查裁剪区域是否有效
            print(f"警告: 裁剪到无效的人脸区域: ({x1},{y1}) to ({x2},{y2})")
            return "人脸区域无效"

        face_image_resized = cv2.resize(
            face_image, (224, 224))  # <--- 修改为 224x224
        # --- 修改结束 ---

        # 将调整大小后的人脸图像转换为PIL Image并应用转换
        # 注意：Image.fromarray 期望 RGB 数组
        face_image_pil = Image.fromarray(face_image_resized)
        face_tensor = transform(face_image_pil)

        # 添加batch维度 (模型期望输入是 [batch_size, channels, height, width])
        face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(device)  # 将tensor发送到设备

        # 进行预测
        with torch.no_grad():  # 预测阶段不需要计算梯度
            outputs = model(face_tensor)
            # outputs 是模型的原始输出 (例如 logits)，需要通过 softmax 得到概率 (可选)
            # probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label_index = predicted.item()

        # 将预测的索引映射到表情类别
        predicted_emotion = emotion_map.get(predicted_label_index, "未知表情")

        # 可以在图像上绘制人脸框和预测结果 (可选)
        # 注意：绘制是在原始图像上进行
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
        # 使用支持中文的字体绘制 (如果您修改了predict_emotion.py来支持中文绘制)
        # 如果您还没有实现中文绘制，这里可能会显示问号
        cv2.putText(image, predicted_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # 显示图像 (如果在一个图形界面环境中运行)
        cv2.imshow("Emotion Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return predicted_emotion  # 返回预测的表情字符串
    else:
        print(f"未在图像 {image_path} 中检测到人脸。")
        # 对于没有检测到人脸的情况，显示原始图片并等待关闭
        cv2.imshow("Emotion Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return "未检测到人脸"


# 主执行逻辑
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='对单张图像进行人脸检测和表情识别。')
    parser.add_argument(
        '-i', '--image_path', type=str, help='需要进行表情识别的图像文件路径')  # <--- 添加图像路径参数

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数获取的图像路径
    image_to_predict_path = args.image_path  # <--- 使用解析到的参数

    # 示例用法：预测一张图片
    # 请替换为你想要预测的图像路径
    # 'test_files\\QQ20250604-113918.jpg'  # <--- 替换为你的测试图像路径
    # test_image_path = os.path.join("test_files", "QQ20250604-113918.jpg")

    # 为了演示，我们可以先创建一个简单的dummy测试图像
    if not os.path.exists(image_to_predict_path):
        print(f"创建 dummy 测试图像: {image_to_predict_path}")
        # 创建一个简单的白色图像作为dummy
        dummy_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # 可以在中心画个矩形模拟人脸
        cv2.rectangle(dummy_image, (50, 50), (150, 150), (0, 0, 0), 2)
        cv2.imwrite(image_to_predict_path, dummy_image)
        print("Dummy 测试图像已创建。")

    predicted_emotion = predict_single_image(image_to_predict_path)

    if predicted_emotion:
        print(f"预测的表情是: {predicted_emotion}")

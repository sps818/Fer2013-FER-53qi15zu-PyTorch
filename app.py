# from train_model import SimpleCNN  # 从train_model.py导入模型定义
import config.config as config
from flask import Flask, request, jsonify
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
import torchvision.models as models  # <--- 添加此行导入models模块

# 导入模型定义和配置文件
# 确保相关目录在Python的搜索路径中
# 假设app.py在项目根目录
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '')))  # 添加项目根目录
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'config')))  # 添加config目录


# 定义表情类别映射 (需要与训练时的类别索引对应)
emotion_map = config.EMOTION_MAP

# 定义图像预处理转换 (与训练时的验证/测试集转换一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])
])

# 初始化MTCNN检测器 (全局初始化，避免每次请求都创建)
detector = MTCNN()
print("MTCNN检测器已初始化。")

# 初始化模型并加载训练好的权重 (全局初始化，避免每次请求都加载)
num_classes = config.NUM_CLASSES

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
model_weight_path = 'best_model_state_dict.pth'  # 确保路径正确，相对于项目根目录
if os.path.exists(model_weight_path):
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print(f"成功加载模型权重: {model_weight_path}")
    model.eval()  # 设置模型为评估模式
else:
    print(f"错误: 未找到模型权重文件 {model_weight_path}，请先训练模型。")
    # 在生产环境中，如果模型文件不存在，应该停止服务或者返回错误
    # 这里为了演示，先简单打印错误
    # exit() # 如果在实际部署中，这里可能需要exit()或者更强的错误处理

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_emotion_api():
    """
    接收 POST 请求，包含图像文件，进行表情识别预测。
    """
    if 'image' not in request.files:
        return jsonify({"error": "请求中未包含图像文件 ('image')."}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "未选择图像文件."}), 400

    if file:
        try:
            # 从文件流中读取图像数据
            image_stream = io.BytesIO(file.read())
            # 使用Pillow从流中打开图像
            pil_image = Image.open(image_stream).convert('RGB')  # 确保是RGB格式

            # 将PIL图像转换为OpenCV格式以便MTCNN处理
            image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 使用MTCNN检测人脸
            results = detector.detect_faces(cv2.cvtColor(
                image_cv2, cv2.COLOR_BGR2RGB))  # MTCNN期望RGB

            predictions = []  # 可能有多个人脸，保存所有预测结果

            if results:
                for result in results:
                    x, y, width, height = result['box']

                    # 裁剪人脸区域
                    # 添加边界检查以处理MTCNN可能检测到超出图像边界的框
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(
                        image_cv2.shape[1], x + width), min(image_cv2.shape[0], y + height)
                    face_image = image_cv2[y1:y2, x1:x2]

                    # --- 关键：将裁剪后的人脸图像调整到模型期望的尺寸 (224x224) ---
                    if face_image.shape[0] == 0 or face_image.shape[1] == 0:  # 检查裁剪区域是否有效
                        print(f"警告: 裁剪到无效的人脸区域: ({x1},{y1}) to ({x2},{y2})")
                        continue  # 跳过无效区域

                    face_image_resized = cv2.resize(
                        face_image, (224, 224))  # <--- 修改为 224x224
                    # --- 修改结束 ---

                    # 将调整大小后的人脸图像转换为PIL Image并应用转换
                    face_image_pil = Image.fromarray(cv2.cvtColor(
                        face_image_resized, cv2.COLOR_BGR2RGB))  # 转换回RGB for transform
                    face_tensor = transform(face_image_pil)

                    # 添加batch维度
                    face_tensor = face_tensor.unsqueeze(0)
                    face_tensor = face_tensor.to(device)  # 将tensor发送到设备

                    # 进行预测
                    with torch.no_grad():
                        outputs = model(face_tensor)
                        # outputs 是模型的原始输出 (例如 logits)，需要通过 softmax 得到概率 (可选)
                        # probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs.data, 1)
                        predicted_label_index = predicted.item()

                    # 将预测的索引映射到表情类别
                    predicted_emotion = emotion_map.get(
                        predicted_label_index, "未知表情")

                    # 记录预测结果和人脸位置
                    predictions.append({
                        # 返回整数坐标
                        "box": [int(x), int(y), int(width), int(height)],
                        "emotion": predicted_emotion
                    })

                return jsonify({"predictions": predictions}), 200

            else:
                return jsonify({"message": "未检测到人脸。", "predictions": []}), 200

        except Exception as e:
            # 打印详细的错误信息到控制台，方便调试
            print(f"处理图像时发生错误: {e}", file=sys.stderr)
            return jsonify({"error": f"处理图像时发生错误: {e}"}), 500

    return jsonify({"error": "未知错误。"}), 500  # 理论上不应该走到这里


if __name__ == '__main__':
    # 在本地开发环境中运行 Flask 应用
    # debug=True 会在代码修改后自动重启，并提供详细错误信息
    app.run(debug=True, host='0.0.0.0', port=5000)  # 监听所有IP，端口5000

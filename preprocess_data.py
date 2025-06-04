import pandas as pd
import numpy as np
import cv2
from mtcnn import MTCNN
import os
from config import config


# 创建输出目录（如果不存在）
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

# 加载CSV文件
print(f"加载数据集：{config.INPUT_PATH}")
data = pd.read_csv(config.INPUT_PATH)

# 初始化MTCNN检测器
detector = MTCNN()
print("MTCNN检测器已初始化。")

# 遍历数据集中的每一行
print("开始处理图像...")
for index, row in data.iterrows():
    # 获取像素数据和标签
    emotion = row['emotion']
    pixels = row['pixels']
    usage = row['Usage']  # 通常有Training, PublicTest, PrivateTest

    # 将像素字符串转换为numpy数组
    # 像素值是以空格分隔的字符串
    pixels = np.array(pixels.split(), dtype='uint8')

    # 将一维像素数组转换为48x48的灰度图像
    image = pixels.reshape(48, 48)

    # 将灰度图像转换为彩色图像（MTCNN需要彩色图像输入）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 使用MTCNN检测人脸
    results = detector.detect_faces(image_rgb)

    # 处理检测结果
    if results:
        # 通常fer2013每张图只有一个人脸
        x, y, width, height = results[0]['box']

        # 裁剪人脸区域
        face_image = image_rgb[y:y+height, x:x+width]

        # 将裁剪后的人脸图像调整到固定大小（例如，48x48）
        face_image_resized = cv2.resize(face_image, (48, 48))

        # 将处理后的图像保存到新的目录
        # 根据Usage和emotion创建子目录
        usage_dir = os.path.join(config.OUTPUT_PATH, usage)
        emotion_dir = os.path.join(usage_dir, str(emotion))

        if not os.path.exists(usage_dir):
            os.makedirs(usage_dir)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

        # 构建输出文件路径
        output_path = os.path.join(emotion_dir, f"image_{index}.png")

        # 保存图像
        cv2.imwrite(output_path, cv2.cvtColor(face_image_resized,
                    cv2.COLOR_RGB2BGR))  # MTCNN输出RGB，cv2保存需要BGR

        if (index + 1) % 1000 == 0:
            print(f"已处理 {index + 1} 张图像...")

print("图像处理完成！")

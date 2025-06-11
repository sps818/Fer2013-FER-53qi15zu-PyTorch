import pandas as pd
import numpy as np
import cv2
from mtcnn import MTCNN
import os
from config import config


# 创建输出目录（如果不存在）
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

# 由于config.INPUT_PATH现在指向fer2013new.csv，我们需要单独加载原始fer2013.csv来获取像素数据
original_fer2013_path = os.path.join(
    os.path.dirname(config.INPUT_PATH), "fer2013.csv")

# 加载FERPlus标签文件 (通过 config.INPUT_PATH 获取)
print(f"加载FERPlus标签文件：{config.INPUT_PATH}")
ferplus_df = pd.read_csv(config.INPUT_PATH)  # fer2013new.csv有表头

# 加载原始fer2013.csv获取像素数据
print(f"加载原始像素数据文件：{original_fer2013_path}")
original_pixels_df = pd.read_csv(original_fer2013_path)

# 初始化MTCNN检测器
detector = MTCNN()
print("MTCNN检测器已初始化。")

# 遍历原始像素数据集中的每一行
print("开始处理图像...")
for index, row in original_pixels_df.iterrows():
    # 获取原始像素数据
    pixels = row['pixels']

    # 从ferplus_df中获取对应索引的Usage和投票数据
    ferplus_row = ferplus_df.iloc[index]

    # 使用列名获取 'Usage' 列的值，更健壮
    usage = ferplus_row['Usage']  # 明确通过列名获取 Usage

    # 获取情绪投票列 (根据fer2013new.csv的实际表头)
    # 假设列名是: 'Usage', 'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'
    # 那么情绪投票列就是 'neutral' 到 'fear'
    emotion_vote_columns = ['neutral', 'happiness',
                            'surprise', 'sadness', 'anger', 'disgust', 'fear']
    emotion_votes = ferplus_row[emotion_vote_columns].values.astype(int)

    # 多数投票法确定最终标签
    if np.sum(emotion_votes) == 0:
        continue

    emotion = np.argmax(emotion_votes)

    # 将像素字符串转换为numpy数组
    pixels = np.array(pixels.split(), dtype='uint8')

    # 将一维像素数组转换为48x48的灰度图像
    image = pixels.reshape(48, 48)

    # 将灰度图像转换为彩色图像（MTCNN需要彩色图像输入）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 使用MTCNN检测人脸
    results = detector.detect_faces(image_rgb)

    # 处理检测结果
    if results:
        x, y, width, height = results[0]['box']
        face_image = image_rgb[y:y+height, x:x+width]
        face_image_resized = cv2.resize(face_image, (48, 48))

        # 将处理后的图像保存到新的目录
        usage_dir = os.path.join(config.OUTPUT_PATH, usage)  # 这里确保使用正确的Usage
        emotion_dir = os.path.join(usage_dir, str(emotion))

        if not os.path.exists(usage_dir):
            os.makedirs(usage_dir)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

        output_path = os.path.join(emotion_dir, f"image_{index}.png")

        cv2.imwrite(output_path, cv2.cvtColor(face_image_resized,
                    cv2.COLOR_RGB2BGR))

        if (index + 1) % 1000 == 0:
            print(f"已处理 {index + 1} 张图像...")

print("图像处理完成！")

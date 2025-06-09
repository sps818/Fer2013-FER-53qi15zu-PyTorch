"""
配置文件，定义数据集路径和模型参数。
"""
import os

# 定义数据集路径
BASE_PATH = os.path.join(os.getcwd(), "datasets", "fer2013")
INPUT_PATH = os.path.join(BASE_PATH, "fer2013.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
TEMP_PATH = os.path.join(os.getcwd(), "temp")

# 定义模型参数
EPOCHS = 51
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 7

EMOTION_MAP = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Surprise',
    6: 'Normal'
}

"""
配置文件，定义数据集路径和模型参数。
"""
import os

# 定义数据集路径
BASE_PATH = os.path.join(os.getcwd(), "datasets", "fer2013")
INPUT_PATH = os.path.join(BASE_PATH, "fer2013new.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
TEMP_PATH = os.path.join(os.getcwd(), "temp")
REPORT_PATH = os.path.join(os.getcwd(), "evaluation_report")

# 定义模型参数
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_CLASSES = 7
PATIENCE = 10

EMOTION_MAP = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear'
}

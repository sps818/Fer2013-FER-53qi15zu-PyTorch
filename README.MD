# 基于PyTorch的FER2013Plus人脸表情识别项目

这是一个利用MTCNN进行人脸检测，并使用PyTorch实现的ResNet18模型进行人脸表情分类的项目。它支持从数据预处理到模型训练、评估、单张图片/实时视频情感检测的完整流程，并集成了一个Streamlit Web前端。

## 功能特性

*   **数据处理**: 使用MTCNN进行人脸裁剪和对齐，支持FERPlus数据集的多数投票标签处理，并按Usage和emotion组织数据。
*   **模型训练**: 基于ResNet18进行迁移学习，支持断点续训、数据增强、权重衰减和Early Stopping。
*   **模型评估**: 生成详细的分类报告和混淆矩阵热力图，全面评估模型性能。
*   **情感识别**: 提供命令行工具进行单张图片或实时视频/摄像头的情感检测。
*   **Web界面**: 集成Streamlit，提供用户友好的Web交互界面进行情感识别。

## 快速开始

### 1. 环境准备

1.  **克隆仓库**:
    ```bash
    git clone <仓库地址>
    cd Fer2013-FER-53qi15zu-PyTorch
    ```

2.  **创建并激活虚拟环境 (推荐)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    `requirements.txt` 包含了项目所需的所有库，例如 `torch`, `torchvision`, `pandas`, `numpy`, `opencv-python`, `mtcnn`, `scikit-learn`, `matplotlib`, `seaborn` 等。

### 2. 数据准备

本项目使用FER2013数据集及其扩展FERPlus的标签。请按以下步骤下载和准备数据：

1.  **下载数据集**:
    *   `fer2013.csv`: 原始FER2013数据集，包含像素数据。
    *   `fer2013new.csv`: FERPlus数据集，包含FER2013图像的投票标签。
    将这两个文件下载到项目根目录下的 `datasets/fer2013/` 文件夹中。
    （如果 `datasets/fer2013/` 不存在，请手动创建）。

2.  **运行数据预处理脚本**:
    ```bash
    python preprocess_data.py
    ```
    此脚本将读取 `fer2013.csv` 和 `fer2013new.csv`，使用MTCNN检测人脸并裁剪，然后将处理后的图像保存到 `datasets/fer2013/output/` 目录下，按照 `Usage` (训练/公共测试/私有测试) 和 `emotion` (表情类别索引) 组织文件夹。

    **重要提示**:
    *   **如果修改了 `config/config.py` 中的 `EMOTION_MAP` 或者 `preprocess_data.py` 中的标签处理逻辑，请务必先删除 `datasets/fer2013/output/` 目录下的所有内容，然后重新运行 `preprocess_data.py` 以确保数据标签的一致性。**
    *   预处理过程可能需要一些时间。

### 3. 模型训练

使用 `train_model.py` 脚本进行模型训练。

```bash
python train_model.py
```

*   **断点续训**: 训练过程中会自动保存最新的检查点 (`latest_checkpoint.pth.tar`)。如果程序中断，重新运行 `train_model.py` 将自动从上次的检查点继续训练。
*   **最佳模型保存**: 训练过程中，如果验证集准确率提升，模型权重 (`best_model_state_dict.pth`) 将被保存。
*   **Early Stopping**: 当验证集准确率连续 `config.PATIENCE` 个 epoch 没有提升时，训练将自动停止。
*   **注意**: 如果修改了模型结构或 `config/config.py` 中的 `EMOTION_MAP`，**请务必删除 `best_model_state_dict.pth` 和 `latest_checkpoint.pth.tar`，并重新运行 `preprocess_data.py` 后再从头开始训练。**

### 4. 模型评估

`train_model.py` 在训练结束后会自动在测试集上进行评估，并生成详细的报告。评估结果将保存到 `evaluation_report/` 目录下。

*   **混淆矩阵**: `confusion_matrix.png`
*   **分类报告**: `classification_report.csv`

### 5. 情感识别 (预测)

项目提供了命令行脚本进行情感识别：

*   **单张图片情感识别**:
    ```bash
    python single_emotion_detector.py -i <图片文件路径>
    # 示例
    # python single_emotion_detector.py -i test_files/example.jpg
    ```

*   **实时情感检测 (视频文件或摄像头)**:
    *   **从视频文件检测**:
        ```bash
        python realtime_emotion_detector.py -sf <跳帧数> -v <视频文件路径>
        # 示例
        # python realtime_emotion_detector.py -sf 6 -v test_files/my_video.mp4
        ```
    *   **从摄像头检测**:
        ```bash
        python realtime_emotion_detector.py
        ```
        （默认使用第一个摄像头，如果只有一个摄像头，通常是 `0`）

### 6. Streamlit Web 前端

启动Web界面：
```bash
streamlit run streamlit_app.py
```
通过浏览器访问，上传图片或视频，实时检测表情。

## 项目结构

```
├── config/                  # 配置文件（如 emotion_map、类别数等）
├── datasets/fer2013/        # 原始数据集及预处理输出
│   ├── fer2013.csv
│   ├── fer2013new.csv       # FERPlus标签文件
│   └── output/Training/0~6/ # 预处理后按类别分文件夹
├── evaluation_report/       # 模型评估报告输出目录
├── temp/                    # 临时文件（视频处理等）
├── test_files/              # 测试图片/视频样例
├── train_model.py           # 训练与评估主脚本
├── preprocess_data.py       # 数据预处理脚本
├── single_emotion_detector.py # 单张图片预测
├── realtime_emotion_detector.py # 视频/摄像头实时检测
├── streamlit_app.py         # Streamlit Web 前端
├── requirements.txt         # 依赖包
└── README.md                # 项目说明文件
```

## 项目配置 (`config/config.py`)

所有核心配置参数都定义在 `config/config.py` 文件中：

*   `BASE_PATH`: 数据集根目录。
*   `INPUT_PATH`: FERPlus标签文件路径 (`fer2013new.csv`)。
*   `OUTPUT_PATH`: 预处理图像的输出目录。
*   `TEMP_PATH`: 临时文件目录。
*   `REPORT_PATH`: 评估报告输出目录。
*   `EPOCHS`: 训练的总轮数。
*   `BATCH_SIZE`: 训练批次大小。
*   `LEARNING_RATE`: 学习率。
*   `NUM_CLASSES`: 表情类别数量 (当前为7)。
*   `PATIENCE`: Early Stopping的耐心值。
*   `EMOTION_MAP`: 表情类别索引与名称的映射关系。**此映射与 `preprocess_data.py` 中的标签处理顺序严格一致，请勿随意修改。**

```python
EMOTION_MAP = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear'
}
```

## 常见问题与注意事项

*   **模型权重未找到/预测结果异常**: 请确保已按"快速开始"中的步骤**完整地**完成了数据预处理和模型训练。每次修改 `config.py` 中的 `EMOTION_MAP` 或模型结构后，都需要重新处理数据并重新训练模型。
*   **灰度图像兼容**: `fer2013.csv` 原始为灰度图，项目已通过 `PIL.Image.convert('RGB')` 和 `cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)` 适配ResNet18的RGB输入要求。
*   **MTCNN 检测性能**: 对于实时视频流，可以通过调整 `realtime_emotion_detector.py` 中的 `-sf` (跳帧数) 参数来优化处理速度。

## 许可证

本项目遵循 MIT License，详见 `LICENSE` 文件。
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from mtcnn import MTCNN
import numpy as np
import os
import sys
import time
import torchvision.models as models
import io
import uuid
import subprocess

# 导入配置文件 (假设config目录在项目根目录)
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'config')))
try:
    import config.config as config
except ImportError:
    st.error("无法导入配置文件 `config/config.py`。请确保文件存在且路径正确。")
    st.stop()

# 定义表情类别映射
try:
    emotion_map = config.EMOTION_MAP
except AttributeError:
    st.error("配置文件 `config/config.py` 中未定义 `EMOTION_MAP`。请在配置文件中添加。")
    st.stop()

# --- 使用Streamlit缓存加载模型和MTCNN ---


@st.cache_resource
def load_detector():
    return MTCNN()


@st.cache_resource
def load_model(num_classes, model_weight_path, device):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(
            model_weight_path, map_location=device))
        model.eval()
        return model
    else:
        st.error(
            f"错误: 未找到模型权重文件 {model_weight_path}。请先运行 train_model.py 进行模型训练。")
        return None


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])
])

detector = load_detector()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = config.NUM_CLASSES
model_weight_path = 'best_model_state_dict.pth'
model = load_model(num_classes, model_weight_path, device)
if model is None:
    st.stop()
model.to(device)

# --- 图片处理函数 ---


def process_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_rgb_np = np.array(pil_image)
    image_bgr_np = cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2BGR)
    processed_image_cv2 = image_bgr_np.copy()
    image_rgb_mtcnn = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb_mtcnn)
    detected_faces_info = []
    if results:
        for result in results:
            x, y, width, height = result['box']
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(
                image_rgb_mtcnn.shape[1], x + width), min(image_rgb_mtcnn.shape[0], y + height)
            face_image = image_rgb_mtcnn[y1:y2, x1:x2]
            if face_image.shape[0] == 0 or face_image.shape[1] == 0:
                detected_faces_info.append({
                    'box': [x, y, width, height],
                    'emotion': '人脸区域无效'
                })
                continue
            try:
                face_image_resized = cv2.resize(face_image, (224, 224))
            except cv2.error as e:
                detected_faces_info.append({
                    'box': [x, y, width, height],
                    'emotion': '调整大小失败'
                })
                continue
            face_image_pil = Image.fromarray(face_image_resized)
            face_tensor = transform(face_image_pil)
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label_index = predicted.item()
            predicted_emotion = emotion_map.get(predicted_label_index, "未知表情")
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
    else:
        return processed_image_cv2, []

# --- 视频处理函数 ---


def process_uploaded_video(uploaded_file):
    import uuid
    import os
    import time
    import subprocess

    if not os.path.exists(config.TEMP_PATH):
        os.makedirs(config.TEMP_PATH)
    temp_input_path = os.path.join(
        config.TEMP_PATH, f"input_{uuid.uuid4().hex}.mp4")
    with open(temp_input_path, 'wb') as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output_path_avi = os.path.join(
        config.TEMP_PATH, f"output_{uuid.uuid4().hex}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_output_path_avi, fourcc, fps, (width, height))

    # 初始化进度条
    progress_bar = st.progress(0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame_cv2 = frame.copy()
        # ====== 检测和绘制 ======
        frame_rgb_mtcnn = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame_rgb_mtcnn)
        if results:
            for result in results:
                x, y, w, h = result['box']
                cv2.rectangle(processed_frame_cv2, (x, y),
                              (x+w, y+h), (0, 255, 0), 2)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(
                    frame_rgb_mtcnn.shape[1], x + w), min(frame_rgb_mtcnn.shape[0], y + h)
                face_image = frame_rgb_mtcnn[y1:y2, x1:x2]
                if face_image.shape[0] > 0 and face_image.shape[1] > 0:
                    try:
                        face_image_resized = cv2.resize(face_image, (224, 224))
                        face_image_pil = Image.fromarray(face_image_resized)
                        face_tensor = transform(face_image_pil)
                        face_tensor = face_tensor.unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(face_tensor)
                            _, predicted = torch.max(outputs.data, 1)
                            predicted_label_index = predicted.item()
                        predicted_emotion = emotion_map.get(
                            predicted_label_index, "未知表情")
                        cv2.putText(processed_frame_cv2, predicted_emotion, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except Exception as e:
                        print(f'处理帧时出错: {e}')
        out.write(processed_frame_cv2)
        frame_idx += 1
        progress = frame_idx / frame_count if frame_count > 0 else 0
        progress_bar.progress(min(progress, 1.0))
    cap.release()
    out.release()
    os.remove(temp_input_path)
    time.sleep(1)

    # ffmpeg转码为mp4(h264)
    temp_output_path_mp4 = temp_output_path_avi.replace('.avi', '.mp4')
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', temp_output_path_avi,
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', temp_output_path_mp4
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    progress_bar.progress(1.0)  # 处理完成
    st.write("视频处理完成。")
    st.video(temp_output_path_mp4)
    st.success("以上是处理后的视频。临时文件会在会话结束后自动清理。")


# --- Streamlit UI ---
st.title("人脸表情识别应用")
st.write("上传一张图片或一个视频文件，进行人脸检测和表情识别。")
file_type = st.radio("选择要上传的文件类型:", ("图片", "视频"))
uploaded_file = st.file_uploader(f"上传{file_type}文件", type=[
                                 "jpg", "jpeg", "png"] if file_type == "图片" else ["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    if file_type == "图片":
        st.write("正在处理图片...")
        processed_image_cv2, detected_faces = process_uploaded_image(
            uploaded_file)
        if processed_image_cv2 is not None:
            processed_image_rgb = cv2.cvtColor(
                processed_image_cv2, cv2.COLOR_BGR2RGB)
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="上传的图片", width=300)
            with col2:
                st.image(processed_image_rgb, caption="检测结果", width=300)
            if detected_faces:
                st.write("检测到的人脸及表情：")
                for face_info in detected_faces:
                    st.write(
                        f"  人脸框: {face_info['box']}, 表情: {face_info['emotion']}")
            else:
                st.write("未检测到人脸。")
    elif file_type == "视频":
        st.write("正在处理视频...")
        process_uploaded_video(uploaded_file)

st.markdown("---")
st.markdown("表情识别应用 Powered by PyTorch, MTCNN, ResNet18 and Streamlit")
st.markdown("模型权重文件: `best_model_state_dict.pth`")
st.markdown("请确保已安装所有依赖库: `pip install -r requirements.txt`")
st.markdown(
    "运行方式: 在项目根目录打开终端，运行 `streamlit run streamlit_app.py --server.fileWatcherType none`")

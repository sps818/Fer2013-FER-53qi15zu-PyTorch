import requests
import os

# 替换为您的 Flask 应用地址和端口
api_url = "http://127.0.0.1:5000/predict"
# 替换为您的测试图片路径
image_path = os.path.join("test_files", "QQ20250604-113918.jpg")  # 或者其他图片的路径

try:
    with open(image_path, 'rb') as f:
        # 'image/jpeg' 或 'image/png' 根据图片类型调整
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(api_url, files=files)

    response.raise_for_status()  # 如果请求失败（非2xx状态码），抛出异常

    result = response.json()
    print("API 响应:", result)

except requests.exceptions.RequestException as e:
    print(f"请求 API 时发生错误: {e}")
except Exception as e:
    print(f"处理响应时发生错误: {e}")

# 安装
1. python版本是3.11
2. 执行：`pip install -r requirements.txt`

# 测试文件
有个叫`test_files`的文件夹，可以往里防图片或者视频。

# 运行
1. 你需要自己下载fer2013数据集，放到datasets/fer2013/文件夹下。 例：./datasets/fer2013/fer2013.csv
2. 要是测单张图片就可以再根目录执行`python single_emotion_detector.py -i  test_files/1.jpg`
3. 要是测视频，设置跳6个帧，就可以再根目录执行`python realtime_emotion_detector.py -sf 6 -v test_files/1.mp4`
   这里我没提供视频，可以自己弄一个

# 自己训练模型
`config.py`文件有写参数可以设置，然后`preprocess_data.py`是做数据预处理用的。

在根目录可以执行`python train_model.py`就开始训练了。
需要说明的是我这里用了 **ResNet18** 的与训练模型，可以根据自己的情况更换模型。

# 其他
`app.py`，`test_api.py`  这两个其实不看也行，就是做成了单张图片检测的API，可以调用接口进行测试。
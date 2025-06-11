import config.config as config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 导入配置文件
# 确保config目录在Python的搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 定义表情类别映射 (需要与训练时的类别索引对应)
# fer2013数据集的表情类别通常是：
emotion_map = config.EMOTION_MAP


# 1. 自定义数据集类
class Fer2013ProcessedDataset(Dataset):
    def __init__(self, data_dir, usage, transform=None):
        """
        Args:
            data_dir (string): 预处理后数据集的根目录 (例如: datasets/fer2013/output/).
            usage (string): 数据集的使用类别 ('Training', 'PublicTest', 'PrivateTest').
            transform (callable, optional): 应用于图像的转换.
        """
        # 根据新的文件结构拼接路径
        self.data_dir = os.path.join(data_dir, usage)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # 使用配置文件中的类别数来构建类别到索引的映射
        # 这里假设emotion文件夹的名称是0-based index的字符串
        self.class_to_idx = {str(i): i for i in range(config.NUM_CLASSES)}

        # 遍历目录加载图像路径和标签
        for emotion_label in os.listdir(self.data_dir):
            emotion_dir = os.path.join(self.data_dir, emotion_label)
            if os.path.isdir(emotion_dir):
                # 确保标签是有效的表情类别
                if emotion_label in self.class_to_idx:
                    label_idx = self.class_to_idx[emotion_label]
                    for img_name in os.listdir(emotion_dir):
                        img_path = os.path.join(emotion_dir, img_name)
                        if os.path.isfile(img_path):
                            self.image_paths.append(img_path)
                            self.labels.append(label_idx)
                else:
                    print(
                        f"Warning: Skipping unknown emotion directory: {emotion_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像，并确保是RGB格式以便后续转换
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# 2. 模型定义 (一个简单的CNN)


# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):  # 接收num_classes参数
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         # 计算全连接层的输入尺寸
#         # 对于48x48的输入图像，经过3次kernel_size=2, stride=2的MaxPool，尺寸变为 48/2/2/2 = 6
#         # 所以展平后尺寸为 128 * 6 * 6
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 使用AdaptiveAvgPool2d更灵活
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(128 * 6 * 6, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(256, num_classes),  # 使用传入的num_classes
#         )

#     def forward(self, x):
#         x = self.features(x)
#         # 如果您想使用AdaptiveAvgPool2d而不是展平，请取消下一行的注释并注释掉 x = x.view(x.size(0), -1)
#         # x = self.avgpool(x)
#         x = x.view(x.size(0), -1)  # 展平
#         x = self.classifier(x)
#         return x

# 3. 训练和验证函数 (支持断点续训)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, start_epoch, device, best_acc, patience=config.PATIENCE):
    model.to(device)

    # 定义checkpoint和最佳模型保存路径
    latest_checkpoint_path = 'latest_checkpoint.pth.tar'
    best_model_save_path = 'best_model_state_dict.pth'
    best_val_acc = best_acc
    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs):
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):  # 验证阶段不需要计算梯度
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 保存最新的 checkpoint
        torch.save({
            'epoch': epoch + 1,  # 保存下一轮的起始epoch
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_val_acc,  # 保存当前的best_acc
            # 保存当前学习率 (如果使用了scheduler，这里需要调整)
            'learning_rate': optimizer.param_groups[0]['lr']
        }, latest_checkpoint_path)
        print(f"最新的 checkpoint 已保存到 {latest_checkpoint_path}")

        # Early Stopping 判断
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_save_path)
            print(f"保存了表现更好的模型权重到 {best_model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"验证集准确率未提升，已连续 {epochs_no_improve} 轮未提升")
            if epochs_no_improve >= patience:
                print(f"验证集准确率连续 {patience} 轮未提升，提前停止训练 (Early Stopping)！")
                break

    print(f'训练完成. Best Val Acc: {best_val_acc:.4f}')
    return model


# 4. 主执行逻辑
if __name__ == '__main__':
    # 从配置文件中读取参数
    data_dir = config.OUTPUT_PATH
    batch_size = config.BATCH_SIZE
    num_epochs = config.EPOCHS
    learning_rate = config.LEARNING_RATE
    num_classes = config.NUM_CLASSES

    # 检查是否有GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义图像转换
    # 对训练集进行数据增强和标准化
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # 对验证集和测试集只进行标准化
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet输入尺寸为224*224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("加载数据集...")
    # 确保usage参数正确，根据fer2013数据集的划分方式
    train_dataset = Fer2013ProcessedDataset(
        data_dir=data_dir, usage='Training', transform=train_transforms)
    # PublicTest通常用于验证，PrivateTest用于最终测试
    val_dataset = Fer2013ProcessedDataset(
        data_dir=data_dir, usage='PublicTest', transform=val_test_transforms)
    test_dataset = Fer2013ProcessedDataset(
        data_dir=data_dir, usage='PrivateTest', transform=val_test_transforms)

    # 使用DataLoader加载数据
    # num_workers > 0 可以加速数据加载，但需要注意在Windows上可能需要将相关代码放在 if __name__ == '__main__': 块中
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("数据集加载完成。")

    # 初始化模型、损失函数和优化器
    print("初始化 ResNet18 模型并加载预训练权重...")

    # 加载预训练的ResNet18模型
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 获取全连接层的输入特征数量
    num_ftrs = model.fc.in_features

    # 替换预训练模型的全连接层为新的线性层，输出维度为num_classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 定义损失函数 (交叉熵)
    criterion = nn.CrossEntropyLoss()

    # 定义优化器 (Adam，增加权重衰减防止过拟合)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 加载 checkpoint (如果存在)
    start_epoch = 0
    best_acc = 0.0
    latest_checkpoint_path = 'latest_checkpoint.pth.tar'

    if os.path.exists(latest_checkpoint_path):
        print(f"找到 checkpoint 文件: {latest_checkpoint_path}, 正在加载...")
        # map_location='cpu' 可以确保在GPU训练的模型也能在只有CPU的机器上加载
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # 将optimizer state中的tensor显式转移到当前设备
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Checkpoint 加载成功，从 Epoch {start_epoch} 继续训练。")
        print(f"加载时的最佳验证集准确率: {best_acc:.4f}")
    else:
        print("未找到 checkpoint 文件，从头开始训练。")

    # 将模型发送到指定设备
    model.to(device)

    # 开始训练模型
    print("开始训练模型...")
    # patience参数用于Early Stopping
    model = train_model(model, train_loader, val_loader, criterion,
                        optimizer, num_epochs=num_epochs, start_epoch=start_epoch, device=device, best_acc=best_acc, patience=config.PATIENCE)

    # 测试集评估
    print("\n在测试集上评估模型...")
    best_model_save_path = 'best_model_state_dict.pth'

    # 确保最佳模型权重文件存在
    if os.path.exists(best_model_save_path):
        print(f"加载最佳模型权重文件: {best_model_save_path} 进行测试评估...")

        # <--- 初始化 ResNet18 模型结构 (不加载ImageNet权重)
        test_model = models.resnet18(weights=None)
        num_ftrs = test_model.fc.in_features
        test_model.fc = nn.Linear(num_ftrs, num_classes)

        # 加载模型权重
        test_model.load_state_dict(torch.load(
            best_model_save_path, map_location=device))
        test_model.to(device)

        # 设置为评估模式
        test_model.eval()

        all_labels = []
        all_predictions = []
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = test_model(inputs)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                running_corrects += torch.sum(preds == labels.data)

        test_acc = running_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc:.4f}')

        print("\n--- 详细评估报告 ---")
        cm = confusion_matrix(all_labels, all_predictions)
        print("\n混淆矩阵:")
        print(cm)

        # 使用emotion_map来生成目标名称
        target_names = [emotion_map[i] for i in range(num_classes)]
        report = classification_report(all_labels, all_predictions,
                                       target_names=target_names)
        print("\n分类报告:")
        print(report)

        # 保存评估结果
        report_dir = config.REPORT_PATH
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        # 保存混淆矩阵热力图
        cm_png_path = os.path.join(report_dir, 'confusion_matrix.png')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(cm_png_path)
        plt.close()
        print(f"混淆矩阵热力图已保存为 {cm_png_path}")

        # 保存分类报告为CSV
        report_dict = classification_report(
            all_labels, all_predictions, target_names=target_names, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        # 格式化浮点数为两位小数
        df_report = df_report.applymap(
            lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        csv_path = os.path.join(report_dir, 'classification_report.csv')
        df_report.to_csv(csv_path, encoding='utf-8-sig')
        print(f"分类报告已保存为 {csv_path}")
    else:
        print(f"未找到最佳模型权重文件: {best_model_save_path}，无法进行测试评估。请先成功训练至少一个epoch。")

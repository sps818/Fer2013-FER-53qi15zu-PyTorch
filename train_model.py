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
# 尝试将普通的CNN换成Resnet18
import torchvision.models as models
from torchvision.models import ResNet18_Weights

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


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):  # 接收num_classes参数
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 计算全连接层的输入尺寸
        # 对于48x48的输入图像，经过3次kernel_size=2, stride=2的MaxPool，尺寸变为 48/2/2/2 = 6
        # 所以展平后尺寸为 128 * 6 * 6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 使用AdaptiveAvgPool2d更灵活
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),  # 使用传入的num_classes
        )

    def forward(self, x):
        x = self.features(x)
        # 如果您想使用AdaptiveAvgPool2d而不是展平，请取消下一行的注释并注释掉 x = x.view(x.size(0), -1)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# 3. 训练和验证函数 (支持断点续训)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, start_epoch, device, best_acc):
    model.to(device)

    # 定义checkpoint和最佳模型保存路径
    latest_checkpoint_path = 'latest_checkpoint.pth.tar'
    best_model_save_path = 'best_model_state_dict.pth'
    # latest_checkpoint_path = 'latest_checkpoint.onnx'
    # best_model_save_path = 'best_model.onnx'

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
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

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 保存最新的 checkpoint
        torch.save({
            'epoch': epoch + 1,  # 保存下一轮的起始epoch
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,  # 保存当前的best_acc
            # 保存当前学习率 (如果使用了scheduler，这里需要调整)
            'learning_rate': optimizer.param_groups[0]['lr']
        }, latest_checkpoint_path)
        print(f"最新的 checkpoint 已保存到 {latest_checkpoint_path}")

        # 保存表现最好的模型权重 (单独保存，方便后续推理加载)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_save_path)
            print(f"保存了表现更好的模型权重到 {best_model_save_path}")

    print(f'训练完成. Best Val Acc: {best_acc:.4f}')
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
        transforms.Resize((224, 224)),  # ResNet输入尺寸为224*224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),  # 随机颜色抖动
        transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为Tensor，并缩放到[0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # ImageNet的均值和标准差，适用于RGB图像
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)  # num_workers可以根据你的CPU核数调整
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("数据集加载完成。")

    # 初始化模型、损失函数和优化器
    # model = SimpleCNN(num_classes=num_classes)  # 使用配置文件中的num_classes

    # 初始化模型、损失函数和优化器
    print("初始化 ResNet18 模型并加载预训练权重...")
    # 加载预训练的 ResNet18 模型
    model = models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1)  # <--- 加载预训练权重
    # 修改最后一层全连接层，使其输出维度等于表情类别数
    # ResNet18 的最后一层是 model.fc
    num_ftrs = model.fc.in_features  # 获取倒数第二层输出的特征数量
    model.fc = nn.Linear(num_ftrs, num_classes)  # <--- 替换最后一层

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用配置文件中的学习率

    # --- 添加加载 checkpoint 的逻辑 ---
    start_epoch = 0
    best_acc = 0.0
    latest_checkpoint_path = 'latest_checkpoint.pth.tar'

    if os.path.exists(latest_checkpoint_path):
        print(f"找到 checkpoint 文件: {latest_checkpoint_path}, 正在加载...")
        # 使用map_location确保在不同设备上都能正确加载
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # --- 关键修改：将优化器状态发送到目标设备 ---
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        # --- 修改结束 ---

        best_acc = checkpoint.get('best_acc', 0.0)  # 兼容旧的没有best_acc的checkpoint

        print(f"Checkpoint 加载成功，从 Epoch {start_epoch} 继续训练。")
        print(f"加载时的最佳验证集准确率: {best_acc:.4f}")
    else:
        print("未找到 checkpoint 文件，从头开始训练。")
    # --- 加载逻辑结束 ---

    # 将模型发送到设备 (这里需要放在加载checkpoint之后)
    model.to(device)  # <--- 放在加载checkpoint之后

    # 训练模型
    print("开始训练模型...")
    # 调用 train_model 函数，传入start_epoch和best_acc
    model = train_model(model, train_loader, val_loader, criterion,
                        optimizer, num_epochs=num_epochs, start_epoch=start_epoch, device=device, best_acc=best_acc)

    # 在测试集上评估最终模型
    print("\n在测试集上评估模型...")
    best_model_save_path = 'best_model_state_dict.pth'  # 加载训练期间保存的最佳模型权重
    if os.path.exists(best_model_save_path):
        print(f"加载最佳模型权重文件: {best_model_save_path} 进行测试评估...")
        # 需要重新初始化一个模型实例，并加载最佳权重
        # 如果train_model返回了最佳模型，也可以直接使用返回的模型
        # 为了清晰，这里重新初始化并加载

        # test_model = SimpleCNN(num_classes=num_classes)
        # <--- 初始化 ResNet18 模型结构 (不加载ImageNet权重)
        test_model = models.resnet18(weights=None)
        num_ftrs = test_model.fc.in_features
        test_model.fc = nn.Linear(num_ftrs, num_classes)  # <--- 修改最后一层

        test_model.load_state_dict(torch.load(
            best_model_save_path, map_location=device))
        test_model.to(device)
        test_model.eval()

        # --- 添加收集真实标签和预测结果的代码 ---
        all_labels = []
        all_predictions = []
        # --- 添加结束 ---

        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = test_model(inputs)
                _, preds = torch.max(outputs, 1)

                # --- 收集标签和预测结果 ---
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                # --- 收集结束 ---

                running_corrects += torch.sum(preds == labels.data)

        test_acc = running_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc:.4f}')

        # --- 计算和打印详细评估指标 ---
        print("\n--- 详细评估报告 ---")

        # 计算混淆矩阵
        # confusion_matrix 函数的参数顺序是 (y_true, y_pred)
        cm = confusion_matrix(all_labels, all_predictions)
        print("\n混淆矩阵:")
        print(cm)

        # 计算精确率、召回率、F1分数等
        # target_names 是表情类别的名称列表，与emotion_map的顺序对应
        target_names = [emotion_map[i] for i in range(num_classes)]
        # classification_report 函数提供了一个方便的总结报告
        report = classification_report(
            all_labels, all_predictions, target_names=target_names)
        print("\n分类报告:")
        print(report)
        # --- 详细评估指标计算结束 ---

    else:
        print(f"未找到最佳模型权重文件: {best_model_save_path}，无法进行测试评估。请先成功训练至少一个epoch。")

import torch
import torch.nn as nn
import torchvision.models as models
from treelib import Tree
import joblib
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
class CustomResNet(nn.Module):
    def __init__(self, num_classes,batch_size,dimension_number):
        super(CustomResNet, self).__init__()
        # 加载预训练的 ResNet50
        self.resnet50 = models.resnet50(pretrained=True)
        
        # 修改第一个卷积层以接受 1D 输入
        self.resnet50.conv1 = nn.Conv2d(1, batch_size, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # 添加自定义的全连接层
        self.fc = nn.Linear(2048, num_classes)  # 假设 ResNet50 的特征大小是 2048
        self.fc2 = nn.Linear(2048, dimension_number)  # 将输出样本控制在

    def forward(self, x):
        # 将 1D 输入调整为 2D 输入
        x = x.unsqueeze(1)  # 假设 x 的维度是 [batch_size, 400]
        # print(x.shape)
        #x = x.unsqueeze(-1)  # 增加一个维度，以匹配 2D 卷积的要求，现在 x 的维度是 [batch_size, 1, 400, 1]

        # 通过修改后的 ResNet50 提取特征
        x = self.features(x)
        
        # 展平特征以用于全连接层
        x = torch.flatten(x, 1)
        
        # 通过全连接层获得最终输出
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x):
        # 将 1D 输入调整为 2D 输入
        x = x.unsqueeze(1)  # 假设 x 的维度是 [batch_size, 400]
        #x = x.unsqueeze(-1)  # 增加一个维度，以匹配 2D 卷积的要求，现在 x 的维度是 [batch_size, 1, 400, 1]

        # 通过修改后的 ResNet50 提取特征
        x = self.features(x)
        
        # 展平特征以用于全连接层
        x = torch.flatten(x, 1)
        x =self.fc2(x)
        return x
    
if __name__ == '__main__':
    batch_size = 64
    dimension_number = 400
    label_encoder = LabelEncoder()
    # 初始化 ResNet50 模型
    resnet50 = models.resnet50(pretrained=True)
    # 将模型修改为特征提取模式
    resnet50.fc = torch.nn.Identity()
    # 设备配置：使用GPU如果可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)
    # 导入原始特征
    origin_train_tree = joblib.load('D:/lgp/lpy/open-set-loss/data/3XX_data/deep_features_tree.pkl')
    all_data = pd.read_csv('D:/lgp/lpy/open-set-loss/data/3XX_data/all_data.csv') # 假设标签列名为'label'
    all_labels =pd.read_csv('D:/lgp/lpy/open-set-loss/data/3XX_data/all_labels.csv')
    all_data_labels =pd.concat([all_data, all_labels], axis=1)
    num_type_mid = all_data_labels.drop_duplicates(subset=['label'])
    for name in num_type_mid.label:
        data_ss_temp = all_data_labels[all_data_labels.label == name]
        origin_train_tree.update_node(name, data=data_ss_temp)
    all_data = all_data#.iloc[[i % 100 <= 1 for i in range(all_data.shape[0])], :]
    all_labels = all_labels#.iloc[[i % 100 <= 1 for i in range(all_labels.shape[0])], :]
    integer_encoded = label_encoder.fit_transform(all_labels.values)

    # 将所有数据和标签转换为 Tensor
    all_data_tensor = torch.tensor(all_data.values, dtype=torch.float).unsqueeze(1)  # 增加一个维度作为通道维
    # 假设你有以下的float字符标签数组

    all_labels_tensor = torch.tensor(integer_encoded, dtype=torch.long)

    # 使用 DataLoader
    dataset = TensorDataset(all_data_tensor, all_labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
    
    # 创建模型实例
    num_classes = len(origin_train_tree.leaves()) 
    model = CustomResNet(num_classes,batch_size,dimension_number)
    # 定义损失函数，CrossEntropyLoss已经包括了softmax操作
    criterion = nn.CrossEntropyLoss()
    # 定义SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 设置训练轮数
    num_epoch =100
    # 初始化最高识别率
    best_accuracy = 0
    # 训练过程
    for epoch in range(num_epoch):
        total_loss = 0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(data_loader):
            # 前向传播
            outputs = model(inputs)
        
            # 计算损失
            loss = criterion(outputs, labels)
        
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 计算正确预测的数量
        avg_loss = total_loss / len(data_loader)# 计算每个epoch的平均损失和识别率
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), "model_best.pth")
        print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    deep_features_tree = Tree(origin_train_tree.subtree(origin_train_tree.root), deep=True)
    # 遍历 train_tree 的所有叶子节点并提取深度特征
    for leaf in origin_train_tree.leaves():
        print(leaf.tag)
    # 获取叶子节点的训练数据
        train_data = leaf.data  # 假设这是一个Tensor或者可以转换为Tensor的数据结构
        train_data = train_data.iloc[:, :-1]
        train_data = torch.tensor(train_data.values, dtype=torch.float).unsqueeze(1)
    # 使用模型提取深度特征
        with torch.no_grad():
            deep_features = model.extract_features(train_data)  # 增加batch维度
    # 将提取的深度特征存储到 deep_features_tree 中对应节点的数据里
            deep_features_tree.update_node(leaf.identifier, data=deep_features)
    joblib.dump(deep_features_tree,'deep_features_tree_0321_number_epoch1.pkl')
    print('end')
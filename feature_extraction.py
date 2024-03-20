import torch
import torch.nn as nn
import torchvision.models as models
from treelib import Tree
import joblib
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
        
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # 加载预训练的 ResNet50
        self.resnet50 = models.resnet50(pretrained=True)
        
        # 修改第一个卷积层以接受 1D 输入
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # 添加自定义的全连接层
        self.fc = nn.Linear(2048, num_classes)  # 假设 ResNet50 的特征大小是 2048

    def forward(self, x):
        # 将 1D 输入调整为 2D 输入
        x = x.unsqueeze(1)  # 假设 x 的维度是 [batch_size, 400]
        x = x.unsqueeze(-1)  # 增加一个维度，以匹配 2D 卷积的要求，现在 x 的维度是 [batch_size, 1, 400, 1]

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
        x = x.unsqueeze(-1)  # 增加一个维度，以匹配 2D 卷积的要求，现在 x 的维度是 [batch_size, 1, 400, 1]

        # 通过修改后的 ResNet50 提取特征
        x = self.features(x)
        
        # 展平特征以用于全连接层
        x = torch.flatten(x, 1)  
        return x
    
if __name__ == '__main__':
    # 初始化 ResNet50 模型
    resnet50 = models.resnet50(pretrained=True)
    # 将模型修改为特征提取模式
    resnet50.fc = torch.nn.Identity()
    # 设备配置：使用GPU如果可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)
    # 导入原始特征
    origin_train_tree = joblib.load('data\3XX_data\train_tree.pkl')
    for leaf_id in origin_train_tree.leaves():
        leaf = origin_train_tree.get_node(leaf_id.identifier)
    # 合并所有叶子节点的数据
    all_data = []
    all_labels = []
    df = leaf.data  # 假设每个叶子节点的数据是一个DataFrame
    all_data.append(df.drop('label', axis=1))  # 假设标签列名为'label'
    all_labels.append(df['label'])

    # 将所有数据和标签转换为 Tensor
    all_data_tensor = torch.tensor(pd.concat(all_data).values, dtype=torch.float).unsqueeze(1)  # 增加一个维度作为通道维
    all_labels_tensor = torch.tensor(pd.concat(all_labels).values, dtype=torch.long)

    # 使用 DataLoader
    dataset = TensorDataset(all_data_tensor, all_labels_tensor)
    data_loader = DataLoader(dataset, batch_size=32)
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
    
    # 创建模型实例
    num_classes = len(origin_train_tree.leaves()) 
    model = CustomResNet(num_classes)
    # 定义损失函数，CrossEntropyLoss已经包括了softmax操作
    criterion = nn.CrossEntropyLoss()
    # 定义SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 设置训练轮数
    num_epoch =10
    # 训练过程
    for epoch in range(num_epoch):
        for i, (inputs, labels) in enumerate(DataLoader):
            # 前向传播
            outputs = model(inputs)
        
            # 计算损失
            loss = criterion(outputs, labels)
        
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}], Loss: {loss.item()}')
    deep_features_tree = Tree(origin_train_tree.subtree(origin_train_tree.root), deep=True)
    # 遍历 train_tree 的所有叶子节点并提取深度特征
    for leaf in origin_train_tree.leaves():
        print(leaf.tag)
    # 获取叶子节点的训练数据
        train_data = leaf.data  # 假设这是一个Tensor或者可以转换为Tensor的数据结构
        train_data = torch.tensor(train_data, dtype=torch.float32)
    # 使用模型提取深度特征
        with torch.no_grad():
            deep_features = model.extract_features(train_data.unsqueeze(0))  # 增加batch维度
    # 将提取的深度特征存储到 deep_features_tree 中对应节点的数据里
            deep_features_tree.get_node(leaf.identifier).data = deep_features
    print('end')
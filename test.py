from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch
import numpy as np
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# 定义模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 100)  # 最终为什么是 10，因为手写数字识别最终是 10分类的，分类任务中有多少，就分几类。 0-9
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, image):
        image_viwed = image.view(-1, 1*28*28)  # 此处需要拍平
        out_1 = self.fc1(image_viwed)
        fc1 = self.relu(out_1)
        out_2 = self.fc2(fc1)
        return out_2
    
# 打开模型文件
def openModelFiles(model):
        
    if os.path.exists("./models/model.pkl"):
        model.load_state_dict(torch.load("./models/model.pkl"))


# 数据集加载 
def loadData():    
    my_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
        ]
    )
    mnist_train = MNIST(root="../MNIST_data", train=False, download=True, transform=my_transforms)
    dataloader = DataLoader(mnist_train, batch_size=8, shuffle=True)
    dataloader = tqdm(dataloader, total=len(dataloader))
    return dataloader

# 运行模型
def runModel(model,dataloader):
    succeed = []
    total_loss = []
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    # 不需要反向传播和计算梯度
    with torch.no_grad():
        for images, labels in dataloader:
            # 获取结果
            output = model(images)
            result = output.max(dim=1).indices
            # img = make_grid(images).permute(1,2,0).numpy()    
            # plt.imshow(img)
            # plt.show()
            print(labels)
            print(result)
            succeed.append(result.eq(labels).float().mean().item())
            # 通过结果计算损失
            loss = loss_function(output, labels)
            total_loss.append(loss.item())
    print(np.mean(total_loss))
    return np.mean(succeed)

# 开始测试
def main():
    # 实例化模型
    model = MnistModel()
    openModelFiles(model)
    # 加载数据
    dataloader = loadData()
    runModel(model,dataloader)

if __name__=="__main__":
    main()








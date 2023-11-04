from torchvision import transforms
from torch import nn
import torch
from PIL import Image

# 模型定义
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

# 初始化模型
model = MnistModel()
model.load_state_dict(torch.load("./models/model.pkl"))


# 加载自己的图片
image = Image.open('test.jpg')
my_transforms = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ]
)
image = my_transforms(image)
# 评估模式
model.eval()
with torch.no_grad():
    output = model(image)
    result = output.max(dim=1).indices
    print(result)


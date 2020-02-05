# 機械学習- 合宿2020

## MNIST

### 1. データ読み込み

```python
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=100,
                                            shuffle=False, 
                                            num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
```

### データの可視化
```python
# データの可視化
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```
### 2. モデル定義

```python
import torch.nn as nn
import torch.nn.functional as F

# モデル定義
class Net(nn.Module):
  def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.dropout1(x)
      x = x.view(-1, 12 * 12 * 64)
      x = F.relu(self.fc1(x))
      x = self.dropout2(x)
      x = self.fc2(x)
      return x

# 特徴抽出用のモデル

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Dropout()
    )
    self.classifier = nn.Sequential(
        nn.Linear(12*12*64, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(128, 10)       
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(-1, 12*12*64)
    x = self.classifier(x)
    return x
```

#### GPUの設定

```python
GPU = True
device = torch.device("cuda" if GPU else "cpu")
```

#### モデルのインスタンス化

```python
model = Model()
model = model.to(device)	
```
#### パラメータの設定
```python
import torch.optim as optim
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```




### 3. 学習

```python
# 学習を行う
model.train()
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = torch.tensor(inputs, dtype=torch.float).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

### モデルの保存

```python
# モデルの保存
import cloudpickle
with open('model.pkl', 'wb') as f:
    cloudpickle.dump(model, f)
```



### 学習したモデルで特徴抽出

```python
### 学習したモデルで特徴抽出を行う．
# 保存したモデルをload
with open('model.pkl', 'rb') as f:
    model = cloudpickle.load(f)
for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = torch.tensor(inputs, dtype=torch.float).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    feature_vector = model.features(inputs)
    print(feature_vector.shape)
    feature_vector = feature_vector.view(-1, 12*12*64)
    print(feature_vector.shape)
    feature_vector = model.classifier(feature_vector)
    print(feature_vector.shape)

    break
```

### 抽出した特徴をUMAPで可視化

```python
import umap
from mpl_toolkits import mplot3d
%matplotlib inline
import matplotlib.pyplot as plt
reducer = umap.UMAP(n_components=3)
embedding = reducer.fit_transform(feature_vector.detach().cpu().numpy())
```

```python
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels.cpu())
```


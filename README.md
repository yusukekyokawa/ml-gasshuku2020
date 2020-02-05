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
\```

### データの可視化
​```python
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

```python

#### モデルのインスタンス化

​```python
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

​```python
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

### 4. CNNの判断根拠の可視化
GradCAMを使用してCNNの予測の判断根拠の可視化を行います．

```python
import cv2
class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad
    
    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        feature_maps = []
        
        for i in range(x.size(0)):
            img = x[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)
    
            feature = x[i].unsqueeze(0)
            
            for name, module in self.model.named_children():
                if name == 'classifier':
                    feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
                    
            classes = F.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()
    
            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
                
            feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = feature_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
                
            feature_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
            
        feature_maps = torch.stack(feature_maps)
        
        return feature_maps
```

#### 適用する画像を表示

```python
%matplotlib inline
import matplotlib.pyplot as plt
def imshow_one(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)).squeeze())
```

#### 判断根拠の可視化

```python
grad_cam = GradCam(model)
# 次元数を増やした後にtorch型に変換．その後，GPUで作動するようにする．
test_image_tensor = torch.tensor(np.expand_dims(images[0], 0))
feature_image = grad_cam(test_image_tensor.cuda()).squeeze(dim=0)
feature_image = transforms.ToPILImage()(feature_image)
pred_idx = model(test_image_tensor.cuda()).max(1)[1]
plt.imshow(feature_image.resize((28, 28)))
```

### 5. モデル構造の可視化

​```python
!pip3 install torchsummary
# モデル構造の可視化
from torchsummary import summary
net = Net().to(device)
summary(net, input_size=(1, 28, 28))		
```

### 6. 畳み込みフィルタの可視化

```python
def plot_filters_single_channel_big(t):
    
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]
    
    
    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    
    npimg = npimg.T
    
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)
```

```python
def plot_filters_single_channel(t):
    
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()
```

```python
# 畳み込みフィルタの可視化
def plot_weights(model, layer_num, single_channel = True, collated = False):
  
  #extracting the model features at the particular layer number
  layer = model.features[layer_num]
  
  #checking whether the layer is convolution layer or not 
  if isinstance(layer, nn.Conv2d):
    #getting the weight tensor data
    weight_tensor = model.features[layer_num].weight.data.cpu()
    
    if single_channel:
      if collated:
        plot_filters_single_channel_big(weight_tensor)
      else:
        plot_filters_single_channel(weight_tensor)
        
    else:
      if weight_tensor.shape[1] == 3:
        plot_filters_multi_channel(weight_tensor)
      else:
        print("Can only plot weights with three channels with single channel = False")
        
  else:
    print("Can only visualize layers which are convolutional")
        
#visualize weights for alexnet - first conv layer
plot_weights(model, 0, single_channel = True)
```



## 問題1 Fashion MNIST

上記のMNISTと同様にFashion MNISTのデータで学習を行ってください．



# 参考文献

GradCAMの実装

https://qiita.com/sasayabaku/items/fd8923cf0e769104cc95

https://www.noconote.work/entry/2019/01/12/231723



学習モデルの可視化

https://qiita.com/yasudadesu/items/1dda5f9d1708b6d4d923



重みフィルタの可視化

https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
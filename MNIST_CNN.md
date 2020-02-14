# ml-gasshuku2020

## MNIST

### 1. データセットの読み込み

#### データの格納方法について
ディープラーニングを学習させるためのデータセットを読み込みます．ディープラーニングでは主に4次元配列でデータを準備します．
フレームワークによって格納されている順番が違うのですが，pytorchでは[データ数、チャネル、画像の縦サイズ、画像の横サイズ]の順番でデータが格納されています．

教師ラベルのデータは、１次元もしくは二次元配列で用意します。 1次元の場合はクラスのインデックス(例えば３クラス分類にて犬なら0、イモリなら1、ヤモリなら2みたいな)を指定するが、二次元の場合はone-hot表現を用いる(犬なら[1,0,0]、イモリなら[0,1,0]、ヤモリなら[0,0,1]みたいな)。 これもフレームワークによって変わります。
pytorchではindex[データ数]のように管理をします．

#### データセットの用意
ディープラーニングで学習するデータセットを用意する方法としては，
1. 自分で用意する
2. オープンソースのデータを用意する
の2つあるのですが，今回はpytorchからダウンロードできるオープンデータセットを使って学習を行います．

#### 実装手順
1. ライブラリのimport
2. データセットの前処理方法の指定
3. trainデータの読み込み
4. trainデータをバッチサイズに固めて返すローダの作成
5. testデータの読み込み
6. testデータをバッチサイズに固めて返すローダの作成
7. 教師ラベルの作成

<details>
<summary>コードはこちら</summary>  
<p>  

```python
# 1. ライブラリのimport
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# 2. データセットの前処理方法の指定
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
# 3. trainデータの読み込み
trainset = torchvision.datasets.FashionMNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
# 4. trainデータをバッチサイズに固めて返すローダの作成
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=2)
# 5. testデータの読み込み
testset = torchvision.datasets.FashionMNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform)
# 6. testデータをバッチサイズに固めて返すローダの作成
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=100,
                                            shuffle=False, 
                                            num_workers=2)
# 7. 教師ラベルの作成
classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
```

</p>
</details>


### 2. データセットの可視化

<details>
<summary>コードはこちら</summary>
<p>

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

</p>  
</details>

### 3. モデルの定義
| Layer | カーネルサイズ | フィルタ数 |  ストライド| パディング |  活性化関数 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 28 x 28 x 1(入力サイズ) |||||
| Convolution | 3 x 3 | 32 | 1 | 0 | ReLU |
| Convolution | 3 x 3 | 64 | 1 | 0 | ReLU |
| MaxPooling | 2 x 2 | - | 1 | 0 | - |
| Dropout | - | - | - | - | - |
| MultiLayerPerceptron | 128 | - | - | - | ReLU |
| Dropout | - | - | - | - | - |
| MultiLayerPerceptron | 10 (クラス) | - | - | - | - |

```python
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
```

### 4. 学習

#### step1. GPUの設定

pytorchではGPUを利用する際には明示的に指定する必要があります．

```python
GPU = True
device = torch.device("cuda" if GPU else "cpu")

net = Net()
net = net.to(device)
```

#### step2. ハイパーパラメータの設定

誤差関数と最適化手法を定義します．

誤差関数には交差エントロピー，最適化手法にはSGDを利用します．

```python
import torch.optim as optim

# 誤差関数の定義
criterion = nn.CrossEntropyLoss().cuda()
# 最適化手法の定義
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)	
```

#### step3. 学習

本格的に学習を行なっていきます．

学習の手順としては，

1. inputs(画像)，labels(ラベル)を読み込む
2. 勾配の初期化
3. モデルに画像を入力
4. lossの計算
5. 誤差を逆伝搬
6. 重みの更新

上記を各エポックごとに行います．

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

#### step4. モデルの保存

予測ステップでも利用できるように学習したモデルを保存しましょう．

```python
# モデルの保存
import cloudpickle
with open('model.pkl', 'wb') as f:
    cloudpickle.dump(net, f)
```

### 5. 予測

## モデルの可視化
基本的な流れは以上になりますが，CNNの学習モデルをチューニングする際に，CNNモデルの中身を理解することは重要です．
ここでは，CNNモデルの可視化手法について以下の3つの方法を紹介します．

1. GradCAM
2. 畳み込みフィルタの可視化

### 1. Grad-CAM
Grad-CAMとは，大雑把に言うとCNNの予測の根拠となった箇所をヒートマップで表示する技術です．

![grad-cam](assets/grad-cam.png)

>原著: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
>https://arxiv.org/abs/1610.02391

#### GradCAMのクラスを用意


今回利用したコードはこちらを使用しています．
https://www.noconote.work/entry/2019/01/12/231723
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
####　可視化の実行

```python
grad_cam = GradCam(model)
test_image_tensor = torch.tensor(np.expand_dims(images[0], 0))
feature_image = grad_cam(test_image_tensor.cuda()).squeeze(dim=0)
feature_image = transforms.ToPILImage()(feature_image)
pred_idx = model(test_image_tensor.cuda()).max(1)[1]
plt.imshow(feature_image.resize((28, 28)))
```



### 2. 畳み込みフィルタの可視化



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


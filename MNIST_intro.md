# ml-gasshuku2020

## MNISTでの練習

### 1. データの読み込み
MNISTデータセットを読み込みます．
### 2. データセットの可視化
データセットを見てみましょう
### 3. モデルの定義
LeNetを実装してください．

| Layer | カーネルサイズ | フィルタ数 |  ストライド| パディング |  活性化関数 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 32 x 32 x 3(入力サイズ) |
| Convolution | 5 x 5 |  6 | 1 | 0 | - |
| MaxPooling | 2 x 2 | - | 2 | 0 | sigmoid |
| Convolution | 5 x 5 | 16 | 1 | 0 | - |
| MaxPooling | 2 x 2 | - | 2 | 0 | sigmoid |
| MultiLayerPerceptron | 120 | - | - | - | - | - |
| MultiLayerPerceptron |  64 | - | - | - | - | - |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|

### 4. 学習

#### step1. GPUの設定
#### step2. 学習パラメータの設定
#### step3. 学習

#### step4. モデルの保存

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


### 2. 畳み込みフィルタの可視化


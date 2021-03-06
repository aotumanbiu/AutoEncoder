# Auto-Encoder

##### 注意：该代码并不是真正的Auto-Encoder，但是可以通过简单的更改即可。(代码自己乱写的，实现简单，但是有助于自己理解)

##### 如果把最终的输出改一下，那么它就是一个语义分割网络了！！！！！！！！



------

*2021.11.19:*

​	**```补充：后来通过了解知道，我这种写法叫做 Convolutional-AutoEncoder。（真让人害羞(✿◡‿◡)）```**

------

### 1. 网络模型	

这个模型结构其实是```SegNet```，而且恰巧也在了解Auto-Encoder，所以当看到这个图的时候，突然就有了想自己尝试写一下的冲动。

于是第一次尝试就是看着这个图写的，中间生成的是 ```特征图``` 而不是一个 ```向量```。



Encoder部分采用的VGG-16， Decoder部分是与之对称的。(其实结构没必要这么复杂)

<img src="./files/model.png" style="zoom: 100%">

```注意：AutoEncoder的两个部分不一定是对称的```

------

### 2. 训练可视化

#### (1) 学习率曲线

```对于学习率的调整，这里采用 WarmUp 的调成测率，它能使模型更容易收敛。```

<img src="./files/warmuplr.png" style="zoom: 100%">



#### (2)  Loss 损失曲线

```什么也不用说，简直就是完美！！！```

<img src="./files/loss.png" style="zoom: 100%">



#### (3) 训练过程中的输入和输出

```输入图片：```

<img src="./files/out.png" style="zoom: 100%">

------

```输出图片：```

<img src="./files/in.png" style="zoom: 100%">



### 3. 文件下载

##### a. 权重：Link: https://pan.baidu.com/s/1pqGgaSRBal8HC3flYVirSA pw: 5c9e 

##### b. 数据集：Link: https://pan.baidu.com/s/1oVo9SE8phibpZWRBskFjaA pw: hc2b 





------

**----------------------------------------------------------------------------------这是分割线-----------------------------------------------------------------------------------------**

------

### 4. 采用 VGG13 结构模型

#### (1)  Loss 损失曲线

```也是非常完美的!!!!!!```

<img src="./files/loss2.png" style="zoom: 100%">



### (2) Encoder激活函数

#### (a).  不采用激活函数

```python
if nums == 1:
    return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1), )
```

<img src="./files/no.png" style="zoom: 100%">



训练100个epoch 结果 如下图：(上图为训练时的输入数据)

<img src="./files/no0.png" style="zoom: 100%">



**！！！！！！！** **通过对比可知，图像整体还可以，但是仿佛蒙上了一层纱，图像亮度较暗，但是细节却都没有损失。**  **！！！！！！！**



#### (b). 采用 Sigmoid 激活函数

```python
if nums == 1:
    return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1),
                         nn.Sigmoid())
```

<img src="./files/yes.png" style="zoom: 100%">



训练100个epoch 结果 如下图：(上图为训练时的输入数据)

<img src="./files/yes0.png" style="zoom: 100%">



**！！！！！！！！！！！！！！！** **通过对比可知，图像的亮度大幅度增加，但是细节却被柔化了。**  **！！！！！！！！！！！！！！！！**



#### (c). 原因分析

在书写代码时，我对数据进行了归一化处理，是每个像素值处于 0-1 之间。同时采用的损失函数为 **均方误差(MSE)**，但是生成的结果却不相同的格式（像素值 0-1 之间）。

=================================================================

##### **!!!! 2021.13.03: 如果利用浅层的特征图，图片的细节会更加的完善 Encoder - Decoder 拥有很大的潜力**

=================================================================

-------

***不同数据处理方式对应的激活函数应认真选择，以及实现某功能所需要的损失函数也不同！！！！！！！！！！！！！***

-------

**注意： 这个分析不一定争正确**

```python
Transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])  # transforms.ToTensor() 归一化处理

train_data = AnimeData(dataRoot=args.dataSet, subFold="train", transform=Transform)
```



####  (d) 权重下载

##### a. 权重_无损失函数：Link: https://pan.baidu.com/s/1EExW9EtKx55JGym0zOgemw   pw: 8xit 

**b. 权重_Sigmoid：Link: https://pan.baidu.com/s/1nL6rM7ID8OrDCjfFXo7T3w   pw: fi18**



-----



### 5. 测试

#### (1). 输入为噪声

##### a. 随机初始化输入

```feed_noise_inputs.py```

```python
inputs = torch.rand((256, 3, 224, 224)).to(device)
data = DataLoader(inputs, batch_size=8)
```

输入可视化：

<img src="./files/lala0.png" style="zoom: 100%">

输出可视化：

<img src="./files/lala2.png" style="zoom: 100%">



##### b. 随机初始Decoder的输出, 然后用Decoder解码

```feed_noise_middle.py```

```python
# ---------------------------------------------- #
# 随机初始化噪声
# ---------------------------------------------- #
inputs = torch.rand((256, 512, 7, 7)).to(device)
data = DataLoader(inputs, batch_size=8)
```

输出可视化：

<img src="./files/noise.png" style="zoom: 100%">

------

#### (2). 输入为测试集

输入可视化：

<img src="./files/test0.png" style="zoom: 100%">

输出可视化：

<img src="./files/test1.png" style="zoom: 100%">



--------
### 6. 图像修复
我们通过认为的添加缺陷，然后利用 AE 模型去尝试修复
#### (1). 默认网络结构

输入图像：

<img src="./files/denoise00.png" style="zoom: 100%">

修复结果：

<img src="./files/denoise01.png" style="zoom: 100%">

-----


#### (2). 融合浅层的特征图

输入图像：

<img src="./files/denoise10.png" style="zoom: 100%">

修复结果：

<img src="./files/denoise11.png" style="zoom: 100%">

------

#### (3). 结论

对于典型的Autoencoder结构(即上述所有实验采用的结构)，进行图像修复时，结果效果还可以，但是修复后的图片损失了很多的细节。

对于上述的情况，我在Decoder部分利用了Encoder部分的浅层特征图，通过结果可知，修复后的图片细节很好，```但是图像的亮度很低，所以具体的原因后续会继续的跟进修改```。










# 未完待续（其它尝试）(AE yyds !!!!!)。。。。。。

## 1. ~~单独把Decoder拿出来， 喂给它随机特征图，看会的到什么~~

## 2. VAE

## 3. 不可说

## 4. 。。。。。。


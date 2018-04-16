# speechFeatures
语音处理，声源定位中的一些基本特征

在语音信号处理的研究中，常常涉及到很多复杂的音频特征提取，而且这些特征大部分用matlab提取较方便，但是如果转为python，就需要花一点时间自己去理解了。
考虑到后续要用python做一些语音处理，可以集成到系统中，因此，自己整理了一些常用的语音特征的提取，后续会不断添加改进，也欢迎志同道合的朋友可以一起改进。

## librosa
librosa 是一个python的语音特征提取工具包，大家也可以安装这个工具包来提取特征，但是我觉得有一些特征自己提反而更加清晰。librosa的安装：
```python
    	pip install librosa
    	conda install librosa	# 前提是你安装了anaconda
```




## 特征文档
# 
下面就简单介绍下每个特征主要在干什么，代码中也有中文注释，一些效果图我就懒得放了，毕竟也不是一个精致的猪猪女孩。

## ```read_wav	```
这个没什么好说的，就是用```scipy```来读取一段音频，读取的音频可能为单声道、双声道和多声道，
一开始我主要是处理双耳声源定位的，所以认为只有两个声道，同样的，你可以仿照例子读取多声道中每个声道的音频。

注释掉的部分为去掉信号前后为0的部分，这个需要每个声道的信号统一，否则在做互相关的时候，结果出错。不要求处理效率，我觉得可以先保留前后为0的信号。


## ```enframe```
看名字都知道是分帧啦，它主要有以下参数：
```python
    	signal: 可以是从read_wav读取的一个声道的信号
	frequency: 信号采样率 
	window_size: 窗大小，以毫秒为单位，如20
	shift_step: 滑动窗口的步长，以毫秒为单位，一般小于窗口大小，表示相邻两帧之间有重叠
	use_window: 加窗，分帧之后一般是需要加窗的，因为两帧之间有重叠，这里可选的窗函数有几种：
                hanning/hamming/kaiser/blackman/none，none表示不加窗
	pre_emhance: 预加重的权重，预加重的原理是为了增强高频的部分，若该值<=0，则不做预加重处理
```
最后返回一个分帧后的二维数组


## ```ERBSpace```
作为求解伽马通滤波器的一组中心频率，输入参数：
```python
    	lowFreq: 最低频率
	highFreq: 最高频率
	N: 滤波器阶数
```
最后返回一维的N个中心频率的数组



## ```MakeERBFilters```
用于产生伽马通滤波器的系数，里面调用了ERBSpace用于产生一组中心频率，输入参数为：
```python
    	fs:	信号的采样频率
	numChannels: 滤波器通道数，即需要多少个滤波器
	lowFreq: 最低频率
	highFreq: 最高频率
```
最后返回一个二维的滤波器系数矩阵，32x10，和中心频率


## ```ERBFilterBank```
输入伽马通滤波器系数，和原始数据，用滤波器系数对原始数据进行滤波，得到一组不同中心频带的信号



## ```STFT```
短时傅里叶变换，对前面分帧后的信号进行傅里叶变换，参数：
```python
    	signal: 分帧后的二维信号，第一维是帧数，第二维是每一帧的信号
	n_point: 傅里叶变换的点数。不足的补0
```
最后输出一个二维数组，表示每一帧的短时傅里叶变换，数值是复数形式




## ```crossCorrelation```
广义互相关函数，提供GCC-PHAT的计算方法，在声源定位中，用于计算两个信号的延时。根据原理分析，两个信号计算互相关之后，最大的值所对应的时间点就是两个信号的延时。
计算方法参考matlab中的xcorr，其中做了归一化的修改，变成GCC-PHAT



## ```ILD```
计算两个信号的双耳能量差，在双耳声源定位中，由于受到头部的影响，导致左右耳朵接收到的信号存在一定的时间差和能量差，时间差由上面的crossCorrelation函数计算得出，能量差计算两个信号的能量比值再取对数。





## ```crossSpectrum```
计算两个信号的互相关时延谱图，其原理为：对左右信号分帧并通过伽马通滤波器后，再对每一帧中的每一个频带的左右信号计算互相关值，得到每一个time-frequency(TF) bin 的时延。这种情况多数用于双耳多声源定位，假设每一个TF bin只由一个声源主导。


# 

## 接下来是其他常用的音频特征

# 



## ```zero_cross_ratio```
过零率体现的是信号过零点的次数，体现是频率特性，因为需要过零点，所以信号处理之前需要中心化处理。
最后返回一帧信号中，共有多少次过零点



## ```short_time_energy```
短时能量，体现的是信号在不同时刻的强弱程度



## ```short_time_average_amplitude```
短时平均幅度差, 音频具有周期性，平稳噪声情况下利用短时平均幅度差可以更好地观察周期特性



## ```spectrum_entropy```
谱熵体现的是不确定性，例如抛骰子一无所知，每一面的概率都是1/6，信息量最大，也就是熵最大  如果知道商家做了手脚，抛出3的概率大，这个时候我们已经有一定的信息量，抛骰子本身的信息量就少了，熵也就变小。对于信号，如果是白噪声，频谱类似均匀分布，熵就大一些；如果是语音信号，分布不均匀，熵就小一些，利用这个性质也可以得到一个粗糙的VAD（有话帧检测）。谱熵有许多的改进思路，滤波取特定频段、设定概率密度上限、子带平滑谱熵，自带平滑通常利用拉格朗日平滑因子，这是因为防止某一段子带没有信号，这个时候的概率密度就没有意义了，这个思路在利用统计信息估计概率密度时经常用到，比如朴素贝叶斯就用到这个思路。



## ```basic_frequency```
基频：也就是基频周期。人在发音时，声带振动产生浊音(voiced)，没有声带振动产生清音（Unvoiced）。浊音的发音过程是：来自肺部的气流冲击声门，造成声门的一张一合，形成一系列准周期的气流脉冲，经过声道（含口腔、鼻腔）的谐振及唇齿的辐射形成最终的语音信号。故浊音波形呈现一定的准周期性。所谓基音周期，就是对这种准周期而言的，它反映了声门相邻两次开闭之间的时间间隔或开闭的频率。基音提取常用的方法有：倒谱法、短时自相关法、短时平均幅度差法、LPC法.



## ```mfcc```
梅尔倒谱系数，由于梅尔刀谱系数的计算过程较复杂，这里直接调用了librosa中的```mfcc```函数。网上博客很多关于梅尔倒谱系数的计算方法，它也是用一种模拟人耳的滤波器对信号进行滤波，一般返回每一帧的梅尔倒谱系数作为特征。



# 
## 关于作者
# 

``` susanna@wull.me```

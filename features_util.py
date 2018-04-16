# encoding=utf-8

import sys
sys.path.append('..')
import wave
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import pylab
from scipy import signal
from scipy.io import wavfile
from scipy.stats import mode
import json
import random
import pandas as pd
# import gammatone
import scipy.io as scio
import librosa

"""读取信号->分帧->伽马通滤波->"""


def read_wav(file_path):
    fs, wav_data = wavfile.read(file_path)
    wav_data = np.array(wav_data / (2.**15))
    n_channels = wav_data.shape[1]

    def find_zero_point(signal):
        l = signal.shape[0]
        front, back = 0, l-1
        for front in range(l):
            if signal[front] != 0:
                break
        for back in range(l-1, -1, -1):
            if signal[back] != 0:
                break
        return front, back

    left_signal = None
    right_signal = None
    if n_channels >= 2:
        left_signal = wav_data[:, 0]
        right_signal = wav_data[:, 1]
        # 去掉信号前后为0的部分，不去除延时的部分
        # l_front, l_back = find_zero_point(left_signal)
        # r_front, r_back = find_zero_point(right_signal)
        # left_signal = left_signal[min([l_front, r_front]):max([l_back, r_back]) + 1]
        # right_signal = right_signal[min([l_front, r_front]):max([l_back, r_back]) + 1]
    return {
        'wav_data': wav_data,
        'left_data': left_signal,
        'right_data': right_signal,
        'fs': fs
    }


def enframe(signal, frequency, window_size, shift_step, use_window="hanning", pre_emhance=0.95):
    """
    :param signal:  input speech
    :param frequency: sampling frequency rate
    :param window_size: in ms
    :param shift_step: in ms
    :param use_window: hanning, hamming, kaiser, blackman, None
    :param pre_emhance: 预加重
    :return: frames array, n_frames*window_size
    """
    signal_frame = list()
    window_size = int(window_size*frequency/1000)
    shift_step = int(shift_step*frequency/1000)
    window = None
    use_window = use_window.lower()
    if use_window == "hanning":
        window = np.hanning(window_size)
    elif use_window == "hamming":
        window = np.hamming(window_size)
    elif use_window == "kaiser":
        window = np.kaiser(window_size, 0)
    elif use_window == "blackman":
        window_size = np.kaiser(window_size, 8.6)
    elif use_window == "none":
        window = np.ones(window_size)
    signal_length = len(signal)

    if pre_emhance > 0:
        i = 1
        while i < signal_length:
            signal[i] = signal[i] - pre_emhance*signal[i-1]
            i += 1
    i = 0
    while i < signal_length - window_size:
        tmp = signal[i:i+window_size]
        tmp = tmp*window
        signal_frame.append(tmp)
        i += shift_step
    signal_frame = np.array(signal_frame, dtype='float32')
    return signal_frame


def ERBSpace(lowFreq=80, highFreq=16000, N=32):
    """
    ERBSpace获得了中心频率f。
    :param lowFreq: 最小频率
    :param highFreq: 最大频率
    :param N: 滤波器通道数
    :return: 滤波器族的各个中心频率
    """
    earQ = 9.26449
    minBW = 24.7
    low = float(lowFreq)
    high = float(highFreq)
    N = float(N)
    cf = -(earQ * minBW) + np.exp(
        (np.arange(N + 1)[1:]) * (-np.log(high + earQ * minBW) + np.log(low + earQ * minBW)) / (N)) * (
                     high + earQ * minBW)
    cf = cf[::-1]   # 沿着通道数对频率从小到大排序，注释掉可反过来
    return cf


def MakeERBFilters(fs, numChannels, lowFreq, highFreq):
    """MakeERBFilters函数产生了GT滤波器系数。输入为：采样频率、滤波器通道数、最小频率、最高频率。输出为：n通道GT滤波器系数。"""
    fs = float(fs)

    T = 1 / fs
    if np.isscalar(numChannels):
        numChannels = [numChannels]
    numChannels = np.array(numChannels)
    if numChannels.ndim > 1:
        print("is not one dimision data")
        return 0
    if numChannels.size == 1:
        cf = ERBSpace(lowFreq, highFreq, numChannels[0])
    else:
        cf = numChannels

    EarQ = 9.26449  # Glasberg and Moore Parameters
    minBW = 24.7
    order = 1.0
    ERB = ((cf / EarQ) ** order + minBW ** order) ** (1 / order)
    B = 1.019 * 2.0 * np.pi * ERB

    arg = 2 * cf * np.pi * T
    vec = np.exp(2j * arg)

    A0 = T
    A2 = 0.0
    B0 = 1.0
    B1 = -2 * np.cos(arg) / np.exp(B * T)
    B2 = np.exp(-2.0 * B * T)

    rt_pos = np.sqrt(3 + 2 ** 1.5)
    rt_neg = np.sqrt(3 - 2 ** 1.5)

    common = -T * np.exp(-(B * T))
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14
    gain_arg = np.exp(1j * arg - B * T)
    gain = np.abs(
        (vec - gain_arg * k11)
        * (vec - gain_arg * k12)
        * (vec - gain_arg * k13)
        * (vec - gain_arg * k14)
        * (T * np.exp(B * T)
           / (-1 / np.exp(B * T) + 1 + vec * (1 - np.exp(B * T)))
           ) ** 4
    )
    allfilts = np.ones_like(cf)
    fcoefs = np.column_stack([
        A0 * allfilts, A11, A12, A13, A14, A2 * allfilts,
        B0 * allfilts, B1, B2,
        gain
    ])
    return fcoefs, cf


def ERBFilterBank(x, fcoefs):
    """
        Gammatone滤波器被广泛用于模拟人类听觉系统对信号的处理方式，作为语音信号的一类听觉分析滤波器（以下简称为GT滤波器）。
        GT滤波器只需要很少的参数就能很好地模拟听觉实验中的生理数据，能够体现基底膜尖锐的滤波特性，
        而且 GT滤波器具有简单的冲激响应函数，能够由此推导出GT函数的传递函数，进行各种滤波器性能分析，同时有利于听觉模型的电路实现。
        GT滤波器的冲击响应函数定义如下：
        g(t)=a*t^(n-1)*cos(a*pi*f*t+fy)*e^(-2*pi*b*t)
        这里n为滤波器阶数，b为滤波器的带宽，f为滤波器的中心频率，a是振幅。
        a=B**n,  B=b1ERB(f)
        ERB(f)为GT滤波器的等价矩形带宽（等价矩形带宽：对于同样的白噪声输入，和指定的滤波器通过一样能量的矩形滤波器的宽度，简称ERB)，
        它同GT滤波器中心频率f，的关系是
        ERB（f）=24.7+0.108f
        b1=1.019是为了让GT函数更好地与生理数据相符而引入的参数。

        ERBFilterBank 函数输入分别为：原始数据和GT滤波器系数。输出为滤波后的数据。该函数实现对原始数据的时域GT滤波。
    """
    A0 = fcoefs[:, 0]
    A11 = fcoefs[:, 1]
    A12 = fcoefs[:, 2]
    A13 = fcoefs[:, 3]
    A14 = fcoefs[:, 4]
    A2 = fcoefs[:, 5]
    B0 = fcoefs[:, 6]
    B1 = fcoefs[:, 7]
    B2 = fcoefs[:, 8]
    gain = fcoefs[:, 9]
    output = np.zeros((gain.size, x.size), dtype=np.float64)
    for chan in range(gain.size):
        y1 = signal.lfilter(np.array([A0[chan] / gain[chan], A11[chan] / gain[chan], A2[chan] / gain[chan]]),
                            np.array([B0[chan], B1[chan], B2[chan]]), x)
        y2 = signal.lfilter(np.array([A0[chan], A12[chan], A2[chan]]), np.array([B0[chan], B1[chan], B2[chan]]), y1)
        y3 = signal.lfilter(np.array([A0[chan], A13[chan], A2[chan]]), np.array([B0[chan], B1[chan], B2[chan]]), y2)
        y4 = signal.lfilter(np.array([A0[chan], A14[chan], A2[chan]]), np.array([B0[chan], B1[chan], B2[chan]]), y3)
        output[chan, :] = y4
    return output


def STFT(signal, n_point):
    """
    语谱图
    短时傅里叶变换，利用离散傅里叶变换，信号长度不足n_point的进行补零
    :param signal: every frame
    :param n_point:
    :return: stft
    """
    spec_list = list()
    for item in signal:
        spec = np.fft.fft(item, n=n_point)
        spec_list.append(spec)
    # 如果是功率谱密度，傅里叶变换后的abs值再除以点数
    spec_list = np.array(spec_list, dtype='float32')
    return spec_list


def crossCorrelation(x1, x2):
    """
    互相关函数，求两个信号的互相关值
    :param x1:
    :param x2:
    :return:
    """
    n_point = 2*x1.shape[0]-1
    X = np.fft.fft(x1, n_point)
    Y = np.fft.fft(x2, n_point)
    XY = X * np.conj(Y)

    # 归一化
    # pl = np.sum(np.square(x1))
    # pr = np.sum(np.square(x2))
    # c = XY / (np.sqrt(pl * pr) + 2.2204e-16)
    # GCC-PHAT
    c = XY / (abs(X)*abs(Y)+2.2204e-16)
    c = np.real(np.fft.ifft(c))
    end = len(c)
    center_point = int(end/2)
    c = np.hstack((c[center_point+1:], c[:center_point+1]))
    lag = np.argmax(abs(c)) - len(x1) + 1       # 返回最大值所对应的下标，如果换算成时间：lag*1000/fs

    # test
    # c, m = cross_correlation(wav_info['right_data'], wav_info['left_data'])
    # print(m * 1000 / wav_info['fs'])
    return c, lag


def ILD(x1, x2):
    """
    计算两个信号的能量差, interaural level differences
    20log10(sum(power(left))/sum(power(right)))
    :param x1:
    :param x2:
    :return:
    """
    fl = np.sum(np.square(x1))
    fr = np.sum(np.square(x2))
    if fl != 0 and fr != 0:
        return 20 * np.log10(fr / fl)
    elif fl == 0 and fr != 0:
        return 10000
    elif fr == 0 and fl != 0:
        return -10000
    else:
        return 0


def crossSpectrum(x1, x2, fs):
    """
    对两个信号计算互相关时延谱图，其中，信号都为分帧分频带后的矩阵, x1:left, x2:right
    :param x1:
    :param x2:
    :param fs:
    :return:
    """
    if x1.shape != x2.shape:
        return None

    frequency_bands = x1.shape[1]
    frames = x1.shape[0]
    res = []
    for i in range(frames):
        tmp = []
        for j in range(frequency_bands):
            c, m = crossCorrelation(x2[i][j], x1[i][j])
            itd = m * 1000 / fs
            tmp.append(itd)
        res.append(tmp)
    res = np.array(res)
    return res


# 常用音频特征
def zero_cross_ratio(frames):
    """
    Zn = 1/2*sum_(m=0~(N-1))(sgn[Xn(m)]-sgn[Xn(m-1)])
    N 是一帧的长度，n为对应的帧数，按帧处理
    理论分析：过零率体现的是信号过零点的次数，体现是频率特性，因为需要过零点，
    所以信号处理之前需要中心化处理
    :param frames: 分帧后的信号
    :return: 每一帧的过零率
    """
    n_frames = frames.shape[0]
    res = np.zeros(n_frames)
    wlen = frames.shape[1]
    for i in range(n_frames):
        for j in range(wlen-1):
            if frames[i][j]*frames[i][j+1] < 0:
                res[i] += 1
    return res


def short_time_energy(frames_signal):
    """
    短时能量, En = sum_(m=0~N-1)(Xn^2(m))
    短时能量体现的是信号在不同时刻的强弱程度
    :param frames_signal:
    :return:
    """
    n_frames = frames_signal.shape[0]
    res = np.zeros(n_frames)
    for i in range(n_frames):
        res[i] = np.sum(frames_signal[i]*frames_signal[i])
    return res


def short_time_average_amplitude(frames_signal):
    """
    短时平均幅度差, Rn(k)=sum_(m=0~N-1)(x(n)-x(n+k))
    音频具有周期性，平稳噪声情况下利用短时平均幅度差可以更好地观察周期特性
    :param frames_signal:
    :return:
    """
    n_frame = frames_signal.shape[0]
    wlen = frames_signal.shape[1]
    res = np.zeros((n_frame, wlen))
    for i in range(n_frame):
        for j in range(wlen):
            res[i][j] = np.sum(abs(frames_signal[i][j:]-frames_signal[i][:wlen-j]))  # 求每个样点的幅度差再累加
    return res


def spectrum_entropy(frames_fft):
    """
    谱熵的定义，首先对每一帧信号的频谱绝对值归一化
    Pi=(Ym(fi))/(sum_(k=0~N-1)(Ym(fk)))
    这样就得到了概率密度，进而求取熵
    Hm=-sum_(i=0~N-1)(P(i)log(P(i))
    分析：熵体现的是不确定性，例如抛骰子一无所知，每一面的概率都是1/6，信息量最大，也就是熵最大。
    如果知道商家做了手脚，抛出3的概率大，这个时候我们已经有一定的信息量，抛骰子本身的信息量就少了，熵也就变小。
    对于信号，如果是白噪声，频谱类似均匀分布，熵就大一些；如果是语音信号，分布不均匀，熵就小一些，
    利用这个性质也可以得到一个粗糙的VAD（有话帧检测）。
    谱熵有许多的改进思路，滤波取特定频段、设定概率密度上限、子带平滑谱熵，自带平滑通常利用拉格朗日平滑因子，
    这是因为防止某一段子带没有信号，这个时候的概率密度就没有意义了，这个思路在利用统计信息估计概率密度时经常用到，
    比如朴素贝叶斯就用到这个思路。

    :param frames_fft: 分帧后每一帧的傅里叶变换
    :return:
    """
    n_frames = frames_fft.shape[0]
    wlen = frames_fft.shape[1]
    H = np.zeros(n_frames)
    for i in range(n_frames):
        Sp = np.abs(frames_fft[i])
        Sp = Sp[:int(wlen/2)+1]
        Ep = Sp*Sp
        prob = Ep/(np.sum(Ep))
        H[i] = -np.sum(prob*np.log(prob+2.2204e-16))
    return H


def basic_frequency(signal, fs):
    """
    基频：也就是基频周期。人在发音时，声带振动产生浊音(voiced)，没有声带振动产生清音（Unvoiced）。
    浊音的发音过程是：来自肺部的气流冲击声门，造成声门的一张一合，形成一系列准周期的气流脉冲，
    经过声道（含口腔、鼻腔）的谐振及唇齿的辐射形成最终的语音信号。故浊音波形呈现一定的准周期性。
    所谓基音周期，就是对这种准周期而言的，它反映了声门相邻两次开闭之间的时间间隔或开闭的频率。
    基音提取常用的方法有：倒谱法、短时自相关法、短时平均幅度差法、LPC法
    自相关函数：
    Rn(k)=sum_(m=0~N-1-k)([x(n+m)w'(m)][x(n+m+k)w'(k+m)])
    归一化处理，因为R(0)最大
    Rn(k)=Rn(k)/Rn(0)
    得到归一化相关函数的时候，归一化的相关函数第一个峰值为k=0，第二个峰值理论上应该对应基频的位置，
    因为自相关函数对称，通常取一半分析即可
    :return:
    """
    wlen = len(signal)
    r, lag = crossCorrelation(signal, signal)
    r = r / np.max(r)
    rhalf = r[wlen:]  # 取延迟量为正值的部分
    # 提取基音，假设介于50~600Hz之间
    lmin = np.round((50 / fs) * wlen)
    lmax = np.round((600 / fs) * wlen)
    tloc = np.argmax(rhalf[int(lmin):int(lmax)])
    pos = lmin + tloc - 1
    pitch = pos / wlen * fs
    return pitch


def mfcc(signal, fs):
    """
    梅尔倒谱系数
    :param signal:
    :param fs:
    :return:
    """
    # S = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=320, fmax=8000)
    # res_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
    res_mfcc = librosa.feature.mfcc(y=signal, sr=fs)
    return res_mfcc


def test_spectrum_entropy():
    a=[-104.9486, -104.6963, -104.9053, -105.1850, -104.9786, -104.6889, -104.8497, -105.1547, -104.9943, -104.6741]
    a = np.array(a)
    a = [np.fft.fft(a)]
    a = np.array(a)
    print(a.shape)
    print(a)
    H = spectrum_entropy(a)
    print(H)


def test_basic_frequency(data_path):
    wav_info = read_wav(data_path)
    signal = wav_info['left_data'][16000:(16000 + 320)]
    pitch = basic_frequency(signal, wav_info['fs'])
    print(pitch)


def test_cross_spectrum(wav_path):
    wav_info = read_wav(wav_path)
    left_frame = enframe(wav_info['left_data'], wav_info['fs'], 20, 10, "hanning", 0.98)
    right_frame = enframe(wav_info['right_data'], wav_info['fs'], 20, 10, "hanning", 0.98)

    c, m = crossCorrelation(wav_info['right_data'], wav_info['left_data'])
    print("两个信号时延", m * 1000 / wav_info['fs'], "ms")
    zcr = zero_cross_ratio(np.array([left_frame[0]]))
    print("过零率", zcr)
    ste = short_time_energy(left_frame)
    print("短时能量", ste.shape)
    saa = short_time_average_amplitude(left_frame)
    print("短时平均幅度差", saa.shape)

    stft = STFT(left_frame, None)
    scio.savemat("stft.mat", {'stft': stft})
    print("短时傅里叶变换（语谱图）", stft.shape)

    res_mfcc = mfcc(wav_info['left_data'], wav_info['fs'])
    print("mfcc", res_mfcc.shape)

    fcoefs, center_freq = MakeERBFilters(wav_info['fs'], 32, 80, 8000)

    left_signal_filtering = []
    right_signal_filtering = []
    frames = len(left_frame)
    for i in range(frames):
        cfs = ERBFilterBank(left_frame[i], fcoefs)
        left_signal_filtering.append(cfs)
        cfs = ERBFilterBank(right_frame[i], fcoefs)
        right_signal_filtering.append(cfs)
    left_signal_filtering = np.array(left_signal_filtering)
    right_signal_filtering = np.array(right_signal_filtering)
    print("通过伽马通滤波器后左右信号的大小", left_signal_filtering.shape, right_signal_filtering.shape)

    cross_spec = crossSpectrum(left_signal_filtering, right_signal_filtering, wav_info['fs'])
    print("交叉互相关谱的大小", cross_spec.shape)

    scio.savemat("cross_spec.mat", {'d30': cross_spec.T})


if __name__ == '__main__':
    data_path = 'E:\pku\papers\experiments\SSL\data\SSL_expriment_data//test_2s.wav'
    d300 = 'E:\pku\papers\experiments\SSL\data\SSL_expriment_data\without_noise\\test\QU_KEMAR_anechoic_AKGK271_0.5m' \
           '\dr5\\fgmd0\sx413_300.wav'
    d30 = 'E:\pku\papers\experiments\SSL\data\SSL_expriment_data\without_noise\\test\QU_KEMAR_anechoic_AKGK271_0.5m' \
          '\dr5\\fasw0\sa2_30.wav'

    test_cross_spectrum(d300)
    # test_basic_frequency(data_path)
    # librosa.feature.zero_crossing_rate()


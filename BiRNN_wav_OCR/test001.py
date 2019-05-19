import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

sample_rate, signal = scipy.io.wavfile.read('OSR_us_000_0010_8k.wav') #读入语言
signal = signal[0:int(3.5 * sample_rate)] #取出前3.5s
time=np.arange(0,int(3.5 * sample_rate))*(1.0/sample_rate)
#1、Pre-Emphasis,预加重
pre_emphasis = 0.97 #滤波器系数，典型值为0.95或0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
#2、Framing 分帧
frame_size = 0.025 #帧大小，单位s
frame_stride = 0.01 #10毫秒的步幅.相对25ms的帧，有15ms的重叠
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # 从秒转换为短时帧数据
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # 确保至少有一个短时帧

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) # 填充信号以确保所有帧具有相同数量的样本而不截断原始信号中的任何样本
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                                                                         (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

#3、Window 将窗口函数应用于每个帧，即加窗
frames *= np.hamming(frame_length)
#4、傅里叶变换和功率谱Power Spectrum
NFFT = 512 #通常为256或512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # FFT结果取绝对值(幅值)
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # 功率谱
#5、Filter Banks ：在Mel刻度上将三角滤波器（通常为40个滤波器）应用于功率谱以提取频带。
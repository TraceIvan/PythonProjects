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
plt.rcParams['figure.figsize'] = (12, 4) # 设置figure_size尺寸
plt.figure()
plt.plot(time,signal,label='OSR_us_000_0010_8k.wav')
plt.xlabel("time(/s)")
plt.ylabel("Amplitude")
plt.legend(loc=1)
plt.title("original signal")
#plt.xticks(np.arange(0,3.5,0.1))
plt.show()

#1、Pre-Emphasis,预加重
#是噪声整形技术在模拟信号的处理中，一项关于噪声整形技术原理的技术。所谓预加重是指在信号发送\
# 之前，先对模拟信号的高频分量进行适当的提升。
#（1）平衡频谱，因为高频通常与较低频率相比具有较小的幅度，\
# （2）避免在傅里叶变换操作期间的数值问题\
# （3）也可以改善信号 - 噪声比（SNR）。
pre_emphasis = 0.97 #滤波器系数，典型值为0.95或0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
plt.figure()
plt.plot(time,emphasized_signal,label='OSR_us_000_0010_8k.wav')
plt.xlabel("time(/s)")
plt.ylabel("Amplitude")
plt.title("signal after Pre-Emphasis")
plt.legend(loc=1)
plt.show()

#2、Framing 分帧
#将信号分成短时帧
#信号中的频率随时间变化，因此在大多数情况下，对整个信号进行傅立叶变换是没有意义的，
# 因为我们会随着时间的推移而丢失信号的频率轮廓。为了避免这种情况，我们可以安全地假设信号
# 中的频率在很短的时间内是平稳的。因此，通过在该短时帧上进行傅里叶变换，我们可以通过连接相邻
# 帧来获得信号频率轮廓的良好近似。

#典型帧大小范围为20ms至40ms，连续帧之间具有50％（+/- 10％）重叠
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
#用于从较长的数据中多次取出部分数据分析，而不是一次性
#抵消FFT假设数据无限并减少频谱泄漏的假设
#使用  Hamming window
alpha=0.46
N=200
n=np.arange(0,200,1)
W=(1-alpha)-alpha*np.cos(2*np.pi*n/(N-1)) #Hamming窗函数
plt.rcParams['figure.figsize'] = (6, 4) # 设置figure_size尺寸
plt.figure()
plt.plot(n,W,label="Hamming window")
plt.xlabel("samples")
plt.ylabel("Ampitude")
plt.legend(loc=1)
plt.show()#展示Hamming窗效果

frames *= np.hamming(frame_length)
## frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))

#4、傅里叶变换和功率谱Power Spectrum
#短时傅里叶变换（STFT）
NFFT = 512 #通常为256或512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # FFT结果取绝对值(幅值)
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # 功率谱


#5、Filter Banks ：在Mel刻度上将三角滤波器（通常为40个滤波器）应用于功率谱以提取频带。
#Mel滤波,通过Mel滤波器组进行滤波，以得到符合人耳听觉习惯的声谱，
# 最后通常取对数将单位转换成db
# Hz转换为Mel:  m=2295*log10(1+f/700)
# Mel转为为Hz: f=700*(10^(m/2296)-1)
#滤波器组中的每个滤波器都是三角形的，在中心频率处响应为1，并且朝向0线性减小，直到它达到响应为0的两个相邻滤波器的中心频率

#得到Mel滤波器组
nfilt = 40 #滤波器数目
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
plt.figure()
colors=['red','orange','coral','yellow','green','lime','blue','navy','purple','pink']
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    fs=np.linspace(hz_points[m-1],hz_points[m+1],100)
    amplitudes=np.zeros(100)
    for id,k in enumerate(fs):
        if k<=hz_points[m]:
            amplitudes[id]=(k-hz_points[m-1])/(hz_points[m]-hz_points[m-1])
        else:
            amplitudes[id]=(hz_points[m + 1]-k)/(hz_points[m+1] - hz_points[m])
    plt.plot(fs,amplitudes,color=colors[(m-1)%10])

plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("The 40 filter Mel Filter Bank")
plt.show()
plt.figure()
for i in range(nfilt):
    plt.plot(range(0,int(np.floor(NFFT / 2 + 1))),fbank[i,:],color=colors[(i-1)%10])
plt.xlabel("FFT bin number")
plt.ylabel("Amplitude")
plt.title("The 40 filter Mel Filter Bank")
plt.show()

print(np.shape(fbank))
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB
print(np.shape(filter_banks))
filter_banks -= (np.mean(filter_banks,axis=0) + 1e-8)#均值归一化,从所有帧中减去每个系数的平均值,用以平衡频谱并改善信噪比（SNR）
plt.rcParams['figure.figsize'] = (12, 4)
plt.figure()
plt.imshow(filter_banks.T, cmap=plt.cm.jet, aspect='auto')
xloc=list(np.arange(0, (filter_banks.T).shape[1],50))
plt.xticks(xloc,
           ['0s', '0.5s', '1s', '1.5s','2s','2.5s','3s'])
ax = plt.gca()
ax.invert_yaxis()
plt.title('信号频谱图(Spectrogram of the Signal)')
plt.show()

#6、梅尔频率倒谱系数（MFCCs）
#应用离散余弦变换（DCT）去相关滤波器组系数并产生滤波器组
# 的压缩表示。通常，对于自动语音识别（ASR），保留所得到的倒频谱系数2-13，其余部分被丢弃
num_ceps = 13
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
print(np.shape(mfcc))
#将正弦提升器Liftering应用于MFCC以去强调更高的MFCC
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter=22#apply a lifter to final cepstral coefficients. 0 is no lifter
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift

plt.figure()
plt.imshow(mfcc.T, cmap=plt.cm.jet, aspect='auto')
xloc=list(np.arange(0, (mfcc.T).shape[1],50))
plt.xticks(xloc,
           ['0s', '0.5s', '1s', '1.5s','2s','2.5s','3s'])
ax = plt.gca()
ax.invert_yaxis()
plt.title('MFCCs')
plt.show()

mfcc -= (np.mean(mfcc, axis=0) + 1e-8)#均值归一化
plt.figure()
plt.imshow(mfcc.T, cmap=plt.cm.jet, aspect='auto')
xloc=list(np.arange(0, (mfcc.T).shape[1],50))
plt.xticks(xloc,
           ['0s', '0.5s', '1s', '1.5s','2s','2.5s','3s'])
ax = plt.gca()
ax.invert_yaxis()
plt.title('Normalized MFCCs')
plt.show()
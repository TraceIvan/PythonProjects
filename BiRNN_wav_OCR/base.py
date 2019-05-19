import os
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

'''读取WAV文件及对应的label'''
def get_wavs_labels(wav_path,label_file):
    wav_files=[]
    for(dirpath,dirnames,filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith(".WAV"):
                filename_path=os.sep.join([dirpath,filename])
                if os.stat(filename_path).st_size<24000:#剔除小文件
                    continue
                wav_files.append(filename_path)

    labels_dict={}
    with open(label_file,'rb') as f:
        for label in f:
            label=label.strip(b'\n')
            label_id=label.split(b' ',1)[0]
            label_text=label.split(b' ',1)[1]
            labels_dict[label_id.decode('ascii')]=label_text.decode('utf-8')#自建数据集时需utf8改为gb2312

    labels=[]
    new_wav_files=[]
    for wav_file in wav_files:
        wav_id=os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)
    return new_wav_files,labels

def audio_to_input_vector(audio_filename,numcep,numcontext):
    #加载wav文件
    fs,audio=wav.read(audio_filename)
    #获得MFCC特征
    orig_inputs=mfcc(audio,samplerate=fs,numcep=numcep)
    orig_inputs=orig_inputs[::2]#每隔一行采样(BiRNN输出包含正反向的结果，相当于每个时序扩展一倍，此处保证总时序不变)
    #print(np.shape(orig_inputs))#(时间序列个数，各个时序的特征数)

    train_inputs=np.array([],np.float32)
    train_inputs.resize((orig_inputs.shape[0],numcep+2*numcep*numcontext))

    empty_mfcc=np.array([])
    empty_mfcc.resize((numcep))
    #扩展特征值：当前时序的MFCC特征=前9个时序的MFCC+当前时序的MFCC+后9个时序的MFCC(numcontext=9)
    time_silces=range(train_inputs.shape[0])
    context_past_min=time_silces[0]+numcontext
    context_future_max=time_silces[-1]-numcontext
    for time_silce in time_silces:
        #前9个补0
        need_empty_past=max(0,context_past_min-time_silce)
        empty_source_past=list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past=orig_inputs[max(0,time_silce-numcontext):time_silce]
        assert (len(empty_source_past)+len(data_source_past)==numcontext)

        #后9个补0
        need_empty_future=max(0,(time_silce-context_future_max))
        empty_source_future=list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future=orig_inputs[time_silce+1:time_silce+numcontext+1]
        assert (len(empty_source_future)+len(data_source_future)==numcontext)

        if need_empty_past:
            past=np.concatenate((empty_source_past,data_source_past))
        else:
            past=data_source_past
        if need_empty_future:
            future=np.concatenate((data_source_future,empty_source_future))
        else:
            future=data_source_future

        past=np.reshape(past,numcontext*numcep)
        now=orig_inputs[time_silce]
        future=np.reshape(future,numcontext*numcep)

        train_inputs[time_silce]=np.concatenate((past,now,future))#扩充时序后的数据
        assert (len(train_inputs[time_silce])==numcep+2*numcep*numcontext)

    #print(np.shape(train_inputs))  # (时间序列个数，各个时序经扩展后的特征数=原各个时序的特征数*（2*numcontext+1）)
    #正态分布标准化（减去均值后除以方差）,方便训练
    train_inputs=(train_inputs-np.mean(train_inputs))/np.std(train_inputs)
    return train_inputs
#从文件读取文本
def get_ch_label(txt_file):
    labels=""
    with open(txt_file,'rb') as f:
        for label in f:
            labels=labels+label.decode('gb2312')
    return labels
#将文件中的文本转为向量
def get_ch_label_v(txt_file,word_num_map,txt_label=None):
    words_size=len(word_num_map)
    to_num=lambda word:word_num_map.get(word,words_size)
    if txt_file!=None:
        txt_label=get_ch_label(txt_file)
    label_vector=list(map(to_num,txt_label))
    return label_vector

#得到转换后的音频MFCC特征以及文本向量
def get_audio_and_transcriptch(txt_files,wav_files,n_input,n_context,word_num_map,txt_labels=None):
    audio=[]
    audio_len=[]
    transcript=[]
    transcript_len=[]
    if txt_files != None:
        txt_labels=txt_files
    #载入音频数据并转化为特征值
    for txt_obj,wav_file in zip(txt_labels,wav_files):
        audio_data=audio_to_input_vector(wav_file,n_input,n_context)
        audio_data=audio_data.astype('float32')
        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))
        #载入音频对应文本
        target=[]
        if txt_files!=None:
            target=get_ch_label_v(txt_obj,word_num_map)
        else:
            target=get_ch_label_v(None,word_num_map,txt_obj)
        transcript.append(target)
        transcript_len.append(len(target))
    audio=np.asarray(audio)
    audio_len=np.asarray(audio_len)
    transcript=np.asarray(transcript)
    transcript_len=np.asarray(transcript_len)
    return audio,audio_len,transcript,transcript_len

#批对齐，保证一批次的音频的时序数统一
def pad_sequences(sequences,maxlen=None,dtype=np.float32,padding='post',truncating='post',value=0.0):
    #print(np.shape(sequences))
    lengths=np.asarray([len(s) for s in sequences],dtype=np.int64)
    #print(np.shape(lengths))
    nb_samples=len(sequences)
    if maxlen is None:
        maxlen=np.max(lengths)
    #从第一个非空的序列中得到样本形状
    sample_shape=tuple()
    for s in sequences:
        if len(s) >0:
            sample_shape=np.asarray(s).shape[1:]
            break

    x=(np.ones((nb_samples,maxlen)+sample_shape)*value).astype(dtype)
    for idx,s in enumerate(sequences):
        if len(s)==0:
            continue
        if truncating=='pre':#前截断
            trunc=s[-maxlen:]
        elif truncating=='post':#后截断
            trunc=s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood'%truncating)

        #检查trunc
        trunc=np.asarray(trunc,dtype=dtype)
        if trunc.shape[1:] !=sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s'
                             %(trunc.shape[1:],idx,sample_shape))

        if padding=='post':#后补0
            x[idx,:len(trunc)]=trunc
        elif padding=='pre':#前补0
            x[idx,-len(trunc):]=trunc
        else:
            raise ValueError('Padding type "%s" not understood'%padding)

    return x,lengths

#用于密集矩阵转稀疏矩阵,返回结果可被tf.SparseTensor调用生成稀疏矩阵
def sparse_tuple_from(sequences,dtype=np.int32):
    indices=[]
    values=[]
    for n,seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq),range(len(seq))))
        values.extend(seq)

    indices=np.asarray(indices,dtype=np.int64)
    values=np.asarray(values,dtype=dtype)
    shape=np.asarray([len(sequences),indices.max(0)[1]+1],dtype=np.int64)
    return indices,values,shape

#字向量转文字
SPACE_TOKEN='<space>'#space为符号
SPACE_INDEX=0 #0为space索引
FIRST_INDEX=ord('a')-1
#将稀疏矩阵的字向量转成文字
def sparse_tuple_to_texts_ch(tuple,words):
    indices=tuple[0]
    values=tuple[1]
    results=['']*tuple[2][0]
    for i in range(len(indices)):
        index=indices[i][0]
        c=values[i]

        c=' ' if c==SPACE_INDEX else words[c]
        results[index]=results[index]+c
    return results
#将密集矩阵的字向量转成文字
def ndarray_to_text_ch(value,words):
    results=''
    for i in range(len(value)):
        results+=words[value[i]]
    return results.replace('`',' ')




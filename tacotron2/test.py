import torch
from hparams import hparams
from model import Tacotron2
import os 
import soundfile as sf
import numpy as np
import librosa
from preprocess import pinyin_2_phoneme
from xpinyin import Pinyin

def text2speech(model,para,coded_text_in,device):
    
    text_in = torch.from_numpy(coded_text_in)
    text_in = text_in.unsqueeze(0).to(device)
    
    # 进行解码
    with torch.no_grad():
        eval_outputs = m_model.inference(text_in)
        mel_out = eval_outputs[1]
        mel_out = mel_out.squeeze(0)
        mel_out = mel_out.cpu().detach().numpy() 
    
    print(mel_out.shape)   
    # 加载统计信息
    file_static = os.path.join(para.path_fea,'static.npy')
    static_mel = np.load(file_static,allow_pickle=True) 
    
    mean_mel = np.float64(static_mel[0])
    std_mel =  np.float64(static_mel[1])
    
    # 反正则
    generated_mel =  mel_out* std_mel + mean_mel
    
    # 进行解码
    inv_fbank = librosa.db_to_power(generated_mel)
    inv_wav = librosa.feature.inverse.mel_to_audio(inv_fbank,
                                           sr =para.fs,
                                           n_fft = para.n_fft,
                                           win_length = para.win_length,
                                           hop_length = para.hop_length,
                                           fmin= para.fmin, 
                                           fmax= para.fmax)
                                           
    return inv_wav
    

def generate_text_code(words,dic_phoneme):
    
    new_words = words.replace('#','')
 
    new_words = ''.join([i for i in new_words if not i.isdigit()])
    p = Pinyin()
    out_pinyin = p.get_pinyin(new_words,' ',tone_marks='numbers')
    sent_phonemes = pinyin_2_phoneme(out_pinyin, words)
    print(sent_phonemes)
    coded_text = [ dic_phoneme[phonemes] for phonemes in sent_phonemes.split()]
    coded_text.append(dic_phoneme['~']) # 添加eos
    return coded_text
    

if __name__ == "__main__":
    # 加载相关参数
    para = hparams()
    
    
    device = torch.device("cuda:0")
    n_model = 24000
    m_model_all = torch.load(os.path.join('save2',str(n_model),'model.pick'))
    
    m_model = Tacotron2(para)
    m_model.to(device)
    m_model.load_state_dict(m_model_all['model'])
    
    m_model.eval()
    
    path_save = os.path.join('eval',str(n_model))
    os.makedirs(path_save,exist_ok = True)
    
    
    
    #输入文字        

    words = "该车#1将#1三人#1撞倒后#3在#1大雾中#1逃窜#4"
    coded_text = generate_text_code(words,para.dic_phoneme)
  
    # 生成语音
    wav_out = text2speech(m_model,para,np.array(coded_text),device)
    wav_out = wav_out/max(wav_out)    
    sf.write(os.path.join(path_save,'2.wav'),wav_out,para.fs)
            
    
   
    
    files = np.loadtxt("scp/test.scp",dtype = 'str',delimiter = '|')    
    
    file_ids = files[:,0].tolist()
    index_phones = files[:,2].tolist()
    
    for file_id,index_phone in zip(file_ids,index_phones):
        text_in = [int(id) for id in index_phone.split()]
        print("Generate speech %s.wav"%(file_id))
        wav_out = text2speech(m_model,para,np.array(text_in),device)
        wav_name = os.path.join(path_save,file_id+'.wav')
        wav_out = wav_out/max(wav_out)
        sf.write(wav_name,wav_out,para.fs)
    
    
    
    
    
    
    
    
    
    
    
    
    


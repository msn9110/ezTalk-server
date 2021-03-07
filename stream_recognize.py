import os
import wave
import numpy as np

os.environ['to_print_results'] = '0'
from config import get_settings
from create_syllable import load_model


*_, settings = get_settings()

path = '/home/hungshing/FastData/ezTalk/users/msn9110/voice_data/sentence-test/目前在做多人語音辨識研究/20201121-173551.wav'

input_name = 'decoded_sample_data:0'
model_mode, [model, *_] = load_model(settings['model_settings'])
if model_mode == 1:
    # 打开WAV文档
    f = wave.open(path, "rb")
    # 读取格式信息
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    # 读取波形数据
    str_data = f.readframes(nframes)
    f.close()

    size = int(framerate * settings['model_settings']['__clip_duration'])
    step = int(framerate * 0.05)

    # 将波形数据转换为数组
    wave_data = np.fromstring(str_data, np.int16)
    wave_data = np.array(wave_data, np.float) / 32768
    wave_data = np.reshape(wave_data, [nframes, 1])
    for i in range(0, nframes - size, step):
        syllable_list, score_list = model.label_wav(wave_data[i:i + size,], input_name=input_name)
        print(syllable_list[0])

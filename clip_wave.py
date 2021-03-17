import numpy as np
import wave
import re
import os
import filter as myfilter


def get_abs_mean(data):
    return max(1.0, sum(abs(data)) / len(data))


class WaveClip:
    def __init__(self, path, toFilter=False):
        self.path = path
        spiltArr = str(path).split('/')
        self.name = spiltArr[len(spiltArr) - 1]
        #print(self.name)
        # 打开WAV文档
        f = wave.open(path, "rb")
        # 读取格式信息
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        params = f.getparams()
        self.params = params
        self.nchannels, self.sampwidth, self.framerate, self.nframes = params[:4]

        # 读取波形数据
        str_data = f.readframes(self.nframes)
        f.close()

        # 将波形数据转换为数组
        wave_data = np.fromstring(str_data, np.int16)
        wave_data.shape = -1, 1
        self.wave_data = wave_data.T[0]
        if toFilter:
            lowcut, highcut = 100, 3500
            self.wave_data = myfilter.butter_bandpass_filter(self.wave_data,lowcut,highcut,self.framerate,order=6)

    def randomClip(self, duration_sec=1.0):
        wave_data = self.wave_data
        duration_size = int(self.framerate * duration_sec)
        if len(wave_data) <= duration_size:
            return np.append(wave_data, np.random.randint(-32, 33, size=duration_size - len(wave_data), dtype=np.int16))
        else:
            max_num = self.nframes - duration_size
            offset = np.random.randint(0, max_num)
            return wave_data[offset:offset + duration_size]

    def getparams(self):
        return self.params

    def clipWithNoise(self, noiseDir, duration_sec=1.0, shift_size=10, scale=1):
        clip_wave = clipWave(self.wave_data, self.framerate, duration_sec, shift_size)
        tmp = np.array(clip_wave, dtype=np.int32)
        files = list(filter(lambda file: file if file.endswith('.wav') else None, os.listdir(noiseDir)))
        num = np.random.randint(0, len(files))
        noisePath = noiseDir + '/' + files[num]
        noiseWave = WaveClip(noisePath)
        noiseClip = noiseWave.randomClip(duration_sec)
        tmp = tmp + scale*np.array(noiseClip, dtype=np.int32)
        for point in tmp:
            if point > 32767:
                point = 32767
            elif point < -32767:
                point = -32767
        return np.array(tmp, dtype=np.int16)

    def clipWave_toFile(self, output_path='', duration_sec=1.0, shift_size=10, add_noise=False,
                        randomly=False):
        if len(output_path) == 0:
            output_path = re.sub(self.name + '$', '', os.path.abspath(self.path)) + \
                         'clip-' + self.name

        if randomly:
            clip_wave = self.randomClip(duration_sec)
        else:
            if add_noise:
                output_path = re.sub('.wav$', '', output_path) + '-with_noise.wav'
                clip_wave = self.clipWithNoise('./_background_noise_', duration_sec, shift_size)
            else:
                clip_wave = clipWave(self.wave_data, self.framerate, duration_sec, shift_size)

        return write_wave(clip_wave, output_path, self.params)

    def is_perfect_wave(self, desire_duration=1.0, frame_sec=0.01):
        base_vol = 327.67
        abs_mean = get_abs_mean(self.wave_data)
        frame_sec = min(0.1, max(frame_sec, 0.01))
        stride_sec = frame_sec / 2
        stride = int(stride_sec * self.framerate)
        offset = int(frame_sec * self.framerate)
        my_frames = [self.wave_data[i:i+offset]
                     for i in range(0, self.nframes, stride)]

        margin = frame_sec * int(20 / round(frame_sec / 0.01)) \
            if frame_sec < 0.05 else frame_sec * max(0, 2 - round(frame_sec / 0.05))
        if frame_sec == 0.05:
            margin += frame_sec

        m_abs_means = list(map(get_abs_mean, my_frames))

        if len(m_abs_means) == 1:
            return False, 0.0

        # find an appropriate threshold
        s_m_abs_means = [a_m for a_m in sorted(set([max(a_m, 100.0) for a_m in
                                                    m_abs_means]))]
        delta_abs_means = [_2 / _1 for _1, _2 in zip(s_m_abs_means[:-1],
                                                     s_m_abs_means[1:])]
        choosing_vector = list(zip(s_m_abs_means[1:], delta_abs_means))
        tmp_abs_mean_threshold = max(choosing_vector, key=lambda it: it[1])[0] \
                    if len(choosing_vector) else max(base_vol * 2, abs_mean)
        th = tmp_abs_mean_threshold if base_vol * 2 <= tmp_abs_mean_threshold <= base_vol * 8 \
            else abs_mean

        # delete overlaps
        m_abs_means = m_abs_means[::2]

        activates = [1 if a_m >= th else 0
                   for a_m in m_abs_means]

        # last frame would not end the voice
        if activates[-1] and m_abs_means[-1] / th >= 1.25:
            return False, 0.0

        count = []
        duration_count = 0
        flag = False
        for i, act in enumerate(activates):
            if not flag and act:
                flag = True
                duration_count += 1
            elif flag and act:
                duration_count += 1
            elif flag and not act:
                duration_count += 3
                count.append(duration_count)
                flag = False
                duration_count = 0
        my_duration = max(count) * frame_sec if count else 0.0
        my_duration += margin
        return min(0.3, desire_duration) <= my_duration <= desire_duration - 0.1, my_duration


def make_noises(timeVector, min_scale=-1000, max_scale=1000, min_freq=100, max_freq=200,
                num_of_waves=5, rand_noise=True):
    scales = [12, -23, 7, -19, 22]#[663, 991, -615, 780, -593]
    freqs = [121, 189, 188, 140, 160]
    if rand_noise:
        scales = np.random.randint(min_scale, max_scale + 1, size=num_of_waves)
        freqs = np.random.randint(min_freq, max_freq + 1, size=num_of_waves)
        print(scales, '\n', freqs)
    signal = np.zeros(len(timeVector), dtype=np.float64)
    for i in range(num_of_waves):
        if i % 2 == 0:
            signal += scales[i] * np.sin(2 * np.pi * freqs[i] * timeVector)
        else:
            signal += scales[i] * np.cos(2 * np.pi * freqs[i] * timeVector)
    return np.array(signal, dtype=np.int16)


def clipWave(wave_data, framerate, duration_sec=0.8, shift_size=10):
    duration_size = int(framerate * duration_sec)
    if len(wave_data) <= duration_size:
        size = duration_size - len(wave_data)
        timeVector = np.arange(0, size) / framerate
        offsets = make_noises(timeVector, rand_noise=False)
        return np.append(wave_data, offsets)
    else:
        max_abs_mean = 0.0
        offset = -1
        for i in range(0, len(wave_data) - duration_size, shift_size):
            current_abs_mean = np.mean(np.abs(wave_data[i:i + duration_size]))
            if current_abs_mean > max_abs_mean:
                max_abs_mean = current_abs_mean
                offset = i
        return wave_data[offset:offset + duration_size]


def write_wave(wave_data, output_path, params):
    outfile = wave.open(output_path, 'wb')
    outfile.setparams(params)
    outfile.setnframes(len(wave_data))
    outfile.writeframes(wave_data)
    outfile.close()
    return os.path.abspath(output_path)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    framerate = 16000
    duration_sec = 0.8
    size = int(framerate * duration_sec)
    timeVector = np.arange(0, size) / framerate
    noise = make_noises(timeVector, rand_noise=False)
    print(np.mean(abs(noise)), min(noise), max(noise))
    #plt.ylim([-32767, 32767])
    plt.plot(timeVector, noise)
    plt.show()

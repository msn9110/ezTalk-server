import matplotlib.pyplot as plt
import numpy as np
import wave


class WavePlot:
    def __init__(self, path):
        f = wave.open(path, 'rb')
        params = f.getparams()
        # 读取格式信息
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        params = f.getparams()
        nchannels, sampwidth, self.framerate, self.nframes = params[:4]

        # 读取波形数据
        str_data = f.readframes(self.nframes)
        f.close()

        # 将波形数据转换为数组
        wave_data = np.fromstring(str_data, np.int16)

        wave_data.shape = -1, 1
        self.wave_data = wave_data.T[0]

    def isNoiseFile(self, threshold=0.5):
        wave_data = np.abs(self.wave_data) / 32767 / self.nframes
        if np.sum(wave_data) > threshold:
            return True
        else:
            return False

    def _createTimeVector(self):
        Fs = self.framerate  # sampling rate
        Ts = 1.0 / Fs  # sampling interval
        t = np.arange(0, self.nframes / Fs, Ts)  # time vector
        return t

    def _fft(self):
        y = self.wave_data
        n = self.nframes  # length of the signal
        k = np.arange(n)
        T = n / self.framerate
        frq = k / T  # two sides frequency range
        frq = frq[range(n // 2)]  # one side frequency range

        Y = np.fft.fft(y) / n  # fft computing and normalization
        Y = Y[range(n // 2)]
        return frq, Y

    def makeFig(self):
        frq, Y = self._fft()
        idx = np.arange(len(Y))
        arr = np.array([idx, np.abs(Y), Y]).T
        arr = sorted(arr, key=lambda keys: keys[1], reverse=True)

        t = self._createTimeVector()
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(t, self.wave_data, c='blue')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Amplitude')
        axes[1].plot(frq, abs(Y), 'r')  # plotting the spectrum
        axes[1].set_xlabel('Freq (Hz)')
        axes[1].set_ylabel('|Y(freq)|')
        fig.set_size_inches(12, 8)

        return arr, fig
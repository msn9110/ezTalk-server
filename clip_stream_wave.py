import wave
import numpy as np
import os
import re
from pypinyin_ext.zhuyin import convert_to_zhuyin
from clip_wave import clipWave, WaveClip


class StreamClip:

    def __init__(self, path):
        self.path = path
        splitArr = str(path).split('/')
        self.name = splitArr[-1]
        # print(self.name)
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

    def clipWave(self, duration_sec=0.8, absMeanThreshold=0.0,
                 scale=1, baseFrameSec=0.1, expected_num=-1,
                 recursive=False, debug=False):

        def get_abs_mean(frame):
            return max(1.0, np.mean(np.abs(frame)))

        base_vol = 327.67
        wave_data = list(self.wave_data)
        # arguments to affect clip performance
        myAbsMean = get_abs_mean(wave_data)
        AbsMeanThreshold = myAbsMean
        if 800.0 <= myAbsMean < 3000.0:
            AbsMeanThreshold = 700.0
        elif myAbsMean > 3000.0:
            AbsMeanThreshold -= (myAbsMean // 500) * 200.0
        else:
            AbsMeanThreshold = 500.0
        # Custom Threshold
        if absMeanThreshold > 0.0:
            AbsMeanThreshold = absMeanThreshold
            expected_num = -2
        ######################################

        frame_sec = baseFrameSec / scale
        frame_size = int(frame_sec * self.framerate)
        stride_sec = frame_sec / 2
        stride_size = int(stride_sec * self.framerate)

        frames = [wave_data[i:i + frame_size]
                  for i in range(0, self.nframes, stride_size)]

        abs_means = list(map(get_abs_mean, frames))

        if expected_num > 0 or expected_num == -2:
            # To calculate more accurate AbsMeanThreshold
            tmp_abs_mean_threshold = AbsMeanThreshold
            m_abs_means = [max(100.0, a_m) for a_m in abs_means]
            s_abs_means = [a_m for a_m in sorted(m_abs_means)]  # sort this list before
            delta_abs_means = [_2 / _1 for _1, _2 in zip(s_abs_means[:-1],
                                                         s_abs_means[1:])]
            choosing_vector = list(zip(s_abs_means[1:], delta_abs_means))
            if len(choosing_vector) > 1:
                tmp_abs_mean_threshold = max(choosing_vector,
                                             key=lambda it: it[1])[0]
                #tmp_abs_mean_threshold += np.sqrt(tmp_abs_mean_threshold) * 5
            AbsMeanThreshold = AbsMeanThreshold if tmp_abs_mean_threshold < 2 * base_vol \
                else tmp_abs_mean_threshold

        if debug:
            print('Threshold :', AbsMeanThreshold)

        # delete overlaps(strided frames)
        frames = frames[::2]
        abs_means = abs_means[::2]

        NumOfFrames = len(frames)

        # end of calculate of AbsMeanThreshold
        frameActs = []
        frameAbsMeanChanges = [] # To improve clip
        previousFrameAbsMean = 1.0 # initial
        for i in range(NumOfFrames):
            abs_mean = abs_means[i]
            act = abs_mean >= AbsMeanThreshold
            frameActs.append(act)
            frameAbsMeanChanges.append(abs_mean / previousFrameAbsMean)
            previousFrameAbsMean = abs_mean
            if debug:
                print(i, abs_mean)

        # Variable to merge frames into voice
        frameUsedTimes = [0] * NumOfFrames
        results = []
        voice = []
        frameCount = 0

        # threshold of frameCount of a voice to check if voice should add to the results
        FrameCountThreshold = 3 * scale

        # Control FLAG
        # To check if voice has achieved maximum amplitude of the voice
        # if yes, its absMeanChange will smaller than or equal to -1500
        # then set this flag to true
        isInDecayedState = False

        def includeFrame(pos, offset=0, MaxUsedCount=2):
            nonlocal voice, frameCount, frameUsedTimes
            index = pos + offset
            # avoid index out of range
            if 0 <= index < NumOfFrames:
                if frameUsedTimes[index] < MaxUsedCount:
                    frame = frames[index]
                    voice.extend(frame)
                    if frameActs[index]:
                        frameCount += 1
                    frameUsedTimes[index] += 1
            return voice, frameCount

        def findIndexOfEndFrame(pos, forward=True):
            offset = 1
            end = NumOfFrames
            if forward:
                offset = -1
                end = -1
            begin = pos + offset
            index = begin
            for i in range(begin, end, offset):
                p_a_m = abs_means[i - offset]
                c_a_m = abs_means[i]
                delta_rate = c_a_m / p_a_m
                change_flag = abs_means[i] <= 100.0 or delta_rate <= 1.025
                shouldInclude = not frameActs[i] and change_flag
                if shouldInclude:
                    index = i
                else:
                    break
            return index

        def addVoiceToResults(voice):
            nonlocal results, FrameCountThreshold
            if frameCount >= FrameCountThreshold:
                voice = clipWave(np.array(voice, dtype=np.int16), self.framerate,
                                 duration_sec=duration_sec)
                absMean = np.mean(np.abs(voice))
                if debug:
                    print(absMean)
                if absMean >= AbsMeanThreshold * 2 / 3:
                    results.append(voice)

            # reset voice
            reset()

        def reset():
            nonlocal voice, frameCount,\
                isInDecayedState
            voice = []
            frameCount = 0
            isInDecayedState = False


        for i in range(NumOfFrames):
            flag = frameActs[i]
            # detect frames[i] should be included
            if flag:
                if isInDecayedState and frameAbsMeanChanges[i] >= 1.2:
                    if debug:
                        print('early')
                        print(len(results)+1, i)
                    addVoiceToResults(voice)

                # include previous frame to prevent distortion
                if len(voice) == 0:
                    numOfPrevious = i - findIndexOfEndFrame(i, True)
                    for j in range(numOfPrevious, 0, -1):
                        voice, frameCount = includeFrame(i, -j)

                if not isInDecayedState and frameAbsMeanChanges[i] <= 0.8:
                    isInDecayedState = True

                # voice include current frame
                voice, frameCount = includeFrame(i)
            else:
                # include next frame to prevent distortion
                if frameCount > 0:
                    numOfNext = findIndexOfEndFrame(i, False) - i
                    for j in range(0, numOfNext, 1):
                        voice, frameCount = includeFrame(i, j)

                    if debug:
                        print(len(results) + 1, i)
                    addVoiceToResults(voice)
        if len(results) == expected_num or expected_num == -2 or recursive:
            return results
        else:
            return self.clipWave(duration_sec, absMeanThreshold, scale,
                                 baseFrameSec, -1, True, debug)


    def clipWave_toFile(self, msg, duration_sec=0.8, outputDir='.', force=False, debug=False,
                        show_num_detect=False, overwrite=False, frame_sec=0.01):
        if len(outputDir) == 0:
            outputDir = '.'

        zhuyin = convert_to_zhuyin(re.sub('[^\u4e00-\u9fa5]+', '', msg))
        all_path = []
        if len(re.sub('[^\u4e00-\u9fa5]+', '', msg)) > 0:
            all_exist = True
            for i in range(len(msg)):
                myDir = outputDir
                label = re.sub('[˙ˊˇˋ]$', '', zhuyin[i][0])
                myDir += '/' + label
                outputPath = myDir + '/' + 'clip-stream' + str(i + 1) + '-' \
                    + self.name
                all_path.append(os.path.abspath(outputPath))
                all_exist = all_exist and os.path.exists(outputPath)
            if not overwrite and all_exist:
                return len(zhuyin), all_path, \
                       [p for p in all_path
                        if WaveClip(p).is_perfect_wave(duration_sec, frame_sec)[0]]

        wanted_num = len(zhuyin)
        if force:
            wanted_num = -2
        myClip = self.clipWave( duration_sec=duration_sec,
                                debug=debug, expected_num=wanted_num)
        if show_num_detect:
            print('system slice a wave into ', len(myClip), 'waves')
        resultPath = []
        results = []
        if len(zhuyin) == len(myClip) or force:
            for i in range(len(myClip)):
                try:
                    outputPath = all_path[i]
                    myDir = os.path.split(outputPath)[0]
                except IndexError:
                    myDir = outputDir + '/tmp'
                    outputPath = myDir + '/' + 'clip-stream' + str(i + 1) + '-' + self.name

                os.makedirs(myDir, exist_ok=True)

                if not os.path.exists(outputPath) or overwrite:
                    outfile = wave.open(outputPath, 'wb')
                    outfile.setparams(self.params)
                    outfile.setnframes(len(myClip[i]))
                    outfile.writeframes(myClip[i])
                    outfile.close()
                resultPath.append(os.path.abspath(outputPath))
            if force:
                results = resultPath
            else:
                results = [p for p in resultPath
                           if WaveClip(p).is_perfect_wave(duration_sec, frame_sec)[0]]
        return len(zhuyin), resultPath, results

if __name__ == '__main__':
    s_c = StreamClip('test_stream/7-20181017-154759.wav')
    s_c.clipWave_toFile('有沒有辦法解決', 'test-stream', debug=True, show_num_detect=True,
                        overwrite=True, force=True)
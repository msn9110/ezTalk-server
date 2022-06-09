from config import get_settings, ROOT
import random
from glob import glob
import os
from clip_wave import WaveClip, write_wave, clipWave
import numpy as np


if __name__ == '__main__':
    unknown_dir = os.path.join(ROOT, 'unknown')
    os.makedirs(unknown_dir, exist_ok=True)
    *_, settings = get_settings('msn9110')
    vdp = settings['voice_data_path']
    background_noise = vdp['_background_noise_']

    n_files = list(glob(os.path.join(background_noise, '*.wav')))
    files = list(glob(os.path.join(vdp['training_set_root'], 'all', '*', '*.wav')))

    random.shuffle(n_files)
    random.shuffle(files)

    files = files[:200]
    n_files = n_files[:10]

    i = 1
    for f in n_files:
        wc = WaveClip(f)
        for _ in range(10):
            dst = os.path.join(unknown_dir, 'unknown_{0:d}.wav'.format(i))
            i += 1
            wc.clipWave_toFile(dst, randomly=True)

    for f in files:
        wc = WaveClip(f)
        dst = os.path.join(unknown_dir, 'unknown_{0:d}.wav'.format(i))
        flag, duration = wc.is_perfect_wave(1.0)
        if flag:
            i += 1
            params, framerate = wc.params, wc.framerate
            offset = int(framerate * duration / 2)
            wave_data = clipWave(wc.wave_data, framerate, duration)
            is_head = random.randint(0, 2)
            if is_head:
                wave_data = wave_data[:offset]
            else:
                wave_data = wave_data[offset:]
            size = framerate
            nframes = len(wave_data)
            start = random.randint(0, size - nframes)
            wave_data = np.pad(wave_data, [[start, size - start - nframes]], 'constant')
            write_wave(wave_data, dst, params)

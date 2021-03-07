import os
import re
import glob
from subprocess import call

from config import get_settings

*_, settings = get_settings()
model_settings = settings['model_settings']


def output_model(name, clip_duration,
                 mode_str='1', wanted_words=None, overwrite=False):

    if not wanted_words or not isinstance(wanted_words, str):
        return ''
    common_path = model_settings['__common_path'] + name
    my_dir = os.path.join(common_path, '_' + mode_str)
    checkpoint_prefix = os.path.join(my_dir, 'cmds', 'ckpt-')
    chks = list(map(lambda p: int(re.sub('^' + checkpoint_prefix, '', p).split('.')[0]),
                    glob.glob(checkpoint_prefix + '*.index')))
    checkpoint = checkpoint_prefix + str(max(chks))
    print(checkpoint)

    pb = os.path.join(common_path, 'model_labels', '_' + mode_str + '_model.pb')

    if not overwrite and os.path.exists(pb):
        return checkpoint

    cmd = 'python3 freeze.py '
    cmd += '--clip_duration_ms ' + str(int(clip_duration * 1000)) + ' '
    cmd += '--wanted_words ' + wanted_words + ' '
    cmd += '--start_checkpoint ' + checkpoint + ' '
    cmd += '--output_file ' + pb
    call(cmd, shell=True)
    for p in glob.glob(checkpoint_prefix + '*'):
        if re.match(checkpoint, p):
            continue
        os.remove(p)
    if os.path.exists(pb):
        return checkpoint
    return ''

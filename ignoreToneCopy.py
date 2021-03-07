from waveTools import *
import multiprocessing as mp
import os
import re
import shutil
import argparse


def task_add_tone_copy(record):
    all_tone = '˙_ˊˇˋ'
    for filepath in record[1]:
        _, name = os.path.split(filepath)
        clip = ''
        if 'clip-' in name:
            clip = 'clip-'
            name = re.sub(clip, '', name)
        tone, origin_name = re.split('[-]+', name, maxsplit=1)
        dirname = record[0]
        tone = int(tone)
        if tone > 5:
            print(filepath, tone)
            continue
        dirname += all_tone[tone]
        newdir = '_with_tone/' + dirname
        os.makedirs(newdir, exist_ok=True)
        new_path = newdir + '/' + clip + origin_name
        if not os.path.exists(new_path):
            shutil.copy(filepath, new_path)


def task_reduce_tone_copy(record):
    label = record[0]
    reduced_label = label.strip('˙ˊˇˋ_')
    if len(re.sub('[\u3105-\u3129]', '', reduced_label)) != 0:
        return
    tone = 1 if len(re.sub(reduced_label, '', label)) == 0 else '˙_ˊˇˋ'.index(label[-1])
    os.makedirs('_no_tone/' + reduced_label, exist_ok=True)
    for f in record[1]:
        name = f.split(record[0] + '/')[1]
        clip = ''
        if 'clip-' in name:
            clip = 'clip-'
            name = re.sub(clip, '', name)
        rpath = '_no_tone/' + reduced_label + '/' + clip + str(tone) + '-' + name
        rpath = '_no_tone/' + reduced_label + '/' + clip + name
        if not os.path.exists(rpath):
            shutil.copyfile(f, rpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tc', '--tone_copy',
                        action='store_true',
                        help="""\
                            To add labels with tone""")
    parser.add_argument('-p', '--path',
                        type=str,
                        default='.',
                        help="""\
                            specific path""")
    args, _ = parser.parse_known_args()
    os.chdir(args.path)
    records = get_all_waves('.', write_info=False)
    pool = mp.Pool()
    if not args.tone_copy:
        print('no tone copy')
        pool.map(task_reduce_tone_copy, records)
    else:
        print('tone copy')
        pool.map(task_add_tone_copy, records)
    pool.close()
    pool.join()
    print('finish')
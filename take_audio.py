import argparse
import re
from random import sample, randint
from math import log10
import create_syllable as cs
import clip_stream_wave as csw
from waveTools import get_all_waves

from config import get_settings

import adjust_relation as adjust
from pypinyin_ext import zhuyin


def randomPath(label):
    labels = set(all_waves.keys())
    if label in labels and all_waves[label]:
        wavs = all_waves[label]
    else:
        label = ls[randint(0, len(labels))]
        return randomPath(label)
    rand_int = sample(wavs,  1)
    wav_path = rand_int[0]
    return wav_path


def multi_syllables(label_list):
    wavs_path = []
    for label in label_list:
        temp_wav = randomPath(label)
        wavs_path.append(temp_wav)
    return wavs_path


def test(label_list, **kwargs):
    defaults = {
        'enable': False,
        'by_construct': True,
        'include_construct': True,
        'number': 5
    }

    for k in defaults.keys():
        if k not in kwargs:
            kwargs[k] = defaults[k]

    paths = multi_syllables(label_list)
    # print('[', '\n'.join(paths), ']')
    s_lists, res, candidates = cs.syllables_to_sentence(paths, settings, intelli_select=True, **kwargs)
    s_lists = [{sy: [i, round(10 + log10(max(1e-9, pr)), 2)]
                for i, (sy, pr) in enumerate(s_l, 1)}
               for s_l in s_lists]
    records = []
    for label, s_l in zip(label_list, s_lists):
        r = s_l[label] if label in s_l else [-1, 0]
        records.append(r)

    print('\n'.join(candidates[::-1]))
    print('final :', res[0])
    print(records)
    return True, candidates


def test2(paths, enable=False):
    print('[', '\n'.join(paths), ']')
    _, res, _ = cs.syllables_to_sentence(paths, settings, number=5, enable=enable,)

    print('final :', res[0])
    return res[0]


def test3(label_list, enable=False):
    paths = multi_syllables(label_list)
    # print('[', '\n'.join(paths), ']')
    sy_lists = cs.syllables_convert(paths, settings, number=-1, enable=enable)
    sy_lists = [{s: i for i, (s, _) in enumerate(l, start=1)}
                for l in sy_lists]
    res = []
    for l, sy_l in zip(label_list, sy_lists):
        rank = sy_l[l] if l in sy_l else -1
        res.append('{0:d}/{1:d}'.format(rank, len(sy_l)))
    print(res)

    return False, res


def recognize_file_to_sentence(path, by_construct=True, settings=None, user=None, **kwargs):
    if not settings:
        if not user:
            user = 'user'
        *_, settings = get_settings(user)

    model_settings = settings['model_settings']
    s_clip = csw.StreamClip(path)
    out_dir = path + '.clips'
    _, files, _ = s_clip.clipWave_toFile(path.split('/')[-2],
                                         duration_sec=model_settings['__clip_duration'],
                                         outputDir=out_dir, force=True,
                                         debug=False)
    *_, candidates = cs.syllables_to_sentence(files, settings, enable=True, num_of_stn=8,
                                              by_construct=by_construct,
                                              **kwargs)
    return candidates


def construct_stn(func, stn, label_list=None, enable=False, train=False, **kwargs):
    if type(label_list) is not list:
        label_list = [res[0].strip('˙ˊˇˋ') for res in zhuyin.convert_to_zhuyin(stn)]
    is_stn, res_stns = func(label_list, enable=enable, **kwargs)

    if train:
        adjust.adjustment(res_stns[0], stn, settings)

    if is_stn:
        return stn in res_stns

    return True


def get_recognized_syllable_lists(sentence, number=-1, settings=None, user=None, paths=None,
                                  **kwargs):
    global all_waves
    valid_sentence = re.sub('[^\u4e00-\u9fa5]', '1', sentence)
    if '1' in valid_sentence:
        raise ("Invalid Sentence")
    if not settings:
        if not user:
            user = 'user'
        *_, settings = get_settings(user)
    if not paths:
        voice_data_path = settings['voice_data_path']
        global_path = voice_data_path['testing_set']

        all_waves = dict(get_all_waves(global_path,
                                       write_info=False))
        label_list = [res[0].strip('˙ˊˇˋ') for res in zhuyin.convert_to_zhuyin(sentence)]
        paths = multi_syllables(label_list)

    sy_lists = cs.syllables_convert(paths, settings, number=number, enable=True, **kwargs)

    return settings['id'], settings, sentence, sy_lists


def recognize_wav_file(path, **kwargs):
    sentence = path.split('/')[-2]
    user = kwargs['user'] if 'user' in kwargs \
        else 'user'
    settings = get_settings(user)[-1] if 'settings' not in kwargs \
        else kwargs['settings']
    model_settings = settings['model_settings']
    s_clip = csw.StreamClip(path)
    out_dir = path + '.clips'
    _, files, _ = s_clip.clipWave_toFile('',
                                         duration_sec=model_settings['__clip_duration'],
                                         outputDir=out_dir, force=True,
                                         overwrite=True, debug=False)
    kwargs['settings'] = settings
    kwargs['user'] = user
    return get_recognized_syllable_lists(sentence, paths=files, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sentence', default='測試', type=str, help="""\
                                please enter a chinese sentence""")
    parser.add_argument('-para', '--paragraph', default='', type=str, help="""\
                                please enter a chinese paragraph""")
    parser.add_argument('-tp', '--train_path', default='', type=str, help="""\
                                please enter a train file path""")
    parser.add_argument('-p', '--path', default='', type=str, help="""\
                                    please enter a path""")
    parser.add_argument('-l', '--label', default='', type=str, help="""\
                                    please enter true label""")
    parser.add_argument('-st', '--syllable_test', action='store_true', help="""\
                                    test for syllable ranks""")
    parser.add_argument('-m', '--more_syllable', action='store_true', help="""\
                                to consider non-recorded syllables""")
    parser.add_argument('-a', '--adjust', action='store_true', help="""\
                                    to adjust list weights""")
    parser.add_argument('-fa', '--force_adjust', action='store_true', help="""\
                                   force to adjust list weights""")
    parser.add_argument('-u', '--user', default='', type=str, help="""\
                                        specify a user""")
    args, _ = parser.parse_known_args()

    user, (_, DATA_DIR, *_), *_, settings = get_settings(args.user)
    voice_data_path = settings['voice_data_path']
    model_settings = settings['model_settings']
    global_path = voice_data_path['testing_set']

    all_waves = dict(get_all_waves(global_path,
                                   write_info=False))
    labels = set(all_waves.keys())
    ls = list(labels)

    import os
    if args.path:
        if os.path.exists(args.path):

            s_clip = csw.StreamClip(args.path)
            out_dir = '/' + '/'.join(os.path.abspath(args.path).split('/')[:-1])
            _, files, _ = s_clip.clipWave_toFile('',
                                                 duration_sec=model_settings['__clip_duration'],
                                                 outputDir=out_dir, force=True,
                                                 overwrite=True, debug=False)
            pred = test2(files, args.more_syllable)
            if not re.sub('[\u4e00-\u9fa5]', '', args.label):
                adjust.adjustment(pred, args.label, settings)
            exit(3)
        exit(-3)
    if args.train_path and os.path.exists(args.train_path):

        with open(args.train_path) as f:
            stns = [s for s in f.read().split('\n')
                    if s and not re.sub('[\u4e00-\u9fa5]', '', s)]
            stns = dict(zip(stns, map(lambda stn: [res[0].strip('˙ˊˇˋ')
                                                   for res in zhuyin.convert_to_zhuyin(stn)],
                                      stns)))
            corrects = []
            import time
            s = time.time()
            try:
                while 1:
                    correct = {}

                    for stn, ls in stns.items():
                        res = construct_stn(test, stn, ls, False, args.adjust,
                                            include_construct=False,
                                            by_construct=True)
                        f = 1 if res else 0
                        correct.setdefault(len(stn), 0)
                        correct[len(stn)] += f
                    rate = sum(map(lambda it: it[0] * it[1], correct.items()))\
                           / sum(map(lambda s: len(s), stns.keys()))
                    corrects.append((rate, correct, sum(correct.values())))
                    print(corrects[-1])

                    if args.adjust:
                        yes = input('continue training([y]es): ')
                        yes = yes.lower()
                        if yes not in ['y', 'yes'] or rate >= 0.95:
                            break
                    else:
                        break
            except KeyboardInterrupt:
                pass
            finally:
                log = {i: {'acc': [correct[0], correct[-1], len(stns)],
                           'detail': correct[1]}
                       for i, correct in enumerate(corrects, 1)}
                print(log)
                print(time.time() - s, 'sec')

                import json
                p = os.path.join(DATA_DIR, 'training_results.json')
                with open(p, 'w') as log_f:
                    json.dump(log, log_f, sort_keys=True, indent=2,
                              ensure_ascii=False)

        exit(2)
    if args.paragraph:
        stns = re.split(r'[^\u4e00-\u9fa5]+', args.paragraph)
        print(stns)
        for stn in stns:
            if stn:
                construct_stn(test, stn, enable=args.more_syllable,
                              train=args.adjust)
        exit(0)
    sentence = args.sentence
    valid_sentence = re.sub('[^\u4e00-\u9fa5]', '1', sentence)
    if '1' in valid_sentence:
        raise("Invalid Sentence")
    else:
        func = test3 if args.syllable_test else test
        args.adjust = False if args.syllable_test else args.adjust
        construct_stn(func, sentence, enable=args.more_syllable,
                      train=args.adjust)

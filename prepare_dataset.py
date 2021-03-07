import os
import re
import glob
import shutil
import random
from subprocess import call
from multiprocessing import Pool

from waveTools import get_all_waves
from utils import read_json, write_json
from config import get_settings, default


def get_path(key):
    return voice_data_path[key]


def task_move_file(args):
    src, dst_dir = args
    label, name = src.split('/')[-2:]
    dst_dir += '/' + label
    os.makedirs(dst_dir, exist_ok=True)
    dst = dst_dir + '/' + name
    print(src, dst)
    shutil.move(src, dst)


def move_files_in_uploads_dir():
    def move_files(src_dir, dst_dir, regex):
        all_dirs = [a_dir for a_dir in os.listdir(src_dir) if os.path.isdir(src_dir + '/' + a_dir)]
        valid_labels = [a for a in all_dirs if len(re.sub(regex, '', a)) == 0]
        wavs = [os.path.abspath(p) for l in valid_labels
                for p in glob.glob(src_dir + '/' + l + '/*.wav')]
        args = zip(wavs, [dst_dir] * len(wavs))
        pool = Pool()
        pool.map(task_move_file, args)
        pool.close()
        pool.join()
        shutil.rmtree(src_dir, ignore_errors=True)
        os.makedirs(src_dir)

    uploads_dir = voice_data_path['uploads_dir']
    u_stn_dir = os.path.join(uploads_dir, 'sentence')
    if os.path.exists(u_stn_dir):
        print('wait for move uploads/sentence')
        move_files(u_stn_dir, stn_dir, '[\u4e00-\u9fa5]')
    u_single_pronounce_dir = os.path.join(uploads_dir, 'train')
    if os.path.exists(u_single_pronounce_dir):
        print('wait for move uploads/train')
        move_files(u_single_pronounce_dir, source_dir, '[\u3105-\u3129]{1,3}[˙_ˊˇˋ]{0,1}')
    # clear uploads dir
    for d in glob.glob(os.path.join(uploads_dir, '_clip_*')):
        shutil.rmtree(d)
    shutil.rmtree(os.path.join(uploads_dir, 'tmp'), ignore_errors=True)
    print('move done!')


def prepare_data_set(jp):
    testing_percentage = training_set_settings['testing_percentage']
    validation_percentage = min(30.0,
                                max(0.0,
                                    training_set_settings['validation_percentage']))
    testing_percentage = testing_percentage if 0.0 <= testing_percentage <= 30 \
        else 10
    if voice_data_path['prepared_testing_set']:
        testing_percentage = 0.0
    using_all = training_set_settings['using_all']
    max_num_of_files = training_set_settings['max_num_of_files']
    min_num_of_testing = training_set_settings['min_num_of_testing']

    train_dir = get_path('training_set_root')

    noise_dir = get_path('_background_noise_')

    # create empty train dir
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir, ignore_errors=True)
    os.makedirs(train_dir, exist_ok=True)

    # create empty test dir
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
        except OSError:
            os.unlink(test_dir)

    os.makedirs(test_dir, exist_ok=True)

    cmd1 = 'python3 waveTools.py -p ' + source_dir + ' -t ' + str(clip_duration)
    cmd2 = 'python3 waveTools.py -p ' + stn_dir + ' -s -t ' + str(clip_duration)

    train_files = []
    test_dict = {}
    if jp:
        m_dict = read_json(jp)
        all_wavefiles = m_dict['train']
        tst = m_dict['test']
        for l, l_files in tst.items():
            files = all_wavefiles[l] if l in all_wavefiles else []
            files.extend(l_files)
            all_wavefiles[l] = files
    else:
        # single pronounce wave
        call(cmd1, shell=True)
        clip_dir = os.path.join(source_dir, '_clip_' + str(clip_duration))

        info1 = get_all_waves(clip_dir, write_info=False)
        all_wavefiles = {l: files for l, files in info1}
        print('sum info1:', sum([len(all_wavefiles[k]) for k in all_wavefiles]))
        # stream pronounces wave
        call(cmd2, shell=True)
        clip_dir = os.path.join(stn_dir, '_clip_' + str(clip_duration))
        info2 = get_all_waves(clip_dir, write_info=False)

        # merge 2 info to all_wavefiles
        for l, l_files in info2:
            files = all_wavefiles[l] if l in all_wavefiles else []
            files.extend(l_files)
            all_wavefiles[l] = files

    f_maps = {f.split('/')[-1]: l
              for l, files in all_wavefiles.items()
              for f in files}

    f_maps_path = os.path.join(train_dir, 'f_maps.json')
    write_json(f_maps, f_maps_path)

    test_size = 0
    train_dict = {}
    tr_val = {'training': [], 'validation': [], 'testing': []}
    # prepare the files needed to be put in train, test dir
    for label, files in all_wavefiles.items():
        # ------------------------------------
        _dir = os.path.join(train_dir, 'all', label)
        for f in files:
            os.makedirs(_dir, exist_ok=True)
            os.symlink(f, os.path.join(_dir, f.split('/')[-1]))
        # ------------------------------------
        num_of_files = len(files)
        num_of_testing = max(num_of_files * testing_percentage // 100,
                             min_num_of_testing)
        num_of_testing = num_of_testing if num_of_testing <= num_of_files // 2 \
            else num_of_files // 2
        num_of_testing = int(num_of_testing)
        num_of_val = int(num_of_files * validation_percentage // 100)
        num_of_training = num_of_files - (num_of_testing + num_of_val)
        random.shuffle(files)

        tmp_train = files[:num_of_training]
        tmp_val = files[num_of_training: num_of_training + num_of_val]
        tmp_test = [label, files[num_of_training + num_of_val:]]

        if using_all:
            train_files.extend(tmp_train + tmp_val)

        else:
            if num_of_training <= max_num_of_files:
                train_files.extend(tmp_train + tmp_val)
            else:
                tmp_test[1].extend(tmp_train[max_num_of_files:])
                tmp_train = tmp_train[:max_num_of_files]
                train_files.extend(tmp_train + tmp_val)

        tr_val['training'].extend(tmp_train)
        tr_val['validation'].extend(tmp_val)
        tmp_train += tmp_val
        train_dict[label] = tmp_train
        test_dict[tmp_test[0]] = tmp_test[1]
        test_size += len(tmp_test[1])

        # make link tokens for test
        dst_dir = os.path.join(test_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        for f in tmp_test[1]:
            name = f.split('/')[-1]
            os.symlink(f, dst_dir + '/' + name)

        # make link tokens for train
        dst_dir = os.path.join(train_dir, '_0', label)
        os.makedirs(dst_dir, exist_ok=True)
        for f in tmp_train:
            name = f.split('/')[-1]
            os.symlink(f, dst_dir + '/' + name)
    os.symlink(f_maps_path, os.path.join(test_dir, 'f_maps.json'))

    tr_val['testing'] = list(glob.glob(os.path.join(test_dir, '*', '*.wav')))

    train_size = len(train_files)
    log = "test size : " + str(test_size) + '\ntrain size : ' + str(train_size)
    print(log)

    # from waveTools import classify
    # classify(wavefiles=train_files, dst_dir=train_dir, same_as_path=False)

    dsts = ['0', '1', '2', '3'][:1]
    # for noise files
    for a in dsts:
        class_dir = train_dir + '/_' + a
        cmd = 'python3 waveTools.py -i -p ' + class_dir
        os.makedirs(class_dir, exist_ok=True)
        os.symlink(f_maps_path, os.path.join(class_dir, 'f_maps.json'))
        n_dir = class_dir + '/_background_noise_'
        os.symlink(noise_dir, n_dir)
        call(cmd, shell=True)

    my_dict = \
        {
            'meta':
                {
                    'num_of_labels': len(train_dict),
                    'all':
                        {
                            'size': sum([len(files) for files in all_wavefiles.values()]),
                            'labels': [{l: len(files)} for l, files in
                                       sorted(all_wavefiles.items(),
                                              key=lambda it: len(it[1]))]
                        },
                    'train':
                        {
                            'size': train_size,
                            'labels': [{l: len(files)}
                                       for l, files in sorted(train_dict.items(),
                                                              key=lambda a: a[0])]
                        },
                    'test':
                        {
                            'size': test_size,
                            'labels': [{l: len(files)}
                                       for l, files in sorted(test_dict.items(),
                                                              key=lambda a: a[0])]
                        },
                },
            'train': train_dict, 'test': test_dict,
            'train_val': tr_val,
            'model_duration': str(clip_duration)
        }
    write_json(my_dict, os.path.join(train_dir, 'data.json'))
    write_json(train_dict, os.path.join(train_dir, 'train.json'))

    ################################
    if voice_data_path['prepared_testing_set']:
        my_testing_set = voice_data_path['prepared_testing_set']
        cmd = 'python3 waveTools.py -f -p ' + my_testing_set + ' -t ' + str(clip_duration)
        call(cmd, shell=True)
        shutil.rmtree(test_dir, ignore_errors=True)

        my_testing_set = os.path.join(my_testing_set, '_clip_' + str(clip_duration))
        info = get_all_waves(my_testing_set, write_info=False)
        f_maps = {p.split('/')[-1]: l
                  for l, fs in info
                  for p in fs}

        write_json(f_maps, os.path.join(my_testing_set, 'f_maps.json'))

        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
            except OSError:
                os.unlink(test_dir)

        os.symlink(my_testing_set, test_dir)
        ################################


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-jp', '--json_path', default='', help='''\
        using previous log json''')
    parser.add_argument('-m', '--move_uploads', action='store_true', help='''\
            only move files''')
    parser.add_argument('-u', '--user', default='', help='''\
            change user''')

    args, _ = parser.parse_known_args()

    default(args.user)
    _, (*_, VOICE_DATA_DIR), _, settings = get_settings(args.user)
    training_set_settings = settings['training_set_settings']
    training_default_values = settings['training_default_values']
    model_settings = settings['model_settings']
    data_path = settings['data_path']
    voice_data_path = settings['voice_data_path']
    clip_duration = settings['clip_duration']

    root = VOICE_DATA_DIR

    source_dir = get_path('source')
    stn_dir = get_path('sentence')
    test_dir = get_path('testing_set')

    move_files_in_uploads_dir()
    if args.move_uploads:
        cmd_1 = 'python3 ignoreToneCopy.py -p ' + source_dir + ' &&\
          python3 waveTools.py -i -p ' + source_dir + '/_no_tone'
        cmd_2 = 'python3 waveTools.py -p ' + stn_dir + ' -s -t ' + str(clip_duration)
        call(cmd_1, shell=True)
        call(cmd_2, shell=True)
        exit(0)
    prepare_data_set(args.json_path)

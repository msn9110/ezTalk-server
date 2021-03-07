import re
import os
import shutil
import datetime
import argparse
from subprocess import call

import exec_test
import output_model

from utils import time_used
from config import get_settings, update_settings, set_pid, trigger_reloading, write_log, default


@time_used
def a_model_train(cmd):
    call(cmd, shell=True)


@time_used
def training(clip_duration=0.8, training_times='7800,2200',
             part_train=True, name=''):
    modes = [1, 2, 3] if part_train else [0]
    if args.pre_suf:
        modes = [4, 5]

    if not name:
        name = re.sub('-', '', str(datetime.datetime.now().date()))

    set_pid('executor')

    print(name)

    training_set_root = voice_data_path['training_set_root']
    common_path = model_settings['__common_path'] + name
    model_dir = os.path.join(common_path, 'model_labels')
    os.makedirs(common_path, exist_ok=True)
    times = sum([int(t) for t in training_times.split(',')])
    success = True
    mode = {1: 'tops', 2: 'mids', 3: 'bots', 4: 'pref', 5: 'suff'}
    opts = []
    ckpts = ' --ckpts '
    num_labels = []
    for i in modes:
        mode_str = str(i)
        if not part_train:
            mode_str = '0'
        training_set = training_set_root + '/_0'
        write_log('state', 'training,{0:d},{1:d},{2:d}'.format(i, times, len(modes)))

        my_dir = os.path.join(common_path, '_' + mode_str)

        summary_dir = os.path.join(my_dir, 'logs')
        train_dir = os.path.join(my_dir, 'cmds')

        cmd = 'python3 train_v1.py --mode ' + mode_str + ' '
        cmd += '--how_many_training_steps ' + training_times + ' '
        cmd += '--clip_duration_ms ' + str(int(clip_duration * 1000)) + ' '
        cmd += '--test_dir ' + voice_data_path['testing_set'] + ' '
        cmd += '--data_dir ' + training_set + ' '
        cmd += '--train_dir ' + train_dir + ' '
        cmd += '--summaries_dir ' + summary_dir + ' '

        if not args.debug:
            pass
            a_model_train(cmd)

        filename = 'labels.txt'
        with open(os.path.join(train_dir, filename)) as a:
            wanted_words = a.read()
            num_labels.append(str(len(wanted_words.split('\n'))))
            wanted_words = ','.join(wanted_words.split('\n'))

        ckpt = output_model.output_model(name, clip_duration, mode_str, wanted_words)
        success = success and bool(ckpt)

        write_log('step', '0')
        if success:
            if i:
                ckpts += ckpt + ','
            src = os.path.join(train_dir, 'labels.txt')
            dst = os.path.join(model_dir, '_{0:d}_labels.txt'.format(i))
            shutil.copy(src, dst)
        else:
            break

    model_name = name

    if success:
        # update config file
        new_settings = {'name': model_name}
        update_settings('model_settings', new_settings)

        exec_test.test('1', model_name, part_train=part_train)

        file = 'data.json'
        src = os.path.join(voice_data_path['training_set_root'],file)
        dst = os.path.join(common_path, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
        #exit()

        if part_train:
            '''
            src_dir = model_settings['__common_path'] + model_name
            dst_dir = model_settings['__common_path'] + \
                      '-'.join(model_name.split('-')[:-1] + ['ncs'])
            try:
                shutil.copytree(src_dir, dst_dir)
            except Exception:
                print('not copy', dst_dir)
            '''

            _set = os.path.join(training_set_root, '_0')

            if not args.debug:
                cmd = 'python3 make_train_json.py -m train -ts ' + _set
                if args.pre_suf:
                    cmd += ' -ps'
                call(cmd, shell=True)

            if not args.debug:
                cmd = 'python3 make_train_json.py -m test -ts ' + voice_data_path['testing_set']
                if args.pre_suf:
                    cmd += ' -ps'
                call(cmd, shell=True)

            # cmodel training
            os.chdir('cmodels')

            train_json = os.path.join(_set, 'train_data.json')
            test_json = os.path.join(voice_data_path['testing_set'], 'test_data.json')

            my_dir = os.path.join(common_path, 'cs')

            summary_dir = os.path.join(my_dir, 'logs')
            train_dir = os.path.join(my_dir, 'cmds')

            cmd = 'python3 trainer.py '
            cmd += '-p ' + train_json + ' -ts ' + test_json + ' '
            cmd += '--how_many_training_steps ' + str(times) + ' '
            cmd += '--train_dir ' + train_dir + ' '
            cmd += '--summaries_dir ' + summary_dir + ' '
            if args.pre_suf:
                cmd += ' -ps'
            a_model_train(cmd)

            import time
            time.sleep(3)

            import glob

            pb_path = os.path.join(common_path, 'model_labels', 'cs.pb')

            checkpoint_prefix = os.path.join(train_dir, 'ckpt-')
            chks = list(map(lambda p: int(re.sub('^' + checkpoint_prefix, '', p)
                                          .split('.')[0]),
                            glob.glob(checkpoint_prefix + '*.index')))
            checkpoint = checkpoint_prefix + str(max(chks))
            cmd = 'python3 freeze.py '
            cmd += '--start_checkpoint ' + checkpoint + ' '
            cmd += '--output_file ' + pb_path
            if args.pre_suf:
                cmd += ' -ps'
            call(cmd, shell=True)

            opts.append(ckpts[:-1])
            opts.append(' --label_counts {0:s} '.format(','.join(num_labels)))
            opts.append(' --cs ' + checkpoint)

            os.chdir('..')

            pb_path = os.path.join(common_path, 'model_labels', '_0_model.pb')

            cmd = 'python3 merge_models.py '
            cmd += ''.join(opts) + ' '
            cmd += '--clip_duration_ms ' + str(int(clip_duration * 1000)) + ' '
            cmd += '--output_file ' + pb_path

            call(cmd, shell=True)
            merged = os.path.exists(pb_path)

            src = os.path.join(train_dir, 'labels.txt')
            dst = os.path.join(model_dir, '_0_labels.txt')
            shutil.copy(src, dst)

            try:
                for p in glob.glob(checkpoint_prefix + '*'):
                    if re.match(checkpoint, p):
                        continue
                    os.remove(p)
            except:
                pass

            exec_test.test('1', model_name,
                           test_part=False, part_train=part_train, cs=merged)

        trigger_reloading()

    write_log('state', 'finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--syllable_train',
        action='store_true',
        help='Directly train syllable', )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Debug mode', )
    parser.add_argument(
        '-c', '--clip_duration',
        type=float,
        default=0.0,
        help='Expected duration in seconds of the wavs', )
    parser.add_argument(
        '-t', '--how_many_training_steps',
        type=str,
        default='',
        help='How many training loops to run', )
    parser.add_argument(
        '-n', '--name',
        type=str,
        default='',
        help='Custom model name', )
    parser.add_argument('-u', '--user', default='', help='''\
                change user''')
    parser.add_argument(
        '-ps', '--pre_suf',
        action='store_true',
        help='use pre suf')

    args, _ = parser.parse_known_args()

    default(args.user)

    *_, settings = get_settings(args.user)
    training_set_settings = settings['training_set_settings']
    training_default_values = settings['training_default_values']
    model_settings = settings['model_settings']
    data_path = settings['data_path']
    voice_data_path = settings['voice_data_path']

    if len(args.how_many_training_steps) == 0:
        args.how_many_training_steps = training_default_values['training_times']
    if args.clip_duration == 0.0:
        args.clip_duration = settings['clip_duration']
    flag = not args.syllable_train
    training(args.clip_duration, args.how_many_training_steps,
             part_train=flag, name=args.name)

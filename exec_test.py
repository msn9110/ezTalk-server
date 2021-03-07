import os
import argparse
from subprocess import call

from utils import time_used
from config import write_log, set_pid, get_settings, default

*_, settings = get_settings()
training_set_settings = settings['training_set_settings']
training_default_values = settings['training_default_values']
model_settings = settings['model_settings']
data_path = settings['data_path']
voice_data_path = settings['voice_data_path']

_test_set = voice_data_path['testing_set'] if 'testing_set' not in os.environ \
    else os.environ['testing_set']


def draw_all_results(name):
    for i in range(4):
        dir_path = model_settings['__common_path'] + name + \
                    '/_' + str(i)
        print('Draw results for :')
        print(dir_path)
        cmd = 'python3 draw_results.py -p ' + dir_path
        call(cmd, shell=True)


def call_task(cmd):
    call(cmd, shell=True)


@time_used
def test(r_suffix, name,
         test_part=False, part_train=True, cs=False):
    prefix = '' if part_train else 'a'
    write_log('state', 'testing')
    cmds = []
    modes = ['all', 'top', 'mid', 'bot']
    py_file = 'label_for_test_v7.py'

    for m in modes:
        cmd = 'python3 ' + py_file + ' -m ' + prefix + m + \
              ' -n ' + name + \
              ' -ts ' + _test_set + \
              ' -s ' + r_suffix
        cmds.append(cmd)
    if part_train and not cs:
        cmd = 'python3 label_wav_syllable_test.py ' + \
              ' -n ' + name + \
              ' -ts ' + _test_set + \
              ' -s ' + r_suffix
        cmds[0] = cmd
    if not test_part:
        cmds = cmds[:1]

    from multiprocessing import Pool
    try:
        pool = Pool()
        print('wait for join')
        pool.map(call_task, cmds)
        pool.close()
        pool.join()
        #draw_all_results(name, train_times, clip_duration)
    except:
        pass
    finally:
        print('done')
        write_log('state', 'finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--draw_results', action='store_true', help="""\
                            draw results in bar form""")
    parser.add_argument('-s', '--output_file_suffix', default='1', help="""\
                        suffix for output file e.g: xxx_(suffix).txt""")
    parser.add_argument('-d', '-n', '--model_name', default='', help="""\
                                model_settings['common_path']_(name)/model_labels/xxx.pb""")
    parser.add_argument('-ts', '--test_set', default='', help="""\
                                path for test set""")
    parser.add_argument('-a', '--syllable_train',
                        action='store_true',
                        help='Directly train syllable', )
    parser.add_argument('-c', '--cs',
                        action='store_true',
                        help='cs nn create syllable', )
    parser.add_argument('--part_test',
                        action='store_true',
                        help='part test', )
    parser.add_argument('-u', '--user', default='', help='''\
                change user''')

    args, _ = parser.parse_known_args()

    default(args.user)

    *_, settings = get_settings(args.user)
    training_set_settings = settings['training_set_settings']
    training_default_values = settings['training_default_values']
    model_settings = settings['model_settings']
    data_path = settings['data_path']
    voice_data_path = settings['voice_data_path']

    _test_set = voice_data_path['testing_set'] if 'testing_set' not in os.environ \
        else os.environ['testing_set']

    if args.test_set:
        _test_set = args.test_set
    # use settings.json for args default value
    if len(args.model_name) == 0:
        args.model_name = model_settings['name']
    r, d = args.output_file_suffix, args.model_name
    if args.draw_results:
        draw_all_results(d)
        exit(0)
    set_pid('executor')
    flag = not args.syllable_train

    test(r, d, test_part=args.part_test, part_train=flag, cs=args.cs)

import os
import argparse
from multiprocessing import Pool
from subprocess import call

from utils import time_used, write_json, read_json
from config import get_settings, write_log

*_, settings = get_settings()
training_set_settings = settings['training_set_settings']
training_default_values = settings['training_default_values']
model_settings = settings['model_settings']
data_path = settings['data_path']
voice_data_path = settings['voice_data_path']
train_dir = voice_data_path['training_set_root']

test_set = ''
train_set = os.path.join(train_dir, '_0')


@time_used
def test(mode):

    cmd = 'python3 label_for_test_v7.py -m ' + mode \
                                    + ' -ts ' + test_set + ' --prepare_train_json'
    call(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ts', '--test_set', default='', help="""\
                                path for test set""")
    parser.add_argument('-m', '--mode', default='train', help="""\
                                train or test""")
    parser.add_argument(
        '-ps', '--pre_suf',
        action='store_true',
        help='use pre suf')
    args, unknown = parser.parse_known_args()

    test_set = args.test_set if args.test_set else train_set
    modes = ['top', 'mid', 'bot']
    start = 0
    if args.pre_suf:
        modes = ['pref', 'suff']
        start = 3
    write_log('state', 'testing')
    pool = Pool()
    pool.map(test, modes)
    pool.close()
    pool.join()
    print('done')
    write_log('state', 'finish')

    result_path = [os.path.join(test_set, '_{0:d}_test.json'.format(i + 1))
                   for i in range(start, start + len(modes))]
    results = []
    for p in result_path:
        results.append(read_json(p))

    f_maps = read_json(os.path.join(test_set, 'f_maps.json'))

    keys = results[0].keys()
    train_dict = {k: [[_[k] for _ in results], f_maps[k]]
                  for k in keys}

    write_json(train_dict, os.path.join(test_set, args.mode + '_data.json'))

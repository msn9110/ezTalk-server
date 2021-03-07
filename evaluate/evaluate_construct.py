import os
import sys
import glob
import argparse
from math import ceil
from multiprocessing import Pool, cpu_count

os.environ['to_print_results'] = '0'
sys.path.insert(1, '..')

from utils import write_json
from utils import time_used
from take_audio import recognize_file_to_sentence
from clip_stream_wave import StreamClip
from config import get_settings


def task_clip(path, duration):
    s_clip = StreamClip(path)
    out_dir = path + '.clips'
    _ = s_clip.clipWave_toFile(path.split('/')[-2], duration_sec=duration,
                               outputDir=out_dir, force=True, debug=False)


@time_used
def clip_all(files, duration):
    args = zip(files, [duration] * len(files))
    with Pool() as pool:
        pool.starmap(task_clip, args)


@time_used
def task(method, files):

    result = {}
    count = 0
    ranking = 0
    total_char = 0
    total_err = 0

    for f in files:
        label, name = f.split('/')[-2:]
        total_char += len(label)
        candidate = recognize_file_to_sentence(f, settings=settings, n_gram_method=method,
                                               multiprocessing=False)
        candidates = {a: i for i, a in enumerate(candidate, 1)}
        r = 9 if label not in candidates else candidates[label]
        count += 1 - r // 9
        ranking += r
        error = 0 if label in candidates else len(label)
        if error:
            cmp = lambda _: 1 - int(_[0] == _[1])
            for predict in candidate:
                error = min(error, sum(map(cmp, zip(label, predict))))
                if error == 1:
                    break

        total_err += error
        result[name] = [r, error, error / len(label), candidate]

    total = {'acc': count / len(files),
             'avg_ranking': ranking / len(files),
             'error_words': total_err,
             'error_rate': total_err / total_char}
    return {'method': method, 'files_detail': result, 'evaluate_total': total}


@time_used
def evaluate(files, methods):
    if not methods:
        methods = [0]

    f_maps = dict([f.split('/')[-2:][::-1] for f in files])
    ncore = max(1, cpu_count() - 1)
    res = []

    print('start to test')

    for i in range(ceil(len(methods) / ncore)):
        methods_ = methods[i * ncore:(i + 1) * ncore]

        args = zip(methods_, [files] * len(methods_))

        with Pool() as pool:

            tmp = pool.starmap(task, args)
            res += [_ for _ in tmp]

    print('finish test')

    # temp merge
    d = {_['method']: {'files_detail': _['files_detail'],
                       'evaluate_total': _['evaluate_total']}
         for _ in res}

    # final merge
    results = {'evaluate_total': {k: d[k]['evaluate_total']
                                  for k in d.keys()},
               'files_detail': {n: [f_maps[n], {k: d[k]['files_detail'][n]
                                                for k in d.keys()}]
                                for n in f_maps.keys()}
               }

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--root_dir', default='',
                        help='root folder for labeled testing wavs.')
    parser.add_argument('-m', '--methods', default='0',
                        help='what n-gram methods do want to test ex: 0,1,2')
    parser.add_argument('-u', '--user', default='msn9110',
                        help='which user')
    parser.add_argument('-r', '--output_path', default='results.json',
                        help='where do you want to store testing result json')

    args, _ = parser.parse_known_args()

    user, *_, settings = get_settings(args.user)

    if user != args.user:
        raise ValueError('Invalid User')

    duration = settings['model_settings']['__clip_duration']

    if not args.root_dir:
        args.root_dir = os.path.join(settings['voice_data_path']['uploads_dir'], 'ttrain')

    root = args.root_dir
    print(root)
    files = list(glob.glob(os.path.join(root, '*', '*.wav')))

    clip_all(files, duration)

    final = evaluate(files, [int(_) for _ in args.methods.split(',') if _])
    output_path = args.output_path
    if not args.output_path.endswith('.json'):
        output_path += '.json'
    write_json(final, output_path)

import argparse
import os
import re, json
import multiprocessing as mp
import clip_wave as cw
import clip_stream_wave as csw
from label_for_test_v7 import get_syllable, _mode_dict

import matplotlib

matplotlib.use('Agg')
import draw_wave as dw

# GLOBAL
cores = 0
args = None
STREAM_MODE = False
suffix = ''
clip_dir = ''
clip_duration = 1.0


def show_percent(counter, total, num_of_steps=50, finished_msg='task has finished'):
    if num_of_steps < 1:
        num_of_steps = 50
    if counter % num_of_steps == 0 or counter == total or counter == 1:
        percent = 100 * counter / total
        print(' ' * 50, end='\r')
        print("{0:.2f}% has finished!".format(round(percent, 2)), end='\r')
    if counter == total:
        print('\n' + finished_msg)


def get_labels(path='.', is_sentence=False):
    regex = '[\u4e00-\u9fa5]' if is_sentence else '[\u3105-\u3129]{1,3}[˙_ˊˇˋ]{0,1}|no_label'
    all_dirs = [a_dir for a_dir in os.listdir(path) if os.path.isdir(path + '/' + a_dir)]
    valid_labels = [a for a in sorted(all_dirs) if len(re.sub(regex, '', a)) == 0]

    return valid_labels


def write_words(names, target_dir='.'):
    out_str = ''
    out_str2 = ''
    for name in sorted(names):
        out_str += name + ','
        out_str2 += name + '\n'
    out_str = re.sub(',$', '', out_str)
    with open(target_dir + '/labelsForCMD.txt', 'w') as f:
        f.write(out_str)
    with open(target_dir + '/labelsForPB.txt', 'w') as f:
        f.write('__unknown__\n')
        f.write('__silence__\n')
        f.write(out_str2)


def get_all_waves(path='.', write_info=True, is_sentence=False):
    root_path = os.path.abspath(path)
    os.makedirs(root_path, exist_ok=True)
    ls = get_labels(path, is_sentence or STREAM_MODE)
    all_path = [root_path + '/' + l for l in ls]
    import glob
    wavefiles = [[f for f in glob.glob(p + '/*.wav')] for p in all_path]
    # flatten
    files = []
    for _ in wavefiles:
        files.extend(_)
    information = zip(ls, wavefiles)
    if not write_info:
        return information
    with open(path + '/information.txt', 'w') as f:
        for rec in information:
            f.write(rec[0] + ' : ' + str(len(rec[1])) + '\n')

    return files, ls


def join_all(ps):
    for p in ps:
        # print(p.name, p.is_alive())
        p.join()
        # print(p.name + ' join')
    return True


def task_clip(num, wavefiles, counter, total, lock, q, error_q):
    paths = []
    error_files = []

    def handle_error_file(f_path, err_move=True):
        nonlocal counter, error_files
        error_files.append(f_path)
        pardir, fname = f_path.split('/')[-2:]
        if err_move:
            os.makedirs('_errorRAWFiles/' + pardir, exist_ok=True)
            os.rename(wavefiles[i], '_errorRAWFiles/' + pardir + '/' + fname)
        lock.acquire()
        if err_move:
            with open('errorRAWFiles.log', 'a') as file:
                file.write('_errorRAWFiles/' + pardir + '/' + fname + '\n')
        counter.value += 1
        show_percent(counter.value, total, finished_msg='clip done')
        lock.release()

    for i in range(num, total, cores):
        # To check the source file is corrupted previously
        if os.path.getsize(wavefiles[i]) <= 44:
            handle_error_file(wavefiles[i])
            continue
        # end of checking previously

        result_path = []
        arr = wavefiles[i].split('/')
        label, fname = arr[-2:]
        if not STREAM_MODE:
            clip = cw.WaveClip(wavefiles[i], toFilter=False)
            target_dir = clip_dir + '/' + label
            output_path = target_dir + '/clip-' + fname

            if not args.estimate:
                if not os.path.exists(output_path):
                    os.makedirs(target_dir, exist_ok=True)
                    clip.clipWave_toFile(output_path, duration_sec=clip_duration)

                # To check the source file is corrupted postly
                if os.path.getsize(output_path) <= 44:
                    os.remove(output_path)
                    handle_error_file(wavefiles[i])
                    continue
                    # end of checking postly

            is_perfect, wanted_duration = clip.is_perfect_wave(clip_duration,
                                                               args.estimate_length)

            if wanted_duration > 0.0:
                lock.acquire()
                with open('wanted_duration.json', 'a') as w_f:
                    w_f.write('  "{0:s}": {1:.2f},\n'.format(wavefiles[i], wanted_duration))
                lock.release()

            if is_perfect or args.force or args.estimate:
                result_path.append(output_path)
            else:
                os.remove(output_path)

        else:
            clip = csw.StreamClip(wavefiles[i])
            sentence = label
            expected, clips, files = clip.clipWave_toFile(sentence, clip_duration, clip_dir,
                                                          frame_sec=args.estimate_length)
            if len(files) != expected:
                for f in clips:
                    os.remove(f)
                handle_error_file(wavefiles[i], err_move=False)
                continue
            result_path.extend(files)

        paths.extend(result_path)
        lock.acquire()
        counter.value += 1
        show_percent(counter.value, total, finished_msg='clip done')
        lock.release()
    q.put(paths)
    error_q.put(error_files)
    # print('\n' + mp.current_process().name + ' done!')


def clip_all_waves(path='.'):
    wavefiles, names = get_all_waves(path)
    print('num of files need to clip :', len(wavefiles))
    if STREAM_MODE:
        print('Expected num of files :', sum([len(p.split('/')[-2])
                                              for p in wavefiles]))
    else:
        with open('wanted_duration.json', 'w') as w_f:
            w_f.write('{\n')
    counter = mp.Value('I', 0)
    lock = mp.Lock()
    ps = []
    total = len(wavefiles)
    q = mp.Queue()
    error_q = mp.Queue()
    for i in range(cores):
        p = mp.Process(target=task_clip, args=(i, wavefiles, counter, total, lock, q, error_q),
                       name='process ' + str(i + 1))
        ps.append(p)
    print('start to clip all waves')
    for p in ps:
        p.start()
    res = []
    error_files = []
    try:
        for _ in ps:
            res.extend(q.get())
            error_files.extend(error_q.get())
        q.close()
        error_q.close()
        join_all(ps)
        print('problem files :', len(error_files))

        if res:
            clip_dir = '/'.join(res[0].split('/')[:-2])
            f_maps = dict(map(lambda it: tuple(it.split('/')[-2:][::-1]), res))
            if os.path.exists(clip_dir):
                with open(os.path.join(clip_dir, 'f_maps.json'), 'w') as f:
                    json.dump(f_maps, f, ensure_ascii=False, indent=2, sort_keys=True)

        if not STREAM_MODE:
            with open('wanted_duration.json', 'r') as d_f:
                d_str = re.sub(',\n$', '}', d_f.read())
                du_d = json.loads(d_str)
                du_d = {'/'.join(k.split('/')[-2:]): v
                        for k, v in du_d.items()}
                durations = list(du_d.values())
                du_distr = {k: durations.count(k)
                            for k in set(durations)}
                du_d = {'files': du_d}
                du_d['distribution'] = du_distr
                avg = sum([d * n for d, n in du_distr.items()]) / sum(du_distr.values())
                std_dev = pow(sum([n * pow(d - avg, 2)
                                   for d, n in du_distr.items()]) /
                              sum(du_distr.values()), 0.5)
                du_d['a'] = ['{0:.2f}'.format(avg), '{0:.2f}'.format(std_dev),
                             '{0:.2f}'.format(avg + std_dev),
                             '{0:.2f}'.format(avg + 2 * std_dev)]
            with open('wanted_duration.json', 'w') as f:
                json.dump(du_d, f, ensure_ascii=False, sort_keys=True, indent=2)
            print('Not Clipped:', total - len(error_files) - len(res))
    except KeyboardInterrupt:
        print('\nStopped clip all waves')
        exit(-1)

    return res, names


def task_draw(num, wavefiles, counter, total, lock, fail_count):
    i = num
    import matplotlib.pyplot as plt
    while i < total:
        arr = wavefiles[i].split('/')
        dirname, fname = arr[-2:]
        fig_path = './_fig_' + suffix + '/' + dirname + '/' + fname + '.png'
        info_path = './_info_' + suffix + '/' + dirname + '/' + fname + '.info.txt'
        a_wave = dw.WavePlot(wavefiles[i])
        if a_wave.isNoiseFile(0.25):
            lock.acquire()
            with open('noise_' + suffix + '.log', 'a') as f:
                f.write(dirname + '/' + fname + '\n')
            lock.release()
        if not os.path.exists(fig_path) or not os.path.exists(info_path):
            try:

                freq, fig = a_wave.makeFig()
                if not os.path.exists(fig_path):
                    fig.savefig(fig_path)
                plt.close(fig)
                if not os.path.exists(info_path):
                    with open(info_path, 'w') as outfile:
                        for rec in freq:
                            info = str(rec[0]) + '(Hz) : ' + str(rec[1]) + ',' + str(rec[2]) + '\n'
                            outfile.write(info)
            except:
                lock.acquire()
                fail_count.value += 1
                os.makedirs('_errorRAWFiles/' + dirname, exist_ok=True)
                if os.path.exists('./_clip_' + suffix + '/' + dirname + '/' + fname):
                    os.remove('./_clip_' + suffix + '/' + dirname + '/' + fname)
                os.rename('./' + dirname + '/' + re.sub('^clip-', '', fname),
                          './_errorRAWFiles/' + dirname + '/' + re.sub('^clip-', '', fname))
                with open('error-files_' + suffix + '.log', 'a') as file:
                    file.write('_clip_' + suffix + '/' + dirname + '/' + fname + '\n')
                lock.release()

        lock.acquire()
        counter.value += 1
        show_percent(counter.value, total, finished_msg='draw done')
        lock.release()
        i += cores
        # print(mp.current_process().name + 'done!')


def draw_all_waves(to_clip=True, path='.'):
    names = get_labels(path)
    if to_clip:
        wavefiles, _ = clip_all_waves(path)
    else:
        wavefiles, _ = get_all_waves(path)
    total = len(wavefiles)
    if os.path.exists('noise_' + suffix + '.log'):
        os.remove('noise_' + suffix + '.log')
    print('total files :', total)
    for name in names:
        os.makedirs(path + '/_info_' + suffix + '/' + name, exist_ok=True)
        os.makedirs(path + '/_fig_' + suffix + '/' + name, exist_ok=True)
    counter = mp.Value('I', 0)
    fail_count = mp.Value('I', 0)
    lock = mp.Lock()
    ps = []
    for i in range(cores):
        p = mp.Process(target=task_draw, args=(i, wavefiles, counter, total, lock, fail_count),
                       name='process ' + str(i + 1))
        ps.append(p)
    print('start to draw all waves')
    for p in ps:
        p.start()
    try:
        join_all(ps)
        print('problem files :', fail_count.value)
        write_words(names)
        print(total - fail_count.value, ' files successfully draw!')
        print('All done')
    except KeyboardInterrupt:
        print('\nStopped draw all waves')
        exit(-1)


def task_classify(tuple_):
    path, dst_dir = tuple_

    def copy(src, dst):
        # shutil.copy(src, dst)
        os.symlink(src, dst)

    def insert_to_dict(my_dict, category, ch, p):
        if ch not in my_dict[category]:
            my_dict[category][ch] = [p]
        else:
            my_dict[category][ch].append(p)
        return my_dict

    ret_dict = {'_1': {}, '_2': {}, '_3': {}}
    label, filename = path.split('/')[-2:]

    for m in ['top', 'mid', 'bot']:
        s = get_syllable(label, mode=m)
        mode = _mode_dict[m]
        newdir = os.path.join(dst_dir, mode, s)
        os.makedirs(newdir, exist_ok=True)
        dst = os.path.join(newdir, filename)
        if not os.path.exists(dst):
            copy(path, dst)
        ret_dict = insert_to_dict(ret_dict, mode, s, path)

    return ret_dict


def classify(path='.', dst_dir='.', same_as_path=True, wavefiles=None):
    if same_as_path:
        dst_dir = path
    if wavefiles is None:
        wavefiles, _ = get_all_waves(path, write_info=True)
    data = zip(wavefiles, [dst_dir] * len(wavefiles))
    from multiprocessing import Pool
    print('wait for classify')
    pool = Pool()
    q = pool.map(task_classify, data)
    pool.close()
    pool.join()
    res = q[0]
    for a in q[1:]:
        for category in res.keys():
            d = a[category]
            for k, files in d.items():
                if k in res[category]:
                    res[category][k].extend(files)
                else:
                    res[category][k] = files
    res['category'] = {
        '_1': {'num_of_labels': len(res['_1']), 'labels': {k: len(res['_1'][k])
                                                           for k in res['_1'].keys()}},
        '_2': {'num_of_labels': len(res['_2']), 'labels': {k: len(res['_2'][k])
                                                           for k in res['_2'].keys()}},
        '_3': {'num_of_labels': len(res['_3']), 'labels': {k: len(res['_3'][k])
                                                           for k in res['_3'].keys()}},
    }
    print('classify done')

    return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    global cores, args, suffix, clip_dir, STREAM_MODE, clip_duration
    cores = mp.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clip_waves',
                        type=str2bool,
                        default='t',
                        help="""\
                            To tailoring waves""")
    parser.add_argument('-s', '--clip_stream_waves',
                        action='store_true',
                        help="""\
                                To tailoring stream waves""")
    parser.add_argument('-i', '--info',
                        action='store_true',
                        help="""\
                            To get information""")
    parser.add_argument('-f', '--force',
                        action='store_true',
                        help="""\
                                To force clip""")
    parser.add_argument('-d', '--draw_waves',
                        action='store_true',
                        help="""\
                            To draw waves""")
    parser.add_argument('-class', '--classify_waves',
                        action='store_true',
                        help="""\
                            To classify waves""")
    parser.add_argument('--add_noises',
                        action='store_true',
                        help="""\
                            To add noise to waves""")
    parser.add_argument('-t', '--clip_duration',
                        type=float,
                        default=1.0,
                        help="""\
                                duration for output WAVE""")
    parser.add_argument('-et', '--estimate_length',
                        type=float,
                        default=0.01,
                        help="""\
                                estimate frame second""")
    parser.add_argument('-p', '--path',
                        type=str,
                        default='.',
                        help="""\
                            specific path""")
    parser.add_argument('-ig', '--ignore_tones',
                        action='store_true',
                        help="""\
                            To ignore tones""")
    parser.add_argument('-e', '--estimate',
                        action='store_true',
                        help="""\
                                To estimate length""")
    args, _ = parser.parse_known_args()
    STREAM_MODE = args.clip_stream_waves
    clip_duration = args.clip_duration

    if args.info:
        path = args.path
        wavefiles, names = get_all_waves(path)
        write_words(names, path)
        print('total files :', len(wavefiles))
        return

    no_tone_dir = ''
    if args.ignore_tones:
        from ignoreToneCopy import task_reduce_tone_copy
        no_tone_dir = '/_no_tone'
        os.chdir(args.path)
        print('wait for no tone copy')
        records = get_all_waves('.', write_info=False)
        pool = mp.Pool()
        pool.map(task_reduce_tone_copy, records)
        pool.close()
        pool.join()
        print('copy done')
        os.chdir('..')

    os.chdir(args.path + no_tone_dir)
    if args.classify_waves:
        classify('.')
        return

    suffix = str(args.clip_duration)
    if args.add_noises:
        suffix += '_with_noises'
    clip_dir = os.path.abspath('./_clip_' + suffix)

    if args.draw_waves:
        if not args.clip_waves:
            suffix = 'RAW'
            if args.add_noises:
                suffix += '_with_noises'
        draw_all_waves(to_clip=args.clip_waves)
    else:
        wavefiles, names = clip_all_waves()
        write_words(names)
        print('total files :', len(wavefiles))


if __name__ == '__main__':
    main()

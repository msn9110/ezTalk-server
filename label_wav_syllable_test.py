import re
import glob
import argparse
from math import sqrt

from utils import write_json
from config import get_settings, write_test_log, on_finish_test

*_, settings = get_settings()
training_set_settings = settings['training_set_settings']
training_default_values = settings['training_default_values']
model_settings = settings['model_settings']
data_path = settings['data_path']
voice_data_path = settings['voice_data_path']
args = None


def get_cdf(rankings, max_i, i_stop_at_v=95.0, no_cdf=False):
    total = len(rankings)
    distr = {r: rankings.count(r) / total * 100
             for r in range(1, max_i + 1)}
    cdf = [0.0]
    _i = 0
    for i in range(max_i):
        cdf.append(cdf[i] + distr[i + 1])
        if not _i and cdf[-1] >= i_stop_at_v:
            _i = i + 1

    if not _i:
        _i = max_i

    if no_cdf:
        return _i
    cdf = [round(_, 2) for _ in cdf]
    return _i, cdf


def main():
    import os
    # environment variable setting
    os.environ['use_gpu'] = '1'
    os.environ['name'] = args.model_name
    os.environ['to_print_results'] = '0'

    from create_syllable import load_model, load_data, creating_syllables_for_test

    model_mode, models = load_model(model_settings, model_mode=3)
    if model_mode not in [2, 3]:
        exit(-1)
    valid, non = load_data(settings)

    _mode = 0

    # _clip name
    _type_name = '_0'
    _name = args.model_name
    _result_suffix = '_' + args.output_file_suffix
    _test_dataset = args.test_set

    _common_path = model_settings['__common_path'] + _name + '/' \
                   + _type_name
    os.makedirs(_common_path, exist_ok=True)

    # how many labels that needs to show 
    how_many_labels = [5, 4, 3, 2, 1]
    # result file path , name
    result_path = [_common_path + '/result_' + str(i) + _result_suffix + '.txt'
                   for i in how_many_labels]

    # global count of labels
    global_counts = [0, 0, 0, 0, 0]

    if not args.prepare_train_json:
        # create result files
        result_files = [open(p, 'w') for p in result_path]

    # partial results
    partial_results = [[], [], [], [], []]

    train_dict = {}
    rankings = []

    partial = {}
    if not args.prepare_train_json:
        wav_labels = [l for l in os.listdir(_test_dataset)
                      if os.path.isdir(_test_dataset + '/' + l) and
                      len(re.sub('[\u3105-\u3129]', '', l)) == 0]
    else:
        import json
        f = open(os.path.join(settings['training_set_root'], 'train.json'))
        train_dict = json.load(f)
        f.close()
        wav_labels = train_dict.keys()

    wav_labels = list(sorted(wav_labels))

    result_dict = {l: [] for l in wav_labels}
    my_files = []
    total = 0
    for wav_label in wav_labels:
        # get wav file
        files = list(sorted(glob.glob(_test_dataset + '/' + wav_label + "/*.wav"))) \
            if not args.prepare_train_json \
            else train_dict[wav_label]
        total += len(files)
        my_files.append(files)
    label_files = zip(wav_labels, my_files)
    steps = 0

    log_t = '{0:s}\n{1:d}\n{2:d}'.format(str(os.getpid()), total, steps)
    write_test_log(_mode, log_t)

    my_rankings = []
    num_predicts = 0
    for wav_label, wav_files in label_files:
        if len(wav_files) == 0:
            continue

        # get 5, 3, 1 labels count
        counts = [0, 0, 0, 0, 0]
        wav_rankings = []
        if not args.prepare_train_json:
            # one line : all files
            for f in result_files:
                f.write("#############---[%s]---#############\n" % wav_label)

        syllable_ = wav_label

        for _wav_file in wav_files:

            # wav
            wav = _wav_file
            # catch words and scores
            syllables_l = creating_syllables_for_test(wav, model_mode, models,
                                                      valid, non)
            sy_idx = {it[0]: i for i, it in enumerate(syllables_l[-1], 1)}
            num_predicts = len(sy_idx)
            ranking = sy_idx[syllable_] if syllable_ in sy_idx else -1
            rankings.append(ranking)
            wav_rankings.append(ranking)

            steps += 1
            if args.steps:
                print(' ' * 80, end='\r')
                print(steps, '/', total, end='\r')

            if args.prepare_train_json:
                result_dict[syllable_].extend([{word: str(score) for word, score
                                                in syllables_l[how_many_labels[2] - 1]}])
                continue

            for i in range(len(how_many_labels)):
                syllables = syllables_l[how_many_labels[i] - 1]
                # write the 5, 3, 1 labels result in txt
                result_files[i].write("[%s]\n" % wav)
                s_l = [word for word, _ in syllables]
                pos = -1
                if syllable_ in s_l:
                    counts[i] += 1
                    pos = s_l.index(syllable_) + 1
                result_files[i].write('%d / %d\n' % (pos, len(s_l)))
            if steps % 50 == 0 or steps == total:
                log_t = '{0:s}\n{1:d}\n{2:d}'.format(str(os.getpid()), total, steps)
                write_test_log(_mode, log_t)

        mu_ranking = sum(wav_rankings) / len(wav_files)
        std_dev_ranking = sqrt(
            sum(map(lambda r: pow(r - mu_ranking, 2), wav_rankings)) / len(wav_rankings))
        my_rankings.append((wav_label, round(mu_ranking, 2),
                            round(std_dev_ranking, 2),
                            get_cdf(wav_rankings, num_predicts, no_cdf=True),
                            len(wav_rankings)))
        # all len of wav_files
        num_of_files = len(wav_files)
        partial[syllable_] = {
            'amount': num_of_files,
            'correct_count':
                {'top ' + str(2 * i + 1): v
                 for i, v in enumerate(counts[::-1])},
            'percentage':
                {'top ' + str(2 * i + 1): '{0:.2f}%'.format(v / num_of_files * 100)
                 for i, v in enumerate(counts[::-1])}
        }
        for i in range(len(how_many_labels)):
            # Global
            global_counts[i] += counts[i]

            # write partial results in file
            partial_results[i].append("Partial(" + wav_label + ") : "
                                      + str(counts[i]) + "/" + str(len(wav_files))
                                      + "\n")

    if not args.prepare_train_json:
        for i in range(len(how_many_labels)):
            result_files[i].write("-----------------------------------\n")
            # write partial at the end
            for partial_result in partial_results[i]:
                result_files[i].write(partial_result)

            # percentage
            percentage = round((global_counts[i] / total), 4) * 100

            # Total results
            result_files[i].write("Total : " + str(global_counts[i]) + "/" +
                                  str(total) + ", percentage : " +
                                  str(percentage) + "%")

    for m in models:
        m.close()

    if args.prepare_train_json:
        path = os.path.join(settings['training_set_root'], '_0_train.json')
        write_json(result_dict, path)
        exit(0)

    my_dict = {
        'partial': partial,
        'total': {
            'amount': total,
            'num_of_categories': len(wav_labels),
            'correct_count':
                {'top ' + str(2 * i + 1): v
                 for i, v in enumerate(global_counts[::-1])},
            'accuracy':
                {'top ' + str(2 * i + 1): '{0:.2f}%'.format(v / total * 100)
                 for i, v in enumerate(global_counts[::-1])}
        }
    }
    my_rankings.sort(key=lambda it: (it[3], it[0]))
    max_ranking = max(rankings)
    _95, cdf = get_cdf(rankings, num_predicts)
    total_mu_ranking = sum(rankings) / total
    total_sigma_ranking = sqrt(sum(map(lambda it: pow(it - total_mu_ranking, 2), rankings)) / total)
    my_dict['ranking'] = \
    {
        'partial': my_rankings,
        'total': [total_mu_ranking, total_sigma_ranking,
                  {
                      'num_of_files': total,
                      'max_ranking': max_ranking,
                      'K': _95,
                      'cdf': cdf
                  }],
    }

    f_name = '/results.json' if _result_suffix == '_1' else '/results' + _result_suffix[1:] + '.json'
    write_json(my_dict, _common_path + f_name)

    for f in result_files:
        f.close()
    on_finish_test(_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', action='store_true', help="""\
                                show steps""")
    parser.add_argument('--prepare_train_json', action='store_true', help="""\
                                prepare_train_json""")
    parser.add_argument('-ts', '--test_set', default='', help="""\
                        path for test set""")
    parser.add_argument('-s', '--output_file_suffix', default='1', help="""\
                        suffix for output file e.g: xxx_(suffix).txt""")
    parser.add_argument('-d', '-n', '--model_name', default='', help="""\
                            model_settings['common_path']_(name)/_clip_(clip_duration)/ooo/xxx.pb""")
    args, unknown = parser.parse_known_args()

    # use settings.json for args default value
    if len(args.model_name) == 0:
        args.model_name = model_settings['name']
    if len(args.test_set) == 0:
        args.test_set = settings['testing_set']
    main()

import os
import re
import glob
import argparse
from math import sqrt

from config import get_settings, write_test_log, on_finish_test
from utils import write_json
from label_wav_syllable_test import get_cdf
from phoneme import get_syllable

*_, settings = get_settings()
training_set_settings = settings['training_set_settings']
training_default_values = settings['training_default_values']
model_settings = settings['model_settings']
data_path = settings['data_path']
voice_data_path = settings['voice_data_path']

mode = ''
true_mode = ''
args = None
_mode_dict = {
    'all': '_0',
    'top': '_1',
    'mid': '_2',
    'bot': '_3',
    'pref': '_4',
    'suff': '_5',
    'aall': '_0',
    'atop': '_0',
    'amid': '_0',
    'abot': '_0',
}


def main():
    global true_mode
    os.environ['to_print_results'] = '0'
    from label_wav import AcousticModels

    steps = 0
    true_mode = mode.strip('a')
    if true_mode == 'll':
        true_mode = 'all'

    _mode = int(_mode_dict[true_mode][1])

    # _clip name
    _type_name = _mode_dict[mode]
    _name = args.model_name
    _result_suffix = '_' + args.output_file_suffix
    _test_dataset = args.test_set

    _common_path = model_settings['__common_path'] + _name

    # graph path
    graph = _common_path + '/model_labels/' + _type_name + '_model.pb'
    # labels path
    labels = _common_path + '/model_labels/' + _type_name + '_labels.txt'
    # load model
    model = AcousticModels([graph], [labels])
    # input name
    input_name = 'wav_data:0'
    # output name
    output_name = 'labels_softmax:0'
    # how many labels that needs to show
    how_many_labels = [5, 4, 3, 2, 1]

    _type_name = _mode_dict[true_mode]
    _common_path = model_settings['__common_path'] \
                   + _name + '/' + _type_name

    os.makedirs(_common_path, exist_ok=True)
    # result file path , name
    result_path = [_common_path + '/result_' + str(i) + _result_suffix + '.txt'
                   for i in how_many_labels]

    tops = [2 * i - 1 for i in how_many_labels] if true_mode == 'all' \
        else how_many_labels

    # global count of labels
    global_counts = [0, 0, 0, 0, 0]

    partial = {}

    if not args.prepare_train_json:
        # create result files
        result_files = [open(p, 'w') for p in result_path]

    # partial results
    partial_results = [[], [], [], [], []]

    # dict count
    count_dicts = [{}, {}, {}, {}, {}]

    # dict wav number
    num_wav_dict = dict()

    wav_labels = [l for l in os.listdir(_test_dataset)
                  if os.path.isdir(_test_dataset + '/' + l) and
                  len(re.sub('[\u3105-\u3129]|no_label', '', l)) == 0]
    wav_labels = list(sorted(wav_labels))
    result_dict = {}
    my_files = []
    total = 0
    rankings = []

    log_t = '{0:s}\n{1:d}\n{2:d}'.format(str(os.getpid()), total, steps)
    write_test_log(_mode, log_t)

    for wav_label in wav_labels:
        # get wav file
        files = list(sorted(glob.glob(_test_dataset + '/' + wav_label + "/*.wav")))
        total += len(files)
        my_files.append(files)
    label_files = zip(wav_labels, my_files)

    trans_dict = {}
    if not true_mode == 'all':
        for k in set(map(get_syllable, wav_labels)):
            trans_dict[k] = {}
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

        actual = get_syllable(wav_label, true_mode)

        for _wav_file in wav_files:

            # wav
            wav = _wav_file
            # catch words and scores
            words, scores = model.label_wav(wav, input_name, output_name)
            sy_idx = {w: i for i, w in enumerate(words, 1)}
            num_predicts = len(sy_idx)
            ranking = sy_idx[actual] if actual in sy_idx else -1
            rankings.append(ranking)
            wav_rankings.append(ranking)

            steps += 1
            if args.steps:
                print(' ' * 80, end='\r')
                print(steps, '/', total, end='\r')
            if steps % 50 == 0 or steps == total:
                log_t = '{0:s}\n{1:d}\n{2:d}'.format(str(os.getpid()), total, steps)
                write_test_log(_mode, log_t)

            if args.prepare_train_json:
                # f = lambda s: s + _mode_dict[true_mode][1] if re.match('[_a-z]+', s) else s
                result_dict[wav.split('/')[-1]] = [(w, str(pr)) for w, pr in zip(words, scores)]
                continue

            for i in range(len(how_many_labels)):
                # write the 5, 3, 1 labels result in txt
                result_files[i].write("[%s]\n" % wav)
                pos = 0
                pos_c = 0
                counted = False
                for word, prob in zip(words[:tops[i]], scores[:tops[i]]):
                    pos_c += 1
                    predict = get_syllable(word, true_mode)
                    result_files[i].write("predict: %s: %.5f\n" % (predict, prob))
                    if not true_mode == 'all' and tops[i] == 1:
                        from_ = trans_dict[actual]
                        from_.setdefault(predict, 0)
                        to_ = from_[predict] + 1
                        from_[predict] = to_
                        trans_dict[actual] = from_

                    if predict == actual and not counted:
                        counts[i] += 1
                        pos = pos_c
                        counted = True
                result_files[i].write("%d / %d \n" % (pos, tops[i]))

        mu_ranking = sum(wav_rankings) / len(wav_files)
        std_dev_ranking = sqrt(
            sum(map(lambda r: pow(r - mu_ranking, 2), wav_rankings)) / len(wav_rankings))
        my_rankings.append((wav_label, round(mu_ranking, 2),
                            round(std_dev_ranking, 2),
                            get_cdf(wav_rankings, num_predicts, no_cdf=True),
                            len(wav_rankings)))
        # all len of wav_files
        num_of_files = len(wav_files)
        partial[wav_label] = {
            'amount': num_of_files,
            'correct_count':
                {'top ' + str(i): v
                 for i, v in zip(tops, counts)},
            'percentage':
                {'top ' + str(i): '{0:.2f}%'.format(v / num_of_files * 100)
                 for i, v in zip(tops, counts)}
        }

        # sum of wav_files
        if actual not in num_wav_dict:
            num_wav_dict[actual] = len(wav_files)
        else:
            num_wav_dict[actual] += len(wav_files)

        for i in range(len(how_many_labels)):
            # Global
            global_counts[i] += counts[i]

            # write partial results in file
            partial_results[i].append("Partial(" + wav_label + ") : "
                                      + str(counts[i]) + "/" + str(len(wav_files))
                                      + "\n")

            # top how_many_labels[i] syllable
            if actual not in count_dicts[i]:
                count_dicts[i][actual] = counts[i]
            else:
                count_dicts[i][actual] += counts[i]

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

    model.close()

    if args.prepare_train_json:
        write_json(result_dict, os.path.join(_test_dataset, _mode_dict[mode] + '_test.json'))
        exit(0)

    d = \
        {
            k: {
                'amount': num_wav_dict[k],
                'correct_count': {
                    'top ' + str(i): c_d[k]
                    for i, c_d in zip(tops, count_dicts)
                },
                'percentage': {
                    'top ' + str(i): '{0:.2f}%'.format(c_d[k] / num_wav_dict[k] * 100)
                    for i, c_d in zip(tops, count_dicts)
                }
            }
            for k in num_wav_dict.keys()
        }

    my_dict = \
        {
            'partial': partial,
            'total':
                {
                    'amount': total,
                    'num_of_categories': len(wav_labels),
                    'correct_count':
                        {'top ' + str(i): v
                         for i, v in zip(tops, global_counts)},
                    'accuracy':
                        {'top ' + str(i): '{0:.2f}%'.format(v / total * 100)
                         for i, v in zip(tops, global_counts)}
                }
        }

    if not true_mode.startswith('a'):
        my_dict[_mode_dict[true_mode]] = d
        my_dict['transfer'] = trans_dict

    my_rankings.sort(key=lambda it: (it[3], it[0]))
    max_ranking = max(rankings)
    _95, cdf = get_cdf(rankings, num_predicts)
    total_mu_ranking = sum(rankings) / total
    total_sigma_ranking = sqrt(sum(map(lambda it: pow(it - total_mu_ranking, 2), rankings)) / total)
    my_dict['ranking'] = \
        {
            'partial': my_rankings,
            'total':
                [
                    total_mu_ranking, total_sigma_ranking,
                    {
                        'num_of_files': total,
                        'max_ranking': max_ranking,
                        'K': _95,
                        'cdf': cdf
                    }
                ],
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
    parser.add_argument('-m', '--mode', default='all', help="""\
                        all, top, mid, bot, pref, suff, aall, atop, amid, abot""")
    parser.add_argument('-ts', '--test_set', default='', help="""\
                            path for test set""")
    parser.add_argument('-s', '--output_file_suffix', default='1', help="""\
                        suffix for output file e.g: xxx_(suffix).txt""")
    parser.add_argument('-d', '-n', '--model_name', default='', help="""\
                            model_settings['common_path']_(name)/model_labels/xxx.pb""")

    args, unknown = parser.parse_known_args()
    mode = args.mode
    if mode not in _mode_dict:
        exit(-1)
    # use settings.json for args default value
    if len(args.model_name) == 0:
        args.model_name = model_settings['name']
    if len(args.test_set) == 0:
        args.test_set = voice_data_path['testing_set']
    main()

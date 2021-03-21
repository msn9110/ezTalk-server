import os
import re
from math import fabs
from math import log2 as log

import make_sentences_v3 as mks
from cmodels.label import CModel
from label_wav import AcousticModels
from utils import normalize, log_to_prob
from config import get_settings, general_data_path

input_name = 'file'
cs_input_name = 'input:0'
output_name = 'labels_softmax:0'

# the types of syllable
_types = ['_0', '_1', '_2', '_3', '_4', '_5']

_to_print_results = False if 'to_print_results' in os.environ \
                             and os.environ['to_print_results'] == '0' else True

if 'use_gpu' not in os.environ:
    # use cpu to run model
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def set_log_visible(visible=True):
    global _to_print_results
    _to_print_results = visible
    if not visible:
        os.environ['to_print_results'] = '0'


def load_model(model_settings, model_mode=0, enable_cmodel=True):
    if not model_mode and 'model_mode' in model_settings:
        model_mode = model_settings['model_mode']

    if 'enable_cmodel' in model_settings:
        enable_cmodel = model_settings['enable_cmodel']

    # setting model by environment variable
    #  from label wav syllable for test
    _key = 'name'
    _name = os.environ[_key] if _key in os.environ else model_settings[_key]

    # the path name
    _total_path = model_settings['__common_path'] + _name

    # the graph paths
    prefix = os.path.join(_total_path, 'model_labels')
    graphs_path = [os.path.join(prefix, _ + '_model.pb')
                   for _ in _types]
    cs_graph_path = os.path.join(prefix, 'cs.pb')

    # the labels paths
    labels_path = [os.path.join(prefix, _ + '_labels.txt')
                   for _ in _types]

    auto = model_mode + 1

    # MODEL LOADING
    try:
        if model_mode < 2:
            # syllable graph
            model = AcousticModels([graphs_path[0]], [labels_path[0]])
            model_mode = 1
            return model_mode, [model]

        if model_mode < 3:
            try:
                # loading 3 graphs
                model = AcousticModels(graphs_path[1:4], labels_path[1:4])
                model_mode = 2
            except:
                # loading 2 graphs
                model = AcousticModels(graphs_path[4:6], labels_path[4:6])
                model_mode = 4

            if enable_cmodel:
                try:
                    cmodel = CModel(cs_graph_path)
                    model_mode += 1
                    return model_mode, [model, cmodel]
                except FileNotFoundError:
                    return model_mode, [model]
            else:
                return model_mode, [model]

    except Exception:

        if auto < 3:
            return load_model(model_settings, auto + 1, enable_cmodel)

    return -1, []


def load_data(settings):
    data_path = settings['data_path']
    # all the syllables
    with open(general_data_path['valid_zhuyin']) as f:
        not_in_header = ['ㄟ', '']
        valid_syllables = [k for k in f.read().split('\n') if k not in not_in_header]

    with open(data_path['__recorded']) as f:
        global_syllables = f.read().split(',')
        non_recorded = [syllable for syllable in valid_syllables
                        if syllable not in global_syllables]
    return valid_syllables, non_recorded


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reset the Graph, this can avoid the graph conflict.
#
# wavs      : [ wav1_path , wav2_path ... ]
# how_many  : top number
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def syllable_creating(wavs, model):
    import time
    # initial parameters
    syllable_lists = []
    score_lists = []
    loop_round = 1

    # create
    for wav in wavs:
        if _to_print_results:
            print('==================================================')
            print('[target]' + wav)
            print('all', ':')
        start = time.time()
        # single syllable
        syllable_temps, score_temps = [], []
        _is_start = True
        for i in range(loop_round):
            syllable_temp, score_temp = model.label_wav(wav, mode=i, input_name=input_name)
            _is_start = _is_start and syllable_temp[0].startswith('_')
            if _to_print_results and i < loop_round - 1:
                print('------------------------------------')
            for syll, score in zip(syllable_temp, score_temp):
                if not syll.startswith('_'):
                    syllable_temps.append(syll)
                    score_temps.append(score)
        if not _is_start:
            temp = list(sorted(zip(syllable_temps, score_temps),
                               key=lambda it: it[1], reverse=True))
            syllable_lists.append([_[0] for _ in temp])
            score_lists.append([_[1] for _ in temp])
        if _to_print_results:
            print('time :', round(time.time() - start, 2), 'seconds')

    return syllable_lists, score_lists


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reset the Graph, this can avoid the graph conflit.
#
# wavs      : [ wav1_path , wav2_path ... ]
# how_many  : top number
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def word_creating(wavs, model, model_mode):
    import time

    if model_mode in [2, 3]:
        modes = ['top', 'mid', 'bot']
    else:
        modes = ['prefix', 'suffix']

    # initial parameters
    syllable_lists = []
    score_lists = []
    vectors = []

    for _ in modes:
        syllable_lists.append([])
        score_lists.append([])

    # create
    for wav in wavs:
        if _to_print_results:
            print('==================================================')
            print('[target]' + wav)
        vector = []
        start = time.time()
        # 3 parts single syllable
        for i, mode in enumerate(modes):
            if _to_print_results:
                print(mode, ":")
            syllable_list, score_list = model.label_wav(wav, mode=i, input_name=input_name)
            syllable_lists[i].append(syllable_list)
            score_lists[i].append(score_list)
            vector.append(list(zip(syllable_list, score_list)))
            if _to_print_results:
                print('------------------------------------------')
        vectors.append(vector)
        if _to_print_results:
            print('time :', round(time.time() - start, 2), 'seconds')

    return syllable_lists, score_lists, vectors


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# combine the top, middle, bottom single syllable to the
# possible syllable. if is 'no_label' that means no single syllable.
#
# top_l, mid_l, bot_l : single syllable list ['a', 'b', 'c']
# top_s, mid_s, bot_s : single syllable score list [a_score, b_score, c_score]
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def constructing_syllable(packet, global_syllables, non_recorded,
                          enable=False, split_results=False):
    (top_l, top_s), (mid_l, mid_s), (bot_l, bot_s) = packet

    all_possible_syllable = [[], [], [], [], [], []]
    all_possible_syllable_non_recorded = [[], [], [], [], [], []]

    idxes = range(max(len(top_l), len(mid_l), len(bot_l)))  # for enumerate
    for x, top_t, pt in zip(idxes[:len(top_l)], top_l, top_s):
        for y, mid_t, pm in zip(idxes[:len(mid_l)], mid_l, mid_s):
            for z, bot_t, pb in zip(idxes[:len(bot_l)], bot_l, bot_s):
                idx = min(5, max([x, y, z]))

                # using log instead of operator 'add'
                multiplication = pt * pm * pb
                total_score = fabs(log(max(multiplication, 1e-40)))
                syllable = re.sub('(no_label)([1-3])?', '', top_t + mid_t + bot_t)

                if syllable in global_syllables:
                    all_possible_syllable[idx].append((syllable, total_score))
                elif syllable in non_recorded:
                    all_possible_syllable_non_recorded[idx].append((syllable, total_score))

    if enable:
        # the following one line indicates the results contain non-recorded valid syllable
        for i in range(len(all_possible_syllable)):
            all_possible_syllable[i].extend(all_possible_syllable_non_recorded[i])

    for i in range(1, len(all_possible_syllable)):
        all_possible_syllable[i].extend(all_possible_syllable[i - 1])

    # prevent the null of syllables
    if len(all_possible_syllable[-1]) == 0:
        all_possible_syllable[-1].append(('ㄨㄛ', 0.0))

    if split_results:
        return all_possible_syllable
    else:
        return all_possible_syllable[-1]


def constructing_syllable2(packet, global_syllables, non_recorded,
                           enable=False, split_results=False):
    (pref_l, pref_s), (suff_l, suff_s) = packet

    all_possible_syllable = [[], [], [], [], [], []]
    all_possible_syllable_non_recorded = [[], [], [], [], [], []]

    idxes = range(max(len(pref_l), len(suff_l)))  # for enumerate
    for x, pref_t, pt in zip(idxes[:len(pref_l)], pref_l, pref_s):
        for y, suff_t, pb in zip(idxes[:len(suff_l)], suff_l, suff_s):
            idx = min(5, max([x, y]))

            # using log instead of operator 'add'
            multiplication = pt * pb
            total_score = fabs(log(max(multiplication, 1e-40)))
            syllable = pref_t[:-1] + suff_t if pref_t[-1] == suff_t[0] \
                    else pref_t + suff_t

            if pref_t[-1] in 'ㄧㄨㄩ' or suff_t[0] in 'ㄧㄨㄩ'\
                    and pref_t[-1] != suff_t[0]:
                continue

            if syllable in global_syllables:
                all_possible_syllable[idx].append((syllable, total_score))
            elif syllable in non_recorded:
                all_possible_syllable_non_recorded[idx].append((syllable, total_score))

    if enable:
        # the following one line indicates the results contain non-recorded valid syllable
        for i in range(len(all_possible_syllable)):
            all_possible_syllable[i].extend(all_possible_syllable_non_recorded[i])

    for i in range(1, len(all_possible_syllable)):
        all_possible_syllable[i].extend(all_possible_syllable[i - 1])

    # prevent the null of syllables
    if len(all_possible_syllable[-1]) == 0:
        all_possible_syllable[-1].append(('ㄨㄛ', 0.0))

    if split_results:
        return all_possible_syllable
    else:
        return all_possible_syllable[-1]


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create the syllables to prepare making sentence.
#
#
# wavs      : [ wav1_path , wav2_path ... ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def creating_syllables(wavs, settings, enable=False, log_process=False, use_transformer=False):
    syllables = []

    def print_syllables(s):
        sorted_list = s
        print()
        for a in normalize([l[:10] for l in sorted_list]):
            print([(_[0], round(_[1], 6)) for _ in a], '\n')

    model_mode, models = load_model(settings['model_settings'])
    if model_mode in [2, 3, 4, 5]:
        log_process = model_mode in [2, 4] or log_process
        syllable_lists, score_lists, vectors = word_creating(wavs, models[0], model_mode)
        num = len(vectors)
        packets = [[(syllable_lists[i][j], score_lists[i][j])
                    for i in range(len(syllable_lists))]
                   for j in range(num)]
        valid, non = load_data(settings)
        # create all possible syllables and make a sequence
        for i, packet in enumerate(packets):

            if model_mode in [3, 5]:
                syllable_l, score_l = models[1].get_labels(vectors[i],
                                                           cs_input_name,
                                                           output_name)
                first = syllable_l[0]
                if not first.startswith('_'):
                    syllables.append([(sy, sc)
                                      for sy, sc in zip(syllable_l, score_l)
                                      if not sy.startswith('_')])
                continue
            else:
                check_str = ''.join([sl[0] for sl, _ in packet])
                # maybe silence or unknown wave
                if 's' in check_str or 'u' in check_str:
                    continue
                func = constructing_syllable if model_mode == 2 \
                    else constructing_syllable2
                # find out all syllable
                possible_syllables = \
                    func(packet, valid, non, enable)

            if len(possible_syllables):
                # if in global_syllables
                syllables.append({syllable: score for syllable, score in possible_syllables})
    elif model_mode == 1:
        def f(s):
            return fabs(log(s)) if log_process else s

        syllable_lists, score_lists = syllable_creating(wavs, models[0])
        syllables = [[(syllable, f(score))
                      for syllable, score in zip(sy_, sc_)]
                     for sy_, sc_ in zip(syllable_lists, score_lists)]
    else:
        raise Exception('no model loaded')

    if len(syllables):
        if log_process:
            syllables = log_to_prob(syllables)

        # transformer adjust
        if use_transformer:
            try:
                from api_transformer import get_reranking_syllable_lists
                syllables = get_reranking_syllable_lists(settings['__id'], syllables)
                print('transform success')
            except:
                pass

        temp = [[(s, pr) for s, pr in sl if not s.startswith('_')]
                for sl in syllables]
        syllables = normalize(temp)

        if _to_print_results:
            print_syllables(syllables)

    for m in models:
        m.close()

    return syllables


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# call API function that we give wavs lists and produce the sentence list.
#
# enable indicates to contain non-recorded syllables
# wavs      : [ wav1_path , wav2_path ... ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def syllables_to_sentence(wavs, settings=None, number=5, enable=False,
                          num_of_stn=8, include_construct=False,
                          by_construct=False, intelli_select=False,
                          **kwargs):
    if not settings:
        *_, settings = get_settings()
    possible_lists = creating_syllables(wavs, settings, enable)
    possible_syll_d_l = [dict(l) for l in possible_lists]

    if not len(possible_lists):
        return possible_lists, None, None

    total_of_stn = 2 * num_of_stn

    by_construct = by_construct or len(possible_lists) == 1
    if by_construct:
        num_of_stn = total_of_stn
    else:
        pass#num_of_stn = max(1, num_of_stn // 2)

    # lists for construct sentence
    possible_lists = normalize([_l[:number * 10] for _l in possible_lists])

    sentences = mks.parse_sentence(possible_lists, settings, topk=num_of_stn,
                                   by_construct=by_construct, **kwargs)[::-1]

    top_stn = sentences[0]

    # filter sentence generated from method of comparison
    if intelli_select:
        # list of syllable rankings
        rankings_d_l = [{s: i
                         for i, s in enumerate([s for s, _ in l])}
                        for l in possible_lists]

        rankings = [round(sum([rankings_d_l[i][s]
                               if s in rankings_d_l[i]
                               else len(rankings_d_l[i])
                               for i, (_, s) in enumerate(stn[1:])]) / len(rankings_d_l), 2)
                    for stn in sentences]

        # filter with average used syllable ranking
        sentences = [stn for i, stn in enumerate(sentences) if rankings[i] < 19.0]

        if sentences:
            top_stn = sentences[0]

    num_stns = len(sentences)
    if not sentences:
        sentences.append(top_stn)

    stns = sentences[:3]
    candidates = [_[0] for _ in sentences]
    new_list = []

    # add syllable from sentence with the method of comparison
    for i, pl in enumerate(possible_lists):
        d = possible_syll_d_l[i]
        l = pl
        keys = dict(l)
        for sentence in stns:
            pron = sentence[i + 1][1]
            if pron not in keys:
                l.append((pron, d[pron]))
        new_list.append(l)

    if not by_construct and include_construct and num_stns < total_of_stn:
        p_lists = normalize(new_list)
        constructed = mks.parse_sentence(p_lists, settings, topk=total_of_stn - num_stns,
                                         by_construct=True, exclusive_stns=candidates, **kwargs)

        for stn_obj in constructed[::-1]:
            stn = stn_obj[0]
            if stn not in candidates:
                candidates.append(stn)

        # activate when intelli_select is true
        if not num_stns and len(candidates) > 1:
            compare_top = top_stn[0]
            del candidates[0]
            top_stn = constructed[-1]
            if len(candidates) > 8:
                candidates.insert(3, compare_top)
            else:
                candidates.append(compare_top)

    return new_list, top_stn, candidates


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# call API function that we give wavs lists and produce the sentence list.
#
# wavs      : [ wav1_path , wav2_path ... ]
# 7/26
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def syllables_convert(wavs, settings, number=-1, enable=False, **kwargs):
    possible_lists = creating_syllables(wavs, settings, enable=enable, **kwargs)
    if number == -1:
        number = 430
    return normalize([_l[:number] for _l in possible_lists])


def creating_syllables_for_test(wav, model_mode, models, valid, non):
    enable = True
    syllable_lists, score_lists, vectors = word_creating([wav], models[0], model_mode)
    syllables = []

    num = len(vectors)
    packets = [[(syllable_lists[i][j], score_lists[i][j])
                for i in range(len(syllable_lists))]
               for j in range(num)]

    # create all possible syllables and make a sequence
    for i, packet in enumerate(packets):
        if model_mode in [3, 5]:
            possible_syllables = models[1].get_labels(vectors[i],
                                                      cs_input_name,
                                                      output_name)
            syllables = possible_syllables
        else:
            func = constructing_syllable if model_mode == 2 \
                else constructing_syllable2
            # find out all syllable
            possible_syllables_l = \
                func(packet, valid, non, enable, split_results=True)

            for possible_syllables in possible_syllables_l:
                # if in global_syllables
                syllables.append(dict(possible_syllables))
    if model_mode is 3:
        syllables, scores = syllables
        syllables = list(zip(syllables, scores))
        number = 5
        return [syllables[:2 * i + 1]
                for i in range(number)] + [syllables]
    else:
        syllables = log_to_prob(syllables)
        return normalize([_l[: 2 * i + 1]
                          for i, _l in enumerate(syllables[:-1])]) + [syllables[-1]
                                                                      + [('_unknown_', 0.0)]]

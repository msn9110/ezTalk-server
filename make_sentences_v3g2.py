import os
import math
import time
from copy import deepcopy
from multiprocessing import Pool
from pypinyin_ext.zhuyin import convert_to_zhuyin as pinyin
from utils import normalize, process_probability, time_used, get_my_sigmoid_base
from verifier import n_gram_verify_bonus
from utils import read_json as open_file
from config import general_data_path

_to_print_results = False if 'to_print_results' in os.environ \
                             and os.environ['to_print_results'] == '0' else True

base = get_my_sigmoid_base(13, 0.75)
"""
2-gram 5
345gram 18
23

2-gram 3
345gram 7
10
"""


@time_used
def load_global_data():
    return open_file(general_data_path['word_default_weights'])


word_default_weights = load_global_data()


def load_data(data_path):
    which = ['header', 'semantic',
             'n_gram_bonus', 'n_gram_prob']

    return {k: open_file(data_path[k])
            for k in which}


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Find out the header words of all syllables.
# syllables : list of syllables
#             {'a':0.8532,'b':0.3236...}
# header    : path of header list
# [20180515 version]
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def header_parse(syllables, headers):
    possible_headers = []
    pos = 0
    # [('syllable_a',0.8532), ...]
    for syllable, syllable_pr in syllables:
        coeff = 10 + math.log(max(1e-9, syllable_pr), 10)
        pos_score = max(0, 10 - pos // 5)
        s = (coeff + pos_score) / 2
        # { word1:562 , word2:263 , word3:123... }
        header_list = headers[syllable]

        # find biggest frequency of word from same syllable
        for need_header, count in \
                list(sorted(header_list.items(),
                            key=lambda it: it[1]))[-2:]:
            # use log() to decrease difference
            score = math.log(count, 4) * coeff
            possible_headers.append(((need_header, syllable), score, pos, s))
        pos += 1

    return possible_headers


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parse the words and next syllable.
# word          : the previous word
# next_syllable : the next syllable
# semantic      : the list of relation about word and syllable
# word_freq     : the list of frequency of word
# [20180528 version]
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def parse_words(word, next_syllable, data):
    semantic = data['semantic']
    # semantic list
    if word in semantic:
        _dict = semantic[word]
    else:
        semantic.update({word: {}})
        _dict = {}

    # if word and syllable are exist in list
    _connect = 0

    # initial parameter
    need_3_word = []
    need_3_weight = []

    _top = 3
    # if word and syllable are exist
    if next_syllable in _dict:
        _dict_syll = _dict[next_syllable]
        *_, _dict_words = _dict_syll
        _connect = 1

        # find out the top 3 weight words
        sorted_dict_words = list(sorted(_dict_words.items(), key=lambda it: it[1],
                                        reverse=True))

        tmp_words = sorted_dict_words[:_top]

        # make the words and weight list
        for word, weight in tmp_words:
            need_3_word.append(word)
            need_3_weight.append(weight)

    # if not exist, we use words frequency list
    # to make a possible words.
    else:
        if next_syllable in word_default_weights:
            word, weight = max(word_default_weights[next_syllable].items(),
                               key=lambda a: a[1])
            need_3_word.append(word)
            need_3_weight.append(weight)

    return need_3_word, need_3_weight, _connect


def find_sentence_in_history(syllable_ls, stn_d, topk=8):
    sentence_len = len(syllable_ls)
    if sentence_len > 1:
        stn_d = {k: [v, [z[0].strip('˙ˊˇˋ') for z in pinyin(k)]]
                 for k, v in stn_d.items()
                 if len(k) == sentence_len and v > 0}
        if not len(stn_d):
            return False, None
        candidates = []
        syllable_ls_ = list(map(lambda s_l: dict(s_l), syllable_ls))
        syllable_rank_ls = [{it[0]: i for i, it in enumerate(s_l)}
                            for s_l in syllable_ls]
        for stn, values in stn_d.items():
            v, zhuyin = values
            # record = [[score(a), [count(sqrt(b)), bias(c^2)]]]
            record = [[1.0, [0, v]]]
            for i, z in enumerate(zhuyin):
                coeff = 1.0
                if z in syllable_ls_[i]:
                    experimental = 0 * (1 - syllable_rank_ls[i][z] / len(syllable_ls_[i]))
                    coeff += syllable_ls_[i][z] + experimental
                    record[0][1][0] += 1
                record[0][0] *= coeff
            record[0][0] += pow(record[0][1][0], 2) + math.sqrt(v)
            record.append(stn)
            record.append(zhuyin)
            candidates.append(record)
        candidates = list(sorted(candidates, key=lambda a: a[0][0]))
        best_stns = []
        for a in candidates[-topk:]:
            b = [a[0][0], a[0][1]]
            b.extend(zip(a[1], a[2]))
            if _to_print_results:
                print(b)
            stn = [a[1]]
            stn.extend(list(zip(a[1], a[2])))
            best_stns.append(stn)
        return True, best_stns
    return False, None


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parse the sequence of syllables , this is the important
# function for creating sentence.
# words     : the list of syllables
#       [{syllable:score , syllable:score , syllable:score },
#        {syllable:score , syllable:score , syllable:score },
#                           ......
#        {syllable:score , syllable:score , syllable:score },
#        {syllable:score , syllable:score , syllable:score }]
# [20180528 version]
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def parse_sentence(syllables_ls, settings, process_prob=False,
                   topk=8, by_construct=False, exclusive_stns=None,
                   **kwargs):
    """
    :param syllables_ls:
    :param settings:
    :param process_prob:
    :param topk:
    :param by_construct:
    :param exclusive_stns:
    :param kwargs: enable_forget(bool), n_gram_method(int)
    :return: list of top k sentence
    """
    data_path = settings['data_path']

    if not by_construct:
        stn_d = open_file(data_path['stn'])
        flag, res = find_sentence_in_history(normalize([l[:10] for l in syllables_ls]),
                                             stn_d, max(1, topk))
        if flag:
            return res
    data = load_data(data_path)
    syllables_ls = process_probability(syllables_ls, process_prob)
    # construct sentence
    s = time.time()
    all_sentences = make_possible_sentence(syllables_ls, data, **kwargs)
    best_possible_sentences = find_best_sentence(all_sentences, topk,
                                                 exclusive_stns)
    if _to_print_results:
        print(time.time() - s, 'sec')

    return best_possible_sentences


# find the best sentence
def find_best_sentence(all_sentences, topk=8, exclusive_stns=None):
    if type(exclusive_stns) == list and len(exclusive_stns):
        all_sentences = [a for a in all_sentences if a[1][-1] not in exclusive_stns]
    all_sentences = list(sorted(all_sentences, key=lambda s: s[0]))
    all_stn = []
    for a in all_sentences[-topk:]:
        stn = a[1][-1]
        del a[1][-1]
        if _to_print_results:
            print(a)
        a[0] = stn
        del a[1]
        all_stn.append(a)
    return all_stn


def get_relation_intensity(sentence, data, n=3):
    stn = sentence[2:]
    idx = len(stn) - 1
    intensity = 0.0
    coeff = 1.0
    w, syll = stn[-1]
    semantic = data['semantic']
    for j in range(2, n + 1):
        k = idx - j
        if k < 0:
            break

        prev_w, prev_s = stn[k]
        record = semantic[prev_w] if prev_w in semantic else {}

        decay = 0.9
        weight = 0.0
        if syll in record:
            record = record[syll]
            if w in record:
                weight = record[w]
        coeff *= decay
        intensity += coeff * weight
    return intensity


# task for construct sentence
def task_construct_sentence(a_stn, data, syllables_l, max_consider_num,
                            enable_forget, n_gram_method):
    prev_score, prev_word = a_stn[0], a_stn[-1][0]
    sentences = []

    length = len(a_stn) - 1

    group = 5
    for i, syllable_prob in enumerate(syllables_l):  # the main reason to construct all possibles
        syllable, prob = syllable_prob  # execute 1 time

        # probability transform
        coeff = 10 + math.log(max(1e-9, prob), 10)

        pos_score = max(0., 10 - i / group)

        o_s = s = (coeff + pos_score) / 2

        p_words, weights, is_connected = parse_words(prev_word, syllable, data)
        p_words, weights = p_words[:max_consider_num], weights[:max_consider_num]

        prev_avg_ranking = a_stn[1][2]
        alpha = a_stn[1][3]
        #alpha = 10 - prev_avg_ranking / group

        for p_word, weight in zip(p_words, weights):  # constant time 1 - 3 times
            # use list instead of deepcopy due to a_stn[2:] can be shared by reference,
            # and a_stn[0] is immutable
            sentence = list(a_stn)
            # only a_stn[1] cannot use same reference
            sentence[1] = deepcopy(a_stn[1])

            c_stn = sentence[1][-1]  # current sentence
            n_gram_score = n_gram_verify_bonus(c_stn, n_gram_method, d=data)

            sentence.append((p_word, syllable))
            sentence[1][-1] += p_word

            # - - - - - - calculation of score - - - - - -
            score = prev_score

            # relation intensity(RI) definition
            '''
            sigmoid(t) = 1 / (1 + pow(e, -t))
            '''

            th = 0.0
            if length >= 4:
                th = 13
                s = max(s, alpha)

            v = get_relation_intensity(sentence, data, 4) + is_connected * weight
            v *= s / 5
            v += n_gram_score * alpha / 5

            gate = 1 / (1 + math.pow(base, -(v - th))) if length >= 4 and enable_forget else 1.0
            dc = 0.7

            # decay of previous RI
            relation_intensity = gate * dc * sentence[1][0] + \
                                 is_connected * pow(weight, 0.5) * s / 2

            relation_intensity = round(relation_intensity, 2)

            score += round(math.pow(relation_intensity, 1.5), 4) #+ pow(n_gram_score * alpha / 10, 1.)

            score += round(weight * coeff, 4)# + pow(n_gram_score, 2)
            #score += min(729., pow(s, 3))

            # - - - - - - - - - - - - - - - - - - -
            sentence[1][1] += is_connected
            sentence[0] = score
            sentence[1][0] = relation_intensity
            sentence[1][2] = (prev_avg_ranking * (length - 1) + i) / length
            sentence[1][3] = (sentence[1][3] * (length - 1) + o_s) / length

            sentences.append(sentence)
            # - - - - - - - - - - - - - - - - - - -

    return sentences


def get_sentences(all_stn, data, syllables_l, max_consider_num, multiprocessing=True,
                  enable_forget=True, n_gram_method=4):
    if multiprocessing:
        size = len(all_stn)
        args = zip(all_stn, [data] * size,
                   [syllables_l] * size, [max_consider_num] * size,
                   [enable_forget] * size, [n_gram_method] * size)
        with Pool() as pool:
            all_sentences = [a_stn for _ in pool.starmap(task_construct_sentence, args)
                             for a_stn in _]
    else:
        all_sentences = [_ for a_stn in all_stn
                         for _ in task_construct_sentence(a_stn, data, syllables_l, max_consider_num,
                                                          enable_forget, n_gram_method)]

    return all_sentences


# we can adjust max_consider_num, the number can be 1, 2 or 3
def make_possible_sentence(syllables_ls, data,
                           max_consider_num=2, k_round_filter=True,
                           multiprocessing=True,
                           **kwargs):
    if _to_print_results:
        print('==================================================')
    sentence_len = len(syllables_ls)
    max_consider_num = \
        min(3, max(1, max_consider_num))
    # choose top k of score sentence every filter round
    top_k = 500

    # find header word
    all_sentences = [[score, [0, 0, pos, s, word[0]], word]
                     for word, score, pos, s in header_parse(syllables_ls[0], data['header'])]

    # first word has parsed, so starts from second index 1
    for i in range(1, sentence_len):

        syllables_l = syllables_ls[i]

        all_sentences = get_sentences(all_sentences, data, syllables_l, max_consider_num,
                                      multiprocessing and (sentence_len >= 15 or len(all_sentences) > 400),
                                      **kwargs)

        '''
        last_words = {a[-1][0] for a in all_sentences}
        stn_dict = {w: list(sorted([a for a in all_sentences if a[-1][0] == w],
                                   key=lambda it: it[0]))[-1:]
                    for w in last_words}
        all_sentences = []
        for _ in stn_dict.values():
            all_sentences.extend(_)
        '''

        # every filter round to choose top k sentence to avoid exponentially grows
        if k_round_filter or len(all_sentences) >= 5000:
            '''
            # tricky
            keys = {a[1][0] for a in all_sentences}
            stn_dict = {k: list(sorted([a for a in all_sentences if a[1][0] == k],
                                       key=lambda it: it[0]))[-1:]
                        for k in keys}
            all_sentences = []
            for _ in stn_dict.values():
                all_sentences.extend(_)
            '''

            # paper
            all_sentences = list(sorted(all_sentences, key=lambda stn: stn[0]))[-top_k:]

            # filter by avg ranking
            avg_ranking_th = max(15., 60. / i)
            all_sentences = [a for a in all_sentences if a[1][2] <= avg_ranking_th]

            # tricky
            score_th = all_sentences[-1][0] * 0.66
            relation_intensity_th = max(all_sentences, key=lambda it: it[1][0])[1][0] / 2
            all_sentences = [a for a in all_sentences if a[1][0] >= relation_intensity_th or a[0] > score_th]

    return all_sentences

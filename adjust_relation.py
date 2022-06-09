import math
import fcntl
from time import sleep

from utils import read_json as load_data
from utils import write_json as write_data
from n_gram_adjustment import n_gram_adjust
from pypinyin_ext.zhuyin import convert_to_zhuyin as pinyin


def sorted_dict(input_dict):
    sorted_list = [k for k, _ in
                   sorted(input_dict.items(), key=lambda it: it[1], reverse=True)]
    return sorted_list


def header_adjustment(word, syllable, equal, header_list, record):
    default = 2
    if syllable in header_list:
        # word_list = {'a':12,'b':34,'c':56 ... }
        word_list = header_list[syllable]

        if word in word_list:
            w_l = sorted_dict(word_list)
            ranking = w_l.index(word)

            # parameter, position and count
            count = 1

            r_word = record[syllable] if syllable in record else {}

            if word in r_word:
                # record, position and count
                pos = r_word[word][1]
                if not equal:
                    # increment count if ranking is equal to pos,
                    # else reset to 1
                    count = r_word[word][0] + 1 if pos == ranking else 1

            r_word.update({word: [count, ranking]})
            record.update({syllable: r_word})
            pos = ranking

            # compute offset of weight
            try:
                # if equal
                add_ = 2
                sub_ = 0
                if not equal:
                    print("pos:", pos, "count:", count)
                    # boyu's trick
                    # top 1 indicates ranking 0 then grows linearly
                    # ranking > 0, grows exponentially
                    add_ = math.pow(pos + 1, count) if pos else 16
                    # punish relative, for ranking 0 increase 15 = 16 - (0 + 1)^n
                    # every adjustment
                    sub_ = math.pow(pos + 1, count - 1)

                # weights adjustment
                # different from boyu: add 1 for modified header at least
                word_list[word] += max(1, add_)
                for k in word_list.keys():
                    word_list[k] = max(default, word_list[k] - sub_)
            except OverflowError:
                print('header update failed')

        else:
            # initial weight of word
            word_list.update({word: default})
    else:
        word_list = {word: default}
    header_list[syllable] = word_list
    print('header updated')

    return header_list, record


# word_prev and word
def words_adjustment_plus(word_prev, word, syllable, equal,
                          semantic_list, record,
                          relation_coeff=1.0):

    # default value start
    equal = int(equal)
    count = max(1.0, relation_coeff * 2)
    syllable_to_word = {word: count}
    record_list = {syllable: syllable_to_word}

    increment = pow(2.0, count)

    # record {'觀':{ㄕㄤ:{賞:2}}}
    if word_prev in record:
        record_list = record[word_prev]

        # {ㄕㄤ:{賞:2}}
        if syllable in record_list:
            syllable_to_word = record_list[syllable]

            if relation_coeff == 1.0:
                count = min(6.0, syllable_to_word[word] + (1 - equal)) \
                    if word in syllable_to_word else count
                increment = (1 - equal) * pow(2.0, count) + equal * 4.0

    syllable_to_word.update({word: count})
    record_list.update({syllable: syllable_to_word})
    record.update({word_prev: record_list})

    # default value
    log_score = math.log2(increment)
    npw_d = {word: log_score}
    # next possible syllable list
    nps_l = {}

    if word_prev in semantic_list:
        nps_l = semantic_list[word_prev]

        if syllable in nps_l:
            # next possible word dict
            *_, npw_d = nps_l[syllable]
            if word in npw_d:
                log_score = npw_d[word]

                word_score = math.pow(2, log_score)
                log_score = math.log2(word_score + increment)

    npw_d.update({word: log_score})
    nps_l.update({syllable: [npw_d]})
    semantic_list.update({word_prev: nps_l})

    return semantic_list, record


# punish incorrect sentence
def words_adjustment_sub(word_prev, word, syllable, semantic_list,
                         record,
                         relation_coeff=1.0):
    # record {'觀':{ㄕㄤ:{賞:2}}}
    if word_prev in record:
        record_list = record[word_prev]

        # {ㄕㄤ:{賞:2}}
        if syllable in record_list:
            syllable_to_word = record_list[syllable]

            if word in syllable_to_word and relation_coeff == 1.0:
                count = max(0.0, syllable_to_word[word] - 2.0)

                syllable_to_word.update({word: count})
                record_list.update({syllable: syllable_to_word})
                record.update({word_prev: record_list})

    if word_prev in semantic_list:
        nps_l = semantic_list[word_prev]

        if syllable in nps_l:
            *_, npw_d = nps_l[syllable]

            if word in npw_d:
                log_score = math.log2(max(2.0, pow(2., npw_d[word]) - relation_coeff * 4.0))

                npw_d.update({word: log_score})

            nps_l.update({syllable: [npw_d]})
        semantic_list.update({word_prev: nps_l})

    return semantic_list, record


def adjust_sentence(o_stn, m_stn, data_path):
    stn_d = load_data(data_path['stn'])
    if o_stn == m_stn and m_stn in stn_d:
        return True
    stn_d[m_stn] = stn_d[m_stn] if m_stn in stn_d else 0
    bonus = 2.0 if len(m_stn) > 1 else 0
    success = False

    if len(o_stn) == len(m_stn) and not o_stn == m_stn:
        punish = sum([0 if oc == mc else 2 for oc, mc in zip(o_stn, m_stn)])
        flag = True if punish >= len(m_stn) \
            else False

        stn_d[o_stn] = stn_d[o_stn] if o_stn in stn_d else 0

        if flag:
            punish /= len(m_stn)
            o_score = stn_d[o_stn]
            new_score = stn_d[o_stn] = max(-25, stn_d[o_stn] - punish)
            if o_score > 0 and new_score <= 0:
                stn_d[o_stn] = o_score
        success = True

    if stn_d[m_stn] <= 0:
        bonus = 4.0
    # tricky
    stn_d[m_stn] = min(25, stn_d[m_stn] + bonus)
    write_data(stn_d, data_path['stn'])

    success = o_stn == m_stn or success

    return success


def _adjustment(original_sentence, modified_sentence, data_path):
    if len(original_sentence) == len(modified_sentence):
        # path define
        semantic_path = data_path['semantic']
        header_path = data_path['header']

        semantic_record_path = data_path['semantic_record']
        header_record_path = data_path['header_record']

        n_gram_path = {k: data_path[k]
                       for k in data_path.keys()
                       if k.startswith('n_gram')}
        # load data
        word_relation_list = load_data(semantic_path)
        word_header_list = load_data(header_path)
        header_record = load_data(header_record_path)
        semantic_record = load_data(semantic_record_path)

        n_gram_data = {k: load_data(p)
                       for k, p in n_gram_path.items()}

        syllables = [_[0].strip('˙ˊˇˋ') for _ in pinyin(modified_sentence)]
        o_syllables = [_[0].strip('˙ˊˇˋ') for _ in pinyin(original_sentence)]

        n_gram_data = n_gram_adjust(list(zip(original_sentence, o_syllables)),
                                    list(zip(modified_sentence, syllables)),
                                    n_gram_data)

        # tricky
        new_header_list, new_header_record = \
            header_adjustment(modified_sentence[0], syllables[0],
                              original_sentence[0] == modified_sentence[0],
                              word_header_list, header_record)
        write_data(new_header_list, header_path)
        write_data(new_header_record, header_record_path)

        length = len(original_sentence)
        n = 4
        print('adjust relation', length)
        for i in range(1, length):
            equal = 1 if modified_sentence[i] == original_sentence[i] \
                else 0
            for j in range(n):
                if i + j >= length:
                    break

                # relation coefficient
                dpower = j
                relation_coeff = round(pow(0.9, dpower), 2)

                # tricky
                word_relation_list, semantic_record = \
                    words_adjustment_plus(modified_sentence[i - 1],
                                          modified_sentence[i + j], syllables[i + j],
                                          equal,
                                          word_relation_list, semantic_record,
                                          relation_coeff, )
            if original_sentence[i] != modified_sentence[i]:
                word_relation_list, semantic_record = \
                    words_adjustment_sub(original_sentence[i - 1],
                                         original_sentence[i], o_syllables[i],
                                         word_relation_list, semantic_record)

        write_data(word_relation_list, semantic_path)
        write_data(semantic_record, semantic_record_path)

        for k, d in n_gram_data.items():
            write_data(d, n_gram_path[k])

    return adjust_sentence(original_sentence, modified_sentence, data_path)


def adjustment(original_sentence, modified_sentence, settings):
    data_path = settings['data_path']
    lockfile = data_path['__lock']
    # use queue to adjust
    timeout = 600
    count = 0
    flag = False
    while count < timeout:

        lock = open(lockfile, 'a')
        try:

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            flag = _adjustment(original_sentence, modified_sentence, data_path)

        except OSError:
            count += 1
            sleep(1)
        finally:
            if flag:
                fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            if flag:
                break

    return flag

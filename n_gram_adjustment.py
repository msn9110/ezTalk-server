from math import log2 as log


def bonus_update(n, term, pred, bonus_d, record_d, add=True, equal=False):
    if 'bonus' not in record_d:
        record_d['bonus'] = {}
    bonus_record_d = record_d['bonus']

    n = str(n)
    if n not in bonus_record_d:
        bonus_record_d[n] = {}

    n_record = bonus_record_d[n]

    if n not in bonus_d:
        bonus_d[n] = {}
    n_bonus = bonus_d[n]

    coeff = 1.0 if add else -1.0
    equal = int(equal)

    base_power = 1.0 if add else 0.0
    max_power = 5.0

    power = 1.0 if add else -1.0
    term_rec = {}

    if term in n_record:
        term_rec = n_record[term]

        if pred in term_rec:
            # if equal, the power of existed term-predict would be remained
            power = max(base_power, min((1 - equal) * power + term_rec[pred], max_power))

    rec = {pred: max(base_power, power)}
    term_rec.update(rec)
    n_record.update({term: term_rec})
    bonus_record_d.update({n: n_record})

    # offset for adjustment
    # if equal, it would strengthen intensity of term by 4
    increment = coeff * ((1 - equal) * pow(2.0, power) + equal * 4.0)

    term_predicts = n_bonus[term] if term in n_bonus else {}

    score = 0.0
    if pred in term_predicts:
        score = pow(2.0, term_predicts[pred])

    score = max(2.0, score + increment)

    # will not update and term not in list when punishment
    if pred in term_predicts or add:
        term_predicts.update({pred: log(score)})
        n_bonus.update({term: term_predicts})
        bonus_d.update({n: n_bonus})

    return bonus_d, record_d


def n_gram_adjust(original_sentence,
                  modified_sentence,
                  data):
    """
    Called when two sentences are equal
    :param original_sentence:
    :param modified_sentence:
    :param data:
    :return:
    """

    original_sentence, _ = list(zip(*original_sentence))
    modified_sentence, _ = list(zip(*modified_sentence))
    original_sentence, modified_sentence = ''.join(original_sentence), ''.join(modified_sentence)

    n_gram_prob_d = data['n_gram_prob']
    n_gram_bonus_d = data['n_gram_bonus']
    n_gram_record_d = data['n_gram_record']

    min_n = 2
    max_n = 4

    for n in range(min_n, max_n + 1):

        # the length of sentence is not enough to update with n-gram
        if len(original_sentence) - 1 < n:
            break

        for i in range(n, len(original_sentence)):
            o_term, o_pred = original_sentence[i - n:i], original_sentence[i]
            m_term, m_pred = modified_sentence[i - n:i], modified_sentence[i]

            if o_pred != m_pred:
                n_gram_bonus_d, n_gram_record_d = \
                    bonus_update(n, o_term, o_pred, n_gram_bonus_d, n_gram_record_d, False)

            n_gram_bonus_d, n_gram_record_d = \
                bonus_update(n, m_term, m_pred, n_gram_bonus_d, n_gram_record_d, True, o_pred == m_pred)
            # todo: implement update code

    print('n gram adjust')

    return {'n_gram_prob': n_gram_prob_d,
            'n_gram_bonus': n_gram_bonus_d,
            'n_gram_record': n_gram_record_d}

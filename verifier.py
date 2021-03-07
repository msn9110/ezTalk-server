import jieba
from math import log10 as log
from n_gram import get_n_gram_prob, get_n_gram_bonus


def n_gram_verify_prob_value(c_stn, d=None):

    max_n = min(4, len(c_stn) - 1)
    comparison = []

    if isinstance(d, dict) and 'n_gram_prob' in d:
        d = d['n_gram_prob']

    for n in range(2, max_n + 1):
        prob = get_n_gram_prob(n, c_stn[-n - 1:], d)
        v = max(0.0, 10.0 + log(prob))
        comparison.append([v, n])

    return comparison


def n_gram_verify_bonus_value(c_stn, d=None):
    max_n = min(4, len(c_stn) - 1)
    comparison = []

    if isinstance(d, dict) and 'n_gram_bonus' in d:
        d = d['n_gram_bonus']

    for n in range(2, max_n + 1):
        v = get_n_gram_bonus(n, c_stn[-n - 1:], d)
        comparison.append([v, n])

    return comparison


def bonus_1(sentence, **kwargs):
    bonus = 0
    values = n_gram_verify_prob_value(sentence, **kwargs)
    if values:
        bonus_item = max(values, key=lambda _: _[0])
        v, n = bonus_item
        bonus = pow(v, n)

    return bonus


def bonus_2(sentence, **kwargs):
    bonus = 0
    values = n_gram_verify_prob_value(sentence, **kwargs)
    if values:
        bonus_item = max(values, key=lambda _: _[0])
        v, n = bonus_item
        bonus = n * v

    return pow(bonus, 2.0)


def bonus_3(sentence, **kwargs):
    return min(2000.0, bonus_2(sentence, **kwargs))


def bonus_4(sentence, **kwargs):
    bonus = 0
    values = n_gram_verify_bonus_value(sentence, **kwargs)
    if values:
        bonus = min(1000, pow(sum(map(lambda v: v[1] * v[0], values)), 2))
        # bonus = min(500, pow(sum(map(lambda v: v[1] * 5 * v[0], values)), 2))

    return bonus


def bonus_5(sentence, **kwargs):
    bonus = 0
    values = n_gram_verify_bonus_value(sentence, **kwargs)
    if values:
        bonus = min(1000, pow(3 * max(map(lambda v: v[1] * v[0], values)), 2))

    return bonus


def n_gram_verify_bonus(sentence, method=0, **kwargs):
    bonus_f = [0, bonus_1, bonus_2, bonus_3, bonus_4, bonus_5]
    method = min(method, len(bonus_f) - 1)
    bonus = bonus_f[method]
    if not method:
        return bonus

    score = bonus(sentence, **kwargs)

    return 2 * score + bonus(sentence)

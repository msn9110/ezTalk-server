from utils import read_json, time_used
from config import general_data_path


@time_used
def load_global_n_gram_data():
    return [read_json(general_data_path[k])
            for k in ['n_gram_bonus', 'n_gram_prob']]


# load data
global_n_gram_bonus_d, global_n_gram_prob_d = \
    load_global_n_gram_data()


def get_n_gram_prob(n, sentence, n_gram_prob_d=None,
                    minimum=1e-10):
    if len(sentence) < n + 1:
        return minimum

    if not n_gram_prob_d:
        n_gram_prob_d = global_n_gram_prob_d

    total_prob = 1.0

    prob_d = n_gram_prob_d[str(n)]
    for i in range(n, len(sentence)):
        term = sentence[i - n:i]
        next_w = sentence[i]
        prob = minimum

        try:
            prob = prob_d[term][1][next_w]
            prob = max(prob, minimum)
        except:
            pass
        finally:
            total_prob = max(1e-30, total_prob * prob)  # prevent overflow

    return total_prob


def get_n_gram_bonus(n, sentence, n_gram_bonus_d=None):
    bonus = 0

    if len(sentence) < n + 1:
        return bonus

    if not n_gram_bonus_d:
        n_gram_bonus_d = global_n_gram_bonus_d

    bonus_d = n_gram_bonus_d[str(n)]
    for i in range(n, len(sentence)):
        term = sentence[i - n:i]
        next_w = sentence[i]

        try:
            bonus += bonus_d[term][next_w]
        except (KeyError, IndexError):
            pass

    return bonus

import os
import sys

sys.path.insert(0, '..')

from config import CONFIG_DIR, accounts, get_settings
from utils import read_json, write_json, write_settings

user_template_settings = os.path.join(CONFIG_DIR, 'user_templete', 'user_settings.json')


def insert_new_key(d, k, v):
    if k not in d:
        d[k] = v
    return d


def delete_key(d, k):
    if k in d:
        del d[k]
    return d


def modify_value_of_key(d, k, v):
    return insert_new_key(delete_key(d, k), k, v)


if __name__ == '__main__':
    all_path = [user_template_settings]

    for user in accounts.keys():
        path, settings = get_settings(user)[-2:]

        # ensure settings without private attr.
        write_settings(settings, path)

        all_path.append(path)

    for p in all_path:
        print(p)
        settings = read_json(p)

        d_ = settings['data_path']

        insert_new_key(d_, 'n_gram_bonus', 'n_gram_bonus.json')
        insert_new_key(d_, 'n_gram_prob', 'n_gram_prob.json')
        insert_new_key(d_, 'n_gram_record', 'n_gram_record.json')

        rp = os.path.join(os.path.dirname(p), 'data', 'n_gram_record.json')

        if not os.path.exists(rp):
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            write_json({}, rp)

        d_ = settings['model_settings']

        insert_new_key(d_, 'enable_cmodel', True)
        insert_new_key(d_, 'model_mode', 0)

        write_json(settings, p)

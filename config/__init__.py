import os
import re
import sys
from os import makedirs, getpid

from config.settings_reload import c  # let flask auto reload

__hpsw_pattern = re.compile(r'^[a-f0-9]{64}$')
__psw_pattern = re.compile(r'^.{8,}$')
__email_pattern = re.compile(r'^([\w\d\-.]+)@([\w\d\-.]+)\.([a-zA-Z]{2,5})$')
__user_pattern = re.compile(r'^[a-zA-Z][\w.\d]{3,19}$')

# config dir
CONFIG_DIR = os.path.realpath(os.path.dirname(__file__))
# system dir
SYSTEM_PATH = os.path.dirname(CONFIG_DIR)
ROOT = os.path.dirname(SYSTEM_PATH)
USERS_DIR = os.path.join(ROOT, 'users')
RELOAD_FILE_PATH = os.path.join(CONFIG_DIR, 'settings_reload.py')

sys.path.append(SYSTEM_PATH)

tmp_dir = os.path.join(SYSTEM_PATH, '.temp')
makedirs(tmp_dir, exist_ok=True)
makedirs(USERS_DIR, exist_ok=True)
test_logs = [os.path.join(tmp_dir, 'test_status_{0:d}.txt').format(i)
             for i in range(6)]

_states_path = os.path.join(tmp_dir, 'states.txt')
_steps_path = os.path.join(tmp_dir, 'steps.txt')
_executor_pid_path = os.path.join(tmp_dir, 'pid1.txt')
_worker_pid_path = os.path.join(tmp_dir, 'pid2.txt')
_restful_pid_path = os.path.join(tmp_dir, 'restful_pid.txt')

_pids_path = {
    'executor': _executor_pid_path,
    'worker': _worker_pid_path,
    'restful': _restful_pid_path
}

_tags = {
    'state': _states_path,
    'step': _steps_path
}

general_settings = {}
restful_settings = {}
general_data_path = {}
accounts = {}
uid = ''


# load general setting
def __load_general_setting(path):
    global general_settings, restful_settings, general_data_path, accounts, uid

    from utils import read_json
    general_settings = read_json(path)

    general_settings['background_noise'] = general_settings['background_noise'] \
        if general_settings['background_noise'].startswith('/') \
        else os.path.join(ROOT, general_settings['background_noise'])
    restful_settings = general_settings['restful_settings']
    uid = general_settings['uid']
    general_data_path = {k: v if v.startswith('/') else os.path.join(SYSTEM_PATH, 'data', v)
                         for k, v in general_settings['data_path'].items()}
    if not general_settings['remote_backup_url']:
        general_settings['remote_backup_url'] = 'http://localhost:{0:s}/get_files'.format(
            restful_settings['service_port'])
    accounts = general_settings['accounts']


_general_setting_path = os.path.join(CONFIG_DIR, 'settings.json')
if not os.path.exists(_general_setting_path):
    import shutil
    shutil.copy(os.path.join(CONFIG_DIR, 'settings_template.json'), _general_setting_path)

__load_general_setting(_general_setting_path)


def get_settings(user_id=None):
    if not accounts:
        raise ValueError('There is no registered user')

    if user_id and user_id not in accounts:
        raise ValueError("%s isn't registered" % user_id)

    if not user_id:
        user_id = uid

    user_dir = os.path.join(USERS_DIR, user_id)
    settings_path = os.path.join(user_dir, 'user_settings.json')

    from utils import read_json

    # read config file
    settings = read_json(settings_path)

    data_dir = os.path.join(user_dir, 'data')
    data_path = {k: v if v.startswith('/') else os.path.join(data_dir, v)
                 for k, v in settings['data_path'].items()}
    data_path['__lock'] = os.path.join(user_dir, '.lock')
    settings['data_path'] = data_path

    model_dir = os.path.join(user_dir, 'models')
    model_settings = settings['model_settings']
    model_settings['__common_path'] = os.path.join(model_dir, 'update_')

    try:
        from utils import read_json
        model_info = read_json(os.path.join(model_settings['__common_path'] + model_settings['name'],
                               'data.json'))
        model_settings['__clip_duration'] = float(model_info['model_duration'])
    except (KeyError, FileNotFoundError):
        model_settings['__clip_duration'] = float(settings['clip_duration'])

    voice_data_dir = os.path.join(user_dir, 'voice_data')
    voice_data_path = settings['voice_data_path']
    if not voice_data_path['_background_noise_']:
        voice_data_path['_background_noise_'] = general_settings['background_noise']
    voice_data_path = {k: v
                       if v.startswith('/') or not v
                       else os.path.abspath(os.path.join(voice_data_dir, v))
                       for k, v in voice_data_path.items()}
    settings['voice_data_path'] = voice_data_path
    if not settings['backup_url']:
        settings['backup_url'] = general_settings['remote_backup_url'] + '/' + user_id
    data_path['__recorded'] = os.path.join(voice_data_path['training_set_root'],
                                           '_0', 'labelsForCMD.txt')
    settings['__id'] = user_id
    settings['__default_path_prefix'] = {
        'voice_data_path': voice_data_dir,
        'data_path': data_dir
    }

    return user_id, (user_dir, data_dir, model_dir, voice_data_dir), settings_path, settings


def __set_password(user, password):
    account = accounts[user]

    if __hpsw_pattern.match(password):
        account['password'] = password

        from utils import write_json
        write_json(general_settings, _general_setting_path)
    return account


def sign_up(account):
    import shutil

    user = account['user_id']
    if user not in accounts and __user_pattern.match(user):
        e_mail = account['e_mail']
        password = account['password']
        if __hpsw_pattern.match(password) and __email_pattern.match(e_mail):
            accounts[user] = {"e_mail": e_mail, "password": password}

            src = os.path.join(CONFIG_DIR, 'user_template')
            dst = os.path.join(USERS_DIR, user)

            if not os.path.exists(dst):
                shutil.copytree(src, dst)
                general_settings['uid'] = user
                from utils import write_json
                write_json(general_settings, _general_setting_path)
                return True
    return False


# Create default user
def add_user():
    import shutil
    import hashlib

    def sha256(msg, times=1):
        if times <= 0:
            return msg
        else:
            # hash object
            hash_ = hashlib.sha256()
            hash_.update(msg.encode('utf-8'))
            return sha256(hash_.hexdigest(), times - 1)

    def format_input(input_type, pattern, input_func):
        while True:
            value = input_func('ENTER {0:s} : '.format(input_type))
            if pattern.match(value) is not None:
                return value
            print('INVALID', input_type, 'format')

    user = format_input('username', __user_pattern, input)

    if user in accounts:
        print('username', user, 'already exists')
        return False

    from getpass import getpass

    password = format_input('password', __psw_pattern, getpass)
    re_type = getpass('ENTER password again : ')
    if re_type != password:
        print('password not equal')
        print(user, ' failed to create.')
        return False

    password = sha256(password, 2)
    e_mail = format_input('e-mail', __email_pattern, input)

    if e_mail in {v['e_mail'] for v in accounts.values()}:
        print('this e-mail has been registered.')
        return False

    user_dir = os.path.join(USERS_DIR, user)
    if os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir)
        except IOError:
            os.system('rm -f ' + user_dir)
    return sign_up({'user_id': user, 'password': password, 'e_mail': e_mail})


def get_account(user):
    account = {'sign_up': False}
    if user in accounts:
        user_info = accounts[user]
        account['user_id'] = user
        account['password'] = user_info['password']
        account['e_mail'] = user_info['e_mail']
    return {'account': account}


def verify(data):
    account = data['account']
    content = dict(data)
    del content['account']

    user_id = account['user_id']
    password = account['password']

    data = {'user': user_id, 'content': content}

    flag = False
    if user_id in accounts:
        user_info = accounts[user_id]
        # reset password when the form of stored password
        # is not in the form of hashed-password (SHA-256)
        if not __hpsw_pattern.match(user_info['password']):
            user_info = __set_password(user_id, password)
        flag = user_info['password'] == password
    data['accept'] = flag
    settings = {}
    if flag:
        *_, settings_path, settings = get_settings(user_id)
        settings['settings_path'] = settings_path
    data['settings'] = settings
    return data


def default(user):
    if user not in general_settings['accounts']:
        return
    general_settings['uid'] = user

    from utils import write_json
    write_json(general_settings, _general_setting_path)


def update_settings(key, new_settings, user_id=None):
    from utils import write_settings
    if user_id not in general_settings['accounts']:
        user_id = uid
    *_, settings_path, settings = get_settings(user_id)
    settings[key] = new_settings
    write_settings(settings, settings_path)


def set_pid(pid_key):
    if pid_key not in _pids_path:
        raise ValueError('Not Support')
    else:
        makedirs(tmp_dir, exist_ok=True)
        with open(_pids_path[pid_key], 'w') as pid_f:
            pid_f.write(str(getpid()))


def get_pid(pid_key):
    if pid_key not in _pids_path:
        raise ValueError('Not Support')
    else:
        with open(_pids_path[pid_key]) as pid_f:
            return pid_f.readline()


def write_log(tag, msg):
    if tag not in _tags:
        raise ValueError('Invalid TAG : ' + tag)
    else:
        makedirs(tmp_dir, exist_ok=True)
        with open(_tags[tag], 'w') as log_f:
            log_f.write(msg)


def read_log(tag):
    if tag not in _tags:
        raise ValueError('Invalid TAG : ' + tag)
    else:
        with open(_tags[tag]) as log_f:
            logs = log_f.readline().split(',')
            while len(logs) < 3:
                logs.append('')
            return logs


def on_finish_test(mode):
    os.remove(test_logs[mode])


def write_test_log(mode, msg):
    with open(test_logs[mode], 'w') as f:
        f.write(msg)


def read_test_log(mode):
    is_finished = not os.path.exists(test_logs[mode])
    msg = '_' + str(mode) + ' test finished' if is_finished \
        else ''
    if not is_finished:
        with open(test_logs[mode]) as l_f:
            msg = msg if is_finished else l_f.read()
    return is_finished, msg


def trigger_reloading():
    with open(RELOAD_FILE_PATH, 'r+') as r_f:
        c = int(r_f.read()[-1]) + 1
        r_f.seek(0)
        r_f.write('c = {0:d}'.format(c))

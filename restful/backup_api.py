from restful import get_all_waves, accounts, get_settings
import json, shutil, os
import urllib.request as request
from flask import Blueprint, Response

backup_api = Blueprint('backup_api', __name__)


def get_files_dict(voice_data_path):
    source_iter = get_all_waves(voice_data_path['source'], write_info=False,
                                is_sentence=False)
    source = {label: files for label, files in source_iter}

    sentence_iter = get_all_waves(voice_data_path['sentence'], write_info=False,
                                  is_sentence=True)
    sentence = {label: [os.path.split(f)[-1] for f in files]
                for label, files in sentence_iter}

    return {'source': source, 'sentence': sentence}


@backup_api.route('/get_files/<user>', methods=['GET'])
def remote_files(user):
    if user not in accounts:
        return "No " + user + "'s data"
    *_, settings = get_settings(user)
    voice_data_path = settings['voice_data_path']
    return Response(json.dumps(get_files_dict(voice_data_path), ensure_ascii=False),
                    mimetype='application/json')


@backup_api.route('/mkcopies/<user>', methods=['GET'])
def bkup(user):
    if user not in accounts:
        return "No " + user + "'s data"
    *_, (*_, VOICE_DATA_DIR), _, settings = get_settings(user)
    voice_data_path = settings['voice_data_path']

    def to_file_list(f_dict):
        ret = []
        for l, files in f_dict.items():
            ret.extend([l + '/' + a.split('/')[-1] for a in files])
        return ret
    url = settings['backup_url']
    root = VOICE_DATA_DIR
    bkup_path = os.path.join(root, 'backup')
    if os.path.exists(bkup_path):
        shutil.rmtree(bkup_path)
    with request.urlopen(url) as res:
        rfs = json.loads(res.read().decode())
        lfs = get_files_dict(voice_data_path)
        rfs_1 = to_file_list(rfs['source'])
        rfs_2 = to_file_list(rfs['sentence'])
        lfs_1 = to_file_list(lfs['source'])
        lfs_2 = to_file_list(lfs['sentence'])

        b_1 = ['source/' + p for p in lfs_1 if p not in rfs_1]
        b_2 = ['sentence/' + p for p in lfs_2 if p not in rfs_2]

        b = {'source': b_1, 'sentence': b_2}
        bkups = []
        bkups.extend(b_1)
        bkups.extend(b_2)

        for f in bkups:
            src = os.path.join(root, f)
            dst = os.path.join(bkup_path, f)
            dst_dir = os.path.split(dst)[0]
            if os.path.exists(src):
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src, dst)
        return Response(json.dumps(b), mimetype='application/json')

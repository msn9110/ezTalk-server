from waveTools import get_all_waves
import json, shutil, os
import urllib.request as request
from flask import Flask, Response

app = Flask(__name__)
PORT = 5008

def get_files_dict(root):
    source_iter = get_all_waves(root + '/source', write_info=False,
                                is_sentence=False)
    source = {label: files for label, files in source_iter}

    sentence_iter = get_all_waves(root + '/sentence', write_info=False,
                                is_sentence=True)
    sentence = {label: [os.path.split(f)[-1] for f in files]
                for label, files in sentence_iter}

    return {'source': source, 'sentence': sentence}

@app.route('/get_files', methods=['GET'])
def remote_files():
    return Response(json.dumps(get_files_dict('.'), ensure_ascii=False),
                    mimetype='application/json')

@app.route('/mkcopies', methods=['GET'])
def bkup():
    def to_file_list(f_dict):
        ret = []
        for l, files in f_dict.items():
            ret.extend([l + '/' + a.split('/')[-1] for a in files])
        return ret
    remote_host = '1.34.132.90'
    url = 'http://' + remote_host + ':' + str(PORT) + '/get_files'
    root = '/home/dmcl/dataset'
    bkup_path = os.path.join(root, 'bkup')
    if os.path.exists(bkup_path):
        shutil.rmtree(bkup_path)
    with request.urlopen(url) as res:
        rfs = json.loads(res.read().decode())
        lfs = get_files_dict(root)
        rfs_1 = to_file_list(rfs['source'])
        rfs_2 = to_file_list(rfs['sentence'])
        lfs_1 = to_file_list(lfs['source'])
        lfs_2 = to_file_list(lfs['sentence'])

        b_1 = ['source/' + p for p in lfs_1 if p not in rfs_1]
        b_2 = ['sentence/' + p for p in lfs_2 if p not in rfs_2]

        b = {'source': b_1, 'sentence': b_2}
        b_1.extend(b_2)

        for f in b_1:
            src = os.path.join(root, f)
            dst = os.path.join(bkup_path, f)
            dst_dir = os.path.split(dst)[0]
            if os.path.exists(src):
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src, dst)
        return Response(json.dumps(b), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='120.126.145.113', port=PORT, debug=True)
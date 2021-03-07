import os
import re
import glob
import json
import subprocess
from multiprocessing import Process, Queue

import requests
from flask import Blueprint, Response, request, render_template, render_template_string

from restful import get_settings, accounts, restful_settings, tmp_dir, read_json

result_api = Blueprint('result_api', __name__)


def request_data(url, q, user):
    info = ''
    try:
        r = requests.get('http://' + restful_settings['ip'] + ':'
                         + restful_settings['service_port']
                         + '/results/{0:s}/'.format(user) + url)
        data = json.loads(r.text)
        get_category = next(d for i, d in enumerate(data["distribution"]) if "category" in d)
        get_trainsize = next(d for i, d in enumerate(data["distribution"]) if "train" in d)
        train_size = str(get_trainsize["train"]["size"])
        info = '   ' + url.split('/')[0] \
               + '    種類: ' \
               + str(get_category["category"]) + '種, ' \
               + '訓練資料量: ' + train_size + ' \n'
    except:
        pass

    q.put(info)


@result_api.route('/available_results/<user>', methods=['GET', 'POST'])
def available(user):
    if user not in accounts:
        return "No " + user + "'s data"
    *_, settings = get_settings(user)
    model_settings = settings['model_settings']
    common = model_settings['__common_path']

    routes = glob.glob(os.path.join(common + '*', '_0', 'results.json'))
    records = []
    for r in routes:
        name = re.sub('^' + common, '', r).split('/')[0]
        records.append([name])
    records = list(sorted(records))
    links = ''
    checkbox1 = '<form action= "/compare/{user:s}" method="post">'.format(user=user)
    links += checkbox1
    for rec in records:
        l_n = '/'.join(rec)
        l = '<a href="/results/' + user + '/' + l_n + '">' + l_n + '</a>'
        links += '<input type="checkbox" name="date" value="' + l_n + '">' + l + '\n<br>'

    checkbox2 = '<input type="submit" value="比較結果"/></form>'
    links += checkbox2

    return render_template_string(links)


def get_results(common_path, name):
    common = os.path.join(common_path + name)
    if not os.path.exists(common):
        return {'exist': False}
    result = {'exist': True, 'name': name}
    d_path = common + '/data.json'
    result['data'] = os.path.exists(d_path)
    if result['data']:
        distr = read_json(d_path)['meta']
        result['distribution'] = [{'category': distr['num_of_labels']},
                                  {'all': distr['all']},
                                  {'train': distr['train']},
                                  {'test': distr['test']}]

    result['test_results'] = []
    for i in range(4):
        r_path = common + '/_' + str(i) + '/results.json'
        if os.path.exists(r_path):
            result['test_results'].append({'_' + str(i): read_json(r_path)})
    return result


@result_api.route('/results/<user>/<name>', methods=['GET'])
def results(user, name):
    if user not in accounts:
        return "No " + user + "'s data"
    *_, settings = get_settings(user)
    model_settings = settings['model_settings']
    common = model_settings['__common_path']

    result = get_results(common, name)
    return Response(json.dumps(result, ensure_ascii=False, sort_keys=True), mimetype='application/json')


@result_api.route('/compare/<user>', methods=['GET', 'POST'])
def compare(user):
    if user not in accounts:
        return "No " + user + "'s data"

    # call draw_comparison2to4.py
    names = []

    selected_items = request.form.getlist("date")
    print(selected_items)
    if selected_items is None:
        with open(os.path.join(tmp_dir, 'last_selected_items.txt')) as f:
            selected_items = f.read().split('\n')[:-1]
    if len(selected_items) >= 1:
        with open(os.path.join(tmp_dir, 'last_selected_items.txt'), 'w') as f:
            for it in selected_items:
                f.write(it + '\n')
        ps = []
        q = Queue()

        op_n = ' -n '
        op_u = ' -u ' + user

        for item in selected_items:
            p = Process(target=request_data, args=(item, q, user,))
            p.start()
            ps.append(p)

            n, *_ = item.split('/')

            op_n += n + ','
            names.append(n)
        cmd_str = 'python3 draw_comparison2to4.py ' \
                  + op_n + op_u
        subprocess.call(cmd_str, shell=True)

        # show image
        img_filename = ''
        for n in names:
            img_filename += n + '_'
        # show training information
        res = [q.get() for _ in ps]
        res = [_ for _ in res if _]
        res.sort()
        q.close()
        for p in ps:
            p.join()

        return render_template('show_comparison.html', name=img_filename, info=res,
                               user=user)
    else:
        return '請至少勾選1個選項'


@result_api.route('/show_waveform/<user>', methods=['GET'])
def show_waveform(user):
    if user not in accounts:
        return "No " + user + "'s data"
    *_, settings = get_settings(user)
    voice_data_path = settings['voice_data_path']

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    html_prev_body = '''\
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Newest Waveform</title>
    </head>
    <body>
    '''
    html_post_body = '''\
    </body>
    </html>
    '''
    body = '<br>\n'
    token_dir = os.path.join(voice_data_path['uploads_dir'], 'token')
    if os.path.exists(token_dir):
        from glob import glob
        import sys
        sys.path.append('..')
        import draw_wave as dw
        files = list(sorted(glob(os.path.join(token_dir, '*.wav'))))
        fig_dir = 'static/' + user + '/waveform'
        if os.path.exists(fig_dir):
            import shutil
            shutil.rmtree(fig_dir)
        os.makedirs(fig_dir)
        for p in files:
            fp = os.path.join(fig_dir, p.split('/')[-1].split('.')[0]) + '.png'
            myplot = dw.WavePlot(p)
            _, fig = myplot.makeFig()
            os.makedirs(fig_dir, exist_ok=True)
            fig.savefig(fp)
            plt.close(fig)
            body += '\t\t<img src="/' + fp + '"/>' + fp.split('/')[-1] + '\n'
        del sys.path[-1]
    html = html_prev_body + body + html_post_body
    return render_template_string(html)


@result_api.route('/get_info/<user>', methods=['GET'])
def get_info(user):
    if user not in accounts:
        return "No " + user + "'s data"
    *_, settings = get_settings(user)
    voice_data_path = settings['voice_data_path']

    cmd = 'python3 prepare_dataset.py -m -u ' + user
    subprocess.call(cmd.split(' '))
    stn_dir = voice_data_path['sentence']
    clip_dir = '_clip_' + str(settings['clip_duration'])
    info_f = 'information.txt'
    p1 = os.path.join(voice_data_path['source'], info_f)
    p2 = os.path.join(stn_dir, clip_dir, info_f)
    p3 = os.path.join(stn_dir, info_f)
    subprocess.call('python3 waveTools.py -i -p ' + os.path.dirname(p2), shell=True)
    with open(p1) as r:
        info = r.read().split('\n')
        info = [_.split(' : ') for _ in info if _]
        info_d = {k: int(v) for k, v in info}
    with open(p2) as r:
        info = r.read().split('\n')
        info = [_.split(' : ') for _ in info if _]
        for k, v in info:
            v = int(v)
            info_d.setdefault(k, 0)
            info_d[k] += v
    with open(p3) as r:
        info = r.read().split('\n')
        info = [_.split(' : ') for _ in info if _]
        info_ds = {k: int(v) for k, v in info}
    my_info = {'pronounces': info_d, 'sentence': info_ds}
    return Response(json.dumps(my_info, ensure_ascii=False,
                               indent=2, sort_keys=True),
                    content_type='application/json')

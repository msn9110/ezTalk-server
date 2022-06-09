#!flask/bin/python
from flask import Flask, Response, request
import os, re, json, shutil, requests
import sys, traceback
from clip_wave import WaveClip
from clip_stream_wave import StreamClip
import create_syllable as cs
from adjust_relation import adjustment as adjust
from multiprocessing import Process
from subprocess import call
from pypinyin_ext.zhuyin import convert_to_zhuyin

from config import restful_settings, set_pid, verify


# sentences_path = uploads_dir + '/sentences_collect.json'
app = Flask(__name__)


def collect_sentence(sentence, zhuyin_of_stn, sentences_path):
    """
    Utility for chinese sentence and its mapping of zhuyin
    :param sentence:
    :param zhuyin_of_stn:
    :param sentences_path:
    :return:
    """
    stn_dict = {}
    if os.path.exists(sentences_path):
        f = open(sentences_path)
        stn_dict = json.load(f)
        f.close()
    count = stn_dict[sentence]['count'] if sentence in stn_dict\
        else 0
    z_sentence = ''
    for z in zhuyin_of_stn:
        z_sentence += z + ','
    z_sentence = re.sub(',$', '', z_sentence)
    if sentence in stn_dict:
        item = stn_dict[sentence]
        mapped_zhuyin = item['mapped_zhuyin']
        z_count = mapped_zhuyin[z_sentence] + 1\
            if z_sentence in mapped_zhuyin else 1
        mapped_zhuyin[z_sentence] = z_count
        item['mapped_zhuyin'] = mapped_zhuyin
        item['count'] = count + 1
        stn_dict[sentence] = item
    else:
        mapped_zhuyin = {z_sentence: 1}
        stn_dict[sentence] = {'count': 1, 'mapped_zhuyin': mapped_zhuyin}
    f = open(sentences_path, 'w')
    json.dump(stn_dict, f, ensure_ascii=False, indent=2)
    f.close()


def move_error_clip_file(f, label, uploads_dir):
    split_res = f.split('/')
    name = split_res[-1]
    if len(label) == 0:
        label = split_res[-2]
    dst_dir = uploads_dir + '/error_clip_files/' + label
    os.makedirs(dst_dir, exist_ok=True)
    dst = dst_dir + '/' + name
    shutil.move(f, dst)


@app.route('/transfer', methods=['POST'])
def get_file():
    data = verify(request.get_json())
    response = {'uploaded': False}
    if data['accept']:
        settings = data['settings']
        uploads_dir = settings['voice_data_path']['uploads_dir']

        data = data['content']

        label = data['label']  # may be tmp

        my_dir = uploads_dir
        if len(re.sub('[\u4e00-\u9fa5]+', '', label)) == 0:
            # label is sentence
            my_dir += '/sentence/'
        filename, raw = data['filename'], data['raw']

        if 'sentence' not in my_dir and label != 'tmp':
            label = re.sub('[_˙ˊˇˋ]', '', label)
            my_dir += '/train'
        try:
            my_dir += '/' + label
            os.makedirs(my_dir, exist_ok=True)
            path = my_dir + '/' + filename
            # write wave from json
            with open(path, 'wb') as f:
                f.write(bytes(data))

            response['uploaded'] = True
        except IOError:
            pass
    return Response(json.dumps(response), mimetype='application/json')


@app.route('/recognize', methods=['POST'])
def recognize():
    print('debug')
    data = verify(request.get_json())
    response = {"success": 0, 'uploaded': False}
    if data['accept']:
        user = data['user']
        settings = data['settings']
        voice_data_path = settings['voice_data_path']
        duration = settings['model_settings']['__clip_duration']

        data = data['content']

        label = data['label']  # may be tmp
        num_of_stn = 8
        if 'num_of_stn' in data:
            num_of_stn = data['num_of_stn']
        my_dir = uploads_dir = voice_data_path['uploads_dir']
        clipPath = my_dir + '/_clip_{0:.1f}'.format(duration)
        if len(re.sub('[\u4e00-\u9fa5]+', '', label)) == 0:
            # label is sentence
            my_dir += '/sentence/'
        filename, raw = data['filename'], data['raw']

        if 'sentence' not in my_dir and label != 'tmp':
            label = re.sub('[_˙ˊˇˋ]', '', label)
            my_dir += '/train'

        my_dir += '/' + label
        os.makedirs(my_dir, exist_ok=True)
        path = my_dir + '/' + filename
        # write wave from json
        with open(path, 'wb') as f:
            f.write(bytes(data['raw']))
        if os.path.exists(path):
            wavefiles = []
            if not label == 'tmp' and 'sentence' not in my_dir:
                clip = WaveClip(path)
                output_dir = clipPath + '/' + label
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir + '/clip-' + filename
                wavefiles.append(clip.clipWave_toFile(output_path, duration_sec=duration))
            else:
                s_clip = StreamClip(path)
                if label == 'tmp':
                    _, files, _ = s_clip.clipWave_toFile('', outputDir=clipPath, force=True,
                                                         duration_sec=duration, show_num_detect=True)
                else:
                    _, files, _ = s_clip.clipWave_toFile(label, outputDir=clipPath,
                                                         duration_sec=duration, show_num_detect=True)
                wavefiles.extend(files)

            response = {"success": len(wavefiles), "uploaded": True}
            if len(wavefiles) > 0:
                try:
                    network_combine = 0
                    if network_combine:
                        possible_lists = cs.syllables_convert(wavefiles, settings, number=50)
                        data = {'data': possible_lists}

                        r = requests.post('http://localhost:5555/construct', json=data)
                        stns = r.json()['response']
                        s = stns[0]
                        zhuyins = [re.sub('[_˙ˊˇˋ]', '', _[0]) for _ in convert_to_zhuyin(s)]
                        i = 0
                        for z, pl in zip(zhuyins, possible_lists):
                            if z not in dict(pl):
                                possible_lists[i].append((z, 0.0))
                            i += 1
                        sentence_rec = [s] + list(zip(s, zhuyins))
                    else:
                        possible_lists, sentence_rec, stns\
                            = cs.syllables_to_sentence(wavefiles, settings, 5, enable=True,
                                                       num_of_stn=num_of_stn,
                                                       by_construct=False, include_construct=True,
                                                       intelli_select=True,
                                                       n_gram_method=4, enable_forget=True
                                                       )
                    # make token to draw wave
                    token_dir = os.path.join(uploads_dir, 'token')
                    if os.path.exists(token_dir):
                        shutil.rmtree(token_dir, ignore_errors=True)
                    all_path = [path]
                    all_path.extend(wavefiles)
                    for i, p in enumerate(all_path):
                        os.makedirs(token_dir, exist_ok=True)
                        token = os.path.join(token_dir, '{0:d}-'.format(i) +
                                             p.split('/')[-1])
                        shutil.copy(p, token)
                    p = Process(target=requests.get,
                                args=('http://' + restful_settings['ip'] + ':'
                                      + restful_settings['service_port']
                                      + '/show_waveform/{0:s}'.format(user),))
                    p.start()
                    # end token

                    if sentence_rec is None:
                        response = {"success": 0, "uploaded": True}
                        response = {'response': response}
                        os.remove(path)
                        return Response(json.dumps(response), mimetype='application/json')

                    sentence = sentence_rec[0]
                    used_syllables = [a[1] for a in sentence_rec[1:]]
                    if 'sentence' in my_dir:
                        p = Process(target=adjust, args=(sentence, label, settings))
                        p.start()
                        # extra_data = request.get_json().get('extraData')
                        # zhuyin_of_label = extra_data['zhuyin']
                        # collect_sentence(label, zhuyin_of_label)
                    recognition_results = [[key for key, score in possible_list]
                                           for possible_list in possible_lists]
                    used_indexes = [recognition_results[i].index(used_syllables[i]) + 1
                                    for i in range(len(possible_lists))]
                    response["result_lists"] = recognition_results
                    response["sentence"] = sentence
                    response["usedIndexes"] = used_indexes
                    response['sentence_candidates'] = stns
                    print(sentence)
                    print(used_syllables)
                    print([str(used_indexes[i]) + '/' + str(len(possible_lists[i]))
                           for i in range(len(possible_lists))])
                except Exception as ex:
                    print('recognition exception')
                    traceback.print_exc(file=sys.stdout)
                    print(ex)
                    if label != 'tmp':
                        os.remove(path)
                finally:
                    response = {'response': response}
                    return Response(json.dumps(response), mimetype='application/json')

            else:
                move_error_clip_file(path, label, uploads_dir)
                response = {'response': response}
                return Response(json.dumps(response), mimetype='application/json')
    return Response(json.dumps(response), mimetype='application/json')


def add_sentence(stn, path, user):
    stns = re.split(r'[^\u4e00-\u9fa5]+', stn)
    for a in stns:
        with open(path) as f:
            stn_d = json.load(f)
            if a and a not in stn_d:
                cmd = 'python3 take_audio.py -a -s ' + a + ' -u ' + user
                call(cmd, shell=True)


@app.route('/updates', methods=['PUT'])
def updates():
    data = verify(request.get_json())
    accept = data['accept']

    user = ''
    stn_d_path = ''
    settings = {}

    if accept:
        user = data['user']
        settings = data['settings']
        uploads_dir = settings['voice_data_path']['uploads_dir']
        data_path = settings['data_path']
        stn_d_path = data_path['stn']

    data = data['content']
    print(data)
    stream_files_move = data['streamFilesMove']

    sentence = data['sentence']

    update_files = data['update_files']
    response = {}

    if update_files:
        for rec in stream_files_move:
            filename = tuple(rec)[0]
            item = rec[filename]
            original = item['original']
            modified = item['modified']
            flag = False  # initial value for response[filename]
            src = uploads_dir + '/tmp/' + filename

            if accept:
                if adjust(original, modified, settings):
                    dst_dir = uploads_dir + '/sentence/' + modified
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = dst_dir + '/' + filename

                    if os.path.exists(src):
                        shutil.move(src, dst)
                        flag = True

                else:
                    move_error_clip_file(src, modified, uploads_dir)
                    print('Cannot adjust')
            response[filename] = flag
    else:
        if accept:
            p = Process(target=add_sentence, args=(sentence, stn_d_path, user,))
            p.start()

    response = {"success": accept, "movedFilesState": response}
    return Response(json.dumps(response), mimetype='application/json')


def remove_file(filename, target_dir):
    rm_cmd = 'find ' + target_dir + ' -name *' + filename + ' -exec rm {} \;'
    os.system(rm_cmd)


@app.route('/remove', methods=['DELETE', 'PUT'])
def remove():
    data = verify(request.get_json())
    print('del call')
    if data['accept']:
        uploads_dir = data['settings']['voice_data_path']['uploads_dir']
        data = data['content']

        label = data['label']
        filename = data['filename']
        print('remove', label, filename)
        remove_file(filename, uploads_dir)
        response = {"success": True}
        return Response(json.dumps(response), mimetype='application/json')


if __name__ == '__main__':
    set_pid('restful')
    app.run(host=restful_settings['ip'], port=restful_settings['port'], debug=True)

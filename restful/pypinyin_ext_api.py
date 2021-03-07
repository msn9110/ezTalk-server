from flask import Blueprint, request, render_template, Response
import json
from restful import convert_to_zhuyin
from restful import dict_path

pypinyin_ext_api = Blueprint('pypinyin_ext_api', __name__)


@pypinyin_ext_api.route("/pinyin_page", methods=['POST','GET'])
def home():
    
    raw_in = request.form.get('raw_in')
    print(raw_in)
    return render_template("pinyin_page.html", raw=raw_in)


@pypinyin_ext_api.route("/api/getpinyin", methods=['GET','POST'])
def get_pinyin():

    get_input = request.form.get('input')
    converted = convert_to_zhuyin(get_input)
    print(converted)
    result = ''
    for it in converted:
        result += it[0] + ','
    result = result[:-1]
    return render_template("post_pinyin.html", get_in=result, raw=get_input)


@pypinyin_ext_api.route("/api/postpinyin", methods = ['POST'])
def post_pinyin():

    post_input = request.form.get('input')
    raw_in = request.form.get('raw_in')
    
    split_post = post_input.split(",")

    with open(dict_path, 'r') as reader:
        data = json.load(reader)
    data[raw_in] = [[r] for r in split_post]
    with open(dict_path, 'w') as writer:
        json.dump(data, writer, ensure_ascii=False, indent=2, sort_keys=True)

    return render_template("pinyin_page.html", post_in=post_input, raw=raw_in)


@pypinyin_ext_api.route('/api/get_pinyin_custom_dict/<word>',
                        methods=["GET"], defaults={'word': None})
def get_custom_dict(word):
    with open(dict_path) as f:
        my_dict = json.load(f)
    if not request.get_json() is None:
        word = request.get_json()['word']
    if word and word in my_dict:
        result = {word: my_dict[word]}
    else:
        result = my_dict

    return Response(json.dumps(result), mimetype='application/json')
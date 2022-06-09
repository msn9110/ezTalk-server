from flask import Flask, Response, request
import json
import make_sentences_v3 as mks
from config import get_settings

app = Flask(__name__)


def process(data):
    print(data)
    res = []
    # todo: do something with data
    return res


@app.route('/construct', methods=['POST'])
def construct():
    data = request.get_json()['data']
    print(data)

    final = process(data)
    response = {'response': final}
    return Response(json.dumps(response), mimetype='application/json')


@app.route('/get_sentences', methods=['POST'])
def get_sentences():
    content = request.get_json()
    response = {}
    try:
        *_, settings = get_settings(content['user'])
        data = content['data']
        kwargs = content['kwargs'] if 'kwargs' in content else {}
        sentences = mks.parse_sentence(data, settings, **kwargs)[::-1]
        response['success'] = 1
        response['sentences'] = [s[0] for s in sentences]
    except:
        response['success'] = 0

    return Response(json.dumps(response), mimetype='application/json')


if __name__ == '__main__':
    ip = 'localhost'
    port = 55555
    app.run(host=ip, port=port, debug=True)

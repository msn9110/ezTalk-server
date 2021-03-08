from flask import render_template, Response, json, request, Blueprint
from restful import get_account, verify
from restful import trigger_reloading
from restful import parse, clean_up_settings, write_settings

# 路由和處理函式配對
settings_api = Blueprint('settings_api', __name__)


def str2bool(v):
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True


def fill(settings, new_settings):
    old = parse(settings)
    new = parse(new_settings)
    for k, v in old.items():
        # this item is sub-dict
        if not v[1]:
            continue

        # check type is bool
        if v[0] == bool:
            v[0] = str2bool

        cast = v[0]
        try:
            # cast new value
            val = cast(new[k.split('/')[-1]][1])
        except:
            continue

        # find the dict of corresponding entry,
        # then assign value to settings by reference
        keys = k.split('/')
        it = settings
        for i, access in enumerate(keys):
            # modify value by reference
            if i == len(keys) - 1:
                it[access] = val
                break
            # goto sub-dict
            it = it[access]

    return settings


def modify(settings, new_settings, json_url, to_fill=True):
    settings = fill(settings, new_settings) if to_fill else new_settings
    write_settings(settings, json_url)
    return settings


@settings_api.route('/settings_page/<user>')
def settings_page(user):
    info = verify(get_account(user))
    if not info['accept']:
        data = {}
        return Response(json.dumps(data), mimetype='application/json')

    del info['settings']['settings_path']

    data = clean_up_settings(info['settings'])

    s = parse(data)
    data = [(k, str(v[1]), v[-1])
            if v[1] is not None else (k.rstrip('/'), '$+$', v[-1])
            for k, v in sorted(s.items(),
                               key=lambda it: (it[1][-1], it[0]))]
    return render_template('settings_page.html', data=data, user=user)


@settings_api.route('/get_settings/<user>')
def showjson(user):
    info = verify(get_account(user))
    if not info['accept']:
        data = {}
        return Response(json.dumps(data), mimetype='application/json')
    del info['settings']['settings_path']

    data = clean_up_settings(info['settings'])

    return Response(json.dumps(data), mimetype='application/json')


@settings_api.route('/modify_settings/<user>', methods=['GET', 'POST'])
def modify_settings(user):
    info = verify(get_account(user))
    if not info['accept']:
        data = {}
        return Response(json.dumps(data), mimetype='application/json')
    json_url = info['settings']['settings_path']

    del info['settings']['settings_path']

    data = info['settings']

    if request.method == 'POST':
        to_fill = True
        try:
            new_settings = request.get_json()['new_settings']
            to_fill = False
        except:
            new_settings = request.form
            print(new_settings)
        data = modify(data, new_settings, json_url, to_fill)
        trigger_reloading()
    return render_template('showjson.html', user=user, data=data)

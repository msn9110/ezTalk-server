#!flask/bin/python
from flask import Flask
from config import restful_settings
from restful.settings_api import settings_api
from restful.pypinyin_ext_api import pypinyin_ext_api
from restful.backup_api import backup_api
from restful.result_api import result_api
from restful.collect_api import collect_api

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.register_blueprint(settings_api)
app.register_blueprint(pypinyin_ext_api)
app.register_blueprint(backup_api)
app.register_blueprint(result_api)
app.register_blueprint(collect_api)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-\
    revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run(host=restful_settings['ip'], port=int(restful_settings['service_port']), debug=True)

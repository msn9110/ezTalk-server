from flask import Blueprint, request, render_template, Response
import json, os, re
from multiprocessing import Process, Lock


collect_api = Blueprint('collect_api', __name__)
lock = Lock()

@collect_api.route('/collect_page', methods=['GET'])
def show_page():
    return render_template('collect_page.html')


@collect_api.route('/post_sentence', methods=['POST'])
def post_sentence():
    if request.method == 'POST' and 'stn' in request.form:
        stn = request.form['stn']
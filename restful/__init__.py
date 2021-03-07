import sys
sys.path.append('../')
from config import *
from utils import read_json, write_json, write_settings, clean_up_settings
from utils.json2form import parse
from pypinyin_ext.zhuyin import convert_to_zhuyin
from pypinyin_ext import dict_path
from waveTools import get_all_waves
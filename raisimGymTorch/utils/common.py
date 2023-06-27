# -*- coding:utf-8 -*-
# @Time : 2022/2/27 14:23
# @Author: zhcode
# @File : common.py

import os
import yaml

def load_args_from_yaml(file_path):
    if not os.path.exists(file_path):
        raise Exception("env is not exist.")

    args = {}
    with open(file_path, 'r') as file:
        args = yaml.safe_load(file)

    return args
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import config_file

option_list = {}

def config_value(option):
    """
    Return the current configuration value foe the given option
    :param option:
    :return:
    """
    return option_list[option]
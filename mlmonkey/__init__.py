from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from mlmonkey import constants

if not os.path.exists(constants.JOBS_DIR):
    os.makedirs(constants.JOBS_DIR)

if not os.path.exists(constants.SCENARIOS_JSON):
    with open(constants.SCENARIOS_JSON, 'a') as f:
        f.write('{}')


# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import os
import logging
from datetime import datetime
log_dir = 'logs'

# =============== Config ============
# =============== Config ============
curDT = datetime.now()
date_time = curDT.strftime("%m%d%H")
current_file = os.path.basename(__file__).split('.')[0]
log_file = '_'.join(['debug', current_file, date_time, '.log'])
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(log_dir, log_file)
                    )
console = logging.StreamHandler()
logging.getLogger().addHandler(console)


# target_col = train_config_detail[dir_mark]['target_col']

logging.info(f"Creating features")

logging.debug(f'debug')
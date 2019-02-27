import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import myDataUtil
logging.basicConfig(
    level=logging.NOTSET,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def get_inputs():
    '''
    :return:模型输入tensor
    '''

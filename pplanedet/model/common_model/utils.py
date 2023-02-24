import math
import paddle.nn as nn
import paddle

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor
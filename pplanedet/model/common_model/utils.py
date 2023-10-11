import math
import paddle.nn as nn
from itertools import repeat

def _ntuple(n):
    def parse(x):
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor
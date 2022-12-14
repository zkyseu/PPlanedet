import copy
import paddle
from paddle.fluid import framework

from .builder import OPTIMIZERS

OPTIMIZERS.register(paddle.optimizer.Adam)
OPTIMIZERS.register(paddle.optimizer.AdamW)
OPTIMIZERS.register(paddle.optimizer.SGD)
OPTIMIZERS.register(paddle.optimizer.Momentum)
OPTIMIZERS.register(paddle.fluid.optimizer.LarsMomentum)
OPTIMIZERS.register(paddle.optimizer.RMSProp)
import os

import numpy as np
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from paddleseg.core import infer

np.set_printoptions(suppress=True)

def evaluate(model,
             eval_dataset,
             work_dir = None,
             aug_eval=False,
             scales=1.0,
             batch_size = 1.,
             flip_horizontal=False,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             precision='fp32',
             amp_level='O1',
             num_workers=0,
             print_detail=True,
             auc_roc=False):
    """
    Launch evalution.
    Args:
        model（nn.Layer): A semantic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric
    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True, )

    total_iters = len(loader)
    accuracy_list_all = paddle.zeros([1], dtype='int64')
    logits_all = None
    label_all = None

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    #TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    predictions = []
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            # label = data['label'].astype('int64')

            if precision == 'fp16':
                with paddle.amp.auto_cast(
                        level=amp_level,
                        enable=True,
                        custom_white_list={
                            "elementwise_add", "batch_norm",
                            "sync_batch_norm"
                        },
                        custom_black_list={'bilinear_interp_v2'}):
                    output = model(data['img'])
                    output = model.get_lanes(model.out)
            else:
                output = model(data['img'])
                output = model.get_lanes(model.out)                

            predictions.extend(output)

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(data['img']))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                                ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

        accuracy = loader.dataset.evaluate(predictions, work_dir)

        # Gather from all ranks
        if nranks > 1:
            accuracy_list = []
            paddle.distributed.all_gather(accuracy_list,
                                            accuracy)

            # Some image has been evaluated and should be eliminated in last iter
            if (iter + 1) * nranks > len(eval_dataset):
                valid = len(eval_dataset) - iter * nranks
                accuracy_list = accuracy_list[:valid]

            for i in range(len(accuracy_list)):
                accuracy_list_all = accuracy_list_all + accuracy_list[
                    i]
        else:
            accuracy_list_all = accuracy_list_all + accuracy
            
    return accuracy_list_all

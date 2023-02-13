import paddle
from paddle import _C_ops, _legacy_C_ops

from paddle import in_dynamic_mode
from paddle.common_ops_import import Variable, LayerHelper, check_variable_and_dtype, check_type, check_dtype
from paddle.fluid.framework import _non_static_mode,in_dygraph_mode

def nms(
    boxes,
    iou_threshold=0.3,
    scores=None,
    category_idxs=None,
    categories=None,
    top_k=None,
):
    r"""
    This operator implements non-maximum suppression. Non-maximum suppression (NMS)
    is used to select one bounding box out of many overlapping bounding boxes in object detection.
    Boxes with IoU > iou_threshold will be considered as overlapping boxes,
    just one with highest score can be kept. Here IoU is Intersection Over Union,
    which can be computed by:
    ..  math::
        IoU = \frac{intersection\_area(box1, box2)}{union\_area(box1, box2)}
    If scores are provided, input boxes will be sorted by their scores firstly.
    If category_idxs and categories are provided, NMS will be performed with a batched style,
    which means NMS will be applied to each category respectively and results of each category
    will be concated and sorted by scores.
    If K is provided, only the first k elements will be returned. Otherwise, all box indices sorted by scores will be returned.
    Args:
        boxes(Tensor): The input boxes data to be computed, it's a 2D-Tensor with
            the shape of [num_boxes, 4]. The data type is float32 or float64.
            Given as [[x1, y1, x2, y2], …],  (x1, y1) is the top left coordinates,
            and (x2, y2) is the bottom right coordinates.
            Their relation should be ``0 <= x1 < x2 && 0 <= y1 < y2``.
        iou_threshold(float32, optional): IoU threshold for determine overlapping boxes. Default value: 0.3.
        scores(Tensor, optional): Scores corresponding to boxes, it's a 1D-Tensor with
            shape of [num_boxes]. The data type is float32 or float64. Default: None.
        category_idxs(Tensor, optional): Category indices corresponding to boxes.
            it's a 1D-Tensor with shape of [num_boxes]. The data type is int64. Default: None.
        categories(List, optional): A list of unique id of all categories. The data type is int64. Default: None.
        top_k(int64, optional): The top K boxes who has higher score and kept by NMS preds to
            consider. top_k should be smaller equal than num_boxes. Default: None.
    Returns:
        Tensor: 1D-Tensor with the shape of [num_boxes]. Indices of boxes kept by NMS.
    Examples:
        .. code-block:: python
            import paddle
            boxes = paddle.rand([4, 4]).astype('float32')
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            print(boxes)
            # Tensor(shape=[4, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [[0.64811575, 0.89756244, 0.86473107, 1.48552322],
            #         [0.48085716, 0.84799081, 0.54517937, 0.86396021],
            #         [0.62646860, 0.72901905, 1.17392159, 1.69691563],
            #         [0.89729202, 0.46281594, 1.88733089, 0.98588502]])
            out = paddle.vision.ops.nms(boxes, 0.1)
            print(out)
            # Tensor(shape=[3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
            #        [0, 1, 3])
            scores = paddle.to_tensor([0.6, 0.7, 0.4, 0.233])
            categories = [0, 1, 2, 3]
            category_idxs = paddle.to_tensor([2, 0, 0, 3], dtype="int64")
            out = paddle.vision.ops.nms(boxes,
                                        0.1,
                                        paddle.to_tensor(scores),
                                        paddle.to_tensor(category_idxs),
                                        categories,
                                        4)
            print(out)
            # Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
            #        [1, 0, 2, 3])
    """

    def _nms(boxes, iou_threshold):
        if in_dygraph_mode():
            return _C_ops.nms(boxes, iou_threshold)

        if _non_static_mode():
            return _legacy_C_ops.nms(boxes, 'iou_threshold', iou_threshold)

        helper = LayerHelper('nms', **locals())
        out = helper.create_variable_for_type_inference('int64')
        helper.append_op(
            type='nms',
            inputs={'Boxes': boxes},
            outputs={'KeepBoxesIdxs': out},
            attrs={'iou_threshold': iou_threshold},
        )
        return out

    if scores is None:
        return _nms(boxes, iou_threshold)

    if category_idxs is None:
        sorted_global_indices = paddle.argsort(scores, descending=True)
        sorted_keep_boxes_indices = _nms(
            boxes[sorted_global_indices], iou_threshold
        )
        return sorted_global_indices[sorted_keep_boxes_indices]

    if top_k is not None:
        assert (
            top_k <= scores.shape[0]
        ), "top_k should be smaller equal than the number of boxes"
    assert (
        categories is not None
    ), "if category_idxs is given, categories which is a list of unique id of all categories is necessary"

    mask = paddle.zeros_like(scores, dtype=paddle.int32)

    for category_id in categories:
        cur_category_boxes_idxs = paddle.where(category_idxs == category_id)[0]
        shape = cur_category_boxes_idxs.shape[0]
        cur_category_boxes_idxs = paddle.reshape(
            cur_category_boxes_idxs, [shape]
        )
        if shape == 0:
            continue
        elif shape == 1:
            mask[cur_category_boxes_idxs] = 1
            continue
        cur_category_boxes = boxes[cur_category_boxes_idxs]
        cur_category_scores = scores[cur_category_boxes_idxs]
        cur_category_sorted_indices = paddle.argsort(
            cur_category_scores, descending=True
        )
        cur_category_sorted_boxes = cur_category_boxes[
            cur_category_sorted_indices
        ]

        cur_category_keep_boxes_sub_idxs = cur_category_sorted_indices[
            _nms(cur_category_sorted_boxes, iou_threshold)
        ]

        updates = paddle.ones_like(
            cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs],
            dtype=paddle.int32,
        )
        mask = paddle.scatter(
            mask,
            cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs],
            updates,
            overwrite=True,
        )
    keep_boxes_idxs = paddle.where(mask)[0]
    shape = keep_boxes_idxs.shape[0]
    keep_boxes_idxs = paddle.reshape(keep_boxes_idxs, [shape])
    sorted_sub_indices = paddle.argsort(
        scores[keep_boxes_idxs], descending=True
    )
      
    if top_k is None:
        return keep_boxes_idxs[sorted_sub_indices]
#    if _non_static_mode():
#        top_k = shape if shape < top_k else top_k
#        _, topk_sub_indices = paddle.topk(scores[keep_boxes_idxs], top_k)
#        return keep_boxes_idxs[topk_sub_indices]
#    print("6666666666666666666666")
    return keep_boxes_idxs,sorted_sub_indices,top_k

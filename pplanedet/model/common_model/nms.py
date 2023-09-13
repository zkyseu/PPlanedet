import paddle
from ..losses.line_iou import line_iou

def lane_nms(predictions, scores, nms_overlap_thresh, top_k, img_w):
    """
    NMS for lane detection.
    predictions: paddle.Tensor [num_lanes,conf,y,x,lenght,72offsets] [12,77]
    scores: paddle.Tensor [num_lanes]
    nms_overlap_thresh: float
    top_k: int
    """
    # sort by scores to get idx
    # reproduce argsort with np.argsort if you meet error
    # score_np = scores.cpu().numpy()
    # np_idx = np.argsort(score_np)[::-1]
    # idx = paddle.to_tensor(np_idx)
    idx = scores.argsort(descending=True)
    keep = []

    condidates = predictions.clone()
    condidates = condidates.index_select(idx)

    while len(condidates) > 0:
        keep.append(idx[0])
        if len(keep) >= top_k or len(condidates) == 1:
            break

        ious = []
        for i in range(1, len(condidates)):
            ious.append(1 - line_iou(
                condidates[i].unsqueeze(0),
                condidates[0].unsqueeze(0),
                img_w=img_w,
                length=15))
        ious = paddle.to_tensor(ious)

        mask = ious <= nms_overlap_thresh
        id = paddle.where(mask == False)[0]

        if id.shape[0] == 0:
            break
        condidates = condidates[1:].index_select(id)
        idx = idx[1:].index_select(id)
    keep = paddle.stack(keep)

    return keep

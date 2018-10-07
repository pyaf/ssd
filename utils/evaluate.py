import numpy as np

# helper function to calculate IoU


def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max(
        [y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)
    
    # https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/64270#378701
    # If the image has no ground-truth bounding box, and the prediction is empty, 
    # the image is not included in the mean. If any bounding boxes are predicted for 
    # the image, the score is 0 and it is included in the mean.
    
    # return None if both are empty, don't count the image in final evaluation
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    if len(boxes_true) == 0 and len(boxes_pred) != 0:
        return 0
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(
            boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        # FP is the bp that not matched to any bt
        fp = len(boxes_pred) - len(matched_bt)
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


if __name__ == '__main__':
    boxes_true = np.array([[0, 0, 0, 0]])
    boxes_pred = np.array([[100, 100, 200, 200]])
    scores = [0.9]

    print(map_iou(boxes_true, boxes_pred, scores))

    # simple test
    box1 = [100, 100, 200, 200]
    box2 = [100, 100, 300, 200]
    print(iou(box1, box2))

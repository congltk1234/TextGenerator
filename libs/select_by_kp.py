import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def max(a, b):
    return a if a >= b else b

def min(a, b):
    return a if a <= b else b

def select_by_kp(imageIntegral, anchors):
    """
    :param imageIntegral:
    :param anchors:
    :return:
    """
    anchor_preserved = []
    length = anchors.shape[0]
    count = 0
    w = imageIntegral.shape[1] - 1
    h = imageIntegral.shape[0] - 1
    image_area = float(w * h)
    
    while count < length:
        x1 = int(min(max(0, anchors[count, 0]), w))
        y1 = int(min(max(0, anchors[count, 1]), h))
        x2 = int(min(max(0, anchors[count, 2]), w))
        y2 = int(min(max(0, anchors[count, 3]), h))
        integral = imageIntegral[y2, x2] - imageIntegral[y2, x1] - imageIntegral[y1, x2] + imageIntegral[y1, x1]
        if integral < 1.0:
            area = float((y2 - y1) * (x2 - x1)) / float(image_area)
            anchor_preserved.append([x1, y1, x2, y2, area])
        count += 1
    return anchor_preserved

def nms(dets, thresh):
    """
    :param dets:
    :param thresh:
    :return:
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep

def vertical_similarity(box1, box2, iou_thresh):
    intersec_height = min(box1[3], box2[3]) - max(box1[1], box2[1])
    union_height = max(box1[3], box2[3]) - min(box1[1], box2[1])
    iou = intersec_height / union_height
    if iou > iou_thresh:
        return 1
    else:
        return 0

import numpy as np

def box_connect(imageIntegral, anchors_in, h_gap, width, height):
    """
    :param imageIntegral:
    :param anchors_in:
    :param h_gap:
    :param width:
    :param height:
    :return:
    """

    length = anchors_in.shape[0]
    anchors = anchors_in[:, 0:4]
    image_area = float(width * height)

    # left to right search
    for box_index1 in range(length):
        box1 = anchors[box_index1, :]
        for box_index2 in range(length):
            box2 = anchors[box_index2, :]
            iou = vertical_similarity(box1, box2, 0.6)
            if box1[2] <= box2[0] <= (box1[2] + h_gap) and iou == 1:
                x1, y1, x2, y2 = int(box1[0]), int(min(box1[1], box2[1])), int(box2[2]), int(max(box1[3], box2[3]))
                integral = imageIntegral[y2, x2] - imageIntegral[y2, x1] - imageIntegral[y1, x2] + imageIntegral[y1, x1]
                area = float((y2 - y1) * (x2 - x1))
                if integral < 1:
                    area /= image_area
                    anchors_in[box_index1, 0] = float(x1)
                    anchors_in[box_index1, 1] = float(y1)
                    anchors_in[box_index1, 2] = float(x2)
                    anchors_in[box_index1, 3] = float(y2)
                    anchors_in[box_index1, 4] = area
                    continue

                x1, y1, x2, y2 = int(box1[0]), int(min(box1[1], box2[1])), int(box2[2]), int(min(box1[3], box2[3]))
                integral = imageIntegral[y2, x2] - imageIntegral[y2, x1] - imageIntegral[y1, x2] + imageIntegral[y1, x1]
                area = float((y2 - y1) * (x2 - x1))
                original_area = float((box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]))
                if integral < 1 and area > original_area:
                    area /= image_area
                    anchors_in[box_index1, 0] = float(x1)
                    anchors_in[box_index1, 1] = float(y1)
                    anchors_in[box_index1, 2] = float(x2)
                    anchors_in[box_index1, 3] = float(y2)
                    anchors_in[box_index1, 4] = area
                    continue

                x1, y1, x2, y2 = int(box1[0]), int(max(box1[1], box2[1])), int(box2[2]), int(max(box1[3], box2[3]))
                integral = imageIntegral[y2, x2] - imageIntegral[y2, x1] - imageIntegral[y1, x2] + imageIntegral[y1, x1]
                area = float((y2 - y1) * (x2 - x1))
                original_area = float((box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]))
                if integral < 1 and area > original_area:
                    area /= image_area
                    anchors_in[box_index1, 0] = float(x1)
                    anchors_in[box_index1, 1] = float(y1)
                    anchors_in[box_index1, 2] = float(x2)
                    anchors_in[box_index1, 3] = float(y2)
                    anchors_in[box_index1, 4] = area
                    continue

    return anchors_in


import numpy as np

NUM_CLASSES = 2

def is_above(rect1, rect2):
    # y1 + height < y2
    return rect1[3] + rect1[1] < rect2[1]
    pass

def is_left_of(rect1, rect2):
    # x1 + width < x2
    return rect1[0] + rect1[2] < rect2[0]
    pass

def is_overlapping(rect1, rect2):
    return not (is_left_of(rect1, rect2) or
                is_left_of(rect2, rect1) or
                is_above(rect1, rect2) or 
                is_above(rect2, rect1))
    pass

""" Returns true if box (x, y, width, height) overlaps any 
    of the regions (x, y, width, height) list"""
def box_overlaps_regions(box, regions):
    for region in regions:
        if is_overlapping(box, region):
            return True
        pass
    return False
    pass

""" takes dict and returns 2d vec representing
    count of blue and red annotations """
def vectorize_dict(dict):
    vec = np.zeros(NUM_CLASSES)
    for annot in dict["annotations"]:
        vec[annot["class_id"]] += 1
        pass
    return vec
    pass

""" calculates sum of difference between # of predicted classes
    and # of actual classes for each image """
def naive_classification_accuracy(actual_dicts, predict_dicts):
    sum_predictions = np.zeros(NUM_CLASSES)
    sum_actuals = np.zeros(NUM_CLASSES)
    dif_sum = np.zeros(NUM_CLASSES)
    for actual, predict in zip(actual_dicts, predict_dicts):
        vec_predict = vectorize_dict(predict)
        vec_actual = vectorize_dict(actual)

        sum_predictions += vec_predict
        sum_actuals += vec_actual

        dif_sum += np.abs(vec_predict - vec_actual)
        pass
    print(sum_predictions)
    print(dif_sum)
    print(sum_predictions / sum_predictions)
    return dif_sum
    pass

import numpy as np

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
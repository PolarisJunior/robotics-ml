
from dataset_format import *

def get_box_corners(box_dict):
    return {
        "xmin": float(box_dict["left"]),
        "xmax": float(box_dict["left"] + box_dict["width"]),
        "ymin": float(box_dict["top"]),
        "ymax": float(box_dict["top"] + box_dict["height"])
    }

def json_to_record(j):
    assert(len(j["image_size"]) == 1)
    assert(len(j["categories"]) == len(j["annotations"]))

    image_size = j["image_size"][0]
    height = float(image_size["height"])
    width = float(image_size["width"])

    filename = os.path.basename(j["file"])

    # what is this
    encoded_image_data = None
    image_format = b'jpeg'

    xmins = []
    xmaxs = []    
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    # class_ids are indexed by 1 for tensorflow
    for annot in j["annotations"]:
        c_name = class_id_to_name(annot["class_id"])
        classes_text.append(c_name)
        classes.append(annot["class_id"] + 1)
        corners = get_box_corners(annot)
        xmins.append(corners["xmin"] / width)
        xmaxs.append(corners["xmax"] / width)
        ymins.append(corners["ymin"] / height)
        ymaxs.append(corners["ymax"] / height)

    return j
    pass

def convert_files_to_record(n = None):
    i = 0
    for file_name in os.listdir(XML_PATH):
        j = file_name_to_json(file_name)
        record = json_to_record(j)
        i += 1
        if n is not None and i >= n:
            break
    pass

convert_files_to_record(n = 1)
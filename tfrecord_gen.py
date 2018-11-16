
from dataset_format import *
import dataset_util
import tensorflow as tf
import io

RECORD_PATH = os.path.join(".", "records")
RECORD_FILE_NAME = "robot_plates.record"

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
    height = image_size["height"]
    width = image_size["width"]

    filename = os.path.basename(j["file"])

    # actual image bytes? refer to dataset_tools/create_pet_tf_record.py
    with tf.gfile.GFile(os.path.join(IMAGE_PATH, filename), "rb") as fid:
        encoded_jpg = fid.read()
        pass
    encoded_image_data = encoded_jpg
    image_format = b'jpeg'

    xmins = []
    xmaxs = []    
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []
    
    for annot in j["annotations"]:
        c_name = class_id_to_name(annot["class_id"])
        classes_text.append(c_name.encode("utf8"))
        # class_ids are indexed by 1 for tensorflow
        classes.append(annot["class_id"] + 1)
        corners = get_box_corners(annot)
        xmins.append(corners["xmin"] / width)
        xmaxs.append(corners["xmax"] / width)
        ymins.append(corners["ymin"] / height)
        ymaxs.append(corners["ymax"] / height)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode("utf8")),
        'image/source_id': dataset_util.bytes_feature(filename.encode("utf8")),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))    
    # print(tf_example)
    return tf_example
    pass

def convert_files_to_record(n = None):
    i = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(RECORD_PATH, RECORD_FILE_NAME))
    for file_name in os.listdir(XML_PATH):
        j = file_name_to_json(file_name)
        tf_example = json_to_record(j)
        writer.write(tf_example.SerializeToString())
        i += 1
        if n is not None and i >= n:
            break
    writer.close()
    pass

convert_files_to_record()
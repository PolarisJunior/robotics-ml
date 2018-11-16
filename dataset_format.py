
# from xml2json import xml2
import xmltodict
import os
import json

XML_PATH = os.path.join(".", "big_dataset", "train_xml")
IMAGE_PATH = os.path.join(".", "big_dataset", "train_plate")
OUT_PATH = os.path.join(".", "out")

CLASSES = ["blue4", "red4"]
CLASSES_MAP = {"blue4": 0, "red4": 1}
CLASS_IDS = [0, 1]

def class_name_fix(class_name):
    if class_name == "blue4ww":
        return "blue4"
    return class_name
    pass

def xml_dict_to_json(xml_dict):
    j = {}
    j["file"] = os.path.join(IMAGE_PATH, xml_dict["annotation"]["filename"])
    size_dict = xml_dict["annotation"]["size"]
    # why does this need to be a list?
    j["image_size"] = [{
        "width": int(size_dict["width"]),
        "height": int(size_dict["height"]),
        "depth": int(size_dict["depth"])
    }]
    cats = []
    annotations = []
    objs = xml_dict["annotation"]["object"]
    if not isinstance(objs, list):
        objs = [objs]
    for o in objs:
        c_id = CLASSES_MAP[class_name_fix(o["name"])]
        cats.append({
            "class_id": c_id,
            "name": class_name_fix(o["name"])
        })
        bndbox = o["bndbox"]
        annotations.append({
            "class_id": c_id,
            "left": int(bndbox["xmin"]),
            "top": int(bndbox["ymin"]),
            "width": int(bndbox["xmax"]) - int(bndbox["xmin"]),
            "height": int(bndbox["ymax"]) - int(bndbox["ymin"])
        })

    j["categories"] = cats
    j["annotations"] = annotations
    return j
    pass

def file_name_to_dict(file_name):
    with open(os.path.join(XML_PATH, file_name), "r") as f:
        xml_string = f.read()
        xml_dict = xmltodict.parse(xml_string)
        return xml_dict
    print("Error opening file", file_name)

def file_name_to_json(file_name):
    xml_dict = file_name_to_dict(file_name)
    return xml_dict_to_json(xml_dict)


def convert():
    for file_name in os.listdir(XML_PATH):
        with open(os.path.join(XML_PATH, file_name), "r") as f:
            xml_string = f.read()
            xml_dict = xmltodict.parse(xml_string)
            json_string = xml_dict_to_json(xml_dict)
        file_no_ext = os.path.splitext(file_name)[0]
        with open(os.path.join(OUT_PATH, file_no_ext + ".json"), "w") as f:
            f.write(json.dumps(json_string, indent=4))
        pass

# convert()
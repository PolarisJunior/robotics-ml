
# from xml2json import xml2
import xmltodict
import os
import json

JSON_PATH = os.path.join(".", "test_IMG", "json")
if __name__ == "__main__":
    XML_PATH = os.path.join(".", "test_IMG", "XML")
    IMAGE_PATH = os.path.join(".", "test_IMG", "train_plate")
    OUT_PATH = JSON_PATH


CLASSES = ["blue4", "red4"]
CLASSES_MAP = {"blue4": 0, "red4": 1}
CLASS_IDS = [0, 1]
CLASS_IDS_MAP = { 0: "blue4", 1: "red4" }

def class_name_fix(class_name):
    if class_name == "blue4ww":
        return "blue4"
    return class_name
    pass

def class_id_to_name(cid):
    return CLASS_IDS_MAP[cid]


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

def file_name_to_dict(file_name, xml_path=None):
    with open(os.path.join(xml_path, file_name), "r") as f:
        xml_string = f.read()
        xml_dict = xmltodict.parse(xml_string)
        return xml_dict
    print("Error opening file", file_name)

def file_name_to_json(file_name, xml_path=None):
    xml_dict = file_name_to_dict(file_name, xml_path=xml_path)
    return xml_dict_to_json(xml_dict)


def convert():
    print("%s XML DIR" % XML_PATH)
    print("%s JSON DIR" % JSON_PATH)
    print("Converting XML Files to JSON")
    for file_name in os.listdir(XML_PATH):
        with open(os.path.join(XML_PATH, file_name), "r") as f:
            xml_string = f.read()
            xml_dict = xmltodict.parse(xml_string)
            json_string = xml_dict_to_json(xml_dict)
        file_no_ext = os.path.splitext(file_name)[0]
        with open(os.path.join(OUT_PATH, file_no_ext + ".json"), "w") as f:
            f.write(json.dumps(json_string, indent=4))
        pass

if __name__ == "__main__":
    convert()

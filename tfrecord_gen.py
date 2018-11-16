
from dataset_format import *


def json_to_record(j):
    print(j)
    pass

def convert_all_json():
    for file_name in os.listdir(XML_PATH):
        json_string = file_name_to_json(file_name)
        print(json_string)
        break
    
    pass

convert_all_json()
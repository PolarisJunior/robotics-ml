""" 
    Composable pipeline for generating
    artificial image datasets from an existing
    dataset

    1. load images and json files into matching lists
    2. feed through the pipeline
    3. save the images
 """

from PIL import Image
import numpy as np
import os
from metrics import box_overlaps_regions
from dataset_format import file_name_to_json

XML_PATH = "./big_dataset/train_xml"
IMAGE_PATH = os.path.join(".", "big_dataset", "train_plate")
OUT_PATH = os.path.join(".", "out")
TEST_PATH = os.path.join(".", "test")
BACKGROUND_PATH = os.path.join(".", "backgrounds")

""" base class for image manipulation strategies """
class ImageManip():
    def __init__(self):
        raise Exception("not implemented")

    """ Takes a PIL Image object """
    def manip(self, image):
        raise Exception("not implemented")


""" Adds gaussian noise to the pixels of an image """
class GaussianManip(ImageManip):
    def __init__(self, mu, variance):
        self.mu = mu
        self.variance = variance
        self.stddev = np.sqrt(self.variance)
    
    def manip(self, image, json_dat):
        noise = np.random.normal(loc=self.mu, scale=self.stddev, size=(image.height, image.width, 3))
        
        res = image + noise
        np.clip(res, 0, 255, out=res)

        return Image.fromarray(np.uint8(res), "RGB")
        pass
    pass

class BackgroundManip(ImageManip):
    def __init__(self, background_path):
        self.img_list = os.listdir(background_path)
        self.background_path = background_path
        pass

    def manip(self, image, json_dat):
        pass

def file_no_ext(file_name):
    return os.path.splitext(file_name)[0]

"""  takes a width and height for a rectangle 
    and a list of (width, height) and returns a list of
    non overlapping boxes inside the rectangle dims 
    
    WARNING, this assumes it has enough room to place all the boxes"""
def randomly_place_boxes(container_width, container_height, dims):
    boxes = []
    for dim in dims:
        x_max = container_width - dim[0]
        y_max = container_height - dim[1]

        attempts = 0
        while True:
            # candidate location
            x = np.random.randint(0, x_max)
            y = np.random.randint(0, y_max)
            box_candidate = (x, y, dim[0], dim[1])
            # only place if not overlapping another box
            if not box_overlaps_regions(box_candidate, boxes):
                break
            attempts += 1
            if attempts > 100:
                raise Exception("Too many attempts to place boxes")
        
        boxes.append(box_candidate)
        pass
    return boxes
    pass

jsons = []
images = []

for file_name in os.listdir(XML_PATH):
    jsons.append(file_name_to_json(file_name))
    img_file_name = file_no_ext(file_name) + ".jpg"
    # print(img_file_name)
    img = Image.open(os.path.join(IMAGE_PATH, img_file_name), "r")
    images.append(img)

background = Image.open(os.path.join(BACKGROUND_PATH, "Forest-Trees.jpg"), "r")


for i in range(10):
    json = jsons[i]
    file_name = os.path.basename(json["file"])
    img = background.copy()

    b_width = background.width
    b_height = background.height
    
    dims = []
    cropped_imgs = []
    for annot in json["annotations"]:
        dims.append((annot["width"], annot["height"]))
        crop_region = (annot["left"], annot["top"],
                       annot["left"] + annot["width"],
                       annot["top"] + annot["height"])
        cropped_imgs.append(images[i].crop(crop_region))

    boxes = randomly_place_boxes(b_width, b_height, dims)
    for cropped_img, box in zip(cropped_imgs, boxes):
        img.paste(cropped_img, (box[0], box[1]))
    
    img.save(os.path.join(TEST_PATH, file_no_ext(file_name) + ".jpg"))


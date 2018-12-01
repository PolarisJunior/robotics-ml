""" 
    Composable pipeline for generating
    artificial image datasets from an existing
    dataset

    1. load images and json files into matching lists
    2. feed through the pipeline
    3. save the images
 """

from PIL import Image, ImageDraw, ImageChops
import numpy as np
import os
from metrics import box_overlaps_regions, naive_classification_accuracy
from dataset_format import file_name_to_json
from json import dump

XML_PATH = "./big_dataset/train_xml"
IMAGE_PATH = os.path.join(".", "big_dataset", "train_plate")
OUT_PATH = os.path.join(".", "out")
TEST_PATH = os.path.join(".", "gaussian")
BACKGROUND_PATH = os.path.join(".", "backgrounds")

OUTPUT_PREFIX = "gaussian"

""" base class for image manipulation strategies """
class ImageManip():
    def __init__(self):
        raise Exception("not implemented")

    """ Takes a PIL Image object """
    def manip(self, image, json):
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

        return Image.fromarray(np.uint8(res), "RGB"), json_dat
        pass
    pass

""" Manip class that stores a background directory
    and pastes the cropped bounding boxes into random backgrounds
    taken from the directory """
class BackgroundManip(ImageManip):
    """ TODO border_size controls how many extra pixels to crop on each side along with the
        bounding boxes """
    def __init__(self, background_dir, circular=False, border_size=0):
        self.img_list = os.listdir(background_dir)
        self.background_dir = background_dir
        self.border_size = border_size

        self.arrangement = np.arange(0, len(self.img_list), 1, dtype="int")
        np.random.shuffle(self.arrangement)
        
        self.idx = 0
        self.circular = circular
        pass
    
    def manip(self, image, json):
        background = Image.open(os.path.join(self.background_dir, self.next_background()), "r")
        b_width = background.width
        b_height = background.height

        dims = []
        cropped_imgs = []
        for annot in json["annotations"]:
            dims.append((annot["width"], annot["height"]))
            crop_region = (annot["left"] - self.border_size, annot["top"] - self.border_size,
                        annot["left"] + annot["width"] + self.border_size,
                        annot["top"] + annot["height"] + self.border_size)
            cropped_imgs.append(image.crop(crop_region))
        
        boxes = randomly_place_boxes(b_width, b_height, dims)
        for cropped_img, box, annot in zip(cropped_imgs, boxes, json["annotations"]):
            background.paste(cropped_img, (box[0], box[1]))
            annot["left"] = box[0]
            annot["top"] = box[1]

        return background, json
        pass

    def next_background(self):
        img =  self.img_list[self.arrangement[self.idx]]
        self.idx += 1
        if self.circular:
            self.idx = self.idx % len(self.img_list)
        if self.idx == 0:
            np.random.shuffle(self.arrangement)
        return img
        pass

    def has_next_background(self):
        return self.idx < len(self.arrangement)
        pass

""" Class to concatenate manipulation strategies """
class ManipPipeline(ImageManip):
    def __init__(self):
        self.manip_pipeline = []
        pass

    def manip(self, image, json):
        for manip in self.manip_pipeline:
            image, json = manip.manip(image, json)
        return image, json
        pass

    def append_chain(self, manip):
        self.manip_pipeline.append(manip)
        return self
        pass

class ContrastManip(ImageManip):
    # TODO implement neg_mag (will slow down significantly so maybe give an option to disable)
    def __init__(self, pos_mag, neg_mag):
        self.pos_mag = pos_mag
        self.neg_mag = neg_mag

        self.blue_class_id = 0
        self.red_class_id = 1
        pass

    def manip(self, image, json):
        width = image.width
        height = image.height
        add_layer = Image.new("RGB", (width, height), (0, 0, 0))
        draw_layer = ImageDraw.Draw(add_layer)

        for annot in json["annotations"]:
            r, g, b = (0, 0, 0)
            pos_0 = (annot["left"], annot["top"])
            pos_1 = (annot["left"] + annot["width"], annot["top"] + annot["height"])
            if annot["class_id"] == self.blue_class_id:
                b = self.pos_mag
            if annot["class_id"] == self.red_class_id:
                r = self.pos_mag
            draw_layer.rectangle([pos_0, pos_1], fill=(r, g, b))                
            pass

        image = ImageChops.add(image, add_layer)
        return image, json
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

def sample_generator():
    file_names = os.listdir(XML_PATH)
    np.random.shuffle(file_names)

    for file_name in file_names:
        json = file_name_to_json(file_name)
        img_file_name = file_no_ext(file_name) + ".jpg"
        img = Image.open(os.path.join(IMAGE_PATH, img_file_name), "r")

        yield img, json
    
    pass

# jsons = []
# images = []

# for file_name in os.listdir(XML_PATH):
#     jsons.append(file_name_to_json(file_name))
#     img_file_name = file_no_ext(file_name) + ".jpg"
    
#     img = Image.open(os.path.join(IMAGE_PATH, img_file_name), "r")
#     images.append(img)


# images, jsons = shuffle()
# state = np.random.get_state()
# np.random.shuffle(images)
# np.random.set_state(state)
# np.random.shuffle(jsons)

background = Image.open(os.path.join(BACKGROUND_PATH, "Forest-Trees.jpg"), "r")

c_manip = ContrastManip(0, 0)
b_manip = BackgroundManip(BACKGROUND_PATH, circular=True, border_size=30)
g_manip = GaussianManip(0, 1000)

p_manip = ManipPipeline()
# p_manip.append_chain(c_manip)
# p_manip.append_chain(b_manip)
p_manip.append_chain(g_manip)

# i = 0
# for image, json in zip(images, jsons):
#     a_image, j_image = p_manip.manip(image, json)
#     # file = file_no_ext(os.path.basename(json["file"])) + "_aug"
#     file = "{}_{}".format(OUTPUT_PREFIX, i)
#     image_path = os.path.join(TEST_PATH, file + ".jpg")
#     a_image.save(image_path)

#     image_dir = os.path.dirname(image_path)
#     json["file"] = os.path.join(image_dir, file + ".json")
        
#     with open(os.path.join(TEST_PATH, file + ".json"), "w") as f:
#         dump(json, f, indent=4)
#         pass
#     i += 1
#     """ if i > 5:
#         break """

def run_on_all_images():
    i = 0
    for image, json in sample_generator():
        # note that dicts are pass by reference
        a_image, _ = p_manip.manip(image, json)

        file_name = "{}_{}".format(OUTPUT_PREFIX, i)

        image_path = os.path.join(TEST_PATH, "image", file_name + ".jpg")
        a_image.save(image_path)

        json["file"] = image_path
        with open(os.path.join(TEST_PATH, "json", file_name + ".json"), "w") as f:
            dump(json, f, indent = 4)
        image.close()
        i += 1
        # if i > 5
        #     break

run_on_all_images()
# naive_classification_accuracy(jsons[:7], jsons[:7])

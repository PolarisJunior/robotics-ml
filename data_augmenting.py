""" 
    Composable pipeline for generating
    artificial image datasets from an existing
    dataset

    1. load images and json files into matching lists
    2. feed through the pipeline
    3. save the images

    TODO when too many attempts to place boxes, resize the image
         Allow placing bounding boxes partially outside of the frame
 """

from PIL import Image, ImageDraw, ImageChops
import numpy as np
import os
from metrics import box_overlaps_regions, naive_classification_accuracy
from dataset_format import file_name_to_json
from json import dump

from operator import itemgetter

XML_PATH = "./big_dataset/train_xml"
IMAGE_PATH = os.path.join(".", "big_dataset", "train_plate")
OUT_PATH = os.path.join(".", "out")
TEST_PATH = os.path.join(".", "mixed")
BACKGROUND_PATH = os.path.join(".", "backgrounds")

OUTPUT_PREFIX = "mixed"

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
        noise = np.random.normal(
            loc=self.mu, scale=self.stddev, size=(image.height, image.width, 3))

        try:
            res = image + noise
        except:
            return image, json_dat
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
        self.border_size_mean = border_size
        self.border_stddev = 10

        self.arrangement = np.arange(0, len(self.img_list), 1, dtype="int")
        np.random.shuffle(self.arrangement)

        self.idx = 0
        self.circular = circular

        self.num_resizes = 0
        pass

    def manip(self, image, json):
        background = Image.open(os.path.join(
            self.background_dir, self.next_background()), "r")
        b_width = background.width
        b_height = background.height
        self.border_size = int(np.abs(np.random.normal(
            self.border_size_mean, self.border_stddev)))

        dims = []
        cropped_imgs = []
        for annot in json["annotations"]:
            dims.append((annot["width"], annot["height"]))
            crop_region = (annot["left"] - self.border_size, annot["top"] - self.border_size,
                           annot["left"] + annot["width"] + self.border_size,
                           annot["top"] + annot["height"] + self.border_size)
            cropped_imgs.append(image.crop(crop_region))

        # ensure cropped images fit in background, resize if not
        max_cropped_width = max(dims, key=itemgetter(0))[0]
        max_cropped_height = max(dims, key=itemgetter(1))[1]
        while (max_cropped_width >= b_width or max_cropped_height >= b_height):
            cropped_imgs, dims = self.resize_images(cropped_imgs, dims)
            max_cropped_width = max(dims, key=itemgetter(0))[0]
            max_cropped_height = max(dims, key=itemgetter(1))[1]
            print(self.num_resizes, self.idx)
            self.num_resizes += 1

        # find box placement or scale down if couldn't find placement
        boxes = randomly_place_boxes(b_width, b_height, dims)
        while boxes is None:
            print(self.num_resizes, self.idx)
            cropped_imgs, dims = self.resize_images(cropped_imgs, dims)
            boxes = randomly_place_boxes(b_width, b_height, dims)
            self.num_resizes += 1
            pass

        # paste images and update dict
        for cropped_img, box, annot in zip(cropped_imgs, boxes, json["annotations"]):
            background.paste(
                cropped_img, (box[0] - self.border_size, box[1] - self.border_size))
            annot["left"] = box[0]
            annot["top"] = box[1]
            annot["width"] = box[2]
            annot["height"] = box[3]

        return background, json
        pass

    def resize_images(self, images, dims, scale_factor=0.75):
        dims = [(int(float(width) * scale_factor),
                 int(float(height) * scale_factor)) for width, height in dims]
        resized_images = [img.resize(
            (int(float(img.width)*scale_factor), int(float(img.height)*scale_factor))) for img in images]
        # dims = [(img.width, img.height) for img in resized_images]
        return (resized_images, dims)

    def next_background(self):
        img = self.img_list[self.arrangement[self.idx]]
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
            pos_1 = (annot["left"] + annot["width"],
                     annot["top"] + annot["height"])
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

 Returns None if boxes cant be placed, 
 dims = list of bounding box dimensions
 returns boxes with the same size as the supplied by dims """


def randomly_place_boxes(container_width, container_height, dims):
    boxes = []
    for dim in dims:
        x_max = container_width - dim[0]
        y_max = container_height - dim[1]
        if (x_max < 1 or y_max < 1):
            return None

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
            if attempts > 200:
                return None
                # raise Exception("Too many attempts to place boxes")

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

c_manip = ContrastManip(0, 0)
b_manip = BackgroundManip(BACKGROUND_PATH, circular=True, border_size=30)
g_manip = GaussianManip(0, 100)

p_manip = ManipPipeline()
# p_manip.append_chain(c_manip)
p_manip.append_chain(b_manip)
p_manip.append_chain(g_manip)



def run_on_all_images():
    i = 0
    DEBUG = False
    for image, json in sample_generator():
        
        # note that dicts are pass by reference
        a_image, _ = p_manip.manip(image, json)

        file_name = "{}_{}".format(OUTPUT_PREFIX, i)

        image_path = os.path.join(TEST_PATH, "image", file_name + ".jpg")
        if DEBUG:
            draw = ImageDraw.Draw(a_image)
            for annot in json["annotations"]:
                draw.rectangle([(annot["left"], annot["top"]),
                                (annot["left"] + annot["width"],
                                 annot["top"] + annot["height"])], outline=(0, 255, 0))

        a_image.save(image_path)

        json["file"] = image_path
        with open(os.path.join(TEST_PATH, "json", file_name + ".json"), "w") as f:
            dump(json, f, indent=4)
        image.close()
        if i % 200 == 199:
            print("Iteration %d" % (i))
        i += 1
        if i > 6000:
            break


# run_on_all_images()
# naive_classification_accuracy(jsons[:7], jsons[:7])

def run_on_all_images_mixed():
    i = 0
    DEBUG = False
    manip_tally = [0, 0, 0, 0]
    for image, json in sample_generator():
        rnd = np.random.rand()
        # note that dicts are pass by reference
        if rnd < .2:
            a_image, _ = image, json
            manip_tally[0] += 1
        elif rnd < .3:
            a_image, _ = g_manip.manip(image, json)
            manip_tally[1] += 1
        elif rnd <= .5:
            a_image, _ = p_manip.manip(image, json)
            manip_tally[2] += 1
        else:
            a_image, _ = b_manip.manip(image, json)
            manip_tally[3] += 1

        file_name = "{}_{}".format(OUTPUT_PREFIX, i)

        image_path = os.path.join(TEST_PATH, "image", file_name + ".jpg")
        if DEBUG:
            draw = ImageDraw.Draw(a_image)
            for annot in json["annotations"]:
                draw.rectangle([(annot["left"], annot["top"]),
                                (annot["left"] + annot["width"],
                                 annot["top"] + annot["height"])], outline=(0, 255, 0))

        a_image.save(image_path)

        json["file"] = image_path
        with open(os.path.join(TEST_PATH, "json", file_name + ".json"), "w") as f:
            dump(json, f, indent=4)
        image.close()
        if i % 200 == 199:
            print("Iteration %d" % (i))
        i += 1
        # print(manip_tally)
        if i > 6000:
            break    
    pass

run_on_all_images_mixed()
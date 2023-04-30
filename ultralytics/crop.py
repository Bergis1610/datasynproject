from PIL import Image
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import re

removed_boxes = 0

crop_width = 2/3
crop_height = 1/3

test = 0

def getWidthHeight(data): 


    ls = re.split("<|>", str(data.findAll("size")))

    wi = ls.index("width")+1
    hi = ls.index("height")+1

    
    image_width = ls[wi]
    image_height = ls[hi]

    return float(image_width), float(image_height)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

for filename in os.listdir("/work/jonasbol/datasynproject/ultralytics/datasets/Norway_cropped/Norway/train/annotations/xmls"):
    test += 1

    xml_path = "/work/jonasbol/datasynproject/ultralytics/datasets/Norway_cropped/Norway/train/annotations/xmls/"+filename
    jpg_path = "/work/jonasbol/datasynproject/ultralytics/datasets/Norway_cropped/Norway/train/images/"+filename[:-3]+"jpg"

    jpg_save_path = "/work/jonasbol/datasynproject/ultralytics/datasets/Norway_Croppppped/images"
    txt_save_path = "/work/jonasbol/datasynproject/ultralytics/datasets/Norway_Croppppped/labels"


    complete_txt_path = txt_save_path + "/" + filename.split("xml")[0] + "txt"
    complete_jpg_path = jpg_save_path + "/" + filename.split("xml")[0] + "jpg"


    txt_file = open(complete_txt_path, "w")

    im = Image.open(jpg_path)

    with open("/work/jonasbol/datasynproject/ultralytics/datasets/Norway_cropped/Norway/train/annotations/xmls/"+filename, 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, "xml")

    bs_classes = bs_data.findAll("name")

    bs_boundingbox = bs_data.findAll("bndbox")

    original_image_width, original_image_height = getWidthHeight(bs_data)

    new_image_width = original_image_width * crop_width
    new_image_height = original_image_height * crop_height

    left = 0 
    right = new_image_width
    top = new_image_height
    bottom = original_image_height

    new_im = im.crop((left, top, right, bottom))

    for c, bb in zip(bs_classes, bs_boundingbox): 

        class_string = re.split("<|>", str(c))[2]

        if class_string == "D00": 
            class_string = "0"
        elif class_string == "D10": 
            class_string = "1"
        elif class_string == "D20": 
            class_string = "2"
        elif class_string == "D40": 
            class_string = "3"
        else: 
            continue

        
        bb_list_temp = [float(x) for x in re.split("<|>", str(bb)) if is_number(x)] ## xmin ymin xmax ymax

        xwidth = ((bb_list_temp[2]) - (bb_list_temp[0]))
        yheight = ((bb_list_temp[3]) - (bb_list_temp[1]))

        xmid = ((bb_list_temp[2]) + (bb_list_temp[0]))/2
        ymid = ((bb_list_temp[3]) + (bb_list_temp[1]))/2

        image_width_limit = original_image_width * crop_width
        image_height_limit = original_image_height * crop_height


        if xmid+xwidth >= image_width_limit : 
            removed_boxes += 1
            continue
        

        if ymid-yheight <= image_height_limit:
            removed_boxes += 1   
            continue
        

        xwidth /= original_image_width
        yheight /= original_image_height
        xmid /= original_image_width
        ymid /= original_image_height

        bb_list_final = [str(xmid), str(ymid), str(xwidth), str(yheight)]

        string = class_string + " " + " ".join(bb_list_final) + "\n"

        txt_file.write(string)

    # plt.imshow(new_im)
    # plt.show()

    # if test == 20: 
    #     break

    new_im.save(filename.split("xml")[0]+"jpg")

print(removed_boxes)
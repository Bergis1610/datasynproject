from bs4 import BeautifulSoup
import re
import os

## Makes txt files for all datasets in YOLO format

datasets =  [ "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]

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
    

#for dataset in datasets: 
for filename in os.listdir("/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/annotations/xmls"):
    
    if os.path.isdir("/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/annotations/xmls/"+filename):
        continue

    with open("/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/annotations/xmls/"+filename, 'r') as f:
        data = f.read()

    save_path = "/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/labels"
    complete_path = save_path+"/"+filename.split("xml")[0] + "txt"
    
    ## opens txt file 
    file = open(complete_path, "w")

    bs_data = BeautifulSoup(data, "xml")

    bs_classes = bs_data.findAll("name")

    bs_boundingbox = bs_data.findAll("bndbox")

    image_widht, image_height = getWidthHeight(bs_data)

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

        if is_number(class_string):
        
            bb_list_temp = [float(x) for x in re.split("<|>", str(bb)) if is_number(x)] ## xmin ymin xmax ymax 
            xwidth = ((bb_list_temp[2]) - (bb_list_temp[0]))
            yheight = ((bb_list_temp[3]) - (bb_list_temp[1]))

            xmid = ((bb_list_temp[2]) + (bb_list_temp[0]))/2
            ymid = ((bb_list_temp[3]) + (bb_list_temp[1]))/2

            xwidth /= image_widht
            yheight /= image_height
            xmid /= image_widht
            ymid /= image_height

            if xwidth > 1 or yheight > 1 or xmid > 1 or ymid > 1: 
                print("VALUES NOT NORMALIZED")
                print(image_widht, image_height)
                print(xmid, ymid, xwidth, yheight)
                break

            bb_list_final = [str(xmid), str(ymid), str(xwidth), str(yheight)]
            string = class_string + " "+ " ".join(bb_list_final) + "\n"

            file.write(string)
        
        else: 
            
            file.write("")
    

    file.close()
    
    

    #print("done with "+dataset)
    
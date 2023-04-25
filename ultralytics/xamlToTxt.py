from bs4 import BeautifulSoup
import re
import os

for filename in os.listdir("/work/jonasbol/datasynproject/ultralytics/datasets/Japan/train/annotations/xmls"):


    with open('/work/jonasbol/datasynproject/ultralytics/datasets/Japan/train/annotations/xmls/'+filename, 'r') as f:
        data = f.read()

    save_path = "/work/jonasbol/datasynproject/ultralytics/datasets/Japan/train/labels"
    complete_path = save_path+"/"+filename.split("xml")[0] + "txt"
    
    ## opens txt file 
    file = open(complete_path, "w")

    bs_data = BeautifulSoup(data, "xml")

    bs_classes = bs_data.findAll("name")

    bs_boundingbox = bs_data.findAll("bndbox")

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
        
        bb_string = re.split("<|>", str(bb))
    
        bb_list = [x for x in bb_string if x.isnumeric()]
        bb_list_temp = [str(int(x)/512) for x in bb_list] ## xmin ymin xmax ymax

        xwidth = float(bb_list_temp[2]) - float(bb_list_temp[0])
        yheight = float(bb_list_temp[3]) - float(bb_list_temp[1])

        xmid = float(bb_list_temp[0]) + xwidth/2
        ymid = float(bb_list_temp[1]) + yheight/2

        bb_list_final = [str(xmid), str(ymid), str(xwidth), str(yheight)]
        string = class_string + " "+ " ".join(bb_list_final) + "\n"

        print(string)
        file.write(string)
    

    file.close()

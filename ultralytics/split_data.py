import os
from sklearn.model_selection import train_test_split
import shutil

## CODE FROM https://blog.paperspace.com/train-yolov5-custom-data/

def move_files_to_folder(files, dest): 
    for file in files: 
        try: 
            shutil.move(file, dest)
        
        except: 
            print(file)
            assert False

datasets = ["China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"] ## all the datasets you want to split, dataset folder structure: datasets/example/train/(images and labels)

for dataset in datasets: 
    print("Working with ", dataset)
    # Change below paths to local path matching pattern. 
    images = [os.path.join('/work/jonasbol/datasynproject/ultralytics/datasets/'+dataset+'/train/images', x) for x in os.listdir('/work/jonasbol/datasynproject/ultralytics/datasets/'+dataset+'/train/images')]
    labels = [os.path.join('/work/jonasbol/datasynproject/ultralytics/datasets/'+dataset+'/train/labels', x) for x in os.listdir('/work/jonasbol/datasynproject/ultralytics/datasets/'+dataset+'/train/labels')]

    images.sort()
    labels.sort()
        
    print(len(images), len(labels))

    ## Partitioning files 

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=1)

    ## Making directories 
    os.mkdir("/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/images/train/")
    os.mkdir("/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/images/val/")

    os.mkdir("/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/labels/train/")
    os.mkdir("/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/labels/val/")

    ## moving files 
    move_files_to_folder(train_images, "/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/images/train/")
    move_files_to_folder(val_images, "/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/images/val/")
    move_files_to_folder(train_labels, "/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/labels/train/")
    move_files_to_folder(val_labels, "/work/jonasbol/datasynproject/ultralytics/datasets/"+dataset+"/train/labels/val/")
    
    print("Done with ", dataset)
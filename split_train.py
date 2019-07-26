import os
import cv2
from random import shuffle


def resize_image():
    path = 'C:/Users/Admin/Desktop/new/'
    savepath = 'C:/Users/Admin/Desktop/RDAUnetcode/raw/'
    files = os.listdir(path)
    shuffle(files)

    i = 0
    for file in files:
        if 'mask' in file:
            continue
        image = cv2.imread(path+file, cv2.IMREAD_GRAYSCALE)
        # resize_image = cv2.resize(image, (96, 96, 1), interpolation=cv2.INTER_CUBIC)
        img_mask_name = file.split('.')[0]+'_mask.tif'
        image_mask = cv2.imread(path+img_mask_name, cv2.IMREAD_GRAYSCALE)
        # resize_image_mask = cv2.resize(image_mask, (96, 96, 1), interpolation=cv2.INTER_CUBIC)

        if i < 20:
            cv2.imwrite(savepath+'test/'+file, image)
            cv2.imwrite(savepath+'test/'+img_mask_name, image_mask)
        elif i < 40:
            cv2.imwrite(savepath + 'validation/' + file, image)
            cv2.imwrite(savepath + 'validation/' + img_mask_name, image_mask)
        else:
            cv2.imwrite(savepath+'train/'+file, image)
            cv2.imwrite(savepath+'train/'+img_mask_name, image_mask)
        i=i+1
resize_image()
print('success')

from res_unet import get_unet_res, preprocess
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.transform import resize


data_path = 'C:/Users/Admin/Desktop/RDAUnetcode/raw/file/'


def load_train_data():
    imgs_train = np.load(data_path + 'train.npy')
    imgs_mask_train = np.load(data_path + 'train_mask.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load(data_path + 'test.npy')
    imgs_mask_test = np.load(data_path + 'test_mask.npy')
    return imgs_test, imgs_mask_test

path = 'C:/Users/Admin/Desktop/RDAUnetcode/raw/file/lr-5-32-100/'

def predict():
    model = get_unet_res()
    # print (model.metrics_names)
    # imgs_train, imgs_mask_train = load_train_data()

    path_to_save_results= path+"UNET_PREDICTIONS/"

    # imgs_train = preprocess(imgs_train)
    # imgs_mask_train = preprocess(imgs_mask_train)
    #
    # # mean= np.mean(img)
    # # std = np.std(imgs_mask_train)

    imgs_test, imgs_test_mask = load_test_data()

    mean = np.mean(imgs_test)
    std = np.std(imgs_test)
    # print(std)

    imgs_test = preprocess(imgs_test)
    imgs_test_mask = preprocess(imgs_test_mask)

    imgs_test_source = imgs_test.astype('float32')
    imgs_test_source -= mean
    imgs_test_source /= std

    imgs_test_mask = imgs_test_mask.astype('float32')
    imgs_test_mask /= 255.  # scale masks to [0, 1]

    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(data_path +'unet.hdf5')
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_predict = model.predict(imgs_test_source, verbose=1)
    res = model.evaluate(imgs_test_source,imgs_test_mask,batch_size=4,verbose=1)
    print(' Test loss:',res[0])
    print(' Test sensitivity:',res[3])
    print(' Test specificity:',res[4])
    #print(' Train mean_length_error:',score_1[3])
    print('Test f1score:',res[5])
    print(' Test precision:',res[6])
    print(' Test mean_iou:',res[8])
    #print('Test recall:',res[6])
    print('Test accuracy:',res[1])
    print('Test dice_coef:',res[2])
   
    res_loss = np.array(res)
    np.save(path+'predict.npy', imgs_mask_predict)
    np.savetxt(path + 'res_loss.txt', res_loss)
    
    predicted_masks=np.load(path+'predict.npy')
    predicted_masks*=255
    imgs_test, imgs_test_mask = load_test_data()

    for i in range(imgs_test.shape[0]):
        img = resize(imgs_test[i], (128, 128), preserve_range=True)
        img_mask = resize(imgs_test_mask[i], (128, 128), preserve_range=True)
        im_test_source = Image.fromarray(img.astype(np.uint8))
        im_test_masks = Image.fromarray((img_mask.squeeze()).astype(np.uint8))
        im_test_predict = Image.fromarray((predicted_masks[i].squeeze()).astype(np.uint8))
        im_test_source= cv2.cvtColor(np.array(im_test_source), cv2.COLOR_RGB2BGR)
        im_test_predict= cv2.cvtColor(np.array(im_test_predict), cv2.COLOR_RGB2BGR)
        im_test_masks= cv2.cvtColor(np.array(im_test_masks), cv2.COLOR_RGB2BGR)
        red = cv2.split(im_test_source)[0]
        im_test_predict = cv2.cvtColor(im_test_predict,cv2.COLOR_RGB2GRAY)
        ret,thresh1 = cv2.threshold(im_test_predict,254,255,cv2.THRESH_BINARY)
        red = cv2.addWeighted(red,1,red*thresh1,1,0)
        im_test_predict = cv2.merge((red,cv2.split(im_test_source)[1],cv2.split(im_test_source)[2]))
        im_test_source_name = "Test_Image_"+str(i+1)+".png"
        im_test_predict_name = "Test_Image_"+str(i+1)+"_Predict.png"
        im_test_gt_mask_name = "Test_Image_"+str(i+1)+"_OriginalMask.png"
        im_test_source= cv2.resize(im_test_source,(520,480))
        im_test_predict= cv2.resize(im_test_predict,(520,480))
        im_test_masks= cv2.resize(im_test_masks,(520,480))
        plt.imsave(os.path.join(path_to_save_results,im_test_source_name), im_test_source)
        plt.imsave(os.path.join(path_to_save_results,im_test_predict_name), im_test_predict)
        plt.imsave(os.path.join(path_to_save_results,im_test_gt_mask_name), im_test_masks)
    message="Successfully Saved Results to "+path_to_save_results
    print (message)

if __name__ == '__main__':
    predict()
    # model = get_unet()
    # print (model.metrics_names)

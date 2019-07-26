from __future__ import print_function

import numpy as np
import os
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
from sklearn.metrics import jaccard_similarity_score
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D,ZeroPadding2D,Convolution2D, Conv2DTranspose,MaxPooling2D,add, UpSampling2D, multiply,Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger
from keras import backend as K,models
from skimage.io import imsave
def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)


K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1.
cardinality=32
img_rows =128
img_cols =128 

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"




data_path = "C:/Users/Admin/Desktop/RDAUnetcode/raw/file/"


# data_path = '/Users/xuchenyang/Documents/sec_exp/file/'


def load_train_data():
    imgs_train = np.load(data_path + 'train.npy')
    imgs_mask_train = np.load(data_path + 'train_mask.npy')
    return imgs_train, imgs_mask_train


def load_validation_data():
    imgs_valid = np.load(data_path + 'validation.npy')
    imgs_mask_valid = np.load(data_path + 'validation_mask.npy')
    return imgs_valid, imgs_mask_valid


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
'''
def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
'''
def dice_coef_loss(y_true, y_pred):
    return (1-dice_coef(y_true, y_pred))
'''
def mean_iou(y_true, y_pred):
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
            
        return score  
'''
def mean_iou(y_true, y_pred):
      
    true_positives=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
    true_negatives=tf.reduce_sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    possible_negatives=tf.reduce_sum(K.round(K.clip((1-y_true), 0, 1)))
    possible_positives=tf.reduce_sum(tf.round(K.clip(y_true, 0, 1)))
    iou = (true_positives)/(possible_negatives - true_negatives + possible_positives + K.epsilon())
    return K.mean(iou)

'''
def mean_iou(y_true, y_pred):
    y_true = K.cast(K.equal(K.argmax(y_true),label='true'), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred),label='pred'), K.floatx())
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.switch(K.equal(union, 0), 1.0, intersection / union)
'''    
    
    
'''
def sensitivity(y_true,y_pred):
    true_positives=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
    possible_positives=tf.reduce_sum(tf.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives+K.epsilon())
'''
def sensitivity(y_true,y_pred):
    true_positives=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
    false_negatives=tf.reduce_sum(tf.round(K.clip(y_true*(1-y_pred), 0, 1)))
    return true_positives / (false_negatives + true_positives + K.epsilon())
'''
def specificity(y_true,y_pred):
    true_negatives=tf.reduce_sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    possible_negatives=tf.reduce_sum(K.round(K.clip((1-y_true), 0, 1)))
    return true_negatives / (possible_negatives+K.epsilon())
'''
def specificity(y_true,y_pred):
    true_negatives=tf.reduce_sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    false_positives=tf.reduce_sum(K.round(K.clip((1-y_true)*(y_pred), 0, 1)))
    return true_negatives / (false_positives + true_negatives + K.epsilon())



def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

'''
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
'''
def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
        false_negatives=tf.reduce_sum(tf.round(K.clip(y_true*(1-y_pred), 0, 1)))
        recall =true_positives / (false_negatives + true_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
 
    return 2*((precision*recall)/(precision+recall+K.epsilon()))      
'''
def accuracy(y_true,y_pred):
    true_positives=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
    true_negatives=tf.reduce_sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    possible_negatives=tf.reduce_sum(K.round(K.clip((1-y_true), 0, 1)))
    possible_positives=tf.reduce_sum(tf.round(K.clip(y_true, 0, 1)))
    accuracy = (true_positives + true_negatives)/(possible_negatives + possible_positives)
    return accuracy
'''

    
def unet4(upsamezhi,chuandi,F_g, F_l,F_int):
   
    up = Conv2D(F_g, (1, 1), activation='relu', padding='same')(upsamezhi)
    up= BatchNormalization()(up)

    down= Conv2D(F_l, (1, 1), activation='relu', padding='same')(chuandi)
    down= BatchNormalization()(down)
    sumadd=add([up,down])
    sumadd = Activation(activation='relu')(sumadd)
    
    
    
    jihe=Conv2D(F_int, (1, 1), activation='relu', padding='same')(sumadd)
    sumhalf= BatchNormalization()(jihe)
    
    
    sum_1=Conv2D(1, (1, 1), activation='sigmoid', padding='same')(sumhalf)
    sum_1= BatchNormalization()(sum_1)
    
    xinchuandi=multiply([chuandi,sum_1])
    return xinchuandi
'''
def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)
'''

def bottleneck(y, nb_channels, _strides): 
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        #depth=6
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality
         
        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y
#    '''

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    #shortcut = Activation(activation= 'relu')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    #shortcut = Activation(activation= 'relu')(shortcut)

    res_path = layers.add([shortcut, res_path])
    #res_path = Activation(activation='relu')(res_path)
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [64, 64], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    
    
    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    
    main_path = res_block(main_path, [512, 512], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    xin_encoder_1=unet4(main_path,from_encoder[4],256, 256,128)
    main_path = concatenate([main_path, xin_encoder_1], axis=3)
    main_path = res_block(main_path, [512, 512], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_2=unet4(main_path,from_encoder[3],128, 128,64)
    main_path = concatenate([main_path, xin_encoder_2], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_3=unet4(main_path,from_encoder[2],64, 64,32)
    main_path = concatenate([main_path, xin_encoder_3], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])
    
    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_4=unet4(main_path,from_encoder[1],32, 32,16)
    main_path = concatenate([main_path, xin_encoder_4], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])
    
    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_5=unet4(main_path,from_encoder[0],16, 16,8)
    main_path = concatenate([main_path, xin_encoder_5], axis=3)
    main_path = res_block(main_path, [32, 32], [(1, 1), (1, 1)])


    return main_path


def build_res_unet( ):
    #inputs = Input(shape=input_shape)
    inputs = Input(shape=(img_rows, img_cols, 1))

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[4], [512,512], [(2, 2), (1, 1)])
    
    
    #bottle = bottleneck(path, filters_bottleneck=256, mode='cascade')
    bottle = bottleneck(path, nb_channels=256, _strides=(1,1))
    path = decoder(bottle, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='hard_sigmoid',padding='same')(path)
    model=Model(input=inputs, output=path)
    

             
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[sensitivity,specificity,f1score,precision,mean_iou,'accuracy',dice_coef])

    return model
    



    




 


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows,img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_valid, imgs_mask_valid = load_validation_data()

    imgs_train = preprocess(imgs_train)
    print(imgs_train.shape)
    imgs_mask_train = preprocess(imgs_mask_train)
    print(imgs_mask_train.shape)
    imgs_valid = preprocess(imgs_valid)
    print(imgs_valid.shape)
    imgs_mask_valid = preprocess(imgs_mask_valid)
    print(imgs_mask_valid.shape)

    imgs_train = imgs_train.astype('float32')
    imgs_valid = imgs_valid.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    val_mean = np.mean(imgs_valid)
    val_std = np.std(imgs_valid)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid -= val_mean
    imgs_valid /= val_std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    imgs_mask_valid = imgs_mask_valid.astype('float32')
    imgs_mask_valid /= 255.

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model =build_res_unet()
    model_checkpoint = ModelCheckpoint('C:/Users/Admin/Desktop/RDAUnetcode/raw/file/unet.hdf5', monitor='val_loss',
                                       save_best_only=True)
    
    

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    #es=EarlyStopping(monitor='val_loss',patience=50,verbose=1)
    his = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=100, verbose=1, shuffle=True,
                    validation_data=(imgs_valid, imgs_mask_valid), callbacks=[model_checkpoint])
  
    
    score_1=model.evaluate(imgs_train,imgs_mask_train,batch_size=32,verbose=1)
    print(' Train loss:',score_1[0])
    print(' Train sensitivity :',score_1[1])
    print(' Train  specificity :',score_1[2])
    print(' Train f1-measure:',score_1[3])
    print(' Train precision :',score_1[4])
    print(' Train mean_iou :',score_1[5])
    #print('Train recall:',score_1[6])
    print(' Train accuracy:',score_1[6])
    print(' Train dice_coef:',score_1[7])
    res_loss_1 = np.array(score_1)
    np.savetxt(data_path+ 'res_loss_1.txt', res_loss_1)
    
    score_2=model.evaluate(imgs_valid,imgs_mask_valid,batch_size=32,verbose=1)
    print(' valid  loss:',score_2[0])
    print(' valid  sensitivity :',score_2[1])
    print(' valid  specificity:',score_2[2])
    print(' valid  f1-measure:',score_2[3])
    print(' valid  precision:',score_2[4])
    print(' valid  mean_iou:',score_2[5])
    #print(' valid  recall:',score_2[6])
    print(' valid  accuracy:',score_2[6])
    print(' valid  dice_coef:',score_2[7])
    res_loss_2 = np.array(score_2)
    np.savetxt(data_path + 'res_loss_2.txt', res_loss_2)
    
    plt.plot()
    plt.plot(his.history['loss'],c='r', label='train')
    plt.plot(his.history['val_loss'], c='black', label='validation')
    plt.title('train and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
        
    plt.plot()
    plt.plot(his.history['sensitivity'],c='r', label='train ')
    plt.plot(his.history['val_sensitivity'], c='black', label='validation')
    plt.title('train  and validation sensitivity')
    plt.xlabel('epoch')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.show()
       
    plt.plot()
    plt.plot(his.history['specificity'],c='r', label='train')
    plt.plot(his.history['val_specificity'], c='black', label='validation')
    plt.title('train  and validation specificity')
    plt.xlabel('epoch')
    plt.ylabel('Specificity')
    plt.legend(loc='lower right')
    plt.show()
    
    
    
    plt.plot()
    plt.plot(his.history['acc'],c='r', label='train')
    plt.plot(his.history['val_acc'], c='black', label='validation')
    plt.title('train  and validation accuraccy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
    
    plt.plot()
    plt.plot(his.history['dice_coef'],c='r', label='train')
    plt.plot(his.history['val_dice_coef'], c='black', label='validation')
    plt.title('train  and validation dice_coef')
    plt.xlabel('epoch')
    plt.ylabel('DSC')
    plt.legend(loc='lower right')
    plt.show()
    


 
   
    
    plt.plot()
    plt.plot(his.history['f1score'],c='r', label='train')
    plt.plot(his.history['val_f1score'], c='black', label='validation')
    plt.title('train  and validation f1-measure')
    plt.xlabel('epoch')
    plt.ylabel('F1-measure')
    plt.legend(loc='lower right')
    plt.show()
    
   
    plt.plot()
    plt.plot(his.history['precision'],c='r', label='train')
    plt.plot(his.history['val_precision'], c='black', label='validation')
    plt.title('train  and validation precision')
    plt.xlabel('epoch')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()
   
    plt.plot()
    plt.plot(his.history['mean_iou'],c='r', label='Train')
    plt.plot(his.history['val_mean_iou'], c='black', label='validation')
    plt.title('train and validation mean_iou')
    plt.xlabel('epoch')
    plt.ylabel('Mean-iou')
    plt.legend(loc='lower right')
    plt.show()
 
    ''' 
    plt.plot()
    plt.plot(his.history['recall'], label='train recall')
    plt.plot(his.history['val_recall'], c='g', label='val_recall')
    plt.title('train  and val recall')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()
    '''
if __name__ == '__main__':
    #model = build_res_unet()
    #print(model.summary())
    train()

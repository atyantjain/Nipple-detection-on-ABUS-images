# Nipple-detection-on-ABUS-images

Breast cancer causes massive deaths every year. To prevent this, the detection of a tumor in its initial stage is necessary. And correctly identifying a tumor from an ultrasound image requires years of experience and due to a few of a number of experienced specialists makes it a challenging task. So, to assist specialists in acceptably ascertaining a tumor in an ultrasound image, in this paper, a modified U-Net model named GRA U-Net is proposed. GRA U-Net is a combination of some of the existing techniques and can segment the nipple from AWBUS images. Nipple segmentation is important as it can help in precisely locating the tumor from the outside of the breast. The segmented nipple can thus be used to locate tumor with respect to its position. There already exist so many segmentation models such as Residual-U-Net, Fcn8, Dense-U-Net, Squeeze U-Net, etc. And on comparing them with the proposed model on parameters like accuracy, sensitivity, specificity, precision and etc. it was found that GRA U-Net delivers better performance than the others in most of the parameters. Thus this method could be used in bio-medical areas for improving the facilities that are present and provide a proper detection of tumor or a lesion in its initial stage.

A.	Data collection

The down flow of the experiment includes many stages and the collection of data is the starting point of this experiment. Thus, it is very important to carefully collect the data. This first stage, performed by the medical imaging hardware AWBUS, which samples data in form of multiple 2D image slices. AWBUS produces 700-800 image slices per scan for each patient. The breast cancer department of the First Affiliated Hospital of Shantou University Medical College (SUMC) provided the data. The data-set consists of breast ultrasound images along with the ground truths which contain nipple segmented in coronal plane for different patients. Some experienced specialists at SUMC manually segmented these ground truths. The nipple in the original image must exactly overlap its ground truth because if it doesn’t then the deep learning model will provide misleading results.  

B.	Experiment and Results

Firstly, from the collected data the input slices of images were down sampled to a 128×128 in flat resolution to simplify computation. From a sample of 25 patients, all the slices which have a nipple on the images together were used, with approximately 2/3 of those randomly picked slices as the training data. There were a total of 131 slices with a nipple used, including 91 slices for training and 20 slices each for validation and test. The selected data for train, validation, and test must be totally unbiased and random. The randomness of data leads to better performance of the U-Net and prevents the case of over-fitting. Thus, the division of data into train, test, and validation was done using a split-train program. The selected data was then loaded into the GRA U-Net.
The experiment was performed on a computer with the configurations: Python 3.6.5: Anaconda, Keras (2.2.4) framework, calls Tensor flow (1.11.0) with Adam optimizer, NVIDIA Quadro P400 GPU with 8GB graphics card’s memory, operating system version windows 10, CUDA version 9.0.176.


DATA .PY

this is program convers the images into numpy arrays and basically ncreates numpy files for each test, train and validation data.

UNETDEEPLIDATE.PY

this program trains accessing numpy data created by DATA.Py

PREDICT.PY

this program usees weights created by UNETDEEPLIDATE.PY to predict on test images and provide various performance metric like accuracy and loss etc.

PLOTONE.PY

this program is only used to plot AUC and ROC curves from predicted data

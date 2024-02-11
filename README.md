# Pneumonia Detection
Pneumonia is a prominent contributor to global mortality rates, necessitating the timely detection of the condition in order to enhance patient prognoses. The current approach utilised for diagnosing pneumonia is chest X-ray imaging, however its application is difficult due to inherent subjective variability. This project introduces a novel ensemble methodology that use convolutional neural networks (CNNs) to effectively diagnose pneumonia. The ensemble consists of four distinct convolutional neural network (CNN) architectures, specifically ResNet, VGG16, Xception, and DenseNet. The models undergo initial training on the ImageNet dataset, followed by fine-tuning on a dataset consisting of chest X-ray images encompassing both pneumonia and normal cases. The evaluation of the ensemble model's performance is conducted on a separate test set, which is not used during the training process. The obtained accuracy of the ensemble model on this test set is 91%. The results suggest that the utilisation of the ensemble approach yields a notable improvement in the precision of pneumonia identification. The ensemble approach exhibits greater resistance to overfitting and demonstrates superior performance in generalisation compared to individual CNN models. Additionally, the training and deployment process of a smaller CNN model is more efficient in terms of speed compared to a single large CNN model. Ensemble approaches can be utilised in the construction of a computer-aided diagnostic system for the identification of pneumonia.


Python version - 3.10.12 

Dataset download - from Kaggle 
 
need to download kaggle.json file from your kaggle account:
 --> Go to your Kaggle account settings page: https://www.kaggle.com/account.
 -->Scroll down to the API section and click on the "Create New API Token" button.
 -->This will download a file named kaggle.json.
 -->then upload that kaggle.json file in the folder section

run the below commands in current directory: 

 !mkdir -p ~/.kaggle
 !mv kaggle.json ~/.kaggle/
 !chmod 600 ~/.kaggle/kaggle.json

run the below command in current directory:

  ! kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

dataset, the unzip folder will be downloaded, to unzip that zip file run the command:

  ! unzip chest-xray-pneumonia.zip

run the whole code, it's successfully executes.

**THE ISSUE** <br />
After the breakout of Coronavirus, researchers from different domains have focused on how the diagnosis could be speeded up to help in early isolation of defected or suspicious cases. Radiologists strongly believe that X-rays or CT scans can support the COVID detection system. In this project, we introduce deep learning-based approach to identify COVID from X-ray images.

**Our Approach** <br />
Diagnostic Decision Support for Medical Imaging (DDSM++) is introduced to detect COVID along with other lung abnormalities. It is based on deep neural network and addresses the lack of COVID X-ray images by using various spatial transform augmentations. The CLAHE is used to pre-process input image. The predictions are interpreted using the LIME which will be further discussed with radiologists.

**How it works** <br />
* Rearing X-ray from Source
* Pre processing the X-ray
* Training and hyper-parameter tuning of DNN
* Evaluate trained model
* Take feedback from Radiologist on model inference






References:

1\. Covid-19 Chest X-ray Dataset:
<https://github.com/ieee8023/covid-chestxray-dataset>
<https://stanfordmlgroup.github.io/projects/chexnet/>

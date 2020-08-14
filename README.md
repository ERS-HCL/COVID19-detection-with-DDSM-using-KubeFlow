

**Submission at the hackathon** <br />

[Link to UI demo video](https://github.com/kaladharanalytics/COVID-19-detection-with-DDSM-using-KubeFlow-/tree/master/Demo_Video) <br />
[Link to_PPT](https://github.com/kaladharanalytics/COVID-19-detection-with-DDSM-using-KubeFlow-/blob/master/Hackton_DDSM_v4.pdf)  <br />
[Link to our Covid-19-X-ray-Image-Augmentation work](https://github.com/ERS-HCL/Covid-19-X-ray-Image-Augmentation-) <br />
[Link to_Setup Document](https://github.com/kaladharanalytics/COVID-19-detection-with-DDSM-using-KubeFlow-/blob/master/Application_Env_Setup%20Documentat.docx) <br />




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

**Setup** <br />
* Fllow the instructions in the document(Application_Env_Setup Documentat.doc) to setup the Application environment.Application_Env_Setup Documentat.doc <br />
*[Link to training code](https://drive.google.com/drive/folders/1rV954AvK1xTGzFnIUGfCv500Jj5i4Anb?usp=sharing)

** Reailtime Demo** <br />
*Realtime Demo hosted on GCP will share the URL based on request <br />



References:

1\. Covid-19 Chest X-ray Dataset:
<https://github.com/ieee8023/covid-chestxray-dataset> <br />
2\.ChexNet Reference:
<https://stanfordmlgroup.github.io/projects/chexnet/><br />
<https://github.com/brucechou1983/CheXNet-Keras> <br />
3\.DataSet:
<https://www.kaggle.com/nih-chest-xrays/data><br />


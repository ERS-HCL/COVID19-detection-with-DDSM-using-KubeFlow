from lime import lime_image
from skimage.segmentation import mark_boundaries  #Return image with boundaries between labeled regions highlighted.
from flask_cors import CORS
import time
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import model_from_json
from skimage.transform import resize
import numpy as np
import cv2
import keras
from keras.layers import Input,Dropout
from keras.layers.core import Dense
from keras.models import Model
from keras.layers import Lambda
from keras.applications.resnet50 import ResNet50
from flask import Flask
from flask import Flask, jsonify, request
# from PIL import Image
import os, io
import base64
import sys
import argparse

app = Flask(__name__)
CORS(app)

v1model=None
v2model=None
input_shape_lay=None
args=None
weights={}


def parse_arguments():
    
    global weights
    parser = argparse.ArgumentParser()

    parser.add_argument("--v1-model-weights", help="Weights for model",
                    dest='v1ModelWeights', default='', required=True)
             
    parser.add_argument("--v2-model-weights", help="Weights for model",
                    dest='v2ModelWeights', default='', required=True)
                    
    args = parser.parse_args()
    print(f'The arguments passed are {args}')
    weights['v1'] = args.v1ModelWeights
    weights['v2'] = args.v2ModelWeights
    

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/test')
def test():
    return "Testing the Explain App !!! " 


def decodeImage(filestr):
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    return img

def enhance_images(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    cl1 = clahe.apply(img_grey)
    cl1 = cl1[..., np.newaxis]
    return cl1

@app.route('/explain', methods=["POST"])
def explain_ddsm():
    print("Entering the explain ddsm ..... ")
    global v1model
    global v2model
    filestr = request.files['file'].read()
    model_ver = request.form['model_ver']
    print("Explain ddsm")
    if(model_ver == 'v1' and v1model is None):
        v1model=build_model(model_ver)
    elif(model_ver == 'v2' and v2model is None):
     	v2model=build_model(model_ver)
    	
    img = decodeImage(filestr)
    h,w,c=img.shape
    print("----------------------",img)
    image_array= enhance_images(img)
    image_array = image_array / 255.
    image_array = resize(image_array,(224,224,3))
    v = np.expand_dims(image_array, axis=0)
    imagenet_mean = np.array([0.449])
    imagenet_std = np.array([0.226])
    batch_x_non = (v - imagenet_mean) / imagenet_std
    print(batch_x_non.shape)
    explainer = lime_image.LimeImageExplainer(feature_selection='lasso_path')  #object to explain predictions on Image data.
    if(model_ver == 'v1'):
        model = v1model
    else:
        model = v2model
    explanation_non_covid = explainer.explain_instance(batch_x_non[0], model.predict, top_labels=5, hide_color=0, num_samples=400)
    print('Explaination is obtained!')
    temp_non_1, mask_non_1 = explanation_non_covid.get_image_and_mask(explanation_non_covid.top_labels[0], positive_only=True,
                                            num_features=10, hide_rest=True)
    temp_non_2, mask_non_2 = explanation_non_covid.get_image_and_mask(explanation_non_covid.top_labels[0], positive_only=False,
                                                    num_features=10, hide_rest=False)
    print("the prediction for non covid", explanation_non_covid.top_labels[0])
    class_names=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',

             'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis',

             'Pleural_Thickening','Hernia','Covid']

    print("Predicted class: ", class_names[explanation_non_covid.top_labels[0]])
    exp_image = cv2.cvtColor(np.asarray(mark_boundaries(((temp_non_2*imagenet_std)+imagenet_mean)*255, mask_non_2),'uint8'), cv2.COLOR_BGR2RGB)

    img = cv2.resize(img,(w,h))
    img = Image.fromarray(exp_image.astype("uint8"))

    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})

### slicing operation

def slicer_fn(the_layer):

    start_slice = [0,0,0,0]
    size_slice = [-1,input_shape_lay[1],input_shape_lay[2],1]
    return keras.backend.slice(the_layer, start_slice, size_slice)

def build_model(model_ver):
    global input_shape_lay
    base_model_class = keras.applications.densenet.DenseNet121
    input_shape=(224, 224, 3)    
    img_input = Input(shape=input_shape)
    input_shape_lay= img_input.shape
    intermediate_layer = Lambda(slicer_fn)(img_input) #,output_shape=(None,224,224,1) 
    print(intermediate_layer)
    base_model = base_model_class(
        include_top=False,
        input_tensor=intermediate_layer,
        input_shape=(224,224,1),
        weights=None,)

    x = base_model.output
    x=keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(512,activation='relu',name="dense_512")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(15, activation="sigmoid", name="my_prediction")(x)
    model = Model(inputs=img_input, outputs=predictions)
    if(model_ver == 'v1'):
#    	model.load_weights("best_weights.h5")
        print('About to load weights from location', weights['v1'])
        model.load_weights(weights['v1'])
        #model.load_weights('gs://ddsmplus/h5_models/weights_v1.h5')
    elif (model_ver == 'v2'):
        print('About to load weights from location', weights['v2'])
        model.load_weights(weights['v2'])
    print('Loaded the model')
    return model

if __name__ == '__main__':
    parse_arguments()
    app.run(host='0.0.0.0', port=8080)

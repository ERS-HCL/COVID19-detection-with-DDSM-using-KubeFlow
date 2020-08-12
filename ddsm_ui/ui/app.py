from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import sys
import argparse

from utils import load_preprocess_image, get_prediction, get_metadata


class_names = np.array(["Atelectasis", "Cardiomegaly", "Effusion", 
               "Infiltration", "Mass", "Nodule", "Pneumonia", 
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
               "Fibrosis", "Pleural_Thickening", "Hernia", "Covid"])


app = Flask(__name__)
CORS(app)

args=None
ips=None

def parse_arguments():

    global args
    global ips

    parser = argparse.ArgumentParser()

    parser.add_argument("--v1-server", help="serving ip for v1",
                    action="store", dest='v1Server', default='ddsm-v1.kubeflow.svc.cluster.local', required=True)

    parser.add_argument("--v1-server-port", help="serving port for v1",
                    action="store", dest='v1ServerPort', default='8500')

    parser.add_argument("--v2-server", help="serving ip for v1",
                    action="store", dest='v2Server', default='ddsm-v2.kubeflow.svc.cluster.local', required=True)

    parser.add_argument("--v2-server-port", help="serving port for v1",
                    action="store", dest='v2ServerPort', default='8500')

    parser.add_argument("--elastic-ip", help="serving ip for elastic",
                    action="store", dest='elasticIp', required=True)

    parser.add_argument("--elastic-port", help="serving port for elastic",
                    action="store", dest='elasticPort', default='8500', required=True)

    parser.add_argument("--explain-url", help="serving port for elastic",
                    action="store", dest='explainURL', default='8500', required=True)

    args = parser.parse_args()
    print(f'The arguments passed are {args}')

    ips = { 
            "v1" : args.v1Server, 
            "v2" : args.v2Server
    }




def decodeImage(filestr):
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    return img


@app.route("/",)
def home():
    print('Entering ... ')
    url = 'http://' + args.explainURL + '/explain'
    print('URL is ', url)
    return render_template('home.html', explainURL  = url)

@app.route("/metadata")
def metadata():
    ip = args.elasticIp
    print(f'The elastic server ip address is {args.elasticIp} elastic server port is {args.elasticPort}')
    metadata = get_metadata(args.elasticIp, args.elasticPort)
    # print(metadata)
    return jsonify(metadata)

# API route
@app.route('/api', methods=['POST'])
def api():
    global ips
    filestr = request.files['file'].read()
    model_ver = request.form['model_ver']
    print(model_ver)
    img = decodeImage(filestr)
    img = load_preprocess_image(img)
    print(img.shape)
    preds = get_prediction(img=img, ver=model_ver, ips=ips)
    preds = np.array(preds['predictions'][0])
    idx = np.argsort(preds)[::-1][:3]
    top_scores = preds[idx]
    top_classes = class_names[idx]
    results = {
        'classes': list(top_classes),
        'scores': list(top_scores)
    }
    return jsonify(results) #"Image received!"  #(jsonify(list(zip(class_names, scores)))) #  # jsonify(preds)


if __name__ == '__main__':
    parse_arguments()
    app.run(host='0.0.0.0', port=8080)

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.utils import secure_filename

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path as osp
import glob
from datetime import datetime
from gps_plotter import GPSPlotter, NoMetaData
from loss import iou, dice_coef, dice_loss

#from models.mobilenet_unet import unet_model
from image_processor import tf_dataset, normalize_mask
from collections import namedtuple

ROI = namedtuple("ROI", ["imgFilename", "pixel_width"])

batch_size = 32
weights = "weights/weights-improvement-max-07-0.900.hdf5"

model = tf.keras.models.load_model(weights, custom_objects={"dice_loss":dice_loss, "dice_coef":dice_coef, "iou":iou})

print ("finished loading model ...")

#fixed failed to convolution algorithm https://github.com/tensorflow/tensorflow/issues/24828
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)

app = Flask(__name__)
app.secret_key = "1234"

basedir = os.path.abspath(os.path.dirname(__file__))
#print ("sqlite:///"+os.path.join(basedir, "tmp.sqlite"))

#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///"+os.path.join(basedir, "tmp.sqlite")
#app.config.from_object(os.environ["APP_SETTINGS"])


app.config[" SQLALCHEMY_TRACK_MODIFICATIONS "] = False
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'JPG'])

apikey = "AIzaSyAODtDkCU-PBBDjpDt5Xjbam8mkKw7dyrg"

db = SQLAlchemy(app)
migrate = Migrate(app, db)
map_file = "map.html"



# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
@app.route("/", methods=["GET"])
def index():
    
    #remove all the files in the upload folder at the very start to prevent plotting previous data
    images = [ f for f in glob.glob(app.config['UPLOAD_FOLDER'] + "*")]
        
    for f in images:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    return render_template("index.html")


@app.route("/predict_image", methods=["POST"])
def predict_image():
    # Get the name of the uploaded files
    print ("it hit thanks");
    uploaded_files = request.files.getlist("file[]")
    
    height = request.form["height"]
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to the upload
            # folder we setup
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Save the filename into a list, we'll use it later
            filenames.append(filename)
            # Redirect the user to the uploaded_file route, which
            # will basicaly show on the browser the uploaded file
    # Load an html page with a link to each uploaded file
    return redirect(url_for("show_map", height=height))
   

@app.route("/show_map/<height>")
def show_map(height):
    #create directory to store results ...
    out_folder= datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    #out_folder = os.path.join(os.getcwd(), out_folder)  
    os.mkdir(os.path.join("result", out_folder))
    
    img_paths = [ f for f in glob.glob(app.config['UPLOAD_FOLDER'] + "*")]
    
    #make predictions...
    test_dataset = tf_dataset(img_paths, batch=len(img_paths))
    test_dataset = test_dataset.as_numpy_iterator().next()
    predictions = model.predict(test_dataset)

    
    imgPaths_Hotspot = []
    for i in range(len(img_paths)):
        
        base = osp.splitext(osp.basename(img_paths[i]))[0]
        im = cv2.imread(img_paths[i])
        h,w = im.shape[:2]
        
        #h, w = cv2.imread(img_paths[i]).shape[:2]
        mask = normalize_mask(predictions[i], threshold=0.7)
        mask = (mask * 255).astype("uint8")
        mask = cv2.resize(mask , (w,h ))
        
        
        mask = [mask for _ in range(3)]
        mask = np.stack(mask, axis=2)
        #convert to red
        mask[np.where((mask==[255, 255, 255]).all(axis=2))] = [0,0,255]

        cv2.addWeighted(mask, 0.6, im, 0.4, 0, im)
        cv2.imwrite(os.path.join("result", out_folder, base+".jpg"), im)

        
        """
        #post processing of masks to remove noise...
        mask_vis = np.zeros(mask.shape)
        contours,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            contour_area = cv2.contourArea(c)
            if contour_area > 20.0:
                cv2.drawContours(mask_vis, [c], 0, 255, thickness=cv2.FILLED)
        cv2.resize(mask_vis, (w,h))
        cv2.imwrite(os.path.join("result", out_folder, base+".jpg"), mask_vis)
        """

        #check if numpy array is all zero hence (no hotspot)
        if np.all((mask==0)):
            continue
        
        mask = mask[:, :, 0] #get just one channel
        #find contours in mask...
        contours,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert(len(contours) > 0)
            
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        
        imgPaths_Hotspot.append(ROI(img_paths[i], w))
        #imgPaths_Hotspot.append(img_paths[i])

    try:
        g = GPSPlotter(apikey, imgPaths_Hotspot, height, os.path.join("result", out_folder, "hotspot_log.csv"))
        g.plot_gps_points(map_file)
   
    except NoMetaData as e:
        return "<h1> Image supplied does not have meta data</h1>"
    
    except IndexError:
        return "<h1> No hotspots detected in the images provided"
    return render_template(map_file)
if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=True)

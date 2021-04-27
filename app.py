from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.utils import secure_filename

from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pickle
import numpy as np
import io
import base64
import cv2
import os
import os.path as osp
import glob
from datetime import datetime
from gps_plotter import GPSPlotter, NoMetaData

from models.mobilenet_unet import unet_model
from image_processor import load_and_preprocess_testImage, normalize_mask


batch_size = 32
weights = "weights/weights-improvement-max-27-0.44.hdf5"
model = unet_model()
model.load_weights(weights)

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
apikey = "AIzaSyA3Je_LyveY2Ot6Z7UD25JLwihOKs1o46M"


db = SQLAlchemy(app)
migrate = Migrate(app, db)
map_file = "map.html"


# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
@app.route("/", methods=["GET"])
@app.route("/predict_image", methods=["GET", "POST"])
def predict_image():
    # Get the name of the uploaded files
    if request.method == "POST":
        uploaded_files = request.files.getlist("file[]")
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
        return redirect(url_for("show_map"))
    else:
        #remove all the files in the upload folder at the very start to prevent plotting previous data
        images = [ f for f in glob.glob(app.config['UPLOAD_FOLDER'] + "*")]
        
        for f in images:
            try:
                os.remove(f)
            except OSError as e:
                 print("Error: %s : %s" % (f, e.strerror))
                
    return render_template("index.html")



@app.route("/show_map")
def show_map():
    #create directory to store results ...
    out_folder= datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    out_folder = os.path.join(os.getcwd(), out_folder)
  
    os.mkdir(out_folder)
    img_paths = [ f for f in glob.glob(app.config['UPLOAD_FOLDER'] + "*")]
    test_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    test_dataset = test_dataset.map(load_and_preprocess_testImage).batch(batch_size)
 
    predictions = model.predict(test_dataset)
    
    imgPaths_noHotspot = []
    for i in range(len(img_paths)):
        
        base = osp.splitext(osp.basename(img_paths[i]))[0]
        mask = normalize_mask(predictions[i], threshold=0.7)
        mask = (mask * 255).astype("uint8")
        
        mask_vis = np.zeros(mask.shape)
        contours,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            contour_area = cv2.contourArea(c)
            if contour_area > 20.0:
                cv2.drawContours(mask_vis, [c], 0, 255, thickness=cv2.FILLED)
                
        cv2.imwrite(os.path.join("result", out_folder, base+".jpg"), mask_vis)
    

        #check if numpy array is all zero hence (no hotspot)
        if np.all((mask_vis==0)):
            continue
        
        imgPaths_noHotspot.append(img_paths[i])

    try:
        g = GPSPlotter(apikey, imgPaths_noHotspot)
        g.plot_gps_points(map_file)
   
    except NoMetaData:
        print ("exception...")
        return "<h1> Image supplied does not have meta data</h1>"
    
    except IndexError:
        return "<h1> No hotspots detected in the images provided"

    return render_template(map_file)
if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')

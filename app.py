from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug import secure_filename

from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont
from hotspotDetection import Roi
import tensorflow as tf
import pickle
import numpy as np
import io
import base64
import cv2
import os
import glob
from gps_plotter import GPSPlotter
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
        print ("post")
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
        print (filenames)
        return redirect(url_for("show_map"))
    return render_template("index.html")

@app.route("/show_map")
def show_map():
    images = [ f for f in glob.glob(app.config['UPLOAD_FOLDER'] + "*")]
    g = GPSPlotter(apikey, images)
    g.plot_gps_points(map_file)
    return render_template(map_file)


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')

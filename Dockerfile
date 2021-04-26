FROM tensorflow/tensorflow:latest

MAINTAINER Temi Odusina "T.J.SamOdusina@ljmu.ac.uk"

#opencv depenedencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y git

RUN pip install --no-cache-dir flask && pip install gmplot && pip install keras \
&& pip install opencv-python && pip install scikit-image && pip install Flask-SQLAlchemy && pip install Flask-Migrate && pip install scikit-learn

WORKDIR /home
RUN git clone https://github.com/temi92/UNET---Segmentation-model-for-hotspot-detection.git \
    && cd UNET---Segmentation-model-for-hotspot-detection \

CMD ["python", "app.py"]
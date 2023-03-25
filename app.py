#!/bin/bash
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
from model import predict
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = 'ids721'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image upload success')
        res, r_max, r_min, r_mean, r_var, g_max, g_min, g_mean, g_var, b_max, b_min, b_mean, b_var, symm, area = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        res_to_pass = 'Benign'
        if res == 1:
            res_to_pass = 'Malignant'
        return render_template('result.html',
                                filename=filename,
                                result=res_to_pass,
                                r_max=r_max,
                                r_min=r_min,
                                r_mean=r_mean, 
                                r_var=r_var, 
                                g_max=g_max, 
                                g_min=g_min, 
                                g_mean=g_mean,
                                g_var=g_var, 
                                b_max=b_max, 
                                b_min=b_min, 
                                b_mean=b_mean, 
                                b_var=b_var, 
                                symm=symm, 
                                area=area)
    
    flash('Please upload an image')
    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# docker images 
# docker container ls  
# docker build -t ids721-v1 .
# docker run -d -p 3000:3000 ids721-v1:latest
# docker kill 516a6fe94515
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
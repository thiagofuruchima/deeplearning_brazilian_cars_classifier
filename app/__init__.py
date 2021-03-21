import os
import time  
import base64
import json

from app import db 
from app.model import make_class_predictions, make_mock_predictions

from flask import (
    Flask, flash, render_template, url_for, redirect, current_app, request,
)


import io
from PIL import Image

def create_app():

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # set default config vaules
    app.config.from_mapping(
        SECRET_KEY='development',
        DATABASE=os.path.join(app.instance_path, 'db.sqlite'),
        UPLOAD_PATH=os.path.join(app.instance_path, '/uploads')
    )
    
    app.logger.info(app.instance_path)

    # override default config values from config.py file located in "instance" folder
    app.config.from_pyfile('config.py', silent=True)

    @app.route('/')
    def home():
        return render_template('home.html')        
        
    @app.route('/about')
    def about():
        return render_template('about.html')                
        
    @app.errorhandler(Exception)
    def handle_exception(e):    
        flash(str(e))    
        return render_template('home.html')
        
    @app.route('/eval', methods=['POST', 'GET'])
    def eval():

        # record start time
        start_time = time.time()
        
        filename = ''
        
        try:
            uploaded_file = request.files['file']        
            
            filename = uploaded_file.filename
        except:
            pass
        
        if (filename == ''):
            
            flash("No file selected. Please, select a picture of your car as a .jpg or .png file.")
            
            return redirect(url_for('home'))
            
        else:
        
            file_ext = os.path.splitext(filename)[1]
            
            if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
                flash('Invalid file format. Please, send a picture as a .jpg, .png file.')
                return redirect(url_for('home'))
            
            file_path = os.path.join(current_app.config['UPLOAD_PATH'], filename)
            
            # log eval start
            current_app.logger.info("Start of Evaluation...")
            
            #uploaded_file.save(file_path)
            
            img = Image.open(uploaded_file) 
            data = io.BytesIO()
            img.save(data, "JPEG")
            
            # use base64 encode to show the picture without saving it to a file
            encoded_img = base64.b64encode(data.getvalue())
            decoded_img = encoded_img.decode('utf-8')
            img_data = f"data:image/jpeg;base64,{decoded_img}"
            
            # make the top 5 class predictions
            prediction = make_class_predictions(uploaded_file)
            
            # log eval finish
            current_app.logger.info("End of Evaluation. Total time: " + str(round((time.time() - start_time),2)) + " seconds.")
                
            return render_template('result.html', img_data=img_data, prediction=prediction)
        
    return app
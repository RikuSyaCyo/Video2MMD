import os
import cv2
from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug import secure_filename
import json

UPLOAD_FOLDER = '/static/video/'
ALLOWED_EXTENSIONS = set(['mp4', 'mov'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def main_page():
    return render_template('webgl_loader_mmd.html')

@app.route('/upload', methods=['POST'])
def upload():
    print('get video file')
    print(request.files['video'])
    print('aaa')
    file = request.files['video']
    basepath = os.getcwd()
    print('basepath: ' + basepath)
    filename = secure_filename(file.filename)
    print('filename: ' + filename)
    filepath = basepath + app.config['UPLOAD_FOLDER'] + filename
    file.save(filepath)
    cam = cv2.VideoCapture(filepath)
    f=open('static/fps/fps_' + filename.split('.')[0] + '.txt', 'w')
    f.write(str(cam.get(5)))
    #TODO add to javascript speed control
    result = os.system('./../run_draft3.sh ./static/video/' + filename + ' -20 20')
    print(result)
    #return jsonify({"result":"upload complete!"})
    return redirect(url_for('main_page'))
    
@app.route('/para_change', methods=['POST'])
def para_change():
    print('get parameters')
    left_angle = request.values['left']
    right_angle = request.values['right']
    filename = request.values['filename']
    if(is_number(left_angle) and is_number(right_angle)):
        basepath = os.getcwd()
        print('basepath: ' + basepath)
        filename = secure_filename(filename)
        print('filename: ' + filename)
        result = os.system('./../run_vmd.sh ./static/video/' + filename + ' ' + left_angle + ' ' + right_angle)
        return_json = {'result': 'true'}
        return json.dumps(return_json)
    else:
        return_json = {'result': 'false', 'text': 'input is not correct data type!'}
        return json.dumps(return_json)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

if __name__ == '__main__':
    app.run()

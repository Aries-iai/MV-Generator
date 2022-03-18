
import os
import time
import argparse
from flask import Flask, render_template, request, redirect, flash, jsonify
from werkzeug.utils import secure_filename
from mvgen import generate_mv as gen

# set FLASK_ENV=development  https://www.cnblogs.com/wuyu1787/p/8919588.html

ALLOWED_EXTENSIONS = {'jpg', 'mp3'}

app = Flask(__name__)
app.config['SECRET_KEY'] = '1253434#@##444463456'
app.config['UPLOAD_FOLDER'] = './static'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_mp4_files():
    files = os.listdir('./static/videos')
    res = []
    for file in files:
        if file.endswith('.mp4'):
            res.append(('videos/' + file, file[:-4]))
    res.sort(key=lambda x:os.path.getmtime(os.path.join('static', x[0])), reverse=True)
    return res

def gen_video(path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', default=path, nargs='?', help='audio input path')
    parser.add_argument('-g','--genre', default='', help='audio input genre (pop, rock, rnb, ...)')
    parser.add_argument('-r', '--resolution', default='16/9', choices=['16/9', '40/17'])
    parser.add_argument('-e', '--extension', default='.mp4', choices=['.mp4', '.avi', '.mkv'])
    args = parser.parse_args()
    gen.main(args)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        videos = get_mp4_files()
        return render_template('index.html', videos=videos)

    if 'music' not in request.files:
        flash('No file part')
        return jsonify({'status': 0, 'info': 'No file！'})
    file = request.files['music']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return jsonify({'status': 0, 'info': 'No file！'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        return jsonify({'status': 1, 'music_file': filename}) #redirect(request.url)
    else:
        return jsonify({'status': 0, 'info': 'Error file type！'})

@app.route('/gen')
def gen_vid():
    filename = request.args.get('music_file')
    if not filename:
        return jsonify({'status': 0, 'info': 'no filename'})
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    gen_video(path)
    time.sleep(6)
    return jsonify({'status': 1, 'info': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
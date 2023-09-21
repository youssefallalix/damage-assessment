from flask import Flask, render_template, request
import requests
from gdal_translate import gdal_translate
from inference import inference

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    url_pre = request.form['url_pre']
    url_post = request.form['url_post']

    response_pre = requests.get(url_pre)
    with open('pre.tif', 'wb') as f:
        f.write(response_pre.content)

    response_post = requests.get(url_post)
    with open('post.tif', 'wb') as f:
        f.write(response_post.content)

    pre_processed_file = gdal_translate('pre.tif')
    post_processed_file = gdal_translate('post.tif')

    inference(pre_fn=pre_processed_file,
              post_fn=post_processed_file, output_fn='damagee_prediction.tif', gpu=1)

    return "Files downloaded successfully!"

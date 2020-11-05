
from random import choice
from flask import Flask, jsonify, request
import  requests
from io import BytesIO
import time

from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_seq2seq')

config['device'] = 'cpu'
config['predictor']['beamsearch']=False

model = Predictor(config)


import time


# imgs = ['/home/hisiter/Downloads/def/322e6cd30fe442feb34ae0763a9b6eec_nghi_.jpg', '/home/hisiter/Downloads/def/64d3d7d8c12f4ac4a0d8a278ad97f246_huyện_.jpg',
#         '/home/hisiter/Downloads/def/880f9e2a393c4f97b9f75cf39a2e078b_c_.jpg',
#         '/home/hisiter/Downloads/def/107_5_.jpg',
#         '/home/hisiter/Downloads/def/111_25-6-2000_.jpg',
#         '/home/hisiter/Pictures/Screenshot_20201102_145853.png']
# inputs = [Image.open(f) for f in imgs]

img = 'image/bien-quang-cao-hiflex-1.jpg'
input = Image.open(img)
w, h  = input.size
print(input.size)
t1 = time.time()
# s = model.predict_batch(inputs)
boxes = [[[{'x':559.375/w, 'y':70.3125/h}], [{'x':575.0/w, 'y':70.3125/h}], [{'x':575.0/w, 'y':81.25/h}], [{'x':559.375/w, 'y':81.25/h}]], [[{'x':354.0754089355469/w, 'y':82.46575927734375/h}], [{'x':499.1781311035156/w, 'y':67.19178009033203/h}], [{'x':502.6369934082031/w, 'y':100.0513687133789/h}], [{'x':357.5343017578125/w, 'y':115.3253402709961/h}]]]
s = model.predict_with_boxes(input, boxes)
print(s)
print(time.time() - t1)


#
# desktop_agents = [
#     'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
#     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
#     'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
#     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
#     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']
#
#
# def random_headers():
#     return {'User-Agent': choice(desktop_agents),
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
#
#
# def download_image(image_url):
#     header = random_headers()
#     response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)
#     image = Image.open(BytesIO(response.content))
#     return image
#
#
# def jsonify_str(output_list):
#     with app.app_context():
#         with app.test_request_context():
#             result = jsonify(output_list)
#     return result
#
#
# app = Flask(__name__)
#
#
# def create_query_result(input_url, results, error=None):
#     if error is not None:
#         results = 'Error: ' + str(error)
#     query_result = {
#         'results': results
#     }
#     return query_result
#
#
# @app.route("/query", methods=['GET', 'POST'])
# def queryimg():
#     if request.method == "POST":
#         data = request.get_data()
#         try:
#             img = Image.open(BytesIO(data))
#         except Exception as ex:
#             print(ex)
#             return jsonify(create_query_result("Upload", "upload error"))
#     else:
#         try:
#             image_url = request.args.get('url', default='', type=str)
#             img = download_image(image_url)
#         except Exception as ex:
#             return jsonify_str(create_query_result("", "", ex))
#     start = time.time()
#     result_text = model.predict(img)
#     time_pred = str(time.time() - start)
#     result = {"result: ": result_text, "predict time" : time_pred}
#     return jsonify_str(result)
#
# if __name__ == "__main__":
#     app.run("localhost", 1912, threaded=True, debug=False)
#

# -*- coding:utf-8 -*-
from flask import Flask,request
from tools.log import Logger
from algrithm import get_curve_points
from werkzeug import secure_filename
import json
log = Logger('./logs/server.log',level='info')

app = Flask(__name__)
app.config.from_object('setting')


@app.route('/get_curve_points',methods=['POST','GET'])
def curve_gcn():
	return curve_point(request)

def curve_point(request):
	if request.method == 'POST':
		log.logger.info('get post info')
		data = str(request.data, encoding='utf-8')
		json_data = json.loads(data)
		log.logger.debug(json_data)
		poly = get_curve_points(json_data['img_path'],json_data['region'],int(app.config['CONTEXT_SCALE']))
		result = {
			'poly':poly.tolist() #numpy array can't convert to json
		}
		log.logger.debug(result)
		return json.dumps(result,indent=4,ensure_ascii=False)
	return 'index html'


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False, port=5101 )
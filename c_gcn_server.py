# -*- coding:utf-8 -*-
from flask import Flask,request
from tools.log import Logger
from algrithm import get_curve_points,get_multi_curve_points
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
		log.logger.info('get single region post info')
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

@app.route('/multi_regions',methods=['POST','GET'])
def multi_regions():
	return multi_curve_points()

def multi_curve_points():
	if request.method == 'POST':
		log.logger.info('get multiple regions post info')
		data = str(request.data, encoding='utf-8')
		json_data = json.loads(data)
		log.logger.debug(json_data[0]['img_path'])
		result = get_multi_curve_points(json_data,app.config)
		# result = {
		# 	'poly':polys.tolist() #numpy array can't convert to json
		# }
		# log.logger.debug(result)
		# return json.dumps(json_data,indent=4,ensure_ascii=False)
		return json.dumps(result,indent=4,ensure_ascii=False)
	return 'multi... html'


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False, port=5101 )
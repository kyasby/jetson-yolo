import requests
import json
import datetime

def send_file(file_name, title):

	TOKEN = ''
	CHANNEL = ''

	files = {'file': open(file_name, 'rb')}
	param = {
	'token': TOKEN,
	'channels': CHANNEL,
	'title': "title"
	}

	requests.post(url = "https://slack.com/api/files.upload", params = param, files = files)


def send_regular_report(content):

	WEB_HOOK_URL = "https://hooks.slack.com/services/"
	
	dt_now = datetime.datetime.now()
	content = str(content).replace("},", "}\n")

	data = json.dumps({'text':        f"{dt_now} \n {content}",  #通知内容
					   'username':    'Jetson',  #ユーザー名
					   'icon_emoji':  ':smile_cat:',})  #アイコン})

	requests.post(WEB_HOOK_URL, data = data)


if __name__ == "__main__":
    
    i = send_file("./main.py", "main.py")
    print(i)

    content = ("cat: 2, 3 \n cat: 2, 3")
    send_regular_report(content)

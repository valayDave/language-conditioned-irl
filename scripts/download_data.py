import gdown
url = 'https://drive.google.com/uc?id=1hxHmeBEWxhaiIFYW4BKpatz_AFnmqNxt'
output = 'GDrive.tgz'
gdown.download(url, output, quiet=False)
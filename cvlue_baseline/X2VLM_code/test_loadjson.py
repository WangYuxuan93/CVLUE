import json

f = '../../en_data/caption/annotations/captions_train2014.json'

with open(f, 'r') as file:

    string = file.read()
    _source = json.loads(string)

print(type(_source['annotations']))
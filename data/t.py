import json

f = json.load(open('data.json', 'r'))
l = len(f)
json.dump(f[:l//16], open('test.json', 'w'), ensure_ascii=False)
json.dump(f[l//16:], open('train.json', 'w'), ensure_ascii=False)

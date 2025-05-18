import json, numpy as np
COLMAP = {0:'bg',1:'cyan',2:'red',3:'blue',4:'yellow',5:'green'}

def load_pair(path):
    d = json.load(open(path))
    return np.array(d['train'][0]['input']), np.array(d['train'][0]['output'])

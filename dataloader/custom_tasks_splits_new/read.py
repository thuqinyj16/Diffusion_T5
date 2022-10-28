import numpy as np
import json
with open("./random.json",'r') as load_f:
    load_dict = json.load(load_f)
    print("len of all:",len(load_dict['all']))
    for task in load_dict['all']:
        print("task:",task)


import os
import random

import numpy as np

SEED = 124
random.seed(SEED)
np.random.seed(SEED)

types = np.array("Ground Water Flying Electric Fire".split())
types_p = np.array([10, 7, 8, 2, 4])
types_p = types_p / types_p.sum()
widths_mu_p = np.array([22, 10, 123, 23, 1, 32, 1, 1])
widths_mu_p = widths_mu_p / widths_mu_p.sum()
widths_mu = np.linspace(0.01, 30, num=len(widths_mu_p))
names = open(os.path.join(os.path.dirname(__file__), "paranoids_names.txt")).readlines()
names = [i.strip() for i in names]
features = "name,type 1,type 2,type 3,width,height,length,weight,cute"
print(features)
features = features.split(",")
for i in names:
    t1, t2, t3 = np.random.choice(types, 3, replace=False, p=types_p)
    if random.random() < 0.6:
        t2, t3 = "", ""
    elif random.random() < 0.8:
        t3 = ""
    mu = np.random.choice(widths_mu, 1, p=widths_mu_p)
    sigma = mu / 4
    width = random.gauss(mu, sigma)
    width = np.clip(width, 0.01, 1e1000).item()
    height = width * (random.random() + 0.5)
    length = width * (random.random() + 0.5)
    weight = width * height * length * (random.random() + 0.5)
    cute = random.random() > 0.38
    p = {
        "name": i,
        "type 1": t1, "type 2": t2, "type 3": t3,
        "width": width, "height": height, "length": length, "weight": weight,
        "cute": cute,
    }
    cols = []
    for i in features:
        value = p[i.lower()]
        if type(value) == str and len(value) == 0:
            value = f'""'
        elif type(value) == float:
            value = f"{value:.6g}"
        value = str(value)
        cols.append(value)
    print(",".join(cols))


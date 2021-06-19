import catboost
import numpy as np

submodels = []

MAP = {0: "HOME", 1: "DRAW", 2: "AWAY", -1: "SKIP"}

def start():
    for i in range(5):
        m = catboost.CatBoostClassifier()
        m.load_model('model_v1_{}.cbm'.format(i+1))
        submodels.append(m)

def model(f):
    pp = [0, 0, 0]
    for m in submodels:
        p = [0, 0, 0]
#         p = m.predict_proba(f)
        for i in range(3):
            pp[i] += p[0][i]
    return np.argmax(pp)

def solve():
    n = int(input())
    for t in range(n):
        features = list(map(float, input().split()))
        d = model([features])
        print(MAP[d], flush=True)
        if t < n - 1:
            aux_info = list(map(float, input().split()))

# start()
solve()

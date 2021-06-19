import catboost
import numpy as np
from collections import defaultdict

submodels = []

MAP = {0: "HOME", 1: "DRAW", 2: "AWAY", -1: "SKIP"}

M = 5
V = 3
TH = 1.1

import tqdm
from scipy.stats import mode

K = 20

stats = defaultdict(list)

def process_before_match(line):
    Division, Time, home_team, away_team, Referee, home_coef, draw_coef, away_coef = list(line)
    def calc_f(seq):
        if len(seq) == 0:
            seq = [np.ones(12) * (-1)]
        seq = np.array(seq)
        return list(np.min(seq, axis=0).reshape(-1)) \
            + list(np.max(seq, axis=0).reshape(-1)) \
            + list(np.median(seq, axis=0).reshape(-1)) \
            + list(mode(seq, axis=0).mode.reshape(-1)) \
            + list(seq[-1])
    features = [Division, Time, home_team, away_team, Referee, home_coef, draw_coef, away_coef]
    home_match_seq = stats[home_team]
    away_match_seq = stats[away_team]

    features += calc_f(home_match_seq)
    features += calc_f(away_match_seq)

    return np.array(features)


def process_after_match(line):
    Division, Time, home_team, away_team, Referee, home_coef, draw_coef, \
    away_coef, full_time_home_goals, \
    full_time_away_goals, half_time_home_goals, half_time_away_goals, \
    home_shots, away_shots, home_shots_on_target, \
    away_shots_on_target, home_fouls, away_fouls, home_corners, \
    away_corners, home_yellow_cards, away_yellow_cards, \
    home_red_cards, away_red_cards = list(line)

    stats[home_team].append((Division, full_time_home_goals, \
    half_time_home_goals, \
    Referee, home_shots, home_shots_on_target, \
    home_fouls, home_corners, \
    home_yellow_cards,
    home_red_cards, home_coef, draw_coef))

    stats[away_team].append((Division, 
    full_time_away_goals, half_time_away_goals, \
    Referee, away_shots, \
    away_shots_on_target, away_fouls, \
    away_corners, away_yellow_cards, \
    away_red_cards, away_coef, draw_coef))

    if len(stats[home_team]) > K:
        stats[home_team] = stats[home_team][-K:]

    if len(stats[away_team]) > K:
        stats[away_team] = stats[away_team][-K:]

def start():
    for i in range(M):
        m = catboost.CatBoostClassifier()
        m.load_model('model_v{}_{}.cbm'.format(V, i+1))
        submodels.append(m)

def get_ext_f(f):
    return process_before_match(f)

def model(f, coefs):
    pp = np.zeros(3)
    f_ext = get_ext_f(f)
    for m in submodels:
        p = m.predict_proba(f_ext.reshape(1,-1))
        for i in range(3):
            pp[i] += p[0][i]
    pp /= M
    d = np.argmax(pp)
    if pp[d] * coefs[d] <= TH:
        d = -1
    return d

def get_coefs(f: list):
    return [f[-3], f[-2], f[-1]] # home, draw, away

def solve():
    n = int(input())
    for t in range(n):
        features = list(input().split())
        for i, f in enumerate(features):
            if i == 1:
                features[i] = -1
                continue
            features[i] = float(features[i])
        coefs = get_coefs(features)
        d = model(features, coefs)
        print(MAP[d], flush=True)
        aux_info = list(map(float, input().split()))
#         process_after_match(list(features) + list(aux_info))

start()
solve()

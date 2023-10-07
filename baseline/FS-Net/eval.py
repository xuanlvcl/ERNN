from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ALL_ = -1
TPR_KEY = 'TPR'
FPR_KEY = 'FPR'
FTF_KEY = 'FTF'


def _fpr_trp_app(real, pred, app_ind):
    real_app = real == app_ind
    pred_app = pred == app_ind
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for r, p in zip(real_app, pred_app):
        if r and p:
            TP += 1
        elif r and not p:
            FN += 1
        elif not r and p:
            FP += 1
        else:
            TN += 1
    return TP, TN, FP, FN


def _evaluate_fpr_and_tpr(real, pred):
    app_num = len(pred)
    real = np.concatenate(real)
    pred = np.concatenate(pred)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    TPR = {}
    FPR = {}
    for app_ind in tqdm(range(app_num), ascii=True, desc='Eval'):
        TP_app, TN_app, FP_app, FN_app = _fpr_trp_app(real, pred, app_ind)
        TP += TP_app
        TN += TN_app
        FP += FP_app
        FN += FN_app
        TPR[app_ind] = TP_app / (TP_app + FN_app)
        FPR[app_ind] = FP_app / (FP_app + TN_app)
    TPR[ALL_] = TP / (TP + FN)
    FPR[ALL_] = FP / (FP + TN)
    return TPR, FPR


def _evaluate_ftf(TPR, FPR, class_num):
    res = 0
    sam_num = np.array(class_num, dtype=np.float)
    sam_num /= sam_num.sum()

    for key in TPR:
        if key == ALL_:
            continue
        res += sam_num[key] * TPR[key] / (1 + FPR[key])
    return res


def save_res(res, filename):
    with open(filename, 'w') as fp:
        json.dump(res, fp, indent=1, sort_keys=True)


def evaluate(real, pred):
    example_len = [len(ix) for ix in real]
    TPR, FPR = _evaluate_fpr_and_tpr(real, pred)
    FTF = _evaluate_ftf(TPR, FPR, example_len)
    res = {
        TPR_KEY: TPR,
        FPR_KEY: FPR,
        FTF_KEY: FTF
    }
    return res

def eve(real,pred):
    acc=accuracy_score(real, pred)
    f1=f1_score(real,pred,average='macro')
    return acc,f1
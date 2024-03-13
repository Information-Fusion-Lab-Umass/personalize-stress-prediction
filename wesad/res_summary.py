from main import *

def calc_res(trail_name):
    scores = {
        "confmats": list(),
        "f1": list(),
        "roc_auc": list(),
        "acc": list()
    }

    for i in range(3):
        with open("data/exp_res/{}_record_{}".format(trail_name, i), "rb") as f:
            record = pickle.load(f)
        curr_scores = eval_res(record["y_preds"], record["y_trues"])
        for s in curr_scores:
            scores[s].append(curr_scores[s])
    
    for s in scores:
        if s == "confmats":
            print(s)
            print(np.sum(scores[s], axis=0))
            continue
        print(s, np.mean(scores[s]), "+-", np.std(scores[s]))

if __name__ == '__main__':
    trail_name = sys.argv[1]
    calc_res(trail_name)
import matplotlib.pyplot as plt
import wandb
import natsort
import numpy as np


np.set_printoptions(precision=3, suppress=True)


def get_mean_and_max_scores(runs):
    mean_scores = []
    for run in runs:
        mean_score_np = run.history(keys=["test/mean_score"], samples=100000).to_numpy()
        mean_scores.append(mean_score_np[:, 1])
    mean_scores = np.array(mean_scores)

    mean_scores = np.sum(mean_scores, axis=0) / len(mean_scores)

    max_scores = []
    for i in range(len(mean_scores)):
        max_scores.append(np.max(mean_scores[:i + 1]))

    max_scores = np.array(max_scores)

    return mean_scores, max_scores


def get_ExpAndNormRuns_AEFH():
    api = wandb.Api()
    runs = api.runs('jianzhou0420-the-university-of-adelaide/DecoupleActionHead_Stage2_Summary')
    normal_runs = []
    exp_runs = []
    for run in runs:
        if "AEFH" in run.name:
            if "Normal" in run.name and "Eval" in run.name:
                normal_runs.append(run)
            if "AEFH1000" in run.name:
                exp_runs.append(run)

    normal_runs = natsort.natsorted(normal_runs, key=lambda x: x.name)
    exp_runs = natsort.natsorted(exp_runs, key=lambda x: x.name)

    return normal_runs, exp_runs


def get_ExpAndNormRuns_ACK():
    api = wandb.Api()
    runs = api.runs('jianzhou0420-the-university-of-adelaide/DecoupleActionHead_Stage2_Summary')
    normal_runs = []
    exp_runs = []
    for run in runs:
        if "ACK" in run.name:
            if "Normal" in run.name and "Eval" in run.name:
                normal_runs.append(run)
            if "ACK1000" in run.name and "Normal" not in run.name:
                exp_runs.append(run)

    normal_runs = natsort.natsorted(normal_runs, key=lambda x: x.name)
    exp_runs = natsort.natsorted(exp_runs, key=lambda x: x.name)

    return normal_runs, exp_runs


def get_ExpAndNormRuns_ACK_new():
    api = wandb.Api()
    runs = api.runs('jianzhou0420-the-university-of-adelaide/Eval')
    normal_runs = []
    exp_runs = []
    for run in runs:
        if "Exp_Multitask_OneStop__ACK" in run.name:
            exp_runs.append(run)

        if "Exp_Multitask_Normal__ACK" in run.name:
            normal_runs.append(run)

    normal_runs = natsort.natsorted(normal_runs, key=lambda x: x.name)
    exp_runs = natsort.natsorted(exp_runs, key=lambda x: x.name)

    return normal_runs, exp_runs


if __name__ == "__main__":

    # normal_runs, exp_runs = get_ExpAndNormRuns_AEFH()
    # normal_runs, exp_runs = get_ExpAndNormRuns_ACK()
    normal_runs, exp_runs = get_ExpAndNormRuns_ACK_new()
    for run in normal_runs:
        print(run.name)

    print(f"Experimental runs: {len(exp_runs)}")
    for run in exp_runs:
        print(run.name)

    exp_mean_scores, exp_max_scores = get_mean_and_max_scores(exp_runs)
    normal_mean_scores, normal_max_scores = get_mean_and_max_scores(normal_runs)

    # print("Experimental mean scores:", exp_mean_scores)
    print("Experimental max scores:", exp_max_scores)
    # print("Normal mean scores:", normal_mean_scores)
    print("Normal max scores:", normal_max_scores)

    plt.figure()
    plt.plot(exp_mean_scores, label="Exp mean", color="red", linestyle="--")
    plt.plot(exp_max_scores, label="Exp max", color="red")
    plt.plot(normal_mean_scores, label="Normal mean", color="blue", linestyle="--")
    plt.plot(normal_max_scores, label="Normal max", color="blue")
    plt.xlabel("Step index")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Experimental vs. Normal Runs")
    plt.legend()
    plt.tight_layout()
    plt.show()

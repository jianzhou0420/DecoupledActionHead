from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import wandb
import natsort
import numpy as np


np.set_printoptions(precision=3, suppress=True)


def get_all_runs():
    api = wandb.Api()
    runs = api.runs('jianzhou0420-the-university-of-adelaide/DecoupleActionHead_Stage2_Summary')
    return runs


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


if __name__ == "__main__":

    runs = get_all_runs()
    runs_flagged = []

    for run in tqdm(runs):
        print(f"--- Run {run.id} ({run.name}) ---")
        try:
            file_obj = run.file("conda-environment.yaml")
            local_path = file_obj.download(replace=True).name

            with open(local_path, "r") as f:
                for lineno, line in enumerate(f, start=1):
                    if "mujoco==3.3.4" in line:
                        print(f"Found in {run.id} [line {lineno}]: {line.strip()}")
                        runs_flagged.append(run)

        except Exception as e:
            # catches missing file, download errors, etc.
            print(f"{run.id}: could not search file ({e})")

    print(f"\nTotal runs flagged: {len(runs_flagged)}")
    for run in runs_flagged:
        print(f"Run {run.id} ({run.name}) has mujoco==3.3.4 in conda-environment.yaml")

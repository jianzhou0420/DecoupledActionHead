from zero.evaluator import evaluate_run
import os
import time


def check_run_ready(run_dir):
    """
    Check if the run directory is ready for evaluation.
    This function checks for the existence of a specific file that indicates readiness.
    """
    ready_file = os.path.join(run_dir, "ready.flag")
    return os.path.exists(ready_file)


if __name__ == "__main__":

    runs_dir = "/data/eval_candidates"
    runs_exists_before = os.listdir(runs_dir)
    runs_exists_current = []
    runs_processed = []

    while True:
        # find all directories in runtime_dir
        runs_exists_current = os.listdir(runs_dir)
        runs_new = [d for d in runs_exists_current if d not in runs_exists_before]
        runs_new_ready = []
        for run in runs_new:

            pass

        time.sleep(2)

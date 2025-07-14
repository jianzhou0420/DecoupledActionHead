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
        runs_new_notprocess = [r for r in runs_new if r not in runs_processed]
        runs_new_notprocess_ready = [r for r in runs_new_notprocess if check_run_ready(os.path.join(runs_dir, r))]
        time.sleep(2)

        for run in runs_new_notprocess_ready:
            run_dir = os.path.join(runs_dir, run)
            print(f"Evaluating run: {run_dir}")
            evaluate_run(run_dir)
            runs_processed.append(run)

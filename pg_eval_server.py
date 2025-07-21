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


def flag_processed(run_dir):
    """
    Flag the run as processed by creating a file in the run directory.
    This function creates a 'processed.flag' file to indicate that the run has been evaluated.
    """
    processed_file = os.path.join(run_dir, "processed.flag")
    with open(processed_file, 'w') as f:
        f.write("This run has been processed.\n")


def check_run_processed(run_dir):
    """
    Check if the run has already been processed.
    This function checks for the existence of a 'processed.flag' file.
    """
    processed_file = os.path.join(run_dir, "processed.flag")
    return os.path.exists(processed_file)


if __name__ == "__main__":
    runs_dir = "/media/jian/data/cached_from_sub_machine/runtime"
    # first check all runs in the directory

    while True:
        not_processed_runs = []
        for run in os.listdir(runs_dir):
            run_dir = os.path.join(runs_dir, run)
            if check_run_ready(run_dir) and not check_run_processed(run_dir):
                not_processed_runs.append(run_dir)
                print(f"Run {run} is ready for evaluation and not processed yet.")
        print(f"Found {len(not_processed_runs)} runs to process.")

        # process not processed runs
        for run in not_processed_runs:
            print(f"Evaluating run: {run}")
            evaluate_run(run_dir=run, wandb_mode="online")
            flag_processed(run_dir)

        time.sleep(2)

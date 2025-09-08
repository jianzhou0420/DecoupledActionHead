import wandb
import os
import json
import pandas as pd

# --- Configuration ---
# The full path to the project in the format 'entity/project'
PROJECT_PATH = 'jianzhou0420-the-university-of-adelaide/ICRA_Decoupled_Final_Experiments'

# The directory where the data will be saved
DOWNLOAD_DIR = "wandb_downloads"

# --- Helper Function to clean the summary object ---


def clean_wandb_summary(summary_obj):
    """
    Recursively converts a wandb summary object (or a sub-dictionary)
    into a standard Python dictionary to make it JSON serializable.
    """
    cleaned_dict = {}
    for key, value in summary_obj.items():
        if isinstance(value, dict) or hasattr(value, 'keys'):
            cleaned_dict[key] = clean_wandb_summary(value)
        else:
            cleaned_dict[key] = value
    return cleaned_dict

# --- Main Script ---


def download_project_data(project_path: str):
    print(f"Connecting to Wandb project: {project_path}")
    api = wandb.Api()

    try:
        runs = api.runs(path=project_path)
        print(f"Found {len(runs)} runs in the project.")

        _, project_name = project_path.split('/')
        project_dir = os.path.join(DOWNLOAD_DIR, project_name)
        os.makedirs(project_dir, exist_ok=True)

        for run in runs:
            print(f"\n--- Processing run: {run.name} ({run.id}) ---")
            run_dir = os.path.join(project_dir, run.id)
            os.makedirs(run_dir, exist_ok=True)

            # --- 1. Download Configuration ---
            print("Downloading run configuration...")
            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(run.config, f, indent=4)
            print(f"  -> Saved config to {config_path}")

            # --- 2. Download Summary Metrics ---
            print("Downloading run summary metrics...")
            summary_path = os.path.join(run_dir, "summary.json")
            with open(summary_path, 'w') as f:
                cleaned_summary = clean_wandb_summary(run.summary)
                json.dump(cleaned_summary, f, indent=4)
            print(f"  -> Saved summary to {summary_path}")

            # --- 3. Download Run History ---
            print("Downloading run history...")
            history_path = os.path.join(run_dir, "history.csv")
            try:
                history_df = run.history(pandas=True)
                history_df.to_csv(history_path, index=False)
                print(f"  -> Saved history to {history_path}")
            except Exception as e:
                print(f"  -> Could not download history for this run. Details: {e}")

            # --- 4. Download Files ---
            print("Downloading run files...")
            files_dir = os.path.join(run_dir, "files")
            os.makedirs(files_dir, exist_ok=True)

            for file in run.files():
                # --- THIS IS THE NEW CODE ---
                if file.name.endswith('.mp4'):
                    print(f"  -> Skipping video file: {file.name}")
                    continue  # Skip to the next file

                # Download the file if it's not a video
                file.download(root=files_dir, replace=True)
                print(f"  -> Downloaded file: {file.name}")

            print("Finished downloading files for this run.")

    except wandb.errors.CommError as e:
        print(f"Error connecting to Wandb. Please check your project path.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    download_project_data(PROJECT_PATH)

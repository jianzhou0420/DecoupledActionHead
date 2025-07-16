import wandb

api = wandb.Api()
runs = api.runs('jianzhou0420-the-university-of-adelaide/DecoupleActionHead_Stage2_Summary')
print(f"Total runs: {len(runs)}")
print("Run names:")
for run in runs:
    print(run.name)
    # You can also access other attributes of the run, such as:
    # print(run.id)  # Run ID
    # print(run.state)  # Run state (e.g., 'finished', 'running', etc.)
    # print(run.created_at)  # Creation time of the run
    # print(run.summary)  # Summary metrics of the run

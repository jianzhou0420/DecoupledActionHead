import pandas as pd
import matplotlib.pyplot as plt
import wandb
import natsort
import numpy as np
from wandb.apis.public import Run
import json
import ast

np.set_printoptions(precision=3, suppress=True)


class SingleRunWrapper:
    name: str
    model: str
    seed: int
    mode: str
    task_alphabet: str
    run_handle: Run
    cfg: dict

    def __repr__(self):
        return f"model={self.model},mode={self.mode} seed={self.seed}, task_alphabet={self.task_alphabet})"


class DH:
    @staticmethod
    def get_ICRA_ALL_Runs(project_name='ICRA_Decoupled_Final_Experiments'):
        api = wandb.Api()
        runs = api.runs(f'jianzhou0420-the-university-of-adelaide/{project_name}')
        wrapped_runs = []

        for run in runs:
            this_run_wrapper = SingleRunWrapper()
            # ignore some runs
            try:
                this_run_wrapper.cfg = ast.literal_eval(run.config['cfg'])
            except:
                continue
            if run.state != 'finished':
                continue
            if 'seed' not in this_run_wrapper.cfg:
                continue

            this_run_wrapper.name = run.name
            this_run_wrapper.run_handle = run

            # mode
            this_run_wrapper.mode = this_run_wrapper.cfg['train_mode'].split('_')[0]
            # model name
            if 'DP_C' in run.name:
                this_run_wrapper.model = 'DP_C'
            elif 'DP_T' in run.name and 'DP_T_FILM' not in run.name:
                this_run_wrapper.model = 'DP_T'
            elif 'DP_T_FILM' in run.name:
                this_run_wrapper.model = 'DP_T_FILM'
            elif 'DP_MLP' in run.name and 'DP_MLP_2' not in run.name:
                this_run_wrapper.model = 'DP_MLP'
            elif 'DP_MLP_2' in run.name:
                this_run_wrapper.model = 'DP_MLP_2'
            else:
                raise ValueError(f"Unknown model in run name: {run.name}")

            # seed
            this_run_wrapper.seed = this_run_wrapper.cfg['seed']

            # task alphabet
            this_run_wrapper.task_alphabet = run.name.split('_')[-1]
            if this_run_wrapper.task_alphabet not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                continue
            wrapped_runs.append(this_run_wrapper)

            # print(history_df['test/mean_score'].to_numpy())
        wrapped_runs = natsort.natsorted(wrapped_runs, key=lambda x: (x.model, x.mode, x.seed, x.task_alphabet))
        return wrapped_runs

    def get_Eval_ALL_Runs(project_name='Eval'):
        api = wandb.Api()
        runs = api.runs(f'jianzhou0420-the-university-of-adelaide/{project_name}')
        wrapped_runs = []
        for run in runs:
            this_run_wrapper = SingleRunWrapper()
            # ignore some runs
            this_run_wrapper.name = run.name
            this_run_wrapper.run_handle = run
            if 'ACG' not in run.name:
                continue
            # model name
            if 'DP_C' in run.name:
                this_run_wrapper.model = 'DP_C'
            elif 'DP_T' in run.name and 'DP_T_FILM' not in run.name:
                this_run_wrapper.model = 'DP_T'
            elif 'DP_T_FILM' in run.name:
                this_run_wrapper.model = 'DP_T_FILM'
            elif 'DP_MLP' in run.name and 'DP_MLP_2' not in run.name:
                this_run_wrapper.model = 'DP_MLP'
            elif 'DP_MLP_2' in run.name:
                this_run_wrapper.model = 'DP_MLP_2'
            else:
                raise ValueError(f"Unknown model in run name: {run.name}")

            # this_run_wrapper.cfg = json.loads(run.json_config)
            # task alphabet
            this_run_wrapper.task_alphabet = run.name.split('_')[-1]
            this_run_wrapper.mode = run.name.split('_')[3].lower()
            this_run_wrapper.seed = int(run.name.split('_')[-4].split('seed')[-1])
            wrapped_runs.append(this_run_wrapper)
        wrapped_runs = natsort.natsorted(wrapped_runs, key=lambda x: (x.model, x.mode, x.seed, x.task_alphabet))
        return wrapped_runs

    @staticmethod
    def get_required_runs(all_runs, model_type, mode, seed, task_alphabet: str):
        filtered_runs = [r for r in all_runs if r.model == model_type and r.mode == mode and r.seed == seed and r.task_alphabet in task_alphabet]
        return natsort.natsorted(filtered_runs, key=lambda x: x.seed)

    @staticmethod
    def print_runs(runs):
        for r in runs:
            print(r)

    @staticmethod
    def get_mean_score_history(run: SingleRunWrapper, samples=100000):
        history_df = run.run_handle.history(keys=["test/mean_score", "epoch"], samples=samples)
        return history_df

    @staticmethod
    def get_max_score_history(run: SingleRunWrapper, samples=100000):
        history_df = run.run_handle.history(keys=["test/max_score", "epoch"], samples=samples)
        return history_df

    @staticmethod
    def get_mean_score_across_runs(runs, samples=100000):
        all_histories = []
        for r in runs:
            history_df = r.run_handle.history(keys=["test/mean_score", "epoch"], samples=samples)
            # 修复: 在访问列之前检查其是否存在
            if 'test/mean_score' in history_df.columns:
                all_histories.append(history_df['test/mean_score'].to_numpy())
            else:
                print(f"Warning: 'test/mean_score' not found in history for run {r.name}. Skipping this run.")

        # 修复: 如果没有找到任何历史数据，则返回一个空数组
        if not all_histories:
            print("Error: No runs had 'test/mean_score' logged. Cannot compute average.")
            return np.array([])

        min_length = min([len(h) for h in all_histories])
        all_histories = [h[:min_length] for h in all_histories]
        this_mean = np.mean(np.stack(all_histories, axis=0), axis=0)
        return this_mean

    @staticmethod
    def get_max_score_across_runs(runs, samples=100000):
        all_histories = []
        for r in runs:
            history_df = r.run_handle.history(keys=["test/max_score", "epoch"], samples=samples)
            # 修复: 在访问列之前检查其是否存在
            if 'test/max_score' in history_df.columns:
                all_histories.append(history_df['test/max_score'].to_numpy())
            else:
                print(f"Warning: 'test/max_score' not found in history for run {r.name}. Skipping this run.")

        # 修复: 如果没有找到任何历史数据，则返回一个空数组
        if not all_histories:
            print("Error: No runs had 'test/max_score' logged. Cannot compute average.")
            return np.array([])

        min_length = min([len(h) for h in all_histories])
        all_histories = [h[:min_length] for h in all_histories]
        this_mean = np.mean(np.stack(all_histories, axis=0), axis=0)
        return this_mean

    @staticmethod
    def show_mean_score_and_max_score(mean_score: list, show=False):
        # accumulated max
        max_score = np.maximum.accumulate(mean_score)
        print(f"Final Mean Score: {mean_score[-1]:.3f}")
        print(f"Final Max Score: {max_score[-1]:.3f}")
        if show:
            plt.plot(mean_score, label='Mean Score')
            plt.plot(max_score, label='Max Score')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.grid()
            plt.legend()
            plt.show()

    @staticmethod
    def show_max_score(max_score: list, show=False):
        print(f"Final Max Score: {max_score[-1]:.3f}")
        if show:
            plt.plot(max_score, label='Max Score')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.grid()
            plt.legend()
            plt.show()


def show_by_task(all_runs):
    table_data = []
    for model_type in ['DP_C', 'DP_T', 'DP_T_FILM', 'DP_MLP']:
        for seed in [0, 42, 420]:
            for train_mode in ['normal', 'stage2']:
                for task_alphabet in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                    test_runs = DH.get_required_runs(all_runs, model_type, train_mode, seed, task_alphabet)
                    if test_runs:
                        # 遍历每个 run 并保存其历史数据
                        for run_wrapper in test_runs:
                            run_history = run_wrapper.run_handle.history()

                            eval_columns = [col for col in run_history.columns if col.startswith('eval/')]
                            columns_to_keep = eval_columns

                            if 'epoch' in run_history.columns:
                                columns_to_keep = ['epoch'] + eval_columns
                            else:
                                print(f"Warning: 'epoch' column not found for run {run_wrapper.name}. CSV might lack epoch numbers.")

                            if not columns_to_keep or not eval_columns:
                                print(f"Warning: No 'eval/' metrics found for run {run_wrapper.name}. Skipping CSV save for this run.")
                                continue

                            eval_df = run_history[columns_to_keep].dropna()

                            filename = f"{run_wrapper.model}_{run_wrapper.mode}_seed{run_wrapper.seed}_task{run_wrapper.task_alphabet}_eval_history.csv"
                            eval_df.to_csv(filename, index=False)
                            print(f"Saved evaluation history for run {run_wrapper.name} to {filename}")

                        # 获取跨 run 的平均分数，并处理可能为空的情况
                        max_scores_across_runs = DH.get_max_score_across_runs(test_runs)

                        if max_scores_across_runs.size > 0:
                            final_max_score = max_scores_across_runs[-1]
                            table_data.append([model_type, train_mode, seed, task_alphabet, f"{final_max_score:.3f}"])
                        else:
                            # 如果没有找到数据，则在表中标记为 N/A
                            table_data.append([model_type, train_mode, seed, task_alphabet, "N/A"])

    headers = ["Model Type", "Train Mode", "Seed", "Task", "Final Max Score"]

    # Calculate column widths
    column_widths = [len(header) for header in headers]
    for row in table_data:
        for i, item in enumerate(row):
            column_widths[i] = max(column_widths[i], len(str(item)))

    print("\nIndividual Scores Per Task and Seed")
    print("---")

    # Print the header row
    header_line = "| "
    for i, header in enumerate(headers):
        header_line += header.ljust(column_widths[i]) + " | "
    print(header_line)

    # Print the separator line
    separator_line = "|-"
    for width in column_widths:
        separator_line += "-" * width + "-|-"
    separator_line = separator_line[:-1]
    print(separator_line)

    # Print the data rows
    for row in table_data:
        row_line = "| "
        for i, item in enumerate(row):
            row_line += str(item).ljust(column_widths[i]) + " | "
        print(row_line)

    # ---
    # Prepare data for the second table, which averages scores across seeds.
    # ---
    average_data = {}
    for row in table_data:
        model_type, train_mode, seed, task_alphabet, score_str = row
        # 处理 N/A 值
        if score_str == "N/A":
            continue
        score = float(score_str)
        key = (model_type, train_mode, task_alphabet)
        if key not in average_data:
            average_data[key] = {'total_score': 0.0, 'count': 0}
        average_data[key]['total_score'] += score
        average_data[key]['count'] += 1

    second_table_data = []
    for (model_type, train_mode, task_alphabet), values in average_data.items():
        if values['count'] > 0:
            avg_score = values['total_score'] / values['count']
            second_table_data.append([model_type, train_mode, task_alphabet, f"{avg_score:.3f}"])
        else:
            second_table_data.append([model_type, train_mode, task_alphabet, "N/A"])

    second_headers = ["Model Type", "Train Mode", "Task", "Average Final Max Score"]

    print("\n")
    print("---")
    print("Average Scores Across Seeds Per Task")
    print("---")

    # Calculate column widths for the second table
    second_column_widths = [len(header) for header in second_headers]
    for row in second_table_data:
        for i, item in enumerate(row):
            second_column_widths[i] = max(second_column_widths[i], len(str(item)))

    # Print the second header row
    second_header_line = "| "
    for i, second_header in enumerate(second_headers):
        second_header_line += second_header.ljust(second_column_widths[i]) + " | "
    print(second_header_line)

    # Print the second separator line
    second_separator_line = "|-"
    for width in second_column_widths:
        second_separator_line += "-" * width + "-|-"
    second_separator_line = second_separator_line[:-1]
    print(second_separator_line)

    # Print the second data rows
    for row in second_table_data:
        row_line = "| "
        for i, item in enumerate(row):
            row_line += str(item).ljust(second_column_widths[i]) + " | "
        print(row_line)


if __name__ == '__main__':
    all_runs = DH.get_ICRA_ALL_Runs(project_name='ICRA_Decoupled_Final_Experiments')
    show_by_task(all_runs)

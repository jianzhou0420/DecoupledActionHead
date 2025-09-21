import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import natsort
import numpy as np
from wandb.apis.public import Run
import json
import ast
np.set_printoptions(precision=3, suppress=True)
sns.set_theme(style="whitegrid")


class SingleRunWrapper:
    name: str
    model: str
    seed: int
    mode: str
    task_alphabet: str
    run_handle: Run
    cfg: dict

    def __repr__(self):
        return f"model={self.model},mode=${self.mode} seed={self.seed}, task_alphabet={self.task_alphabet})"


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
        history_df = run.run_handle.history(keys=["test/mean_score"], samples=samples)
        return history_df['test/mean_score'].to_numpy()

    @staticmethod
    def get_max_score_history(run: SingleRunWrapper, samples=100000):
        history_df = run.run_handle.history(keys=["test/max_score"], samples=samples)
        return history_df['test/max_score'].to_numpy()

    @staticmethod
    def get_mean_score_across_runs(runs, samples=100000):
        all_histories = []
        for r in runs:
            all_histories.append(DH.get_mean_score_history(r, samples=samples))
        min_length = min([len(h) for h in all_histories])
        all_histories = [h[:min_length] for h in all_histories]
        this_mean = np.mean(np.stack(all_histories, axis=0), axis=0)
        return this_mean

    @staticmethod
    def get_max_score_across_runs(runs, samples=100000):
        all_histories = []
        for r in runs:
            all_histories.append(DH.get_max_score_history(r, samples=samples))
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
                        max_scores_across_runs = DH.get_max_score_across_runs(test_runs)
                        final_max_score = max_scores_across_runs[-1]
                        table_data.append([model_type, train_mode, seed, task_alphabet, f"{final_max_score:.3f}"])

    headers = ["Model Type", "Train Mode", "Seed", "Task", "Final Max Score"]

    # Calculate column widths
    column_widths = [len(header) for header in headers]
    for row in table_data:
        for i, item in enumerate(row):
            column_widths[i] = max(column_widths[i], len(str(item)))

    print("Individual Scores Per Task and Seed")
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
        score = float(score_str)
        key = (model_type, train_mode, task_alphabet)
        if key not in average_data:
            average_data[key] = {'total_score': 0.0, 'count': 0}
        average_data[key]['total_score'] += score
        average_data[key]['count'] += 1

    second_table_data = []
    for (model_type, train_mode, task_alphabet), values in average_data.items():
        avg_score = values['total_score'] / values['count']
        second_table_data.append([model_type, train_mode, task_alphabet, f"{avg_score:.3f}"])

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
    for i, header in enumerate(second_headers):
        second_header_line += header.ljust(second_column_widths[i]) + " | "
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


def show_by_task_all_data_to_csv(all_runs, output_filename="all_run_max_scores_comparison"):
    all_scores = {}
    for model_type in ['DP_C', 'DP_T', 'DP_T_FILM', 'DP_MLP']:
        for seed in [0, 42, 420]:
            for train_mode in ['normal', 'stage2']:
                for task_alphabet in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                    test_runs = DH.get_required_runs(all_runs, model_type, train_mode, seed, task_alphabet)
                    if test_runs:
                        max_scores = DH.get_max_score_across_runs(test_runs)
                        if max_scores.size > 0:
                            column_name = f"{model_type}_{train_mode}_seed{seed}_{task_alphabet}"
                            all_scores[column_name] = max_scores

    if not all_scores:
        print("Error: No runs had 'test/max_score' data. No comparison CSV will be created.")
        return

    # Pad arrays to the same length with NaN
    max_len = max(len(arr) for arr in all_scores.values())
    padded_scores = {
        key: np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=np.nan)
        for key, arr in all_scores.items()
    }

    # Create and save the DataFrame
    df = pd.DataFrame(padded_scores)
    df.index.name = 'Epoch'
    output_filename = output_filename if output_filename.endswith('.csv') else output_filename + '.csv'
    df.to_csv(output_filename)

    print(f"\nAll run max scores comparison has been saved to: {output_filename}")


def get_averaged_group_data(csv_file: str, groups_to_plot: list):
    """
    根据 CSV 文件和多组参数提取和平均数据。

    Args:
        csv_file (str): 包含所有运行数据的 CSV 文件路径。
        groups_to_plot (list): 一个列表，包含多个组的参数。
                                每个组都是一个字典，包含 'label' 和 lists of 'model_type', 'train_mode', 'seed', 'task_alphabet'。

    Returns:
        tuple: (pd.DataFrame, dict)
               第一个元素是包含每组平均数据的 DataFrame。
               第二个元素是包含每组最终得分的字典。
    """
    try:
        df = pd.read_csv(csv_file, index_col='Epoch')
        averaged_df = pd.DataFrame()
        final_scores = {}

        from itertools import product

        for group in groups_to_plot:
            label = group['label']
            params = {k: v for k, v in group.items() if k != 'label'}

            column_names = []
            combinations = list(product(*params.values()))

            for combo in combinations:
                model, mode, seed, task = combo
                column_name = f"{model}_{mode}_seed{seed}_{task}"
                if column_name in df.columns:
                    column_names.append(column_name)
                else:
                    print(f"Warning: Column '{column_name}' not found for group '{label}'. Skipping this run.")

            if column_names:
                group_data = df[column_names]
                group_mean = group_data.mean(axis=1)

                # 将平均数据添加到新的DataFrame中
                averaged_df[label] = group_mean
                # 记录最终分数
                final_scores[label] = group_mean.iloc[-1]
            else:
                print(f"Warning: No valid runs found for group '{label}'. Skipping this group.")

        return averaged_df, final_scores

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found. Please make sure the file exists.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


def plot_advanced_seaborn_with_annotations(averaged_df: pd.DataFrame, final_scores: dict, plot_title: str, output_filename: str = None):
    """
    使用 Seaborn 绘制更美观的平均数据，并直接标注最终得分。

    Args:
        averaged_df (pd.DataFrame): 包含平均数据的 DataFrame。
        final_scores (dict): 包含每组最终得分的字典。
        plot_title (str): 图表的标题。
        output_filename (str, optional): 保存图表的文件名。如果为None，则不保存。
    """
    plt.figure(figsize=(12, 8))

    if averaged_df is not None and not averaged_df.empty:
        # 定义一个包含指定颜色的调色板
        colors = ['#5BC5DB', '#F0434F', '#545454', '#FFC300']

        # 将宽格式数据转换为长格式，以方便 Seaborn 绘图
        df_long = averaged_df.reset_index().melt(
            id_vars='Epoch',
            var_name='Group',
            value_name='Score'
        )

        sns.lineplot(
            data=df_long,
            x='Epoch',
            y='Score',
            hue='Group',
            style='Group',
            markers=True,
            dashes=False,
            palette=colors
        )

        # 在每条曲线的末尾直接标注最终得分
        for i, column in enumerate(averaged_df.columns):
            final_score = final_scores.get(column, np.nan)
            if not np.isnan(final_score):
                last_epoch = averaged_df.index[-1]
                last_score = averaged_df[column].iloc[-1]
                plt.text(
                    last_epoch + 1,
                    last_score,
                    f'{final_score:.3f}',
                    ha='left',
                    va='center',
                    fontsize=10,
                    weight='bold',
                    color=colors[i % len(colors)]  # 使用与曲线相同的颜色
                )

    plt.title(plot_title, fontsize=16, weight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title='Group', loc='best', fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    else:
        plt.show()


def plot_ICRA_subfigure(averaged_df: pd.DataFrame, final_scores: dict, plot_title: str, output_filename: str):
    """
    生成一个适合作为 ICRA 论文单列子图的高质量图表。
    图表大小为单列，并保存为 PDF。

    Args:
        averaged_df (pd.DataFrame): 包含用于绘图的平均数据。
        final_scores (dict): 包含每个组的最终分数。
        plot_title (str): 图表标题。
        output_filename (str): 保存图表的文件名（例如：'figure.pdf'）。
    """
    # 设置单列图表的大小，以英寸为单位（8.5 厘米 = 3.37 英寸）。
    fig_width = 3.37
    fig_height = fig_width / 1  # 保持良好的宽高比
    plt.figure(figsize=(fig_width, fig_height))

    if averaged_df is not None and not averaged_df.empty:
        # 定义一个颜色调色板。可以使用预定义的 Seaborn 调色板或自定义。
        colors = ['#5BC5DB', '#F0434F', '#545454', '#FFC300']

        # 将宽格式数据转换为长格式，以便使用 Seaborn 绘图
        df_long = averaged_df.reset_index().melt(
            id_vars='Epoch',
            var_name='Group',
            value_name='Score'
        )

        # 为图例创建自定义标签，包含最终分数
        legend_labels = []
        for label in averaged_df.columns:
            final_score = final_scores.get(label, np.nan)
            if not np.isnan(final_score):
                legend_labels.append(f'{label} ({final_score:.3f})')
            else:
                legend_labels.append(label)

        # 使用 sns.lineplot 绘制数据
        ax = sns.lineplot(
            data=df_long,
            x='Epoch',
            y='Score',
            hue='Group',
            style='Group',
            markers=False,  # 不使用标记点
            dashes=False,
            palette=colors,
            linewidth=1.5  # 较粗的线条以获得更好的可见性
        )

        # 使用自定义的图例标签
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=legend_labels, loc='upper left', fontsize=9, title_fontsize=9)

        # 设置带有适当字体大小的坐标轴标签和标题
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Score', fontsize=10)
        plt.title(plot_title, fontsize=10, weight='bold')

        # 微调刻度和刻度标签
        plt.tick_params(axis='both', which='major', labelsize=7)

        # 设置图表限制并添加网格
        plt.ylim(0, 1.05)  # 在顶部添加一个小缓冲区
        plt.grid(True, linestyle='--', alpha=0.6)

    # 使用 tight_layout 确保所有元素都能够无重叠地容纳
    plt.tight_layout()

    # 以高 DPI 以 PDF 格式保存图表以供发表
    if not output_filename.endswith('.pdf'):
        output_filename = output_filename.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")


def draw():
    DP_C_Normal_dict = {
        'label': 'DP-C-Normal',
        'model_type': ['DP_C'],
        'train_mode': ['normal'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }
    DP_C_Decoupled_dict = {
        'label': 'DP-C-Decoupled',
        'model_type': ['DP_C'],
        'train_mode': ['stage2'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }

    DP_T_Normal_dict = {
        'label': 'DP-T-Normal',
        'model_type': ['DP_T'],
        'train_mode': ['normal'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }
    DP_T_Decoupled_dict = {
        'label': 'DP-T-Decoupled',
        'model_type': ['DP_T'],
        'train_mode': ['stage2'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }
    DP_T_FILM_Normal_dict = {
        'label': 'DP-T-FILM_Normal',
        'model_type': ['DP_T_FILM'],
        'train_mode': ['normal'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }
    DP_T_FILM_Decoupled_dict = {
        'label': 'DP-T-FILM_Decoupled',
        'model_type': ['DP_T_FILM'],
        'train_mode': ['stage2'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }
    DP_MLP_Normal_dict = {
        'label': 'DP-MLP-Normal',
        'model_type': ['DP_MLP'],
        'train_mode': ['normal'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }

    DP_MLP_Decoupled_dict = {
        'label': 'DP-MLP-Decoupled',
        'model_type': ['DP_MLP'],
        'train_mode': ['stage2'],
        'seed': [0, 42, 420],
        'task_alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }

    def DP_C_vs_DP_MLP_Normal():
        groups_to_compare = [DP_C_Normal_dict, DP_MLP_Normal_dict]
        averaged_data_df, final_scores_dict = get_averaged_group_data(
            csv_file="in_distribution_all_run_max_scores_comparison.csv",
            groups_to_plot=groups_to_compare
        )
        plot_ICRA_subfigure(
            averaged_df=averaged_data_df,
            final_scores=final_scores_dict,
            plot_title="DP-C vs DP-MLP Normal",
            output_filename="DP_C_vs_DP_MLP_Normal.pdf"
        )

    def DP_C_vs_DP_MLP_Decoupled():
        groups_to_compare = [DP_C_Decoupled_dict, DP_MLP_Decoupled_dict]
        averaged_data_df, final_scores_dict = get_averaged_group_data(
            csv_file="in_distribution_all_run_max_scores_comparison.csv",
            groups_to_plot=groups_to_compare
        )
        plot_ICRA_subfigure(
            averaged_df=averaged_data_df,
            final_scores=final_scores_dict,
            plot_title="DP-C vs DP-MLP Decoupled",
            output_filename="DP_C_vs_DP_MLP_Decoupled.pdf"
        )

    def DP_T_vs_DP_T_FILM_Normal():
        groups_to_compare = [DP_T_Normal_dict, DP_T_FILM_Normal_dict]
        averaged_data_df, final_scores_dict = get_averaged_group_data(
            csv_file="in_distribution_all_run_max_scores_comparison.csv",
            groups_to_plot=groups_to_compare
        )
        plot_ICRA_subfigure(
            averaged_df=averaged_data_df,
            final_scores=final_scores_dict,
            plot_title="DP-T vs DP-T-FiLM Normal",
            output_filename="DP_T_vs_DP_T_FILM_Normal.pdf"
        )

    def DP_T_vs_DP_T_FILM_Decoupled():
        groups_to_compare = [DP_T_Decoupled_dict, DP_T_FILM_Decoupled_dict]
        averaged_data_df, final_scores_dict = get_averaged_group_data(
            csv_file="in_distribution_all_run_max_scores_comparison.csv",
            groups_to_plot=groups_to_compare
        )
        plot_ICRA_subfigure(
            averaged_df=averaged_data_df,
            final_scores=final_scores_dict,
            plot_title="DP-T vs DP-T-FiLM Decoupled",
            output_filename="DP_T_vs_DP_T_FILM_Decoupled.pdf"
        )

    DP_C_vs_DP_MLP_Normal()
    DP_C_vs_DP_MLP_Decoupled()
    DP_T_vs_DP_T_FILM_Normal()
    DP_T_vs_DP_T_FILM_Decoupled()


def table1_():
    all_runs = DH.get_ICRA_ALL_Runs()
    show_by_task(all_runs)
    # show_by_task_all_data_to_csv(all_runs, output_filename="in_distribution_all_run_max_scores_comparison")


if __name__ == '__main__':
    # draw()
    table1_()

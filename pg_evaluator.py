from zero.evaluator import evaluate_run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a policy on Robomimic tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--results_dir", type=str, default="data/outputs/eval_results",)
    parser.add_argument("--run_dir", type=str, default="/media/jian/data/outputs/Normal/23.27.09_normal_ACK_1000",)
    parser.add_argument("--n_envs", type=int, default=28, help="Number of environments to evaluate.")
    parser.add_argument("--n_test_vis", type=int, default=6, help="Number of test environments to visualize.")
    parser.add_argument("--n_train_vis", type=int, default=3, help="Number of training environments to visualize.")
    parser.add_argument("--n_train", type=int, default=6, help="Number of training episodes to run.")
    parser.add_argument("--n_test", type=int, default=50, help="Number of test episodes to run.")

    args = parser.parse_args()

    evaluate_run(seed=args.seed,
                 run_dir=args.run_dir,
                 results_dir=args.results_dir,
                 n_envs=args.n_envs,
                 n_test_vis=args.n_test_vis,
                 n_train_vis=args.n_train_vis,
                 n_train=args.n_train,
                 n_test=args.n_test
                 )


import argparse
from datautils import bed_datasets as datasets
from algorithms import alg_selector
DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]
def get_args():
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('--command', choices=['launch', 'delete_incomplete'], default='launch')
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=alg_selector.ALGORITHMS)
    parser.add_argument('--sub_algorithms', nargs='+', type=str, default=[])
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default=r'.\outputs\sweep_outs')
    parser.add_argument('--data_dir', type=str, default=r'.\datasets\MNIST')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, default='plain')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--is_clean_label',  action='store_true')  # True: use clean labels to train, else use corrupted labels
    parser.add_argument('--label_flip_p', type=float, default=0)  # corrupted the label by flipping
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--single_test_envs', type=bool, default=True)
    parser.add_argument('--skip_confirmation', type=bool, default=True)
    args = parser.parse_args()
    return args
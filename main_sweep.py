# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
python -m main_sweep\
        --command launch\
        --data_dir=./datasets/MNIST\
        --output_dir=./outputs/sweep_outs\
        --command_launcher plain\
        --algorithms ERM\
        --sub_algorithm 'None'\
        --datasets ColoredMNIST\
        --single_test_envs\
        --skip_confirmation\
        --steps 1000\
        --n_hparams 2\
        --n_trials 1

"""

import time
from algorithms import  alg_launchers
from params.sweep_params import get_args

if __name__ == "__main__":
    sweep_start_time = time.time()
    args = get_args()

    args_list = alg_launchers.make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        sub_algorithms=args.sub_algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        skip_model_save = skip_model_save,
        holdout_fraction=args.holdout_fraction,
        is_clean_label = args.is_clean_label,
        label_flip_p=args.label_flip_p,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams
    )
    is_cmd_launcher = False if args.command_launcher in ['plain'] else True
    jobs = [alg_launchers.Job(train_args, args.output_dir, is_cmd_launcher) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == alg_launchers.Job.DONE]),
        len([j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]),
        len([j for j in jobs if j.state == alg_launchers.Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == alg_launchers.Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            alg_launchers.ask_for_confirmation()
        launcher_fn = alg_launchers.REGISTRY[args.command_launcher]
        alg_launchers.Job.launch(to_launch, launcher_fn, is_cmd_launcher)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            alg_launchers.ask_for_confirmation()
        alg_launchers.Job.delete(to_delete)

    sweep_stop_time = time.time()
    print('#'*10, ' total_time = ', str((sweep_stop_time - sweep_start_time) / 60), ' min ', '#'*10)
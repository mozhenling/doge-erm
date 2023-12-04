"""
Adapted from DomainBed hparams_registry.py
"""
import numpy as np
import hashlib

def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)

def _hparams(algorithm, dataset, random_seed, args):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = [
        'BaseMNIST',
        'ColoredNoiseMNIST',
        'ColoredMNISTNoise', # 'ColoredNoiseMNIST',
        'MNISTColoredNoise',
        'NoiseColoredMNIST',

        'BaseFashion',
        'ColoredNoiseFashion',
        'NoiseColoredFashion',
        'FashionColoredNoise',

        'BaseCIFAR10',
        'ColoredNoiseCIFAR10',
        'NoiseColoredCIFAR10',
        'CIFAR10ColoredNoise',
    ]

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert (name not in hparams)
        random_state = np.random.RandomState(
                        seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    if args.steps is None:
        # log_min = np.log10(500)
        # log_max = np.log10(5000)
        # _hparam('steps', 1000, lambda r: round( 10**r.uniform(log_min, log_max)))
        _hparam('steps', 1000, lambda r:int(r.choice([500,  1000, 1500, 2000, 2500,
                                                      3000, 3500, 4000, 4500, 5000])))
    else:
        hparams['steps'] = (args.steps, args.steps)
    args.steps = hparams['steps'][0] if args.hparams_seed ==0 else hparams['steps'][1]
    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.
    # ----------------------------------- DoYoJo framework --------------------------
    if algorithm in ['DoYoJo','DoYoJoAlpha', "IRM", "SD", "VREx", "IGA", "Mixup", "MMD", "CORAL",
                     "CausIRL_CORAL", "CausIRL_MMD", 'Fish', "GroupDRO", "ANDMask",
                     "IB_ERM", "IB_IRM", "CAD", "CondCAD", "MLDG", "MTL", "RSC",
                     "SANDMask", "SagNet"]:

        if algorithm in ['DoYoJo','DoYoJoAlpha']:
            #------------------------------- Auto-Loss ------------------------------
            #-- disable auto-weighted loss
            _hparam('is_auto_loss', False, lambda r: bool(r.choice([False, False])))
            _hparam('awl_anneal_count_thresh',int(args.steps * 0.1),
                        lambda r: int(r.uniform(0, int(args.steps * 0.9))))
            #--future use
            _hparam('auto_loss_type', 'coef_var_loss', lambda r: r.choice(['coef_var_loss', 'coef_var_loss']))

            #------------------------------- Loss Weight -----------------------------
            _hparam('recon_weight',     10,  lambda r: 10 ** r.uniform(0, 2)   )
            _hparam('kld_weight',       1,   lambda r: 10 ** r.uniform(0, 2)   )
            _hparam('erm_alpha_weight', 1,   lambda r: 10 ** r.uniform(0, 3) )

            _hparam('erm_beta_weight',  100, lambda r: 10 ** r.uniform(0, 3)   )
            _hparam('erm_gamma_weight', 0.01,   lambda r: 10 ** r.uniform(-5, 0) )

            #'subalg_name'
            sub_algorithm = args.sub_algorithm if args.sub_algorithm is not None and args.sub_algorithm !='None' else 'ERM'
            hparams['subalg_name'] = (sub_algorithm, sub_algorithm)
            #---------------------------------
        else:
            sub_algorithm = 'None'

        # -- IRM
        if algorithm in ["IRM"] or (sub_algorithm in ["IRM"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('irm_penalty_weight', 1e2, lambda r: 10 ** r.uniform(-1, 5))
            _hparam('irm_penalty_anneal_iters', int(args.steps* 0.1),
                        lambda r: int(r.uniform(0, int(args.steps* 0.9))))

        # -- SD
        elif algorithm in ["SD"] or (sub_algorithm in ["SD"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('sd_penalty_weight', 0.1, lambda r: 10 ** r.uniform(-5, -1))

        # -- VREx
        elif algorithm in ["VREx"] or (sub_algorithm in ["VREx"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('vrex_penalty_weight', 1e1, lambda r: 10 ** r.uniform(-1, 5))
            _hparam('vrex_penalty_anneal_iters',int(args.steps * 0.1),
                    lambda r: int(r.uniform(0, int(args.steps * 0.9))))
        # -- IGA
        elif algorithm in ["IGA"] or (sub_algorithm in ["IGA"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('iga_penalty_weight', 1e3, lambda r: 10 ** r.uniform(1, 5))

        # -- Mixup
        elif algorithm in ["Mixup"] or (sub_algorithm in ["Mixup"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('mixup_alpha', 0.2, lambda r: 10 ** r.uniform(-1, -1))

        elif algorithm in ["MMD", "CORAL", "CausIRL_CORAL", "CausIRL_MMD"] \
                or (sub_algorithm in ["MMD", "CORAL", "CausIRL_CORAL", "CausIRL_MMD"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('mmd_gamma_weight', 1., lambda r: 10 ** r.uniform(-1, 1))

        elif algorithm in ['Fish'] or (sub_algorithm in ['Fish'] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('meta_lr', 0.5, lambda r: r.choice([0.05, 0.1, 0.5]))

        elif algorithm in ["GroupDRO"] or (sub_algorithm in ["GroupDRO"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

        elif algorithm in ["ANDMask"] or (sub_algorithm in ["ANDMask"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

        elif algorithm in ["IB_ERM"] or (sub_algorithm in ["IB_ERM"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('ib_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
            _hparam('ib_penalty_anneal_iters', int(args.steps* 0.1),
                        lambda r: int(r.uniform(0, int(args.steps* 0.9))))

        elif algorithm in ["IB_IRM"] or (sub_algorithm in ["IB_IRM"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('irm_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
            _hparam('irm_penalty_anneal_iters', int(args.steps* 0.1),
                        lambda r: int(r.uniform(0, int(args.steps* 0.9))))
            _hparam('ib_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
            _hparam('ib_penalty_anneal_iters', int(args.steps* 0.1),
                        lambda r: int(r.uniform(0, int(args.steps* 0.9))))

        elif algorithm in ["CAD", "CondCAD"]  or (sub_algorithm in ["CAD", "CondCAD"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('bn_los_weight', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
            _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
            _hparam('is_normalized', False, lambda r: False)
            _hparam('is_project', False, lambda r: False)
            _hparam('is_flipped', True, lambda r: True)

        elif algorithm in ["MLDG"] or  (sub_algorithm in ["MLDG"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('mldg_beta', 1., lambda r: 10 ** r.uniform(-1, 1))
            _hparam('n_meta_test', 2, lambda r: int(r.choice([1, 2]))) # replaced by random pairs in MLDG

        elif algorithm in ["MTL"] or (sub_algorithm in ["MTL"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

        elif algorithm in ["RSC"] or (sub_algorithm in ["RSC"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('rsc_f_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))
            _hparam('rsc_b_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))

        elif algorithm in ["SANDMask"] or (sub_algorithm in ["SANDMask"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
            _hparam('k', 1e+1, lambda r: 10 ** r.uniform(-3, 5))

        elif algorithm in ["SagNet"] or  (sub_algorithm in ["SagNet"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('sag_w_adv_weight', 0.1, lambda r: 10 ** r.uniform(-2, 1))

        elif algorithm == "Fishr" or  (sub_algorithm in ["Fishr"] and algorithm in ['DoYoJo','DoYoJoAlpha']):
            _hparam('lambda', 1000., lambda r: 10 ** r.uniform(1., 4.))
            _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
            _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10 ** r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10 ** r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2 ** r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10 ** r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10 ** r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10 ** r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.
    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-8, -4))
    else:
        _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2 ** r.uniform(3, 9))) # default 64
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2 ** r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2 ** r.uniform(3, 5.5)))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10 ** r.uniform(-6, -2))

    return hparams

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def default_hparams(args):
    algorithm = args.algorithm
    dataset = args.dataset
    random_seed = 0 # default
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, random_seed, args).items()}


def random_hparams(args):
    algorithm = args.algorithm
    dataset = args.dataset
    random_seed = seed_hash(args.hparams_seed, args.trial_seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_seed, args).items()}


def get_nets(input_shape, num_classes, num_domains, hparams, args):
    if args.sub_algorithm is None or args.sub_algorithm in [[],'None']:
        from networks import bed_nets as nets
        featurizer = nets.Featurizer(input_shape, hparams)
        n_outputs = featurizer.n_outputs * 2 if args.algorithm in ['MTL'] else featurizer.n_outputs
        classifier = nets.Classifier(n_outputs,  num_classes)
        return featurizer, classifier
    else:
        if args.nets_base in ['dense', 'Dense', 'dense_nets']:
            from networks import dense_nets as nets
            if args.algorithm in ['DoYoJo','DoYoJoAlpha']:
                return  nets.Dense_VAE_CMNIST(input_shape, num_classes, num_domains, hparams, args)
            else:
                featurizer = nets.Featurizer(input_shape, num_classes, num_domains, hparams, args)
                n_outputs = featurizer.n_outputs *2  if args.algorithm in ['MTL'] else featurizer.n_outputs
                classifier = nets.Classifier(n_outputs, input_shape, num_classes, num_domains,hparams, args)
                return featurizer, classifier
        else:
            raise NotImplementedError # future
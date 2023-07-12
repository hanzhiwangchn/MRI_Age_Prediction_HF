import math, torch
import torch.nn as nn
import monai
from torchsummary import summary

def build_model(args):
    """
    In general, *_stride represents the models using stride=2.
    We implement *_stride architecture to increase the training speed for ABIDE data-sets
    """
    if args.model == 'densenet':
        model = build_densenet121_monai().to(args.device)

    # parameter initialization
    def weights_init(m):
        if isinstance(m, nn.Conv3d):
            if args.params_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight.data)
            elif args.params_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight.data)
    if args.params_init != 'default':
        model.apply(weights_init)

    # model_config for output files
    model_config = f'{args.model}_loss_{args.loss_type}_skewed_{args.skewed_loss}_' \
                   f'correlation_{args.correlation_type}_dataset_{args.dataset}_' \
                   f'{args.comment}_rnd_state_{args.random_state}'
    return model, model_config


# ------------------- Scratch DenseNet from MONAI ---------------------
def build_densenet121_monai():
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
    return model

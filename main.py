import argparse, logging, os

from utils.common_utils import RunManager
from utils.build_dataset import build_dataset
from utils.build_processor import build_processor
from utils.build_model import build_model
from utils.build_loss_function import build_loss_function
from utils.build_training_loop import build_loader, build_optimizer, train_val_test_pt
from utils.common_utils import update_args

import config
config.init()

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results'
os.makedirs(results_folder, exist_ok=True)

"""
Reference:
H. Wang, M. S. Treder, D. Marshall, D. K. Jones and Y. Li, 
"A Skewed Loss Function for Correcting Predictive Bias in Brain Age Prediction," 
in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3231730.
"""


def build_parser():
    """
    build parser for MRI Age Prediction.
    A template for running the code through the terminal is listed below:
    For the skewed loss, python main.py --skewed-loss --compact-dynamic --comment run0
    For two-stage correction, python main.py --two-stage-correction --comment run1
    """
    parser = argparse.ArgumentParser(description='Brain MRI Age Prediction')
    parser.add_argument('--model', type=str, default='densenet', choices=['densenet'],
                        help='model configurations')
    parser.add_argument('--loss-type', type=str, default='L1', choices=['L1', 'L2', 'SVR'],
                        help='normal loss function configurations')
    parser.add_argument('--correlation-type', type=str, default='pearson', choices=['pearson', 'spearman'],
                        help='correlation metric configurations')
    parser.add_argument('--skewed-loss', action='store_true', default=False,
                        help='use skewed loss function')
    # dynamic lambda strategy config
    parser.add_argument('--compact-dynamic', action='store_true', default=False,
                        help='a compact dynamic-lambda algorithm for the skewed loss')
    parser.add_argument('--compact-target', type=str, default='validation', choices=['train', 'validation'],
                        help='compact dynamic-lambda config: '
                             'specify on which data-set we want the correlation to move toward zero')
    parser.add_argument('--compact-update-interval', type=int, default=2,
                        help='compact dynamic-lambda config: '
                             'update lambda value every a certain number of epoch')
    parser.add_argument('--compact-init-multiplier', type=float, default=1.4,
                        help='compact dynamic-lambda config: '
                             'initialize a multiplier in the stage-2 when updating lambda')
    # apply the two-stage bias correction algorithm
    parser.add_argument('--two-stage-correction', action='store_true', default=False,
                        help='use the two-stage correction approach for the normal loss')
    # frequently used settings
    # NOTE: In the manuscript, we add experiments for distribution shifts.
    #  The updated code is not included in this script for the sake of simplicity.
    parser.add_argument('--dataset', type=str, default='camcan', choices=['camcan'],
                        help='specify which data-set to use')
    parser.add_argument('--random-state', type=int, default=1000,
                        help='used in train test data-set split')
    parser.add_argument('--comment', type=str, default='run0',
                        help='comments to distinguish different runs')
    # default settings
    parser.add_argument('--val-test-size', type=float, default=0.2,
                        help='proportion of validation & test set of the total data-set')
    parser.add_argument('--test-size', type=float, default=0.5,
                        help='proportion of test set of the "validation & test" set')
    parser.add_argument('--init-lambda', type=float, default=1.0,
                        help='default lambda value for the skewed loss')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-train-epochs', type=int, default=20, help='number of epoch')
    parser.add_argument('--params-init', type=str, default='kaiming_uniform',
                        choices=['default', 'kaiming_uniform', 'kaiming_normal'],
                        help='weight initializations')
    parser.add_argument('--acceptable-correlation-threshold', type=float, default=0.05,
                        help='acceptable threshold for correlation when selecting best model')
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='save models with the lowest validation loss in training to prevent over-fitting')
    parser.add_argument('--run-code-test', action='store_true', default=True,
                        help='a compact dynamic-lambda algorithm for the skewed loss')
    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--lr-scheduler-type', type=str, default='linear', help='lr scheduler type')
    parser.add_argument('--num-warmup-steps', type=int, default=0, help='num warmup steps')
    parser.add_argument('--test', action='store_true', default=False,
                        help='testing')
    return parser


def main():
    """overall workflow of MRI Age Prediction"""
    # build parser
    args = build_parser().parse_args()
    
    # update args based on different datasets
    args = update_args(args)
    logger.info(f'Parser arguments are {args}')

    # build dataset
    dataset_train, dataset_val, dataset_test, args = build_dataset(args=args)
    
    # build processor
    dataset_train, dataset_val, dataset_test = build_processor(dataset_train=dataset_train, 
        dataset_val=dataset_val, dataset_test=dataset_test)

    # build model
    model, model_config = build_model(args)

    # build loss function
    loss_fn_train, _, _ = build_loss_function(args=args)

    # build dataloader
    train_loader, val_loader, test_loader = build_loader(args, dataset_train, dataset_val, dataset_test)

    # build optimizer
    optimizer, lr_scheduler = build_optimizer(model=model, train_loader=train_loader, args=args)

    # build RunManager to save stats from training
    m = RunManager(args=args)
    m.begin_run(train_loader, val_loader, test_loader)

    # train and evaluate
    train_val_test_pt(args=args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, m=m, loss_fn_train=loss_fn_train)

    logger.info('Model finished!')


if __name__ == '__main__':
    main()

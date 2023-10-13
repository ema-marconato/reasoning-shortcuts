from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models



def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    # dataset
    parser.add_argument('--dataset', default='addmnist', type=str, choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--task', default='addition', type=str, choices=['addition', 'product', 'multiop'],
                        help='Which operation on two-MNIST.')
    # model settings
    parser.add_argument('--model', type=str, default="mnistdpl", help='Model name.', choices=get_all_models())
    parser.add_argument('--c_sup', type=float, default=0,   help='Fraction of concept supervision on concepts')
    parser.add_argument('--which_c', type=int, nargs='+', default=[-1], help='Which concepts explicitly supervise (-1 means all)')
    parser.add_argument('--joint', action='store_true', default=False, help='Process the image as a whole.')
    parser.add_argument('--splitted', action='store_true', default=False, help='Create different encoders.')
    parser.add_argument('--entropy', action='store_true', default=False, help='Activate entropy on batch.')
    # weights of logic
    parser.add_argument('--w_sl',  type=float, default=10, help='Weight of Semantic Loss')
    # weight of mitigation
    parser.add_argument('--gamma', type=float, default=1, help='Weight of mitigation')
    # additional hyperparams
    parser.add_argument('--w_rec', type=float, default=1, help='Weight of Reconstruction')
    parser.add_argument('--beta',  type=float, default=2, help='Multiplier of KL')
    parser.add_argument('--w_h',   type=float, default=1, help='Weight of entropy')
    parser.add_argument('--w_c',   type=float, default=1, help='Weight of concept sup')
    
    # optimization params
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=2, help='Warmup epochs.')
    parser.add_argument('--exp_decay', type=float, default=0.99, help='Exp decay of learning rate.')
    
    # learning hyperams
    parser.add_argument('--n_epochs',   type=int, default=50, help='Number of epochs per task.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')

def add_management_args(parser: ArgumentParser) -> None:
    # random seed
    parser.add_argument('--seed', type=int, default=None, help='The random seed.')
    # verbosity
    parser.add_argument('--notes', type=str, default=None, help='Notes for this run.')
    parser.add_argument('--non_verbose', action='store_true')
    # logging
    parser.add_argument('--wandb', type=str, default=None,  help='Enable wandb logging -- set name of project')
    # checkpoints
    parser.add_argument('--checkin',  type=str, default=None, help='location and path FROM where to load ckpt.' )    
    parser.add_argument('--checkout', type=str, default=None, help='location and path  TO  where to store ckpt.' )    
    # post-hoc evaluation
    parser.add_argument('--posthoc',  action='store_true', default=False, help='Used to evaluate only the loaded model')
    parser.add_argument('--validate', action='store_true', default=False, help='Used to evaluate on the validation set for hyperparameters search')

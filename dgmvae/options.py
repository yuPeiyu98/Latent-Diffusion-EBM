from dgmvae.utils import str2bool, process_config
import argparse
import logging
# import dgmvae.models.sent_models as sent_models
# import dgmvae.models.sup_models as sup_models
# import dgmvae.models.dialog_models as dialog_models

def add_default_training_parser(parser):
    parser.add_argument('--op', type=str, default='adam')
    parser.add_argument('--backward_size', type=int, default=5)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--init_w', type=float, default=0.08)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--lr_hold', type=int, default=3)
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--improve_threshold', type=float, default=0.996)
    parser.add_argument('--patient_increase', type=float, default=2.0)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--print_step', type=int, default=500)
    parser.add_argument('--fix_batch', type=str2bool, default=False)
    parser.add_argument('--ckpt_step', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--preview_batch_num', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='greedy')
    parser.add_argument('--avg_type', type=str, default='seq')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--forward_only', type=str2bool, default=False)
    parser.add_argument('--load_sess', type=str, default="", help="Load model directory.")
    return parser

def add_default_cond_training_parser(parser):
    parser.add_argument('--op', type=str, default='adam')
    parser.add_argument('--backward_size', type=int, default=30)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    parser.add_argument('--init_w', type=float, default=0.1)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--lr_hold', type=int, default=3)
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--improve_threshold', type=float, default=0.996)
    parser.add_argument('--patient_increase', type=float, default=4.0)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--loss_type', type=str, default="e2e")

    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--fix_batch', type=str2bool, default=False)
    parser.add_argument('--ckpt_step', type=int, default=500)
    parser.add_argument('--freeze_step', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--preview_batch_num', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='greedy')
    parser.add_argument('--avg_type', type=str, default='seq')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--forward_only', type=str2bool, default=False)
    parser.add_argument('--load_sess', type=str, default="", help="Load model directory.")

    parser.add_argument('--embedding_path', type=str, default="data/word2vec/smd.txt", help="word embedding file.")
    return parser

def add_default_variational_training_parser(parser):
    # KL-annealing
    parser.add_argument('--anneal', type=str2bool, default=True)
    parser.add_argument('--anneal_function', type=str, default='logistic')
    parser.add_argument('--anneal_k', type=float, default=0.0025)
    parser.add_argument('--anneal_x0', type=int, default=2500)
    parser.add_argument('--anneal_warm_up_step', type=int, default=0)
    parser.add_argument('--anneal_warm_up_value', type=float, default=0.000)

    # Word dropout & posterior sampling number
    parser.add_argument('--word_dropout_rate', type=float, default=0.0)
    parser.add_argument('--post_sample_num', type=int, default=20)
    parser.add_argument('--sel_metric', type=str, default="elbo", help="select best checkpoint base on what metric.",
                        choices=['elbo', 'obj'],)

    # Other:
    parser.add_argument('--aggressive', type=str2bool, default=False)
    return parser

def add_default_data_parser(parser):
    # Data & logging path
    parser.add_argument('--data', type=str, default='ptb')
    parser.add_argument('--data_dir', type=str, default='data/ptb')
    parser.add_argument('--log_dir', type=str, default='logs/ptb')
    # Draw points
    parser.add_argument('--fig_dir', type=str, default='figs')
    parser.add_argument('--draw_points', type=str2bool, default=False)
    return parser



def get_parser(model_class="sent_models"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GMVAE")
    parser = add_default_data_parser(parser)
    parser = add_default_training_parser(parser)
    parser = add_default_variational_training_parser(parser)

    config, unparsed = parser.parse_known_args()

    try:
        model_name = model_class + "." + config.model
        model_class = eval(model_name)
        parser = model_class.add_args(parser)
    except Exception as e:
        raise ValueError("Wrong model" + config.model)

    config, _ = parser.parse_known_args()
    print(config)
    config = process_config(config)
    return config


def get_parser_cond(model_class="dialog_models"):
    # Conditional generation

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GMVAE")
    parser = add_default_data_parser(parser)
    parser = add_default_cond_training_parser(parser)
    parser = add_default_variational_training_parser(parser)

    config, unparsed = parser.parse_known_args()

    try:
        model_name = model_class + "." + config.model
        model_class = eval(model_name)
        parser = model_class.add_args(parser)
    except Exception as e:
        raise ValueError("Wrong model" + config.model)

    config, _ = parser.parse_known_args()
    print(config)
    config = process_config(config)
    return config




import argparse

from src.utils import get_data_root, get_project_root, get_project_root_path

MODEL_NAMES = ['bidaf', 'roberta-qa']

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_common_args(parser=None):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--project_root', type=str, default=get_project_root())
    parser.add_argument('--data_root', type=str, default=get_data_root())
    parser.add_argument('--dataset_name', type=str, default='squad')

    parser.add_argument('--train_record_file', type=str, default='train.npz')
    parser.add_argument('--dev_record_file', type=str, default='dev.npz')
    parser.add_argument('--test_record_file', type=str, default='test.npz')

    parser.add_argument('--use_pt_we', type=str2bool, default=True, help="Use pre-trained word embeddings")
    parser.add_argument('--use_roberta_token', type=str2bool, default=False, help="Use RobertaTokenizer to map words to indices")
    parser.add_argument('--word_emb_file', type=str, default='glove_word_emb.json', help='file name where to save relevant word embeddings')
    parser.add_argument('--char_emb_file', type=str, default='char_emb.json')

    parser.add_argument('--train_eval_file', type=str, default='train_eval.json')
    parser.add_argument('--dev_eval_file', type=str, default='dev_eval.json')
    parser.add_argument('--test_eval_file', type=str, default='test_eval.json')

    parser.add_argument('--use_squad_v2',
                        type=str2bool,
                        default=False,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--debug', type=str2bool, default=False)

    return parser


def add_preproc_args(parser):
    """Get arguments needed for pre-processing SQuAD """
    #parser = argparse.ArgumentParser('Download and pre-process SQuAD')
    if parser is None:
        parser = add_common_args()
    #add_common_args(parser)

    parser.add_argument('--download', type=int, default=0)

    # note: we used Squad v1.1
    parser.add_argument('--train_url', type=str, default='https://github.com/chrischute/squad/data/train-v2.0.json') # 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
    parser.add_argument('--dev_url', type=str, default='https://github.com/chrischute/squad/data/dev-v2.0.json')
    parser.add_argument('--test_url', type=str, default='https://github.com/chrischute/squad/data/test-v2.0.json')

    parser.add_argument('--train_file', type=str, default='train-v1.1.json')
    parser.add_argument('--dev_file', type=str, default='dev-v1.1.json')

    parser.add_argument('--tokenizer', type=str, default='spacy', choices=['spacy', 'roberta'])

    parser.add_argument('--dev_meta_file', type=str, default='dev_meta.json')
    parser.add_argument('--test_meta_file', type=str, default='test_meta.json')
    parser.add_argument('--word2idx_file', type=str, default='word2idx.json')
    parser.add_argument('--char2idx_file', type=str, default='char2idx.json')
    parser.add_argument('--answer_file', type=str, default='answer.json')

    parser.add_argument('--vocab_size', type=int, default=30000, help='Max number of word in vocabulary')
    parser.add_argument('--max_seq_len', type=int, default=400, help='Max number of words in question + context')
    parser.add_argument('--context_limit', type=int, default=400, help='Max number of words in a context')
    parser.add_argument('--ques_limit', type=int, default=50, help='Max number of words to keep from a question')
    parser.add_argument('--ans_limit', type=int, default=30, help='Max number of words in a training example answer')
    parser.add_argument('--doc_stride', type=int, default=128, help='Number of tokens for context truncation')

    parser.add_argument('--test_context_limit', type=int, default=1000, help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit', type=int, default=100, help='Max number of words in a question at test time')

    parser.add_argument('--char_limit', type=int, default=16, help='Max number of chars to keep from a word')
    parser.add_argument('--char_dim', type=int, default=64, help='Size of char vectors (char-level embeddings)')

    parser.add_argument('--glove_url', type=str, default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--glove_dir', type=str, default='glove')
    parser.add_argument('--glove_dim', type=int, default=300, help='Size of GloVe word vectors to use')
    parser.add_argument('--we_dim', type=int, default=300, help='Size of word embeddings vectors')
    parser.add_argument('--glove_num_vecs', type=int, default=2196017, help='Number of GloVe vectors')

    parser.add_argument('--include_test_examples',
                        type=str2bool,
                        default=False,
                        help='Process examples from the test set')

    args = parser.parse_args()

    return args


def get_train_args(parser=None):
    """Get arguments needed in train.py."""
    if parser is None:
        parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000, #50000
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.') #0.5
    parser.add_argument('--l2_wd', type=float, default=0, help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10, #30
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name', type=str, default='F1', choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints', type=int, default=5, help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed', type=int, default=224, help='Random seed for reproducibility.')
    parser.add_argument('--use_ema', type=str2bool, default=True, help='use Exp Moving Average for model parameters')

    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    if args.set_finetune_def:
        args = set_default_finetune_args(args)

    return args


def get_test_args(parser=None):
    """Get arguments needed in test.py."""

    if parser is None:
        parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args



def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_name', type=str, default='bidaf', choices=MODEL_NAMES)

    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adadelta'])
    parser.add_argument('--num_workers', type=int, default=0, help='Number of sub-processes to use per data loader.')

    parser.add_argument('--max_ans_len', type=int, default=15, help='Maximum length of a predicted answer.')

    parser.add_argument('--save_dir',
                        type=str,
                        default=str(get_project_root_path().joinpath('save')),
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size per GPU')

    parser.add_argument('--d_hidden', type=int, default=100, help='Number of features in encoder hidden layers.')

    parser.add_argument('--num_visuals', type=int, default=10, help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')

    parser.add_argument('--freeze_bert_encoder', type=str2bool, default=True, help='Freeze layers of *BERT encoder')
    parser.add_argument('--freeze_we_embs', type=str2bool, default=False, help='Freeze word embeddings')
    parser.add_argument('--set_finetune_def', type=str2bool, default=False, help='Activate default finetuning parameters (e.g. lr=2e-5)')

    parser.add_argument('--char_n_filters', type=int, default=100)
    parser.add_argument('--char_kernel_size', type=int, default=5)


def set_default_finetune_args(args_):

    args_.lr = 2e-5
    args_.l2_wd = 0.001
    args_.num_epochs = 3

    return args_
from src.options import *
from src.train import *


if __name__ == '__main__':
    args = get_test_args()

    if 'roberta-qa' == args.model_name:
        #eval_bert(args)
        raise NotImplementedError()
    elif 'bidaf' == args.model_name:
        #create dataloader for test set?
        #eval_bidaf(args)
        raise NotImplementedError()
    else:
        raise ValueError("{} is not a valid model".format(args.model_name))
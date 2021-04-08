from src.train import *
from src.preprocessing import prep_squad_v1


if __name__ == '__main__':
    parser = get_common_args()
    args = parser.parse_args()

    if 'preproc' == args.mode:
        args = add_preproc_args(parser)

        prep_squad_v1.pre_process(args)

    elif 'train' == args.mode:
        args = add_train_args(parser)

        if 'roberta-qa' == args.model_name:
            train_bert(args)
        elif 'bidaf' == args.model_name:
            train_bidaf(args)
        else:
            raise ValueError("{} is not a valid model".format(args.model_name))

    elif 'test' == args.mode:
        args = add_test_args(parser)

        raise NotImplementedError()
    else:
        raise ValueError(args.mode)
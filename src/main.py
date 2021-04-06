from src.train import *



if __name__ == '__main__':
    args = get_train_args()

    if 'roberta-qa' == args.model_name:
        train_bert(args)
    elif 'bidaf' == args.model_name:
        train_bidaf(args)
    else:
        raise ValueError("{} is not a valid model".format(args.model_name))
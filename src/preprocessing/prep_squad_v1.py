"""

Code adapted from:
    > https://github.com/minggg/squad/blob/master/setup.py

"""

import numpy as np
import os
import spacy
import json
import urllib.request

from pathlib import Path
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
from transformers import RobertaTokenizer

from src.options import get_setup_args


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(data_root: Path, url):
    return data_root.joinpath(url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(args.data_root, url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

def clean_tokens(sent, tokenize=True):
    doc = nlp(sent)
    if tokenize:
        return [token.text for token in doc]
    else:
        return doc.text


def convert_idx(text):
    current = 0
    spans = []
    for token in clean_tokens(text, tokenize=True):
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def proc_file_with_counter(filename, data_type, word_counter, char_counter, tokenize=True, debug=False):
    """
    Iterate through the raw data structure of SQuAD to build nested dictionaries of iterable examples,
    each comprising context, question and answer.

    :param filename:
    :param data_type:
    :param word_counter: counting total occurrences of words in context, questions and answers (used to build vocabulary)
    :param char_counter: counting total occurrences of characters

    :return:
        list(dict) examples : each example   and comprises context, question, answer and id

            {'context_tokens', 'context_chars', 'ques_tokens', 'ques_chars', 'y1s', 'y2s', 'id'}
        dict(dict) eval_examples =

            'id': {"context": context, "question": ques, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
    """


    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total_qs = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        data = source['data'] if not debug else source['data'][:100]

        for article in tqdm(data):
            for para in article["paragraphs"]:
                # context
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')

                spans = convert_idx(context)

                if tokenize:
                    context_tokens = clean_tokens(context)
                    context_chars = [list(token) for token in context_tokens]
                    for token in context_tokens:
                        word_counter[token] += len(para["qas"]) # increase count for each question in whose context 'token' appears
                        for char in token:
                            char_counter[char] += len(para["qas"])

                # questions
                for qa in para["qas"]:
                    total_qs += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')

                    if tokenize:
                        ques_tokens = clean_tokens(ques)
                        ques_chars = [list(token) for token in ques_tokens]
                        for token in ques_tokens:
                            word_counter[token] += 1
                            for char in token:
                                char_counter[char] += 1

                    # answers
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1) # answer start, i.e. word position in the context
                        y2s.append(y2) # answer end, i.e. position of last word of the answer

                    if tokenize:
                        example = {"context_tokens": context_tokens,
                                   "context_chars": context_chars,
                                   "ques_tokens": ques_tokens,
                                   "ques_chars": ques_chars,
                                   "y1s": y1s,
                                   "y2s": y2s,
                                   "id": total_qs}
                    else:
                        example = {"context_tokens": context,
                                   "context_chars": None,
                                   "ques_tokens": ques,
                                   "ques_chars": None,
                                   "y1s": y1s,
                                   "y2s": y2s,
                                   "id": total_qs}
                    examples.append(example)

                    eval_examples[str(total_qs)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        print(f"{len(examples)} questions in total")

    return examples, eval_examples


def get_embedding(counter, data_type, min_freq=-1, vocab_size=30000, emb_file=None, vec_size=None, num_vectors=None, normal_scale=1):
    """
    With pre-trained embs: for each word in the vocab, load its corresponding embedding
    NO pre-trained embs provided: initialise random numpy vectors sampled from normal distribution

    :param counter:
    :param data_type:
    :param min_freq:
    :param vocab_size (int): maximum number of words in vocabulary
    :param emb_file (Path): txt file containing pre-trained word embeddings
    :param vec_size:
    :param num_vectors:
    :return:
    """

    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filt_tokens = [token for token, counts in counter.most_common(vocab_size) if counts > min_freq] # choose limit to filter out infrequent words

    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in filt_tokens:
                    embedding_dict[word] = vector
                if len(embedding_dict) == vocab_size:
                    break
        print(f"{len(embedding_dict)} / {len(filt_tokens)} tokens have corresponding {data_type} embedding vector")
    else:
        # randomly initialise word embeddings
        assert vec_size is not None
        for token in filt_tokens:
            embedding_dict[token] = list(np.random.normal(scale=normal_scale, size=vec_size))
        print(f"{len(filt_tokens)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--" # is this '<pad>' token??
    OOV = "--OOV--" # replace with '<unk>' to match BERT terminology
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)} # start at 2 to reserve pos 0 and 1 for special tokens
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_roberta_features(args, examples, tokenizer: RobertaTokenizer, data_split: str, out_file: str, is_test=False):
    """

    :param args:
    :param examples: {"context_tokens", "context_chars", "ques_tokens",
                        "ques_chars", "y1s",  "y2s", "id"}
    :param tokenizer:
    :param is_test:
    :return:
    """
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit

    assert para_limit + ques_limit < tokenizer.max_len_sentences_pair

    valid_ex = 0
    total_ex = 0
    meta = {}
    context_ques_idxs = []
    attention_masks = []

    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ex += 1

        if drop_example(example, para_limit, ques_limit, ans_limit, is_test):
            continue

        valid_ex += 1


        # decision: jointly condition Roberta model on context+question, hence encode them together
        # pair of sequences: <s> A </s></s> B </s>
        # need to store Mask to avoid performing attention on padding token indices
        context = example["context_tokens"]
        question = example["ques_tokens"]
        c_q_idxs, attn_mask = tokenizer(context, question, padding=True).values()
        # TODO: check length

        context_ques_idxs.append(c_q_idxs)
        attention_masks.append(attn_mask)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    # save as numpy array
    np.savez(build_data_path(args, 'roberta_' + out_file),
             c_q_idxs=np.array(context_ques_idxs),
             attn_mask=np.array(attention_masks),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))

    print(f"Built {valid_ex} / {total_ex} instances of features in total")
    meta["total"] = valid_ex
    return meta

#currently not in use

# def convert_text2indices(args, data, word2idx_dict, char2idx_dict, is_test):
#
#
#
#     example = {}
#     context, question = data
#     context = context.replace("''", '" ').replace("``", '" ')
#     question = question.replace("''", '" ').replace("``", '" ')
#     example['context_tokens'] = word_tokenize(context)
#     example['ques_tokens'] = word_tokenize(question)
#     example['context_chars'] = [list(token) for token in example['context_tokens']]
#     example['ques_chars'] = [list(token) for token in example['ques_tokens']]
#
#     para_limit = args.test_para_limit if is_test else args.para_limit
#     ques_limit = args.test_ques_limit if is_test else args.ques_limit
#     char_limit = args.char_limit
#
#     def filter_func(example):
#         return len(example["context_tokens"]) > para_limit or \
#                len(example["ques_tokens"]) > ques_limit
#
#     if filter_func(example):
#         raise ValueError("Context/Questions lengths are over the limit")
#
#     context_idxs = np.zeros([para_limit], dtype=np.int32)
#     context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
#     ques_idxs = np.zeros([ques_limit], dtype=np.int32)
#     ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
#
#     def _get_word(word):
#         for each in (word, word.lower(), word.capitalize(), word.upper()):
#             if each in word2idx_dict:
#                 return word2idx_dict[each]
#         return 1
#
#     def _get_char(char):
#         if char in char2idx_dict:
#             return char2idx_dict[char]
#         return 1
#
#     for i, token in enumerate(example["context_tokens"]):
#         context_idxs[i] = _get_word(token)
#
#     for i, token in enumerate(example["ques_tokens"]):
#         ques_idxs[i] = _get_word(token)
#
#     for i, token in enumerate(example["context_chars"]):
#         for j, char in enumerate(token):
#             if j == char_limit:
#                 break
#             context_char_idxs[i, j] = _get_char(char)
#
#     for i, token in enumerate(example["ques_chars"]):
#         for j, char in enumerate(token):
#             if j == char_limit:
#                 break
#             ques_char_idxs[i, j] = _get_char(char)
#
#     return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs

def drop_example(ex, c_limit, q_limit, a_limit, is_test_=False):
    if is_test_:
        drop = False
    else:
        drop = len(ex["context_tokens"]) > c_limit or \
               len(ex["ques_tokens"]) > q_limit or \
               (is_answerable(ex) and
                ex["y2s"][0] - ex["y1s"][0] > a_limit)

    return drop

def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0



def build_features(args, examples, data_split: str, out_file, word2idx_dict, char2idx_dict, is_test=False):
    """
    For each example, map the words to corresponding indices in vocabulary and
    truncate paragraphs and questions to the corresponding word limits

    :param args:
    :param examples:
    :param data_split:
    :param out_file:
    :param word2idx_dict:
    :param char2idx_dict:
    :param is_test:
    :return:
    """
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    print(f"Converting {data_split} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, para_limit, ques_limit, ans_limit, is_test):
            continue

        total += 1

        def _get_word(word):
            """
            Check if the word or any capital variation of it appear in the vocabulary,
            otherwise return the index for OOV
            """
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        # initialise word sequence with '0' and replace each position with a valid word index from the vocabulary
        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    #out_file = args.data_root.joinpath(args.dataset_name, out_file) # build modular path for output file

    # save as numpy array
    np.savez(build_data_path(args, out_file),
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))

    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save_as_json(args, filename, obj, message=None):
    f_name = build_data_path(args, filename)
    if message is not None:
        print(f"Saving {message}...")
        with open(f_name, "w") as fh:
            json.dump(obj, fh)

def build_data_path(args, file_name):
    return args.data_root.joinpath(args.dataset_name, file_name)

def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = proc_file_with_counter(args.train_file, "train", word_counter, char_counter,
                                                        tokenize=not args.use_roberta_token, debug=True)

    if args.use_roberta_token:

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        word2idx = tokenizer.get_vocab()
        char2idx = None

        build_roberta_features(args, train_examples, tokenizer, "train", args.train_record_file)

    else:
        # build embeddings and features from hand-crafted vocabulary
        word_emb_mat, word2idx = get_embedding(
            word_counter, 'word', emb_file=args.we_file, vec_size=args.glove_dim,
            num_vectors=args.glove_num_vecs, vocab_size=args.vocab_size)
        char_emb_mat, char2idx = get_embedding(
            char_counter, 'char', emb_file=None, vec_size=args.char_dim)

        save_as_json(args, args.word_emb_file, word_emb_mat, message="word embedding")
        save_as_json(args, args.char_emb_file, char_emb_mat, message="char embedding")

        save_as_json(args, args.char2idx_file, char2idx, message="char dictionary")

        build_features(args, train_examples, "train", args.train_record_file, word2idx, char2idx)

    save_as_json(args, args.train_eval_file, train_eval, message="train eval")
    save_as_json(args, args.word2idx_file, word2idx, message="word dictionary")

    #
    # Process dev and test sets
    #
    dev_examples, dev_eval = proc_file_with_counter(args.dev_file, "dev", word_counter, char_counter)

    if args.include_test_examples:
        # Process test set
        test_examples, test_eval = proc_file_with_counter(args.test_file, "test", word_counter, char_counter)
        save_as_json(args, args.test_eval_file, test_eval, message="test eval")

    if args.use_roberta_token:

        dev_meta = build_roberta_features(args, dev_examples, tokenizer, "dev", args.dev_record_file)

        if args.include_test_examples:
            test_meta = build_roberta_features(args, test_examples, tokenizer, "test", args.test_record_file,
                                               is_test=True)
            save_as_json(args, args.test_meta_file, test_meta, message="test meta")

    else:
        build_features(args, train_examples, "train", args.train_record_file, word2idx, char2idx)

        dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx, char2idx)

        if args.include_test_examples:
            test_meta = build_features(args, test_examples, "test",
                                       args.test_record_file, word2idx, char2idx, is_test=True)
            save_as_json(args, args.test_meta_file, test_meta, message="test meta")

    save_as_json(args, args.dev_eval_file, dev_eval, message="dev eval")
    save_as_json(args, args.dev_meta_file, dev_meta, message="dev meta")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    if args_.download:
        download(args_)
        #args_.train_file = url_to_data_path(args_.data_root, args_.train_url)
        #args_.dev_file = url_to_data_path(args_.data_root, args_.dev_url)
        #if args_.include_test_examples:
        #    args_.test_file = url_to_data_path(args_.test_url)

    args_.train_file = args_.data_root.joinpath(args_.dataset_name, args_.train_file)
    args_.dev_file = args_.data_root.joinpath(args_.dataset_name, args_.dev_file)
    if args_.include_test_examples:
        args_.test_file = args_.data_root.joinpath(args_.dataset_name, args_.test_file)

    if args_.use_pt_we:
        glove_dir = url_to_data_path(args_.data_root, args_.glove_url.replace('.zip', ''))
        glove_ext = f'.txt' if str(glove_dir).endswith('d') else f'.{args_.glove_dim}d.txt'
        args_.we_file = glove_dir.joinpath(os.path.basename(glove_dir) + glove_ext)
    else:
        args_.we_file = None
    # Import spacy language model
    nlp = spacy.blank("en")

    # Preprocess dataset
    pre_process(args_)
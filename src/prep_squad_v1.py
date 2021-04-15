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
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast

import src.options

# Import spacy language model
NLP = spacy.blank("en")


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


def url_to_data_path(data_root: str, url) -> str:
    return os.path.join(data_root, url.split('/')[-1])


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


def clean_tokens(sent, tokenize=False):
    doc = NLP(sent)
    if tokenize:
        return [token.text for token in doc]
    else:
        return doc.text


def convert_idx(text):
    # create list of character spans for each token
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


def proc_raw_file(filename, data_type, debug=False):
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
                context = clean_tokens(para["context"].replace("''", '" ').replace("``", '" ')).strip()

                # spans = convert_idx(context) # What is this used for later?

                # questions
                for qa in para["qas"]:
                    total_qs += 1
                    ques = clean_tokens(qa["question"].replace("''", '" ').replace("``", '" ')).strip()

                    # answers
                    start_chars, end_chars = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"].strip()
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)

                        answer_texts.append(answer_text)

                        start_chars.append(answer_start)  # start position of char of answer text in context
                        end_chars.append(answer_end)

                    example = {"context": context,
                               "question": ques,
                               "answers": answer_texts,
                               'start_chars': start_chars,
                               'end_chars': end_chars,
                               "id": total_qs}
                    examples.append(example)

                    eval_examples[str(total_qs)] = {"context": context,
                                                    "question": ques,
                                                    "answers": answer_texts,
                                                    'start_chars': start_chars,
                                                    'end_chars': end_chars,
                                                    "uuid": qa["id"]}
        print(f"{len(examples)} questions in total")

    return examples, eval_examples


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
                    context_tokens = clean_tokens(context, tokenize=tokenize)
                    context_chars = [list(token) for token in context_tokens]
                    for token in context_tokens:
                        word_counter[token] += len(
                            para["qas"])  # increase count for each question in whose context 'token' appears
                        for char in token:
                            char_counter[char] += len(para["qas"])

                # questions
                for qa in para["qas"]:
                    total_qs += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')

                    if tokenize:
                        ques_tokens = clean_tokens(ques, tokenize=tokenize)
                        ques_chars = [list(token) for token in ques_tokens]
                        for token in ques_tokens:
                            word_counter[token] += 1
                            for char in token:
                                char_counter[char] += 1

                    # answers
                    y1s, y2s = [], []
                    start_chars, end_chars = [], []
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
                        y1s.append(y1)  # answer start, i.e. word position in the context
                        y2s.append(y2)  # answer end, i.e. position of last word of the answer
                        start_chars.append(answer_start)
                        end_chars.append(answer_end)

                    if tokenize:
                        example = {"context_tokens": context_tokens,
                                   "context_chars": context_chars,
                                   "ques_tokens": ques_tokens,
                                   "ques_chars": ques_chars,
                                   "ans_texts": answer_texts,
                                   'start_chars': start_chars,
                                   'end_chars': end_chars,
                                   "y1s": y1s,
                                   "y2s": y2s,
                                   "id": total_qs}
                    else:
                        example = {"context_tokens": context,
                                   "context_chars": None,
                                   "ques_tokens": ques,
                                   "ques_chars": None,
                                   "ans_texts": answer_texts,
                                   'start_chars': start_chars,
                                   'end_chars': end_chars,
                                   "y1s": y1s,
                                   "y2s": y2s,
                                   "id": total_qs}

                    examples.append(example)
                    eval_examples[str(total_qs)] = {"context": context,
                                                    "question": ques,
                                                    "spans": spans,
                                                    "answers": answer_texts,
                                                    'start_chars': start_chars,
                                                    'end_chars': end_chars,
                                                    "uuid": qa["id"]}
        print(f"{len(examples)} questions in total")

    return examples, eval_examples


def get_embedding(counter, data_type, min_freq=-1, vocab_size=30000, emb_file=None, vec_size=None, num_vectors=None,
                  normal_scale=1):
    """
    With pre-trained embs: for each word in the vocab, load its corresponding embedding
    NO pre-trained embs provided: initialise random numpy vectors sampled from normal distribution

    :param counter:
    :param data_type:
    :param min_freq:
    :param vocab_size (int): maximum number of words in vocabulary
    :param emb_file (str): txt file containing pre-trained word embeddings
    :param vec_size:
    :param num_vectors:
    :return:
    """

    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filt_tokens = [token for token, counts in counter.most_common(vocab_size) if
                   counts > min_freq]  # choose limit to filter out infrequent words

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

    NULL = "<PAD>"
    OOV = "<UNK>"
    token2idx_dict = {token: idx for idx, token in
                      enumerate(embedding_dict.keys(), 2)}  # start at 2 to reserve pos 0 and 1 for special tokens
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_bert_features(args, examples, tokenizer, data_split: str, out_file: str, is_test=False):
    """

    :param args:
    :param examples: {"context_tokens", "context_chars", "ques_tokens",
                        "ques_chars", "y1s",  "y2s", "id"}
    :param tokenizer:
    :param is_test:
    :return:
    """

    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    max_len = args.max_seq_len
    valid_ex = 0
    total_ex = 0

    ques_cont_ids = []
    attention_masks = []

    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ex += 1
        # jointly encode question-context pair with Transformer
        # pair of sequences: <s> A </s></s> B </s>
        # need to store Mask to avoid performing attention on padding token indices
        question = example["question"]
        context = example["context"]

        # Note: implicit right-side padding
        example_tokenized = tokenizer(question, context,
                                      max_length=max_len,
                                      padding='max_length',
                                      truncation="only_second",
                                      return_overflowing_tokens=True,
                                      return_offsets_mapping=True)
        # except Exception as e:
        #     print(n)
        #     print(e)

        q_c_ids = example_tokenized['input_ids'][0]
        cls_index = q_c_ids.index(tokenizer.cls_token_id)
        attn_mask = example_tokenized['attention_mask'][0]
        offsets = example_tokenized['offset_mapping'][0]
        # example: [(0, 0), (0, 2), (3, 7), (8, 11), (12, 15), (16, 22), (23, 27), (28, 37), (38, 44), (45, 47)]
        # tuple at each position corresponds to the indices and span of characters
        # of the original sequence. using the tokenizer may split up words into word-parts
        # (0, 0) corresponds to tokens that were not in the original sequence (e.g. [CLS])
        sequence_ids = example_tokenized.sequence_ids()

        # correct answer span for Tokenizer offset
        # start_word_pos, end_word_pos = example["y1s"][-1], example["y2s"][-1]
        start_char, end_char = example['start_chars'][-1], example['end_chars'][-1]

        # distinguish which parts belong to question and which to context
        # context tokens have sequence id = 1
        c_start_idx = 0  # start token index
        while sequence_ids[c_start_idx] != 1:
            c_start_idx += 1

        c_end_idx = len(q_c_ids) - 1  # end token index
        while sequence_ids[c_end_idx] != 1:
            c_end_idx -= 1

        context_token_span = c_end_idx - c_start_idx
        ques_token_span = c_start_idx - 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if (offsets[c_start_idx][0] <= start_char and offsets[c_end_idx][1] >= end_char):
            # Move (char) start/end_position to start/end of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while c_start_idx < len(offsets) and offsets[c_start_idx][0] <= start_char \
                    and offsets[c_start_idx][1] < end_char:
                c_start_idx += 1
            start_position = c_start_idx - 1

            # if start_position >= 380:
            #     print(n)

            while offsets[c_end_idx][1] >= end_char:
                c_end_idx -= 1
            end_position = c_end_idx + 1
            # print(start_position, end_position)
        else:
            start_position, end_position = cls_index, cls_index
            # print("The answer is not in this feature.")

        answer_token_span = end_position - start_position

        # TODO: truncate longer context passages instead of dropping them
        if drop_example(args, ques_token_span, context_token_span, start_position, end_position, is_test):
            continue

        valid_ex += 1
        #
        if valid_ex % 2000 == 0 and args.debug:
            # validate with ground truth answer
            print(tokenizer.decode(q_c_ids[start_position:end_position + 1]))
            print(example['answers'][0])

        ques_cont_ids.append(q_c_ids)
        attention_masks.append(attn_mask)
        y1s.append(start_position)
        y2s.append(end_position)
        ids.append(example["id"])

        if start_position > 380:
            print(example["id"])

    print(f"Built {valid_ex} / {total_ex} instances of features in total")

    # save as numpy array
    # TODO: change to 'q_c_idxs' and adapt key usage elsewhere
    f_name = build_data_path(args, 'roberta_' + out_file)
    np.savez(f_name,
             c_q_idxs=np.array(ques_cont_ids),
             attn_mask=np.array(attention_masks),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))

    return f_name


# currently not in use

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

def drop_example(args, q_len, c_len, ans_start, ans_end, is_test_=False):
    '''
    Filter out examples that exceed certain token limits or are not answerable
    Return 'True' if example should be dropped
    '''

    c_limit = args.test_context_limit if is_test_ else args.context_limit
    q_limit = args.test_ques_limit if is_test_ else args.ques_limit
    a_limit = args.ans_limit
    squad_v2 = args.use_squad_v2

    if is_test_:
        return False
    elif c_len > c_limit:
        return True
    elif q_len > q_limit:
        return True
    elif is_answerable(ans_start, ans_end) and ans_end - ans_start > a_limit:
        return True
    elif not is_answerable(ans_start, ans_end) and not squad_v2:
        return True
    else:
        return False


def is_answerable(ans_start, ans_end):
    # return len(example['y2s']) > 0 and len(example['y1s']) > 0
    # and ans_start <= ans_end#and ans_start <= ans_end
    return ans_start > 0 and ans_end > 0


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
    max_len = args.max_seq_len
    context_limit = args.test_context_limit if is_test else args.context_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    print(f"Converting {data_split} examples to indices...")
    valid = 0
    total = 0

    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total += 1

        start, end = example["y1s"][-1], example["y2s"][-1]
        if not is_answerable(start, end):
            start, end = -1, -1

        # args, q_len, c_len, ans_start, ans_end, is_test_=False
        q_len = len(example['ques_tokens'])
        if drop_example(args,
                        q_len=len(example['ques_tokens']),
                        c_len=len(example["context_tokens"]),
                        ans_start=start, ans_end=end, is_test_=is_test):
            continue

        valid += 1

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
        context_idx = np.zeros([context_limit], dtype=np.int32)
        context_char_idx = np.zeros([context_limit, char_limit], dtype=np.int32)
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

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    print(f"Built {valid} / {total} instances of features in total")

    # save as numpy array
    f_name = build_data_path(args, out_file)
    np.savez(f_name,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))

    return f_name


def save_as_json(args, filename, obj, message=None):
    f_name = build_data_path(args, filename)
    if message is not None:
        print(f"Saving {message}...")
    with open(f_name, "w") as fh:
        json.dump(obj, fh)

    return f_name


def build_data_path(args, file_name) -> str:
    return os.path.join(args.data_root, args.dataset_name, file_name)


def pre_process(args):
    setup_pre_process(args)

    # Process training set and use it to decide on the word/character vocabularies
    debug = args.debug
    prep_records = {'name': None,
                    'prefix': None}  # record which specific files are generated and shall be used after preprocessing
    print(f'{json.dumps(vars(args), indent=2, sort_keys=True)}')

    if args.use_roberta_token:
        # TODO: add function to get tokenizer (for other models)

        prep_records['name'] = 'roberta01'
        prefix = 'roberta_'
        prep_records['prefix'] = prefix
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        word2idx = tokenizer.get_vocab()
        prep_records['word_dict'] = save_as_json(args, prefix + args.word2idx_file, word2idx, message="word dictionary")

        train_examples, train_eval = proc_raw_file(args.train_file, "train", debug=debug)
        prep_records['train_npz'] = build_bert_features(args, train_examples, tokenizer, "train",
                                                        args.train_record_file)

        prep_records['train_eval'] = save_as_json(args, prefix + args.train_eval_file, train_eval, message="train eval")

        dev_examples, dev_eval = proc_raw_file(args.dev_file, "dev")
        prep_records['dev_npz'] = build_bert_features(args, dev_examples, tokenizer, "dev", args.dev_record_file)

        prep_records['dev_eval'] = save_as_json(args, prefix + args.dev_eval_file, dev_eval, message="dev eval")

        if args.include_test_examples:
            # Process test set
            test_examples, test_eval = proc_raw_file(args.test_file, "test")
            prep_records['test_npz'] = build_bert_features(args, test_examples, tokenizer, "test",
                                                           args.test_record_file, is_test=True)

    else:
        # TODO: add prefix to dynamically indicate Wordembedding type and char emb

        prep_records['name'] = 'bidaf01'
        word_counter, char_counter = Counter(), Counter()
        train_examples, train_eval = proc_file_with_counter(args.train_file, "train", word_counter, char_counter,
                                                            debug=debug)

        # build embeddings and features from hand-crafted vocabulary
        if debug and os.path.exists(build_data_path(args, args.word2idx_file)):
            with open(build_data_path(args, args.word2idx_file), 'r') as f_in:
                word2idx = json.load(f_in)
        else:
            word_emb_mat, word2idx = get_embedding(
                word_counter, 'word', emb_file=args.we_file, vec_size=args.glove_dim,
                num_vectors=args.glove_num_vecs, vocab_size=args.vocab_size)
            prep_records['word_dict'] = save_as_json(args, args.word2idx_file, word2idx, message="word dictionary")
            prep_records['word_emb'] = save_as_json(args, args.word_emb_file, word_emb_mat, message="word embedding")

        if debug and os.path.exists(build_data_path(args, args.char2idx_file)):
            with open(build_data_path(args, args.char2idx_file), 'r') as f_in:
                char2idx = json.load(f_in)
        else:
            char_emb_mat, char2idx = get_embedding(
                char_counter, 'char', emb_file=None, vec_size=args.char_dim)
            prep_records['char_dict'] = save_as_json(args, args.char2idx_file, char2idx, message="char dictionary")
            prep_records['char_emb'] = save_as_json(args, args.char_emb_file, char_emb_mat, message="char embedding")

        prep_records['train_npz'] = build_features(args, train_examples, "train", args.train_record_file, word2idx,
                                                   char2idx)
        prep_records['train_eval'] = save_as_json(args, args.train_eval_file, train_eval, message="train eval")

        # Process dev and test sets
        #
        dev_examples, dev_eval = proc_file_with_counter(args.dev_file, "dev", word_counter, char_counter,
                                                        tokenize=not args.use_roberta_token)

        if args.include_test_examples:
            # Process test set
            test_examples, test_eval = proc_file_with_counter(args.test_file, "test", word_counter, char_counter,
                                                              tokenize=not args.use_roberta_token)
            prep_records['test_eval'] = save_as_json(args, args.test_eval_file, test_eval, message="test eval")

        prep_records['dev_npz'] = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx, char2idx)

        if args.include_test_examples:
            prep_records['test_npz'] = build_features(args, test_examples, "test",
                                                      args.test_record_file, word2idx, char2idx, is_test=True)

        prep_records['dev_eval'] = save_as_json(args, args.dev_eval_file, dev_eval, message="dev eval")

    prep_records['args'] = {k: v for k, v in vars(args).items()}
    f_name = prep_records['name'] + '_records.json'

    try:
        print(save_as_json(args, f_name, prep_records, message='prep records'))
    except:
        for k, v in vars(args).items():
            if isinstance(v, Path):
                print(f'{k}: {v}')
            elif isinstance(v, set):
                print(f'{k}: {v}')

def setup_pre_process(args_):
    # Download resources
    if args_.download:
        download(args_)
        # args_.train_file = url_to_data_path(args_.data_root, args_.train_url)
        # args_.dev_file = url_to_data_path(args_.data_root, args_.dev_url)
        # if args_.include_test_examples:
        #    args_.test_file = url_to_data_path(args_.test_url)
    args_.train_file = os.path.join(args_.data_root, args_.dataset_name, args_.train_file)
    args_.dev_file = os.path.join(args_.data_root, args_.dataset_name, args_.dev_file)

    if args_.include_test_examples:
        args_.test_file = os.path.join(args_.data_root, args_.dataset_name, args_.test_file)
    if args_.use_pt_we:
        glove_dir = os.path.join(args_.data_root, args_.glove_dir) if args_.glove_dir is not None else os.path.join(
            args_.data_root, 'glove')
        glove_f_name = args_.glove_url.replace('.zip', '').split('/')[-1]
        glove_ext = f'.txt' if str(glove_f_name).endswith('d') else f'.{args_.glove_dim}d.txt'
        args_.we_file = os.path.join(glove_dir, glove_f_name + glove_ext)
    else:
        args_.we_file = None


if __name__ == '__main__':
    # Get command-line args
    args_ = src.options.add_preproc_args(parser=None)
    # Preprocess dataset
    pre_process(args_)

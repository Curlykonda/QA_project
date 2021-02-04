
import torch
import torch.utils.data as data
import numpy as np
import json


class SQuAD(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).
    Each item in the dataset is a tuple with the following entries (in order):
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
        - question_idxs: Indices of the words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.
    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """
    def __init__(self, data_path, use_v2=True):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()


        if use_v2:
            # SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
            batch_size, c_len, w_len = self.context_char_idxs.size()
            ones = torch.ones((batch_size, 1), dtype=torch.int64)
            self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
            self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

            ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
            self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
            self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

            self.y1s += 1
            self.y2s += 1

        # SQuAD 1.1: Ignore no-answer examples
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or self.y1s[idx].item() >= 0]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.ids[idx])

        return example

    def __len__(self):
        return len(self.valid_idxs)

class SQuAD_Roberta(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).
    Each item in the dataset is a tuple with the following entries (in order):
        - context_ques_idxs: Indices of the words in the context and the question
            plus special tokens and already padded, Shape (max_input_len,).
        - attention_mask: mask to avoid attention on padding tokens, Shape (max_input_len)
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.
    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """
    def __init__(self, data_path, use_v2=True):
        super(SQuAD_Roberta, self).__init__()

        dataset = np.load(data_path)
        self.context_ques_indxs = torch.from_numpy(dataset['c_q_idxs']).long()
        self.attn_mask = torch.from_numpy(dataset['attn_mask']).long()
        #self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        #self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        #self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()


        if use_v2:
            # SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
            batch_size, c_len, w_len = self.context_char_idxs.size()
            ones = torch.ones((batch_size, 1), dtype=torch.int64)
            self.context_ques_indxs = torch.cat((ones, self.context_ques_indxs), dim=1)
            self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

            ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
            self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
            self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

            self.y1s += 1
            self.y2s += 1

        # SQuAD 1.1: Ignore no-answer examples
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or self.y1s[idx].item() >= 0]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_ques_indxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.ids[idx])

        return example

    def __len__(self):
        return len(self.valid_idxs)

def squad_collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `SQuAD.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.
    Args:
        examples (list): List of tuples of the form (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, y1s, y2s, ids).
    Returns:
        examples (tuple): Tuple of tensors (context_idxs, context_char_idxs, question_idxs,
        question_char_idxs, y1s, y2s, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # Group by tensor type
    context_idxs, context_char_idxs, \
        question_idxs, question_char_idxs, \
        y1s, y2s, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    ids = merge_0d(ids)

    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)

# class SQuAD(AbstractDataset):
#     def __init__(self, args):
#         path = '.data/squad'
#         dataset_path = path + '/torchtext/'
#         train_examples_path = dataset_path + 'train_examples.pt'
#         dev_examples_path = dataset_path + 'dev_examples.pt'
#
#         print("preprocessing data files...")
#         if not os.path.exists('{}/{}l'.format(path, args.train_file)):
#             self.preprocess_file('{}/{}'.format(path, args.train_file))
#         if not os.path.exists('{}/{}l'.format(path, args.dev_file)):
#             self.preprocess_file('{}/{}'.format(path, args.dev_file))
#
#         self.RAW = data.RawField()
#         # explicit declaration for torchtext compatibility
#         self.RAW.is_target = False
#         self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
#         self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
#         self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
#         self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)
#
#         dict_fields = {'id': ('id', self.RAW),
#                        's_idx': ('s_idx', self.LABEL),
#                        'e_idx': ('e_idx', self.LABEL),
#                        'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
#                        'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}
#
#         list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
#                        ('c_word', self.WORD), ('c_char', self.CHAR),
#                        ('q_word', self.WORD), ('q_char', self.CHAR)]
#
#         if os.path.exists(dataset_path):
#             print("loading splits...")
#             train_examples = torch.load(train_examples_path)
#             dev_examples = torch.load(dev_examples_path)
#
#             self.train = data.Dataset(examples=train_examples, fields=list_fields)
#             self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
#         else:
#             print("building splits...")
#             self.train, self.dev = data.TabularDataset.splits(
#                 path=path,
#                 train='{}l'.format(args.train_file),
#                 validation='{}l'.format(args.dev_file),
#                 format='json',
#                 fields=dict_fields)
#
#             os.makedirs(dataset_path)
#             torch.save(self.train.examples, train_examples_path)
#             torch.save(self.dev.examples, dev_examples_path)
#
#         #cut too long context in the training set for efficiency.
#         if args.context_threshold > 0:
#             self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]
#
#         print("building vocab...")
#         self.CHAR.build_vocab(self.train, self.dev)
#         self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))
#
#         print("building iterators...")
#         device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
#         self.train_iter = data.BucketIterator(
#             self.train,
#             batch_size=args.train_batch_size,
#             device=device,
#             repeat=True,
#             shuffle=True,
#             sort_key=lambda x: len(x.c_word)
#         )
#
#         self.dev_iter = data.BucketIterator(
#             self.dev,
#             batch_size=args.dev_batch_size,
#             device=device,
#             repeat=False,
#             sort_key=lambda x: len(x.c_word)
#         )
#
#         # self.train_iter, self.dev_iter = \
#         #    data.BucketIterator.splits((self.train, self.dev),
#         #                               batch_sizes=[args.train_batch_size, args.dev_batch_size],
#         #                               device=device,
#         #                               sort_key=lambda x: len(x.c_word))
#
#     def preprocess_file(self, path):
#         dump = []
#         abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']
#
#         with open(path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             data = data['data']
#
#             for article in data:
#                 for paragraph in article['paragraphs']:
#                     context = paragraph['context']
#                     tokens = word_tokenize(context)
#                     for qa in paragraph['qas']:
#                         id = qa['id']
#                         question = qa['question']
#                         for ans in qa['answers']:
#                             answer = ans['text']
#                             s_idx = ans['answer_start']
#                             e_idx = s_idx + len(answer)
#
#                             l = 0
#                             s_found = False
#                             for i, t in enumerate(tokens):
#                                 while l < len(context):
#                                     if context[l] in abnormals:
#                                         l += 1
#                                     else:
#                                         break
#                                 # exceptional cases
#                                 if t[0] == '"' and context[l:l + 2] == '\'\'':
#                                     t = '\'\'' + t[1:]
#                                 elif t == '"' and context[l:l + 2] == '\'\'':
#                                     t = '\'\''
#
#                                 l += len(t)
#                                 if l > s_idx and s_found == False:
#                                     s_idx = i
#                                     s_found = True
#                                 if l >= e_idx:
#                                     e_idx = i
#                                     break
#
#                             dump.append(dict([('id', id),
#                                               ('context', context),
#                                               ('question', question),
#                                               ('answer', answer),
#                                               ('s_idx', s_idx),
#                                               ('e_idx', e_idx)]))
#
#         with open('{}l'.format(path), 'w', encoding='utf-8') as f:
#             for line in dump:
#                 json.dump(line, f)
#                 print('', file=f)
#
#     def load_dataset(self):
#         self.preprocess()
#         dataset_path = self._get_preprocessed_dataset_path()
#         dataset = pickle.load(dataset_path.open('rb'))
#         return dataset
#
#     def _generate_examples(self, filepath):
#         pass
#     #     """This function returns the examples in the raw (text) form."""
#     #     logging.info("generating examples from = %s", filepath)
#     #     with open(filepath, encoding="utf-8") as f:
#     #         squad = json.load(f)
#     #         for article in squad["data"]:
#     #             title = article.get("title", "").strip()
#     #             for paragraph in article["paragraphs"]:
#     #                 context = paragraph["context"].strip()
#     #                 for qa in paragraph["qas"]:
#     #                     question = qa["question"].strip()
#     #                     id_ = qa["id"]
#     #
#     #                     answer_starts = [answer["answer_start"] for answer in qa["answers"]]
#     #                     answers = [answer["text"].strip() for answer in qa["answers"]]
#     #
#     #                     # Features currently used are "context", "question", and "answers".
#     #                     # Others are extracted here for the ease of future expansions.
#     #                     yield id_, {
#     #                         "title": title,
#     #                         "context": context,
#     #                         "question": question,
#     #                         "id": id_,
#     #                         "answers": {
#     #                             "answer_start": answer_starts,
#     #                             "text": answers,
#     #                         },
#     #                     }
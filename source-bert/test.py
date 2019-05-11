# -*- coding: utf-8 -*-
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# 
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# 
# import csv
# import os
# import logging
# import argparse
# import random
# from tqdm import tqdm, trange
# 
# import numpy as np
# import torch
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# 
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import SentenceBert
# 
# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)
# 

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        logger.info("Samples: {}".format(len(lines)))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SentProcessor(object):
    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, sents):
        # sents: list of sentence
        """See base class."""
        return self._create_examples(sents, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line
            text_b = None
            label = '1'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PairProcessor(object):
    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, sents1, sents2):
        # sents: list of sentence
        """See base class."""
        return self._create_examples(sents1, sents2, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines1, lines2, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines1):
            guid = "%s-%s" % (set_type, i)
            text_a = line
            text_b = lines2[i]
            label = '1'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i


    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

class BertEncoding():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.processor = SentProcessor()
        bert_path = '/home/liuxg/.pytorch_pretrained_bert'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)

        # Prepare model
        self.model = SentenceBert.from_pretrained(bert_path)
        self.model.to(self.device)
        self.model.eval()

    def get_encoding(self, sents, max_seq_length=15):
        label_list = self.processor.get_labels()
        eval_examples = self.processor.get_dev_examples(sents=sents)
        # for e in eval_examples:
        #     print('----------------', e.text_a)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        _,representation = self.model(input_ids, segment_ids, input_mask) # 1,768

        return representation.detach()

    def get_representation(self, sents, max_seq_length=15):
        label_list = self.processor.get_labels()
        eval_examples = self.processor.get_dev_examples(sents=sents)
        # for e in eval_examples:
        #     print('----------------', e.text_a)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        representation,_ = self.model(input_ids, segment_ids, input_mask) # 1,768

        return representation.detach()

class BertMaskedLM():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.processor = SentProcessor()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load pre-trained model (weights)
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.to(self.device)
        self.model.eval()
        self.mask_token = '[MASK]'


    def predictwords(self, sents, masked_index, max_seq_length=15):
        sents_l = sents.strip().split()
        sents_l[masked_index] = self.mask_token
        sents = '[CLS] '+' '.join(sents_l)+ ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(sents)
        masked_index_shift = tokenized_text.index(self.mask_token)
        # Mask a token that we will try to predict back with `BertForMaskedLM`
        print(tokenized_text)
        #assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0]*len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)

        # Predict all tokens
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors) #1,L,vocab
        predicted_index = torch.argsort(predictions[0, masked_index_shift])[-100:]
        predicted_index = predicted_index.tolist()
        predicted_token = self.tokenizer.convert_ids_to_tokens(predicted_index)
        return predicted_token


if __name__ == "__main__":
    sents2 = ['which computer do you like', 'what app are you most using']
    text0 = "[CLS] What do you feel is the purpose of life [SEP]"
    text1 = "What do you feel is the purpose of life"
    text2 = "who was jim henson ? jim henson was a puppeteer"
    predictmodel = BertMaskedLM()
    word_candidates = predictmodel.predictwords(text0, 7)
    print('--')
    print(word_candidates)
    word_candidates = predictmodel.predictwords(text1,6)
    print('--')
    print(word_candidates)
    word_candidates = predictmodel.predictwords(text2,7)
    print('--')
    print(word_candidates)

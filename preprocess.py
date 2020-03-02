from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

class ParsingDataset(Dataset):
    def __init__(self, input_file):
        """
        Read and parse the train file line by line. Create a vocabulary
        dictionary that maps all the unique tokens from your data as
        keys to a unique integer value. Then vectorize your
        data based on your vocabulary dictionary.

        :param input_file: the data file pathname
        """
        # Read
        with open(input_file, 'r') as f:
            raw = f.read().split('\n')

        # Create inputs (START) / label (STOP)
        inputs = [['START'] + s.split(' ')[:-2] for s in raw[1:]]
        labels = [s.split(' ')[:-2] + ['STOP'] for s in raw[1:]]
        length = [len(s) for s in inputs]
        
        # Build dictionary
        tokens = ['START']
        for s in labels:
            tokens.extend(s)        
        words = sorted(list(set(tokens)))
        vocab = {w: i for i, w in enumerate(words)}

        # Convert to ID and tensor
        prepad_inputs, prepad_labels = [], []
        for i in range(len(inputs)):
            input_seq = torch.tensor([vocab[w] for w in inputs[i]])
            label_seq = torch.tensor([vocab[w] for w in labels[i]])
            prepad_inputs.append(input_seq)
            prepad_labels.append(label_seq)
        length = torch.tensor(length)

        # Pad
        inputs = pad_sequence(prepad_inputs, batch_first=True)
        labels = pad_sequence(prepad_labels, batch_first=True)

        # Class dict
        self.word2id = vocab
        self.dataset = {'inputs': inputs,
                        'labels': labels,
                        'length': length}
  
    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return self.dataset['inputs'].shape[0]

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        return {'inputs': self.dataset['inputs'][idx], 
                'labels': self.dataset['labels'][idx],
                'length': self.dataset['length'][idx]}



class RerankingDataset(Dataset):
    def __init__(self, parse_file, gold_file, word2id):
        """
        Read and parse the parse files line by line. Unk all words that has not
        been encountered before (not in word2id). Split the data according to
        gold file. Calculate number of constituents from the gold file.

        :param parse_file: the file containing potential parses
        :param gold_file: the file containing the right parsings
        :param word2id: the previous mapping (dictionary) from word to its word
                        id
        """
        # Read
        with open(parse_file) as f1, open(gold_file) as f2:
            rank = f1.read().split('\n')
            gold = f2.read().split('\n')

        # Create inputs (START) from parse_file
        parse_trees, num_parse_trees = [], []
        num_correct, num_total = [], []
        for r in rank:
            if not r.split(' ')[:-1]:
                if r != '':
                    num_parse_trees.append(int(r))
                else:
                    pass
            else:
                w = r.split(' ')[:-1]

                num_correct.append(int(w[0]))
                num_total.append(int(w[1]))

                parse_trees.append(['START'] + w[2:])
        
        # Count actual constituents from gold_file
        gold_parse = [g.split(' ')[:-2] for g in gold[:-1]]
        
        num_actual = []
        for s in gold_parse:
            cnt = 0
            for w in s:
                if '(' in w:
                    cnt += 1
            num_actual.append(cnt)

        # Convert to ID and tensor
        inputs = [torch.tensor([word2id[w] if w in word2id else word2id['*UNK'] for w in s]) for s in parse_trees]
        inputs_length = torch.tensor([len(s) for s in inputs])

        num_correct = torch.tensor(num_correct)
        num_total = torch.tensor(num_total)  
        num_actual = torch.tensor([n for n, i in zip(num_actual, num_parse_trees) for _ in range(i)])

        sentence_id = torch.tensor([n for n, i in enumerate(num_parse_trees) for _ in range(i)])

        # Pad
        inputs = pad_sequence(inputs, batch_first=True)
        
        self.dataset = {'inputs': inputs,
                        'length': inputs_length,
                        'num_correct': num_correct,
                        'num_total': num_total,
                        'num_actual': num_actual,
                        'id': sentence_id}

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return self.dataset['inputs'].shape[0]

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        return {'inputs': self.dataset['inputs'][idx],
                'length': self.dataset['length'][idx],
                'num_correct': self.dataset['num_correct'][idx],
                'num_total': self.dataset['num_total'][idx],
                'num_actual': self.dataset['num_actual'][idx],
                'id': self.dataset['id'][idx]}
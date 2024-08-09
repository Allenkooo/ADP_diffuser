import numpy as np
import functools, itertools, operator
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

# from nt_augmentation import aa_to_nt_iterative, aa_to_nt_random, aa_to_nt_random_ed, aa_to_nt_random_prob

torch_seed = 0
torch.manual_seed(torch_seed)
torch.cuda.manual_seed_all(torch_seed)
torch.cuda.manual_seed(torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

AMINOACID = {
    "G": 0,
    "A": 1,
    "V": 2,
    "L": 3,
    "I": 4,
    "F": 5,
    "W": 6,
    "Y": 7,
    "D": 8,
    "H": 9,
    "N": 10,
    "E": 11,
    "K": 12,
    "Q": 13,
    "M": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "C": 18,
    "P": 19,
    "X": 20, #Padding
}

NUCLEOTIDE = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
}

nt_aa_dict_std = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'N': ['AAT', 'AAC'],
        'D': ['GAT', 'GAC'],
        'C': ['TGT', 'TGC'],
        'Q': ['CAA', 'CAG'],
        'E': ['GAA', 'GAG'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAT', 'CAC'],
        'I': ['ATT', 'ATC', 'ATA'],
        'L': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'],
        'K': ['AAA', 'AAG'],
        'M': ['ATG'],
        'F': ['TTT', 'TTC'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'W': ['TGG'],
        'Y': ['TAT', 'TAC'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        '*': ['TAA', 'TGA', 'TAG']
}

def find_3grams(seq):
    '''
    input: sequence of nucleotides
    output: string of sequential in-frame codons / 3grams
    '''
    out_seq = []
    for i in range(2,len(seq) , 3):
        threeGram = seq[i-2],seq[i-1],seq[i]
        thrgrm_join = ''.join(threeGram)
        out_seq.append(thrgrm_join)
    return out_seq


def pad_seq(seqs, maxlen):
    output = []
    for s in seqs:
        if len(s) < maxlen:
            for i in range(maxlen, len(s), -1):
                s += 'X'
        output.append(s)
    return output


def onehot2AA(seqs):
    cate = torch.argmax(seqs, dim=1)
    AA_seq = []
    for s in cate:
        seq = ''
        for c in s:
            AA = list(AMINOACID.keys())[list(AMINOACID.values()).index(c)]
            seq += AA
        AA_seq.append(seq)
    return AA_seq


def onehot(seqs, maxlen, mode):
    if  mode=="AA":
        encoder = AMINOACID
    elif mode=="unigram":   # Nucleotide unigram
        encoder = NUCLEOTIDE
    elif mode == "augment" or mode == "DNA_base":     # Nucleotide trigram
        vocabulary = pd.read_csv('../NTA/data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']
        encoder = {word: i for i, word in enumerate(vocabulary)}
        trigram_seqs = [find_3grams(seq) for seq in seqs]
        return np.array(
            [
                np.pad(
                    np.eye(len(encoder))[[encoder[codon] for codon in trigram_seq]],
                    ((0, maxlen - len(trigram_seq)),(0,0)),
                )
                for trigram_seq in trigram_seqs
            ]
        )
    else:
        print("Unknow tpye")
        return 0
    return np.array(
        [
            np.pad(
                np.eye(len(encoder))[[encoder[c] for c in seq]],
                ((0, maxlen - len(seq)),(0,0)),
            ).T
            for seq in seqs
        ]
    )


def sep_word(data, num):
    res = []

    for i, seq in enumerate(data):
        res.append([seq[j: j+ num] for j in range(len(seq) - num + 1)])

    return res


def emb_seq_w2v(seq_mat, w2v_model, num, flatten=False):
    num_sample = len(seq_mat)
    for j in range(num_sample):
      seq=seq_mat[j]
      if j == 0:
         seq_emb = np.array([np.array(w2v_model.wv[seq[i:i+num]]) for i in range(len(seq) - num + 1)])
      else:
         seq_enc = np.array([np.array(w2v_model.wv[seq[i:i+num]]) for i in range(len(seq) - num + 1)])
         seq_emb = np.append(seq_emb, seq_enc, axis=0)

    seq_emb = seq_emb.reshape(num_sample,len(seq) - num + 1, -1)
    if flatten:
        seq_emb = seq_emb.reshape(num_sample,1,-1).squeeze()
    return seq_emb


def Load_Data(Data_path: str) -> tuple:    # seqs, labels
    with open(Data_path,"r") as f:
        lines = f.readlines()
        y_label = []
        seq = []
        neg = 0
        pos = 0
        for i, line in enumerate(lines):
            if i % 2 == 0:
                if line.__contains__("|0|") or line.__contains__("Neg"):
                    y_label.append(0)
                    neg += 1
                else:
                    pos += 1
                    y_label.append(1)
            else:
                # line = line[:-1]
                if len(line) <= 41:
                    line = line[:-1]
                else:
                    line = line[:40]
                seq.append(line)
        y_label = y_label    # (160,)
        seqs = pad_seq(seq, len(max(seq, key=len)))
    return seqs, np.array(y_label)


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.Xs = X
        self.ys = y

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index]

    def __len__(self):
        return len(self.Xs)

def performance(pred, prob, target):
    tn, fp, fn, tp = confusion_matrix(pred,target).ravel()
    return accuracy_score(pred,target), tp/(tp+fn), tn/(tn+fp), matthews_corrcoef(pred,target), roc_auc_score(target, prob) # acc, sen, spe, mcc, auc


class collater():
    def __init__(self, mode, maxlen, aug=10, aug_type='iterative', nt_aa_dict=nt_aa_dict_std, is_val=False):
        self.mode = mode
        self.maxlen = maxlen
        self.aug = aug
        self.aug_type = aug_type
        self.nt_aa_dict = nt_aa_dict
        self.is_val = is_val
    def __call__(self, data):
        seq_list = []
        target_list =[]
        for x, y in data:
            seq_list.append(x)
            target_list.append(y)
        if self.mode == "PCP":
            encoded_x = sequence_based(seq_list, self.mode, self.maxlen, norm="zscore")
            y = target_list
        elif self.mode == "AA":
            encoded_x = onehot(seq_list, self.maxlen, self.mode)
            y = target_list
        elif self.mode == 'DNA_base':
            self.is_val = True
            x, y = aa_to_nt_iterative(seq_list, target_list, self.aug, self.nt_aa_dict, self.is_val) # do augmentation
            encoded_x = onehot(x, self.maxlen, self.mode)
        else:   # augment
            if self.aug_type == 'iterative':
                x, y = aa_to_nt_iterative(seq_list, target_list, self.aug, self.nt_aa_dict, self.is_val) # do augmentation
            elif self.aug_type == 'random':
                x, y = aa_to_nt_random(seq_list, target_list, self.aug, self.nt_aa_dict, self.is_val) # do augmentation
            elif self.aug_type == 'random_ed':
                x, y = aa_to_nt_random_ed(seq_list, target_list, self.aug, self.nt_aa_dict, self.is_val) # do augmentation
            elif self.aug_type == 'random_prob':
                x, y = aa_to_nt_random_prob(seq_list, target_list, self.aug, self.nt_aa_dict, self.is_val) # do augmentation
            else:
                print("Unknown augmentation type")
                exit()
            encoded_x = onehot(x, self.maxlen, self.mode)
        return torch.tensor(encoded_x), torch.tensor(y)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, result_path='./result', name='best_model', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): model name for the checkpoint to be saved to.
                            Default: 'best_model'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.result_path = result_path
        self.name = name
        self.trace_func = trace_func
        self.stop_epoch = 0

    def __call__(self, val_loss, model, stop_epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                f = open(f'{self.result_path}.txt', 'a')
                print(f'{self.name}, stop_epoch: {self.stop_epoch}, training_loss: {self.val_loss_min}', file=f)
                f.close()
        else:
            self.best_score = score
            self.stop_epoch = stop_epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, f'./models/{self.name}.pt')
        self.val_loss_min = val_loss

from gensim.models import word2vec
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import configargparse
import torch
import yaml

from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D, PositionalEncoding
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_word_embedding(w2v_args: dict) -> tuple:
    w2v_model = word2vec.Word2Vec.load(w2v_args['model_path'] + f"{w2v_args['kmer']}_{w2v_args['emb_dim']}_{w2v_args['epochs']}_{w2v_args['window_size']}_{w2v_args['sg']}.pt")
    seqs, labels = Load_Data(args.Data_path)
    emb_seq = emb_seq_w2v(seqs, w2v_model, w2v_args['kmer'], flatten=False)
    emb_seq = torch.tensor(emb_seq)
    labels = torch.tensor(labels)
    emb_seq = torch.permute(emb_seq, (0, 2, 1))  # (n, c , l)
    return emb_seq, labels


def generate_augmented_data(args: object, emb_seq) -> torch.tensor:
    print(f'Load checkpoint {args.checkpoint} diffusion model')
    dataset = Dataset1D(emb_seq)
    model = Unet1D(**args.Unet_args)
    diffusion = GaussianDiffusion1D(model, **args.Diffusion_args)
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 10000,
        results_folder = args.results_folder,
        amp = True,                       # turn on mixed precision
    )
    trainer.load(milestone=args.checkpoint)
    sampled_emb = trainer.model.sample(batch_size = args.n_samples, torch_seed = 0).cpu()
    sampled_emb = sampled_emb.reshape(sampled_emb.shape[0],1,-1).squeeze()
    return sampled_emb


def compute_RD_and_diver(x_tr: torch.tensor, aug_sample: torch.tensor) -> tuple:
    ## Compute reconstruction degree and diversity (optional) ##
    pairwise_distances_between_groups_ = pairwise_distances(x_tr, aug_sample, metric='euclidean')
    RD = sum(np.min(pairwise_distances_between_groups_, axis=1)) / len(np.min(pairwise_distances_between_groups_, axis=1))

    diver = pairwise_distances(aug_sample, metric='euclidean').sum() / 2
    diver = diver / ((len(aug_sample) + 1) * len(aug_sample) / 2)
    return RD, diver


def SMOTE_for_each_data(x_tr: torch.tensor, y_tr: torch.tensor) -> tuple:
    smote = SMOTE(sampling_strategy='auto', k_neighbors=7, random_state=42)
    x_pos = x_tr[torch.where(y_tr==1)[0]]    # (n_pos, c)
    x_neg = x_tr[torch.where(y_tr==0)[0]]    # (n_neg, c)
    x_pseudo_neg = torch.randn(x_pos.shape[0]*2, x_pos.shape[1])   # pseudo neg for pos data (n_pos*2, c)
    x_pseudo_pos = torch.randn(x_neg.shape[0]*2, x_pos.shape[1])   # pseudo pos for neg data (n_neg*2, c)

    X_pos_all = torch.cat((x_pos, x_pseudo_neg))    # (n_pos*3, c)
    X_neg_all = torch.cat((x_neg, x_pseudo_pos))    # (n_neg*3, c)

    ## Generate augmented positive data ##
    pos_labels = torch.ones(len(x_pos))
    neg_labels = torch.zeros(len(x_pseudo_neg))
    labels = torch.cat((pos_labels, neg_labels))
    X_resampled, y_resampled = smote.fit_resample(X_pos_all, labels)
    x_aug_pos = torch.tensor(X_resampled[(x_pos.shape[0]*3):])

    ## Generate augmented negative data ##
    neg_labels = torch.zeros(len(x_neg))
    pos_labels = torch.ones(len(x_pseudo_pos))
    labels = torch.cat((neg_labels, pos_labels))
    X_resampled, y_resampled = smote.fit_resample(X_neg_all, labels)
    x_aug_neg = torch.tensor(X_resampled[(x_neg.shape[0]*3):])
    aug_sample = torch.cat((x_aug_pos, x_aug_neg))

    x_tr_with_aug = torch.cat((x_tr, aug_sample))
    y_tr_with_aug = torch.cat((y_tr, torch.ones(len(x_aug_pos)), torch.zeros(len(x_aug_neg))))
    RD, diver = compute_RD_and_diver(x_tr, aug_sample)
    return x_tr_with_aug, y_tr_with_aug, RD, diver


def process_augmented_data(
    args: object,
    x_tr: torch.tensor,
    y_tr: torch.tensor,
    sampled_emb: torch.tensor,
) -> tuple:       # x_tr_with_aug, y_tr_with_aug

    ## Decide labels of generated samples
    if args.labeling == 'KNN': ### use distance(KNN) to decide labels ###
        neigh = KNeighborsClassifier(n_neighbors=1)     # K=1, 1NN
        neigh.fit(x_tr, y_tr)
        aug_labels = torch.tensor(neigh.predict(sampled_emb))

    if args.labeling == 'IC':  ### use psuedo label to decide labels ###
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        clf.fit(x_tr, y_tr)
        aug_labels = torch.tensor(clf.predict(sampled_emb))

    ## Select augmented data from generated samples ##
    if args.Select_method == 'Euclidean': ## compute the distance, find the closest aug_sample for each training data ###
        pairwise_distances_between_groups = pairwise_distances(x_tr, sampled_emb, metric='euclidean')

        # find the kth close sample
        k = 0
        index_list = []
        for j in range(999):
            index_ = np.argmin(pairwise_distances_between_groups, axis=1)
            pairwise_distances_between_groups[np.arange(x_tr.shape[0]), index_] = float("inf")
            index_list.append(index_)

        aug_sample = sampled_emb[index_list[k]]
        aug_labels = aug_labels[index_list[k]]

    if args.Select_method == 'kmeans':  ### kmeans ###
        pos_sample = sampled_emb[torch.where(aug_labels==1)]
        neg_sample = sampled_emb[torch.where(aug_labels==0)]
        pos_kmeans = KMeans(n_clusters=len(torch.where(y_tr==1)[0]), random_state=0, n_init="auto").fit(pos_sample)
        pos_sample = torch.tensor(pos_kmeans.cluster_centers_)
        neg_kmeans = KMeans(n_clusters=len(torch.where(y_tr==0)[0]), random_state=0, n_init="auto").fit(neg_sample)
        neg_sample = torch.tensor(neg_kmeans.cluster_centers_)
        aug_sample = torch.cat((pos_sample, neg_sample), dim=0)

    pos_sample = aug_sample[torch.where(aug_labels==1)[0]]
    neg_sample = aug_sample[torch.where(aug_labels==0)[0]]

    pos_label = torch.ones(len(pos_sample))
    neg_label = torch.zeros(len(neg_sample))

    x_tr_with_aug = torch.cat((x_tr, pos_sample, neg_sample), 0)
    y_tr_with_aug = torch.cat((y_tr, pos_label, neg_label), 0)

    RD, diver = compute_RD_and_diver(x_tr, aug_sample)
    return x_tr_with_aug, y_tr_with_aug, RD, diver


def train_Word2Vec(args: object) -> None:
    seqs, _ = Load_Data(args.Data_path)
    Word2Vec_args = args.Word2Vec_args
    print("Start training Word2Vec model")
    words = sep_word(seqs, Word2Vec_args["kmer"])
    model = word2vec.Word2Vec(words, vector_size = Word2Vec_args["emb_dim"], min_count = 1, window = Word2Vec_args["window_size"] - Word2Vec_args["kmer"] + 1, epochs = Word2Vec_args["epochs"], sg = Word2Vec_args["sg"])
    model.save(Word2Vec_args["model_path"] + str(Word2Vec_args["kmer"]) + '_' + str(Word2Vec_args["emb_dim"]) + '_' + str(Word2Vec_args["epochs"]) + '_' + str(Word2Vec_args["window_size"]) + '_' + str(Word2Vec_args["sg"])  + "_new.pt")


def train_Diffusion(args: object) -> None:
    w2v_args = args.Word2Vec_args
    emb_seq, _ = get_word_embedding(w2v_args)
    dataset = Dataset1D(emb_seq)
    model = Unet1D(**args.Unet_args)
    diffusion = GaussianDiffusion1D(model, **args.Diffusion_args)
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 10000,
        results_folder = args.results_folder,
        amp = True,                       # turn on mixed precision
    )
    print("Start training diffusion model")
    trainer.train()


def train_Random_Forest(args: object) -> None:
    print("Start training random forest")
    w2v_args = args.Word2Vec_args
    emb_seq, labels = get_word_embedding(w2v_args)

    if args.use_augmented_data:
        print(f'Use {args.Augmentation_method} to generate data')
        if args.Augmentation_method == 'Diffusion':    # Load diffusion model
            sampled_emb = generate_augmented_data(args, emb_seq)

    acc_times = []
    mcc_times = []
    auc_times = []
    sen_times = []
    spec_times = []
    RD_times = []
    diver_times = []
    for i in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        acc_folds = []
        mcc_folds = []
        auc_folds = []
        sen_folds = []
        spec_folds = []
        if args.use_augmented_data:
            RD_folds = []
            diver_folds = []
        for k, (tr_idx, te_idx) in enumerate(kf.split(emb_seq)):
            x_tr, y_tr = emb_seq[tr_idx], labels[tr_idx]
            x_te, y_te = emb_seq[te_idx], labels[te_idx]
            x_tr = x_tr.reshape(x_tr.shape[0],1,-1).squeeze()
            x_te = x_te.reshape(x_te.shape[0],1,-1).squeeze()
            clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

            if args.use_augmented_data:
                if args.Augmentation_method == 'Diffusion':    # Process data generated by diffusion model
                    x_tr_with_aug, y_tr_with_aug, RD, diver = process_augmented_data(args, x_tr, y_tr, sampled_emb)
                if args.Augmentation_method == 'SMOTE':   # Use SMOTE to generate augmented data
                    x_tr_with_aug, y_tr_with_aug, RD, diver = SMOTE_for_each_data(x_tr, y_tr)
                clf.fit(x_tr_with_aug, y_tr_with_aug)
                RD_folds.append(RD)
                diver_folds.append(diver)
            else:
                clf.fit(x_tr, y_tr)
            prob = clf.predict_proba(x_te)
            prob_ = prob[:, 1]
            acc, sen, spec, mcc, auc = performance(np.argmax(prob, axis=1), prob_, y_te)
            acc_folds.append(acc)
            mcc_folds.append(mcc)
            auc_folds.append(auc)
            sen_folds.append(sen)
            spec_folds.append(spec)
        print(f'{i+1}time:|acc: {np.round(np.mean(acc_folds), 4)}±{np.round(np.std(acc_folds), 4)} | mcc: {np.round(np.mean(mcc_folds), 4)}±{np.round(np.std(mcc_folds), 4)} | auc: {np.round(np.mean(auc_folds), 4)}±{np.round(np.std(auc_folds), 4)} | sen: {np.round(np.mean(sen_folds), 4)}±{np.round(np.std(sen_folds), 4)} | spec: {np.round(np.mean(spec_folds), 4)}±{np.round(np.std(spec_folds), 4)}|')
        if args.use_augmented_data:
            print(f'{i+1}time: RD: {np.round(np.mean(RD_folds), 4)} | diversity: {np.round(np.mean(diver_folds), 4)}')
            RD_times.append(np.mean(RD_folds))
            diver_times.append(np.mean(diver_folds))
        acc_times.append(np.mean(acc_folds))
        mcc_times.append(np.mean(mcc_folds))
        auc_times.append(np.mean(auc_folds))
        sen_times.append(np.mean(sen_folds))
        spec_times.append(np.mean(spec_folds))
    print(f'average:|acc: {np.round(np.mean(acc_times), 4)}±{np.round(np.std(acc_times), 4)} | mcc: {np.round(np.mean(mcc_times), 4)}±{np.round(np.std(mcc_times), 4)} | auc: {np.round(np.mean(auc_times), 4)}±{np.round(np.std(auc_times), 4)} | sen: {np.round(np.mean(sen_times), 4)}±{np.round(np.std(sen_times), 4)} | spec: {np.round(np.mean(spec_times), 4)}±{np.round(np.std(spec_times), 4)}|')
    if args.use_augmented_data:
        print(f'average: RD: {np.round(np.mean(RD_times), 4)} | average: diver: {np.round(np.mean(diver_times), 4)}')


def create_parser() -> object:
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument("-c", "--config", help="config file", is_config_file=True)
    parser.add_argument(
        "--Data_path",
        help="Path of ADP Data",
    )
    parser.add_argument(
        "--Word2Vec_args",
        help="Arguments of Word2Vec",
        type=yaml.safe_load,
    )
    parser.add_argument(
        "--Unet_args",
        help="Arguments of Unet",
        type=yaml.safe_load,
    )
    parser.add_argument(
        "--Diffusion_args",
        help="Arguments of diffusion",
        type=yaml.safe_load,
    )
    parser.add_argument(
        "--results_folder",
        help="Path of diffusion model",
    )
    parser.add_argument(
        "-tw",
        "--train_Word2Vec",
        action="store_true",
        help="Pretrain word2Vec model"
    )
    parser.add_argument(
        "-td",
        "--train_Diffusion",
        action="store_true",
        help="Pretrain diffusion model"
    )
    parser.add_argument(
        "-tr",
        "--train_Random_Forest",
        action="store_true",
        help="Train random forest for prediction"
    )
    parser.add_argument(
        "-aug",
        "--use_augmented_data",
        action="store_true",
        help="Use the augmented data"
    )
    parser.add_argument(
        "--n_samples",
        default=1000,
        type=int,
        help="number of generated sample"
    )
    parser.add_argument(
        "-ck",
        "--checkpoint",
        default=10,
        type=int,
        help="checkpoint of diffusion model"
    )
    parser.add_argument(
        "--labeling",
        help="Method of labeling augmented data",
    )
    parser.add_argument(
        "--Select_method",
        help="Method of selecting augmented data from generated samples",
    )
    parser.add_argument(
        "--Augmentation_method",
        help="Method to generate augmented data",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = create_parser()
    if args.train_Word2Vec:
        train_Word2Vec(args)
    if args.train_Diffusion:
        train_Diffusion(args)
    if args.train_Random_Forest:
        train_Random_Forest(args)

import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, EsmModel
from utils.utils import load_config_file, mkdir
from utils.paths import DATA_DIR, MODEL_DIR

def generate_esm2_feature(config, dataset, split):
    print('start generating esm2 feature')

    if split in ['split_double_cold', 'split_drug_cold', 'split_protein_cold', 'split_random']:
        dataset_path = DATA_DIR / dataset
        mkdir(dataset_path)
    else:
        dataset_path = DATA_DIR / dataset / split
        mkdir(dataset_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if split == 'cluster':
        dfs = [pd.read_csv(dataset_path / f'{dataset}_with_id.csv') for dataset in
               ['source_train', 'target_train', 'target_test']]
        df = pd.concat(dfs)
    elif split in ['random', 'cold', 'augmented']:
        dfs = [pd.read_csv(dataset_path / f'{dataset}_with_id.csv') for dataset in ['train', 'val', 'test']]
        df = pd.concat(dfs)
    else:
        df = pd.read_csv(DATA_DIR / dataset / f'{dataset}_with_id.csv')

    # else:
    #     dfs = [pd.read_csv(f'{dataset_path}{dataset}_with_id.csv') for dataset in ['train', 'val', 'test']]
    # df = pd.concat(dfs)

    ems2_model_path = MODEL_DIR / 'protein' / 'esm2_model'
    tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
    model = EsmModel.from_pretrained(ems2_model_path).to(device)
    model.eval()  # disables dropout for deterministic results
    prlist = list()

    for protein_id in tqdm(df['pr_id'].unique(), desc='Processing'):
        protein_seq = df[df['pr_id'] == protein_id]['Protein'].iloc[0]
        inputs = tokenizer(protein_seq, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sr = outputs.last_hidden_state.squeeze()
        prlist.append(sr)

    save_path = os.path.join(dataset_path, config.TRAIN.PR_PATH)
    file = open(save_path, 'wb')
    pickle.dump(prlist, file)
    print('finish generating esm2 feature')


def kmeans_c(featurelist, config, d_model):
    matrix = np.vstack(featurelist)

    kmeans = KMeans(n_clusters=config.TRAIN.DICT_SIZE)
    kmeans.fit(matrix)

    labels = kmeans.labels_

    cluster_centers = np.zeros((d_model, config.TRAIN.DICT_SIZE))

    prior = list()
    total = 0
    for i in range(config.TRAIN.DICT_SIZE):
        cluster_position = matrix[labels == i]
        num = cluster_position.shape[0]
        prior.append(num)
        total += num
        cluster_centers[:, i] = np.mean(cluster_position, axis=0)
    prior = torch.tensor(prior)
    prior = prior / total
    cluster_dict = dict()
    cluster_dict['cluster_centers'] = cluster_centers
    cluster_dict['prior'] = prior
    return cluster_dict


def kmeans_for_c(train_config, train_df, save_path):
    print('start generating confounder dict & aa dict')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    df = train_df
    ems2_model_path = MODEL_DIR / 'protein' / 'esm2_model'
    max_length = 512
    d_model = 1280
    tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
    model = EsmModel.from_pretrained(ems2_model_path).to(device)
    model.eval()  # disables dropout for deterministic results

    featurelist = list()
    # store every aa feature
    aa_features = defaultdict(list)
    for protein_id in tqdm(df['pr_id'].unique(), desc='Processing'):
        protein_seq = df[df['pr_id'] == protein_id]['Protein'].iloc[0]
        inputs = tokenizer(protein_seq, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sr = outputs.last_hidden_state.squeeze()
        featurelist.append(sr.mean(0).cpu())
        sr_2 = sr[1:-1, :].cpu()
        for i in range(sr_2.size(0)):
            aa = protein_seq[i]
            vector = sr_2[i]  # Hidden state of the i-th amino acid
            aa_features[aa].append(vector)

    # Calculate the average feature for each amino acid type
    aa_avg_list = []
    for aa in sorted(aa_features.keys()):  # Sorting ensures consistent order
        vectors = torch.stack(aa_features[aa])  # Stack the list into a tensor
        avg_vector = torch.mean(vectors, dim=0)  # Compute the mean along the rows
        aa_avg_list.append(avg_vector)

    # Combine the average features of all amino acids into a single tensor
    aa_tensor = torch.stack(aa_avg_list)

    cluster_dict = kmeans_c(featurelist, train_config, d_model)
    cluster_dict['aa'] = aa_tensor


    file = open(os.path.join(save_path, train_config.TRAIN.C_PATH), 'wb')
    pickle.dump(cluster_dict, file)
    torch.cuda.empty_cache()
    print('finish generating confounder dict & aa dict')

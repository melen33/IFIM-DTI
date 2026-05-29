import torch
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import Dataset, DataLoader
import random
from rdkit import Chem
import numpy as np

class DTIDataset(Dataset):
    def __init__(self, list_IDs, df, pr_features):
        self.list_IDs = list_IDs
        self.df = df
        self.pr_features = pr_features
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        # smiles
        SMILES = self.df.iloc[index]['SMILES']
        # SMILES = randomize_smile(SMILES)
        # proteins
        pr_id = self.df.iloc[index]['pr_id']
        pr_seq = self.df.iloc[index]['Protein']
        Protein = self.pr_features[pr_id]
        # labels
        y = self.df.iloc[index]["Y"]
        return {
            'SMILES': SMILES,
            'Protein': Protein,
            'Protein_seq': pr_seq,
            'Y': y
        }

def randomize_smile(sml):
    """Function that randomizes a SMILES sequence. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequence to randomize.
    Return:
        randomized SMILES sequence or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        smiles = Chem.MolToSmiles(nm, canonical=False)

        return smiles

    except:
        return sml


def randomize_smile_with_mapping(sml):
    try:
        # Parse the SMILES string into a molecule object
        m = Chem.MolFromSmiles(sml)
        # Create a mapping from atom index to its position in the SMILES
        atom_indices = list(range(m.GetNumAtoms()))
        np.random.shuffle(atom_indices)  # Shuffle the indices

        # Renumber the atoms according to the shuffled indices
        nm = Chem.RenumberAtoms(m, atom_indices)

        # Generate the new SMILES string
        new_smiles = Chem.MolToSmiles(nm, canonical=False)

        # Create a mapping from original atom indices to new positions
        original_to_new_positions = {i: atom_indices.index(i) for i in range(len(atom_indices))}

        return new_smiles, original_to_new_positions
    except:
        return sml, {}

def convert_batch_pr(batch):
    max_len = max([tensor.shape[0] for tensor in batch])
    mask = torch.zeros(len(batch), max_len)
    for i, tensor in enumerate(batch):
        mask[i, :tensor.shape[0]] = 1
    padded_batch = pad(batch, batch_first=True)
    return {'input_ids': padded_batch, 'attention_mask': mask}

def mask_tokens(inputs, attention_mask, tokenizer, probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    mask = attention_mask.clone()
    mask[:, 0] = 0  # cls token
    eos_pos = mask.sum(dim=1).unsqueeze(1)
    mask.scatter_(1, eos_pos, 0)
    mask = mask.bool()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, probability)).bool()
    masked_indices = masked_indices * mask
    labels[~masked_indices] = -1  # We only compute loss on masked tokens
    #pos = (labels + 1).nonzero()[;]
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def drop_tokens(batch, drop_prob=0.7, aa_avg_f=None,mutation_rate=0):
    batch_id, batch_mask = batch['input_ids'], batch['attention_mask']
    # Step 1: Target random deletion
    if 0 < drop_prob:
        num_to_retain = int(max(1, batch_id.size(1) * (1-drop_prob)))

        indices = sorted(random.sample(range(1, batch_id.size(1)), num_to_retain))
        indices.insert(0, 0)
        batch_id = batch_id[:, indices]
        batch_mask = batch_mask[:, indices]

    # Step 2: Random mutation
    if aa_avg_f is not None and mutation_rate > 0:
        # Excluding CLS and EOS positions
        mask = batch_mask.clone()
        mask[:, 0] = 0  # CLS token
        eos_pos = mask.sum(dim=1).unsqueeze(1).to(dtype=torch.int64)
        mask.scatter_(1, eos_pos, 0)  # EOS token
        mask = mask.bool()
        # generate replace mask
        replace_mask = torch.rand(mask.shape) < mutation_rate
        replace_mask &= mask

        # generate random indices
        random_indices = torch.randint(
            0, aa_avg_f.size(0),
            batch_id.shape[:2], device=batch_id.device
        )

        # replace amino acid feature
        replacements = aa_avg_f[random_indices]

        batch_id[replace_mask] = replacements[replace_mask]

    return batch_id, batch_mask

def get_dataLoader(batch_size, dataset, drug_tokenizer,aa=None, shuffle=False, MLM=False,
                   mask_rate=0, target_random_deletion_ratio=0, mutation_rate=0):
    def collate_fn(batch_samples):
        batch_Drug, batch_Protein, batch_seq, batch_label = [], [], [], []
        for sample in batch_samples:
            batch_Drug.append(sample['SMILES'])
            batch_Protein.append(sample['Protein'])
            batch_seq.append(sample['Protein_seq'])
            batch_label.append(sample['Y'])
        batch_pr = convert_batch_pr(batch_Protein)
        batch_inputs_drug = drug_tokenizer(batch_Drug, padding='longest', return_tensors="pt", truncation=True, max_length=200)
        batch_inputs_drug_m, masked_drug_labels = None, None
        # if 0 < target_mask_rate:
        batch_pr['input_ids'], batch_pr['attention_mask'] = drop_tokens(batch_pr, target_random_deletion_ratio,
                                                                            aa_avg_f=aa, mutation_rate=mutation_rate)

        if MLM:
            batch_inputs_drug_m = batch_inputs_drug
            batch_inputs_drug_m['input_ids'], masked_drug_labels\
                = mask_tokens(batch_inputs_drug_m['input_ids'], batch_inputs_drug_m['attention_mask'], drug_tokenizer,mask_rate)

        return {
            'batch_inputs_drug': batch_inputs_drug,
            'batch_inputs_drug_m': batch_inputs_drug_m,
            'masked_drug_labels': masked_drug_labels,
            'batch_inputs_pr': batch_pr,
            'labels': batch_label
        }
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

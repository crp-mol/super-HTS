#!/usr/bin/env python3
import numpy as np
import pandas as pd
import re
import scipy
import os
import argparse
import time
import matplotlib.pyplot as plt
from typing import Union
import warnings
import tensorflow as tf
import spektral
from spektral.transforms.normalize_adj import NormalizeAdj
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
from MDAnalysis.analysis import distances

def load_dataset(csv_file: str) -> Union[pd.DataFrame, float, float]:
    '''
    Dataset contains the list of mutants scored by Rosetta or any other algorithm. 
    E.g.,
    
    Return the original mean and standard deviation. Useful for de-normalizing the dataset
    '''
    df = pd.read_csv(csv_file, names=['score','stdev','replicas','mutant'], header=0, sep='\s+')
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True) # shuffle
    
    # normalize data: mean=0.0, stdev=1.0
    mean , std = df['score'].mean() , df['score'].std()
    n = df.count()[0]
    df['score'] = (df['score'] - mean) / std
    print (f'Loaded and shuffled dataset from "{csv_file}" with {n} mutants. {mean=:.4f} and {std=:.4f} before normalization')
    return df, mean, std

def load_aaindex(csv_file: str, normalize: bool=True):
    '''
    Aminoacid properties across 566 rows (566 properties). Every column is one amino acid. The last column is the name of the property.
    E.g., 
        A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,AAIndex
        0.83,0.83,0.09,0.64,1.48,0.00,0.65,0.10,1.10,3.07,2.52,1.60,1.40,2.75,2.70,0.14,0.54,0.31,2.97,1.79,Hydrophobicity (Zimmerman et al. 1968)
        11.50,14.28,12.82,11.68,13.46,14.45,13.57,3.40,13.69,21.40,21.40,15.71,16.25,19.80,17.43,9.47,15.77,21.67,18.03,21.57,Bulkiness (Zimmerman et al. 1968)
    '''
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f'AAIndex file `{csv_file}` not found, you can download it using `aaindex.py`.')

    df = pd.read_csv(csv_file, sep=',', index_col=0, header=0)
    print (f'Loaded {len(df.index) - 1} AA properties from "{csv_file}"')
    if normalize:
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        df = df.sub(mean, axis=0)
        df = df.div(std, axis=0)
    return df

def list_hotspots(mutants: list) -> list:
    '''
    Given a list of mutants, return the list of positons that were allowed to mutate following to the input list.
    input: ['F19A','F19A_F85A','F19A_F85C',...]
    output: [19,85,...]
    '''
    positions = []
    for m in mutants:
        p = re.findall(r'\d+', m) # format ['236','238',...]
        positions.extend(p)
    mutable_pos = list(set(positions))
    print (f'Found {len(mutable_pos)} mutable positions: {mutable_pos}')
    return mutable_pos

def find_active_site(mutable_pos: list, pdb_reference: str, cutoff: float=2.8, ligand_name: str = None):
    '''
    Find the residues that make up the active site. 
    mutable_pos is a list of positions that were allowed to mutate: ['236','238',...]
    cutoff is the distance (in Angstroms) from any ligand heavy atom to define the active site.
    pdb_reference is the name of the pdb file to use as reference. It must be the wild-type enzyme.
    output: {56: ('A', 'L'), 57: ('A', 'W'), 116: ('A', 'S'), 117: ('A', 'G'), 118: ('A', 'S'), ...} 
    '''
    def _find_ligand_name(u: mda.Universe):
        resnames = u.select_atoms('not (protein or resname HOH WAT WTR SOL)').residues.resnames
        if resnames.size == 0:
            raise ValueError(f'Input structure {pdb_reference} does not have any ligand')
        if resnames.size > 1:
            raise ValueError(f'Input structure {pdb_reference} contains more than one ligand, define it with the argument `ligand_name`. {resnames=}')
        return resnames[0] 

    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K','ILE': 'I', 'PRO': 'P', 
            'THR': 'T', 'PHE': 'F', 'ASN': 'N','GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 
            'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    u = mda.Universe(pdb_reference)   
    ligand_name = ligand_name if ligand_name is not None else _find_ligand_name(u)
    neighbours = u.select_atoms(f'protein and around {cutoff} resname {ligand_name}')
    active_site = {k:(v,d3to1[u]) for k,v,u in zip(neighbours.residues.resnums, neighbours.residues.segids, neighbours.residues.resnames)}
    print (f'Loaded {pdb_reference=}, found {active_site=}.')
    return active_site

def generate_distogram(active_site: dict, pdb_reference: str, inverse: bool = False) -> np.ndarray:
    '''
    Produce a np array of [n,n] with the pairwise distances of all the resides listed in the active_site. 
    Distances are defined from the CA atoms.
    active_site = {417: ('A', 'L'), 85: ('B', 'F'),...}
    pdb_reference='4e3q.pdb'
    '''
    u = mda.Universe(pdb_reference)   

    # unfortunately, distances.distance_array of MDAnalysis does not respect atom ordering, so the distances need to be calculated manually
    atoms = tuple(u.select_atoms(f'name CA and resnum {resnr} and segid {segid_resname[0]}') for resnr,segid_resname in active_site.items())
    distogram = np.zeros([len(atoms), len(atoms)])
    for i,atom_i in enumerate(atoms):
        for j,atom_j in enumerate(atoms):
            dist = distances.dist(atom_i, atom_j)[2][0]
            distogram[i,j] = dist

    # To fix: 1/0 =inf, it may not be a good idea to make 1/0 = 0, maybe 1/0 = max of all other inverse?
    if inverse:
        distogram = 1 / distogram

    return distogram

def featurize_dataset(mutants: list, aaindex: pd.DataFrame, features: list, active_site: dict) -> np.ndarray:
    '''
    Output is a matrix with shape [5000, 23, 30] => [n_mutants, n_nodes, n_features]
    active_site =  {417: ('A', 'L'), 85: ('B', 'F'),... 
    mutants=['F19S_W57D', 'F85P_F19E', 'F19W_F85D_Y150N',...]
    features=[0, 20, 34, 67, 99, ..,]
    '''
    assert features == sorted(features), f'Feature list not sorted.'

    start = time.time()

    # create an (nr_mutants, nr_nodes) array containing the AA for each node
    seqs = []
    for mutant in mutants: 
        _,  positions, to_ = mutant_id_split(mutant)
        d = {p:t for p,t in zip(positions,to_)} # temporary dictionary containing the mutant active site {180:'F', 191:'C',...}
        resnames = [v[1] if k not in positions else d[k] for k,v in active_site.items()] 
        seqs.append(resnames)
    seqs = np.array(seqs) # array([['L', 'D', 'S', ..., 'F', 'F', 'T'], ['L', 'W', 'S', ..., 'F', 'F', 'T'],...] shape is (nr_mutants, nr_nodes)

    # create empty output array 
    nr_mutants , nr_nodes , nr_features = len(mutants), len(active_site),  len(features)
    X = np.zeros([nr_mutants, nr_nodes, nr_features]) # (10050, 21, 30)

    # featurize
    for i,feature in enumerate(features):
        d_property = aaindex.iloc[feature].to_dict()   # {'A': -0.263, 'R': -0.1468, 'N': 1.293,}
        x = np.vectorize(d_property.__getitem__)(seqs) # x.shape = (10050, 21)
        X[:,:,i] = x

    end = time.time()
    print (f'Created feature array {X.shape=} in {end-start:.2f} seconds')
    return X

def create_edgematrix(distogram: np.ndarray) -> scipy.sparse._csr.csr_matrix:
    '''
    From a distogram (np array of nxn) create an edge matrix of type sparse.
    Format: (0,0) 0.0
            (0, 1) 3.853
            (0, 2) 5.3865
    '''
    m, n = distogram.shape
    assert m==n, f'distogram is not symmetric, dimensions are: {distogram.shape}'
    row = [j for j in range(n) for i in range (n)] # [0,0,0,0,1,1,1,1,2,2...]
    col = [i for i in range(n)] * n # [0,1,2,3,0,1,2,3,0,1,2...]
    data = [distogram[i,j] for i in range(n) for j in range(n)] # [17.75141, 20.9143, 16.502, 17.97709,..]
    edge_matrix = scipy.sparse._csr.csr_matrix((data, [row, col]), shape=[n, n])
    return edge_matrix

class EnzDataset(spektral.data.Dataset):
    def __init__(self, X, labels, edge_matrix, **kwargs):
        self.X = X
        self.n_features = X.shape[-1]
        self.labels = labels
        self.edge_matrix = edge_matrix
        self.n_samples = X.shape[0]
        super().__init__(**kwargs)

    def read(self):
        def _make_graph(i):
            x = self.X[i]
            y = np.array([self.labels[i]])
            a = self.edge_matrix
            return spektral.data.Graph(x=x,a=a,y=y)
        return [_make_graph(i) for i in range(self.n_samples)]

    def get_example(self):
        def _make_graph(i,a):
            x = self.X[i]
            y = np.array([self.labels[i]])
            return spektral.data.Graph(x=x,a=a,y=y)
        graph = _make_graph(1, self.edge_matrix).numpy()
        print (f'The output graphs have length of {len(graph)}')
        for i in range(len(graph)):
            print (f'Shape of graph[{i}] is {graph[i].shape}')
        return graph

class Enzyme:
    def __init__(self, pdb_path, mutable_pos):
        self.pdb_path = pdb_path
        self.mutable_pos = mutable_pos
        self.active_site = None
        self.distogram = None

    def find_active_site(self):
        self.active_site = find_active_site(self.mutable_pos, self.pdb_path)

    def generate_distogram(self):
        self.distogram = generate_distogram(active_site=self.active_site, pdb_reference=self.pdb_path)

    def create_edgematrix(self):
        self.edge_matrix = create_edgematrix(self.distogram)

def create_loaders(dataset: EnzDataset, batch_size: int, epochs: int=1, training: bool=False, **kwargs):
    '''
    Create loaders (DisjointLoader) from dataset. It is possible to pass the split proportions for train/val/test in **kwargs
    keys are "tr" "val" and "te". At least "tr" must be provided.
    '''
    # Determine split from **kwargs
    tr = kwargs['tr'] if 'tr' in kwargs.keys() else None
    val = kwargs['val'] if 'val' in kwargs.keys() else None
    te = kwargs['te'] if 'te' in kwargs.keys() else None
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs.keys() else True

    if tr == val == te == None:
        tr, val, te = 80, 20, None

    nr = len(dataset)
    if shuffle:
        np.random.seed(821)
        idxs = np.random.permutation(nr) # [0,1,2,3,4...]
    else:
        print (f'Will not shuffle dataset with {nr} mutants and {tr=}, {val=}, {te=}.')
        idxs = np.arange(nr)

    # Calculate indexes for the loaders
    if val is not None and te is not None:
        k = tr/100
        l = (tr/100) + (val/100)
        val_cut, te_cut = int(k * nr), int(l * nr)
        idx_tr, idx_va, idx_te = np.split(idxs, [val_cut, te_cut]) # [2,6,3...], [], []

    if val is not None and te is None:
        val_cut = int((tr/100) * nr)
        idx_tr, idx_va = np.split(idxs, [val_cut]) 

    if val is None and te is None:
        idx_tr = idxs

    # Make loaders
    if training:
        loader_tr = spektral.data.DisjointLoader(dataset[idx_tr], batch_size=batch_size, epochs=epochs, shuffle=True)
    else:
        loader_tr = spektral.data.DisjointLoader(dataset[idx_tr], batch_size=batch_size, epochs=None, shuffle=False)
    loader_va = None if val is None else spektral.data.DisjointLoader(dataset[idx_va], batch_size=batch_size, epochs=None, shuffle=False)
    loader_te = None if te is None else spektral.data.DisjointLoader(dataset[idx_te], batch_size=batch_size, epochs=None, shuffle=False)
    return loader_tr, loader_va, loader_te

def fit_model(model, loader_tr, loader_va, loss_fn, acc_fn, opt, es_patience):
    global evaluate

    '''
     input_signature: ((TensorSpec(shape=(None, 3), dtype=tf.float64, name=None), SparseTensorSpec(TensorShape([None, None]), tf.float64), TensorSpec(shape=(None,), dtype=tf.int64, name=None)),
    TensorSpec(shape=(None, 1), dtype=tf.float64, name=None)).
    '''
    @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model((inputs[0], inputs[1]), training=True)
            loss = loss_fn(target, predictions)
            loss += sum(model.losses)
            acc = acc_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, acc

    def evaluate(loader, optimizers_list):
        output=[]
        step = 0
        # Loop through each element in the dataset
        while step < loader.steps_per_epoch:
            step += 1
            inputs, target = loader.__next__()
            pred = model((inputs[0], inputs[1]), training=False)
            outs = [o(target,pred) for o in optimizers_list]
            output.append(outs)
        return np.mean(output, 0) 

    current_batch = epoch = model_loss = model_acc = 0 
    best_val_loss, best_weights, patience = np.inf, None, es_patience
    loss_list , acc_list , val_loss_list , val_acc_list = [], [], [], []
    steps_per_epoch = loader_tr.steps_per_epoch
    for batch in loader_tr:
        start = time.time() # time for each epoch
        loss, acc = train_step(batch[0], batch[1])
        model_loss += loss
        model_acc += acc
        current_batch += 1
        if current_batch == steps_per_epoch: 
            model_loss /= steps_per_epoch
            model_acc /= steps_per_epoch
            loss_list.append(model_loss)
            acc_list.append(model_acc)

            epoch += 1
            val_loss, val_acc = evaluate(loader_va, [loss_fn, acc_fn])
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            end = time.time()
            elapsed_t = end - start
            print (f'{epoch=}, {model_loss=:.2f}, {model_acc=:.2f}, {val_loss=:.2f}, {val_acc=:.2f}, in {elapsed_t:.2f} seconds')

            # Check if loss improved for early stopping
            if val_loss < best_val_loss:
                best_val_loss, patience, best_weights = val_loss, es_patience, model.get_weights()
            else:
                patience -= 1
                if patience == 0:
                    print ('Early stopping.')
                    break

            model_loss , model_acc, current_batch = 0, 0, 0

    return np.array(loss_list), np.array(acc_list), np.array(val_loss_list), np.array(val_acc_list), best_weights

def evaluate_model(model, loader):
    '''
    Evaluate trained model with data loader. Print out predictions.
    For now the loader needs to be with batch size of zero.
    '''
    preds=[]
    targets=[]
    step = 0
    # Loop through each element in the dataset
    while step < loader.steps_per_epoch:
        step += 1
        if step%500 == 0:
            print (f'Done with step {step}')
        inputs, target = loader.__next__()
        pred = model((inputs[0], inputs[1]), training=False)
        pred = pred.numpy()[0][0] # <- change if you want to allow more batch sizes
        preds.append(pred)
        target = target[0][0]
        targets.append(target)
    return preds, targets

def mutant_id_split(mutant_id: str):
     '''
     mutant_id='F19S_W57D'
     from_ = ('F', 'W')
     positions = ('19', '57')
     to_ = ('S', 'D')
     ''' 
     mutations = re.findall(r'([A-Z])(\d+)([A-Z])', mutant_id) # [('F', '19', 'S'), ('W', '57', 'D')]
     from_ = tuple(mutation[0] for mutation in mutations)
     positions = tuple(int(mutation[1]) for mutation in mutations)
     to_ =  tuple(mutation[2] for mutation in mutations)
     return from_, positions, to_

def check_validity_mutants(mutants: list, mutable_pos: list=None, active_site: dict=None):
    '''
    Check that the proposed mutants are valid if:
    1) consistency list of mutants              # <- error
    2) they have mutations in mutable positions # <- warning
    3) they have mutations in valid positions   # <- error
    4) the mutations (from_, to_) are correct   # <- error
    mutants = ['F86L_W85G',...] 
    mutable_pos = [86,85,...] or empty
    active_site={56: ('A', 'L'), 118: ('A', 'S'), ...} or empty
    '''
    from_, positions, to_ = list(zip(*[mutant_id_split(mutant_id) for mutant_id in mutants]))

    # 1)
    d_pos2aa = {k:v for k,v in zip(list(np.concatenate(positions).flat), list(np.concatenate(from_).flat))} # {150: 'Y', 57: 'W', 19: 'F', 85: 'F'}
    for k,v in zip(list(np.concatenate(positions).flat), list(np.concatenate(from_).flat)):
        if d_pos2aa[k] != v:
            raise ValueError(f'Position {k} has one or more from_ values in mutant list: {d_pos2aa[k]} and {v}')
    pos = set(np.concatenate(positions).flat)
    # 2)
    if mutable_pos:
        mutable_pos = set([int(p) for p in mutable_pos])
        if len(pos - mutable_pos):
            warnings.warn(f'Positions {(pos - mutable_pos)} are not part of the mutable positions in the dataset used for training, {mutable_pos=}.')
    if not active_site:
        return     
    # 3)
    for p in pos:
        if p in active_site.keys():
            continue
        raise ValueError(f'Position {p} provided in the list of mutants is not valid. Valid positions are {active_site.keys()}.')
    # 4)
    aa_pos = set([str(k)+str(v) for k,v in zip(list(np.concatenate(from_).flat), list(np.concatenate(positions).flat))]) # {'F19', 'Y150', 'W57', 'F85'}
    aa_pos_ref = set([str(v[1]) + str(k) for k,v in active_site.items()])
    diff = (aa_pos - aa_pos_ref)
    if len(diff)>0:
        raise ValueError(f'Positions assigned incorrectly in mutant dataset: {diff}')
    return

def evaluate_mutants(mutants: list, model, aaindex: pd.DataFrame, features: list, active_site: dict, mutable_pos: list, edge_matrix: scipy.sparse._csr.csr_matrix, mean: float, std: float):
    '''
    Evalualuate a user-provided list of mutants.
    mutants = ['F86L_W85G',...]
    model= trained model
    features=list of features to take from aaindex [=] [0,1,3,4,5]
    active_site={56: ('A', 'L'), 118: ('A', 'S'), ...}
    mean, std = average and stdev of the scores of the dataset (useful for denormalization)
    returns = [-13.7, -11.78, -12.14,...]
    '''
    check_validity_mutants(mutants, mutable_pos, active_site)

    # Create datasets from mutants
    X=featurize_dataset(mutants, aaindex, features=features, active_site=active_site)
    labels = [0.0]*len(mutants)
    enzdataset = EnzDataset(X, labels, edge_matrix, transforms=NormalizeAdj()) 
    loader, _, _ = create_loaders(enzdataset, batch_size=1, epochs=1, training=False, tr=100, shuffle=False)
    
    # make predictions
    start = time.time()    
    preds, _ = evaluate_model(model, loader)
    end = time.time()
    elapsed_t = end - start
    print (f'Predicted score of {len(mutants)} mutants in {elapsed_t:.2f} seconds')

    # Denormalize the predictions 
    preds = [(pred*std)+mean for pred in preds]

    return preds

def evaluate_custom_csv(input_csv: str, output_csv: str, model, aaindex, features, active_site, mutable_pos, edge_matrix, mean, std):
    '''
    Evaluate a list of mutants provided by the user. Output is a DataFrame object containing the predictions. 

Input csv format:
    score     std   replicas mutant_id
    -12.1670 0.2833 10      Y150G_W57Y_F19Q_F85H
    -11.2190 0.2641 10      Y150D_F19D_F85E
    -13.2430 0.1451 10      F85K_F19C
or 
    mutant_id
    Y150G_W57Y_F19Q_F85H
    Y150D_F19D_F85E
    F85K_F19C
or 
    score   mutant_id
    -12.1670    Y150G_W57Y_F19Q_F85H
    -11.2190    Y150D_F19D_F85E

    '''
    df = pd.read_csv(input_csv, sep='\s+', names=['score','std','replicas','mutant_id'])
    df =  df[['mutant_id','score']]
    mutants = df['mutant_id'].tolist()
    preds = evaluate_mutants(mutants, model, aaindex, features, active_site, mutable_pos, edge_matrix, mean, std)
    df['predictions'] = preds
    df['error'] = (df['score'] - df['predictions']) 
    mean_error = df['error'].abs().mean()
    print (f'The average absolute deviation was {mean_error} kcal/mol over the {len(mutants)} mutants evaluated.')
    return df

def plot_training(loss, acc, val_loss, val_acc):
    '''
    Print the training history.
    Two panes: A) train and validation losses
               B) train and validation accuracies
    '''
    from matplotlib.ticker import FormatStrFormatter

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    
    axes[0].plot(loss, label='training', color='black', marker='o', markersize=6)
    axes[0].plot(val_loss, label='validation', color='gray', marker='s', markersize=6)
    axes[0].set_xlabel('Epochs', fontsize=13) 
    axes[0].set_ylabel('loss', color = 'black', fontsize=13) 
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f')) 
    axes[0].legend()
    
    axes[1].plot(acc, label='training', color='black', marker='o', markersize=6)
    axes[1].plot(val_acc, label='validation', color='gray', marker='s', markersize=6)
    axes[1].set_xlabel('Epochs', fontsize=13) 
    axes[1].set_ylabel('acc.', fontsize=13) 
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[1].legend()
    
    # Show plot
    fig.tight_layout()
    fig.patch.set_facecolor('white')
    fig.savefig('train_history.png', dpi=300)
    print (f'Printed plot "train_history.png" with training history of loss and accuracy across {len(loss)} epochs.')

def plot_regression(x, y, out_file, title):
    '''
    Plot the predicted vs ground truth figure. Include R2 and fit line.
    '''
    # Calculate the point density
    xy = np.vstack([x,y])
    z = scipy.stats.gaussian_kde(xy)(xy)
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=20)
    
    # regression
    xseq = np.linspace(x.min(),x.max(), num=100)
    b, a = np.polyfit(x, y, deg=1)
    ax.plot(xseq, a + b * xseq, color="k", lw=2.0)
    res = scipy.stats.linregress(x, y)
    plt.text(0.08, 0.82, f'R-squared: {res.rvalue:.2f}', fontsize = 12, transform=ax.transAxes)
    plt.text(0.6, 0.20, f'y = {res.slope:.3f}x + {res.intercept:.3f}', fontsize = 11, transform=ax.transAxes)
    plt.text(0.6, 0.10, f'n={len(x)}', fontsize = 11, transform=ax.transAxes)
    plt.title(title, fontsize=13)
    
    ax.set_xlabel('target (kcal/mol)', fontsize=13)
    ax.set_ylabel('predicted (kcal/mol)', fontsize=13)
    
    fig.patch.set_facecolor('white')
    plt.savefig(out_file, dpi=300)
        
    print (f'Printed plot {title=} "{out_file}" (n={len(x)}) with R-squared: {res.rvalue:.2f}, slope: {res.slope:.3f}, and intercept: {res.intercept:.3f}')

def parse_args():
    description='Train/Evaluate a GCN model to predict the score of a set of mutants.'
    epilog=f'Example: ./{os.path.basename(__file__)} -f dataset.csv -r 4e3q.pdb --aa_index AAIndex.csv --train_model'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description, epilog=epilog)
    parser.add_argument('-f', '--train_dataset', type=str, default=None, help='Input csv file with the list of mutants and their corresponding score to be used for training and validation.')
    parser.add_argument('-r', '--pdb_reference', type=str, required=True, help='Reference structure of the protein with the ligand. Only needed to list the residues near the binding site and build the distogram.')
    parser.add_argument('--train_model', action='store_true', help='Train the model instead of loading pretrained weights from "./trained_models/best_weights.ckpt"')
    parser.add_argument('--aa_index', type=str, required=True, help='Input csv with amino acid features. e.g., "AAIndex.csv"')
    parser.add_argument('--input_eval', type=str, required=False, help='File containing a list of mutants the user wants to evaluate.')
    parser.add_argument('--mutants', type=str, nargs='+', required=False, help='List of mutants to evaluate. Format: Y150D_F19D_F85E,F85K_F19C,Y150A')
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()

    # Prevent the GPU from eating all the RAM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2046)])

    # Load dataset. Calculate the mutable positions and the active site positions around the mutable positions. Calc. the distogram too. 
    dataset, mean, std = load_dataset(args.train_dataset) # 'mutant', 'score'
    mutants, labels = dataset['mutant'].tolist(), dataset['score'].tolist()
    mutable_pos = list_hotspots(mutants)

    # Enzyme
    enzyme = Enzyme(args.pdb_reference, mutable_pos)
    enzyme.find_active_site()
    enzyme.active_site = {18: ('A','G'), 19: ('A','F'), 55: ('A','G'), 56: ('A','L'), 57: ('A','W'), 118: ('A','S'), 150: ('A','Y'),
               151: ('A','H'), 223: ('A','E'), 225: ('A','V'), 227: ('A','G'), 228: ('A','A'), 256: ('A','D'),
               257: ('A','E'), 415: ('A','R'), 417: ('A','L'), 424: ('A','C'), 82: ('B','Y'), 83: ('B','H'), 84: ('B','A'),
               85: ('B','F'), 86: ('B','F'), 88: ('B','R')}
    enzyme.generate_distogram()
    enzyme.create_edgematrix()

    _ = check_validity_mutants(mutants=mutants, mutable_pos=mutable_pos, active_site=enzyme.active_site)

    # Featurization of dataset and loaders creation
    n_epochs = 40
    es_patience = 5
    features = [i for i in range(30)] ## <------ user-defined, random also works
    aaindex = load_aaindex(args.aa_index) # AAIndex.csv
    X = featurize_dataset(mutants, aaindex, features=features, active_site=enzyme.active_site) # shape=(10050, 21, 30)
    enzdataset = EnzDataset(X, labels, enzyme.edge_matrix, transforms=NormalizeAdj()) #enzdataset[0].x
    loader_tr, loader_va, loader_te = create_loaders(enzdataset, batch_size=1, epochs=n_epochs, training=True, tr=80, val=20)
    
    # Build the model
    X_in = tf.keras.layers.Input(shape=(enzdataset.n_node_features,), name='X_in')
    A_in = tf.keras.layers.Input(shape=(None,), sparse=True, name='A_in')
    X_1 = spektral.layers.GCSConv(64, activation='relu')((X_in, A_in))
    pool1 , adj1 = spektral.layers.MinCutPool(k=10, activation='relu')([X_1, A_in])
    X_2 = spektral.layers.GCSConv(8, activation='relu')([pool1, adj1])
    pool2, adj2 = spektral.layers.MinCutPool(k=5, activation='relu')([X_2, adj1])
    X_3 = spektral.layers.GlobalAvgPool()([pool2])
    output = tf.keras.layers.Dense(1)(X_3) # output dimension is 1
    model = tf.keras.models.Model(inputs = (X_in, A_in), outputs=output)
    model.summary()

    # Fit model
    loss_fn = tf.keras.losses.MeanSquaredError()
    acc_fn = tf.keras.metrics.MeanAbsoluteError()
    opt = tf.keras.optimizers.Adam()
    if args.train_model:
        print (f'Fitting model for {n_epochs=} with early stop patience: {es_patience}')
        loss, acc, val_loss, val_acc, best_weights = fit_model(model, loader_tr, loader_va, loss_fn, acc_fn, opt, es_patience)
        print (f'{loss.mean()=:.2f}, {acc.mean()=:.2f}, {val_loss.mean()=:.2f}, {val_acc.mean()=:.2f}')

        # Save weights
        model.save_weights('./trained_models/best_weights.ckpt')

        # Print traing history 
        plot_training(loss, acc, val_loss, val_acc)

        # Evaluate model and print regression plots
        loader_tr, loader_va, loader_te = create_loaders(enzdataset, batch_size=1, epochs=None, training=False, tr=80, val=20)

        preds, targets = evaluate_model(model, loader_tr)
        preds = [(i * std )+ mean for i in preds] # Denormalize
        targets = [(i * std )+ mean for i in targets]
        plot_regression(x=np.array(preds), y=np.array(targets), out_file='training_scatter.png', title='Training dataset')

        preds, targets = evaluate_model(model, loader_va)
        preds = [(i * std )+ mean for i in preds]
        targets = [(i * std )+ mean for i in targets]
        plot_regression(x=np.array(preds), y=np.array(targets), out_file='validation_scatter.png', title='Validation dataset')

    else:
        print (f'Loading weights best_weights.ckpt for pre-trained model. Will not re-train the model because the --train_model argument was not passed.')
        model.load_weights('./trained_models/best_weights.ckpt')
    
    # Evaluate mutants provided by user
    mutants = ['F19A_F85C','F19A_F85D','F19L_W57E_Y150G']
    evaluate_mutants(args.mutants, model=model, aaindex=aaindex, features=features, active_site=enzyme.active_site, mutable_pos=mutable_pos, edge_matrix=enzyme.edge_matrix, mean=mean, std=std)

    if args.input_eval is not None:
        _ = evaluate_custom_csv(args.input_eval,'out.csv', model=model, aaindex=aaindex, features=features, active_site=enzyme.active_site, mutable_pos=mutable_pos, edge_matrix=enzyme.edge_matrix, mean=mean, std=std)


if __name__ == '__main__':
    main()

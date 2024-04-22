## placeholder for preprocess.py
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
from multiprocessing import Process, Queue, Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np 
import pickle
import lmdb
import sys
import glob
import os
import re


def normalize_atoms(atom):
    return re.sub("\d+", "", atom)

def car_parser(car_path):
    return None

def cif_parser(cif_path, primitive=True):
    """
    Parser for single cif file
    """
    id = cif_path.split('/')[-1][:-4]
    s = Structure.from_file(cif_path, primitive=primitive)
    # analyzer = SpacegroupAnalyzer(s)
    # sym_cell = analyzer.get_symmetrized_structure()
    # spacegroup_info = analyzer.get_space_group_info()   # e.g. ('Fm-3m', 225)
    # wyckoff_symbol = sym_cell.wyckoff_symbol()

    lattice = s.lattice
    abc = lattice.abc # lattice vectors
    angles = lattice.angles # lattice angles
    volume = lattice.volume # lattice volume
    lattice_matrix = lattice.matrix # lattice 3x3 matrix

    df = s.as_dataframe()
    atoms = df['Species'].astype(str).map(normalize_atoms).tolist()
    coordinates = df[['x', 'y', 'z']].values.astype(np.float32)
    abc_coordinates = df[['a', 'b', 'c']].values.astype(np.float32)
    assert len(atoms) == coordinates.shape[0]
    assert len(atoms) == abc_coordinates.shape[0]

    return pickle.dumps({'ID':id, 
            'atoms':atoms, 
            'coordinates':coordinates, 
            'abc':abc, 
            'angles':angles, 
            'volume':volume, 
            'lattice_matrix':lattice_matrix, 
            'abc_coordinates':abc_coordinates
            }, protocol=-1)

def single_parser(cif_path):
    try:
        return cif_parser(cif_path, primitive=True)
    except:
        return None

def collect_cifs():
    cif_paths = []
    base = "nanocrystals"
    print("base:", base)
    # base = "debug"
    for folder in os.listdir(base):
        print(folder)
        files = sorted(os.listdir(os.path.join(base, folder)))
        for q in tqdm(files):
            name = q[:-4]
            if q.endswith(".cif"):
                # 测试 CIF 文件路径
                cif_file_path = os.path.join(base, folder, q)
                # cif_data = cif_parser(cif_file_path)
                cif_paths.append(cif_file_path)
    return cif_paths 

def write_lmdb(outpath='./', nthreads=40):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    cif_paths = collect_cifs()
    np.random.seed(42)
    cif_paths = np.random.permutation(cif_paths)
    for name, cifs in [('train.lmdb', cif_paths)]:
        outputfilename = os.path.join(outpath, name)
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(10e12),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(single_parser, cifs), total=len(cifs)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
                    if i % 1000 == 0:
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()

def token_collect(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    outputfilename = os.path.join(dir_path, 'train.lmdb')
    env = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(10e9),
        )
    txn = env.begin()
    atoms_collects = []
    _keys = list(txn.cursor().iternext(values=False))
    for idx in tqdm(range(len(_keys))):
        datapoint_pickled = txn.get(f'{idx}'.encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        atoms_collects.extend(data['atoms'])
        
    import numpy as np
    
    atoms = np.unique([v for v in atoms_collects])
    atoms = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"] + atoms.tolist()
    
    with open(os.path.join(dir_path, "dict.txt"), "w") as f:
        for v in atoms:
            f.write(v+"\n")

if __name__ == '__main__':
    save_path = "ours0422"
    write_lmdb(save_path)
    token_collect(save_path)

from unimol_tools import UniMolRepr
from pymatgen.io.cif import CifParser
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pymatgen.io.xyz import XYZ
import pickle as pkl

def read_xyz_file(file_path):
    # Read XYZ file using pymatgen
    xyz = XYZ.from_file(file_path)
    structure = xyz.molecule

    # Initialize a dictionary for data
    custom_data = {'atoms': [], 'coordinates': []}

    # Get atomic symbols and coordinates
    atoms = [site.species_string for site in structure]
    coords = [site.coords for site in structure]

    # Add to the dictionary
    custom_data['atoms'].append(atoms)
    custom_data['coordinates'].append(coords)

    return custom_data

def read_cif_file(file_path):
    parser = CifParser(file_path)
    structure = parser.get_structures(primitive=False)[0]

    custom_data = {'atoms': [], 'coordinates': []}

    atoms = [site.species_string for site in structure.sites]
    coords = [site.coords for site in structure.sites]

    custom_data['atoms'].append(atoms)
    custom_data['coordinates'].append(coords)

    return custom_data

clf1 = UniMolRepr(data_type='mof')

feats = {}

base = "nanocrystals"
# base = "debug"
for folder in os.listdir(base):
    print(folder)
    files = sorted(os.listdir(os.path.join(base, folder)))
    for q in tqdm(files):
        name = q[:-4]
        if q.endswith(".cif"):
            cif_file_path = os.path.join(base, folder, q)
            custom_data_cif = read_cif_file(cif_file_path)
            # custom_data_xyz = read_xyz_file(cif_file_path[:-4]+".xyz")
            reprs1 = clf1.get_repr(data=custom_data_cif)
            # reprs1_xyz = clf1.get_repr(data=custom_data_xyz)
            feats[name] = reprs1['cls_repr'][0]


feats["PlaceHolder"] = np.zeros(feats[name].shape)
print(feats["PlaceHolder"].shape)

# with open("unioml_feats_20240309_1x_ours_1x_mof.pkl", "wb") as f:
with open("debug.pkl", "wb") as f:
    pkl.dump(feats, f)
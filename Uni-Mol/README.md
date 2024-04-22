Extract substance descriptors using Uni-Mol
==================================================

## Abstract

Uni-Mol is a universal 3D molecular pretraining framework that offers a significant expansion of representation capacity and application scope in drug design. 

Check this [subfolder](./unimol/) for more detalis.

Uni-Mol tools is a easy-use wrappers for property prediction,representation and downstreams with Uni-Mol. It includes the following tools:
* molecular property prediction with Uni-Mol.
* molecular representation with Uni-Mol.
* other downstreams with Uni-Mol.

Check this [subfolder](./unimol_tools/) for more detalis.

Documentation of Uni-Mol tools is available at https://unimol.readthedocs.io/en/latest/

## Installation

To install or run our codes, please `cd` to subfolders first:

- [Uni-Mol](./unimol/)
- [Uni-Mol Tools](./unimol_tools/)

## Extract descriptors from official pretrained models

First, please your ".cif" data into one folder "Uni-Mol/nanocrystals/quantums":

```
nanocrystals
├── quantums
    ├── xxx.cif
    ├── yyy.cif
    ├── zzz.cif
    ├── ...
    ├── xxx@zzz.cif
```

Then run:

```
python extract.py
```

You can modify the **"data_type"** in extract.py line 41:

```
clf1 = UniMolRepr(data_type='mof')
```

Five official options are available: **"molecule", "oled", "protein", "crystal", and "mof"**.

And modify the save path of the ".pkl" file in extract.py line 65/66.

## Train or finetune your own models

### Process dataset

Training or fine-tuning your own data, empirically speaking, leads to better performance.

First, to process the ".cif" files into the format by Uni-Mol for training, run:

```
python convert_data.py
```

This code processes the ".cif" files into ".lmdb" format. Please modify "save_path" in convert_data.py line 150 to decide where to save your processed data.

One example of the processed data should be like:

```
ours0422
├── dict.txt
├── train.lmdb
```

We do not split validation set as we want to train mdoel on all the ".cif" files.

### Finetune your model

First, run:

```
python convert_weights.py 
```

to generate checkpoints to be loaded as pretrained weights.

This code generates a new ".pt" file. We take **"mof"** model as example:

unimol_tools/unimol_tools
├── weights
    ├── mof_pre_no_h_CORE_MAP_20230505.pt
    ├── mof_pre.pt
    ├── ...

where "mof_pre.pt" are the generated weights to be loaded.

Then, modify the "weight_path" in "run.sh", and run:

```
bash run.sh
```

This starts the finetuning process. Disable "weight_path" to train from scratch.

After training, copy the trained ".pt" into "unimol_tools/unimol_tools/weights/ours.pt".

And copy the dict file "dict.txt" into "unimol_tools/unimol_tools/weights/ours.dict.txt".

## Extract descriptors from your models

You can modify the **"data_type"** into "ours" in extract.py line 41:

```
clf1 = UniMolRepr(data_type='ours')
```

Then run:

```
python extract.py
```
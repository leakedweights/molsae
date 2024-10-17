import os
import tarfile
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from lm.tokenizer import SmilesTokenizer, preprocess

tf.config.set_visible_devices([], 'GPU')
for logical_device in tf.config.list_logical_devices('TPU'):
    tf.config.set_logical_device_configuration(
        logical_device,
        [tf.config.LogicalDeviceConfiguration(), ]
    )

def save_tokenized_ds(project_dir):
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_10M_2D.tar.gz"
    dataset_path = "zinc15_10M_2D.tar.gz"
    file_path = "zinc15_10M_2D.csv"
    tokenized_mol_path = f"{project_dir}/zinc_tokenized.npy"
    tokenizer_path = f"{project_dir}/zinc_tokenizer.pkl"

    if not os.path.exists(tokenizer_path):
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        with open(dataset_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with tarfile.open(dataset_path, "r:gz") as tar:
            tar.extractall()

        print("Creating tokenizer...")
        tokenizer = SmilesTokenizer(csv_source=file_path,
                                    dest=tokenizer_path,
                                    template=preprocess)
    else:
        print("Loading tokenizer...")
        tokenizer = SmilesTokenizer()
        tokenizer.load(tokenizer_path)

    if not os.path.exists(tokenized_mol_path):
        print("Tokenizing molecules...")
        data = pd.read_csv(file_path)
        smiles_list = data["smiles"].tolist()

        tokenized_molecules = [tokenizer.encode(smiles, use_template=True) for smiles in tqdm(smiles_list)]
        tokenized_molecules = np.array(tokenized_molecules, dtype=object)

        np.save(tokenized_mol_path, tokenized_molecules)

def create_lm_dataset(fpath, train_bsize, pad_token_id, buffer_size=50_000, eval_size=10_000, eval_bsize=4096):
    data = np.load(fpath, allow_pickle=True)

    train_data = data[:-eval_size]
    val_data = data[-eval_size:]

    train_dataset = tf.data.Dataset.from_generator(
        lambda: iter(train_data),
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.padded_batch(
        train_bsize,
        padded_shapes=[None],
        padding_values=pad_token_id
    )
    train_dataset = train_dataset.repeat()

    def get_eval_ds():
        val_dataset = tf.data.Dataset.from_generator(
            lambda: iter(val_data),
            output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
        val_dataset = val_dataset.padded_batch(
            eval_bsize,
            padded_shapes=[None],
            padding_values=pad_token_id
        )
        return val_dataset.as_numpy_iterator()

    return train_dataset.as_numpy_iterator(), get_eval_ds
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

        tokenized_molecules = [tokenizer.encode(
            smiles, use_template=True) for smiles in tqdm(smiles_list)]
        tokenized_molecules = np.array(tokenized_molecules, dtype=object)

        np.save(tokenized_mol_path, tokenized_molecules)


def create_lm_dataset(fpath, train_bsize, pad_token_id, buffer_size=500_000, eval_size=10_000, eval_bsize=4096):
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
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

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
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        return val_dataset.as_numpy_iterator()

    return train_dataset.as_numpy_iterator(), get_eval_ds


def create_sae_mol_dataset(fpath, batch_size, num_examples, pad_token_id, sort_fn=None):
    data = np.load(fpath, allow_pickle=True)

    if sort_fn is not None:
        data = sorted(data, key=sort_fn)

    data = data[:num_examples]

    molecule_dataset = tf.data.Dataset.from_generator(
        lambda: iter(data),
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    molecule_dataset = molecule_dataset.padded_batch(
        batch_size,
        padded_shapes=[None],
        padding_values=pad_token_id
    )

    return molecule_dataset.as_numpy_iterator()


def create_activation_dataset(base_dir, include_layers, include_sites, d_model, batch_size, num_epochs):
    file_lists = []
    for layer_id in include_layers:
        for site in include_sites:
            data_dir = os.path.join(base_dir, f"block_{layer_id}", site)
            file_list = [os.path.join(data_dir, f)
                         for f in os.listdir(data_dir) if f.endswith('.npy')]
            file_list.sort(key=lambda x: int(
                os.path.splitext(os.path.basename(x))[0]))
            file_lists.append(file_list)

    num_files = len(file_lists[0])
    for fl in file_lists:
        assert len(
            fl) == num_files, "All file lists must have the same number of files"

    files_per_example = list(zip(*file_lists))

    dataset = tf.data.Dataset.from_tensor_slices(files_per_example)

    def load_npy_files(file_paths):
        activations = []
        for file_path in file_paths:
            file_path = file_path.numpy().decode('utf-8')
            npy = np.load(file_path)
            activations.append(npy.astype(np.float32))
        return activations

    def tf_load_npy_files(file_paths):
        npy_tensors = tf.py_function(func=load_npy_files, inp=[file_paths], Tout=[
                                     tf.float32]*len(file_paths))
        for tensor in npy_tensors:
            tensor.set_shape([None, d_model])
        return npy_tensors

    dataset = dataset.map(
        tf_load_npy_files, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.as_numpy_iterator()

    return dataset

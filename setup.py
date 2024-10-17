from setuptools import setup, find_packages


setup(
    name="mol_sae",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.34",
        "jaxlib>=0.4.34",
        "flax>=0.10.0",
        "optax>=0.1.9",
        "tensorflow>=2.17.0",
        "git+https://github.com/google/orbax/#subdirectory=checkpoint",
        "numpy",
        "scipy",
        "rdkit",
        "py3dmol",
        "wandb",
        "tqdm",
        "requests",
    ],
)
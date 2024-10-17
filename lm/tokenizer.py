import os
import re
import pickle
import pandas as pd
from tqdm import tqdm

def preprocess(sequence, extras):
    return f"{extras['start']}{sequence}{extras['end']}"

class SmilesTokenizer:
    def __init__(self, csv_source=None, num_reserved_tokens=0, dest=None, template=None):
        self.template = template
        self._create_special_tokens(num_reserved_tokens)
        self.mappings = {}
        self.itostr = {}
        self.strtoi = {}
        if csv_source is not None:
            self._build(csv_source)
        if dest is not None:
            self.save(dest)

    @property
    def unk_token_id(self):
        self._check_init()
        return self.strtoi[self.special_tokens["unknown"]]

    @property
    def bom_token_id(self):
        self._check_init()
        return self.strtoi[self.special_tokens["start"]]

    @property
    def eom_token_id(self):
        self._check_init()
        return self.strtoi[self.special_tokens["end"]]

    @property
    def pad_token_id(self):
        self._check_init()
        return self.strtoi[self.special_tokens["padding"]]

    @property
    def vocab_size(self):
        self._check_init()
        return len(self.mappings.values())

    def _create_special_tokens(self, num_reserved_tokens):
        unk_token = "<?>"
        pad_token = "<PAD>"
        bom_token = "<BOM>"
        eom_token = "<EOM>"

        reserved_tokens = [f"<RESERVED{i+1}>" for i in range(num_reserved_tokens)]

        self.special_tokens = {
            "padding": pad_token,
            "unknown": unk_token,
            "start": bom_token,
            "end": eom_token,
            **{f"reserved_{i+1}": tok for i, tok in enumerate(reserved_tokens)}
        }

    def _build(self, source):
        df = pd.read_csv(source)
        smiles = df["smiles"]

        special_token_values = list(self.special_tokens.values())

        print("Extracting chemical tokens...")

        chem_tokens = set().union(
            *[
                self._tokenize(self._strip_smiles(smiles_str, special_token_values), special_token_values)
                for smiles_str in tqdm(smiles)
            ]
        )

        all_tokens = special_token_values + sorted(list(chem_tokens))
        self.mappings = {i: tok for i, tok in enumerate(all_tokens)}
        self.itostr = self.mappings
        self.strtoi = {tok: i for i, tok in self.itostr.items()}

    def _check_init(self):
        if self.mappings == {}:
            raise Exception("Tokenizer uninitialized!")

    def encode(self, input_str, use_template=False):
        if use_template and self.template is not None:
            input_str = self.template(input_str, self.special_tokens)

        tokens = self._tokenize(input_str, list(self.special_tokens.values()))
        return [self.strtoi.get(tok, self.unk_token_id) for tok in tokens]

    def decode(self, input_tokens, as_str=False):
        decoded = [self.itostr.get(token, self.special_tokens["unknown"]) for token in input_tokens]
        if as_str:
            return "".join(decoded)
        return decoded

    def _strip_smiles(self, seq, extras):
        for extra in extras:
            seq = seq.replace(extra, "")
        return seq

    def _tokenize(self, seq, extras):
        extras_re = r"(" + "|".join(map(re.escape, extras)) + r")"

        split_extras = re.split(extras_re, seq)

        smiles_re = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>|\*|\$|\%\d{2}|\d)"

        split_smiles = [
            [part] if part in extras else re.findall(smiles_re, part)
            for part in split_extras if part
        ]

        result = [item for sublist in split_smiles for item in sublist]

        return result

    def save(self, path):
        data = {
            "special_tokens": self.special_tokens,
            "mappings": self.mappings,
            "itostr": self.itostr,
            "strtoi": self.strtoi,
            "template": self.template,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: \"{path}\"")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.special_tokens = data["special_tokens"]
        self.mappings = data["mappings"]
        self.itostr = data["itostr"]
        self.strtoi = data["strtoi"]
        self.template = data["template"]
        print(f"Tokenizer loaded from {path}")

    def __getitem__(self, token_id):
        return self.itostr.get(token_id, self.special_tokens["unknown"])
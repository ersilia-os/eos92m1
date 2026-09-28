"""Pharmacophore-conditioned sampling with the pretrained 72-bit TransPharmer model."""
from types import SimpleNamespace

import torch
from rdkit import Chem, RDLogger

from .model import GPT, sample
from .pharm import get_pharma_fp
from .tokens import tokenize, untokenize

RDLogger.DisableLog("rdApp.*")

# architecture of guacamol_pc_72bit.pt (configs/generate_pc.yaml upstream)
MODEL_CONFIG = dict(
    N_LAYERS=8,
    N_HEADS=8,
    N_EMBD=256,
    MAX_LEN=100,
    NUM_PROPS=72,
    EMBD_PDROP=0.1,
    RESID_PDROP=0.1,
    ATTN_PDROP=0.1,
    VOCAB_SIZE=94,
)
START_CONTEXT = "C"
TEMPERATURE = 0.7


def _canonical(mol):
    """Canonical SMILES without stereochemistry; the model vocabulary cannot emit stereo tokens."""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


class Generator:
    def __init__(self, checkpoint_path, temperature=TEMPERATURE):
        config = SimpleNamespace(DEVICE="cpu", MODEL=SimpleNamespace(**MODEL_CONFIG))
        self.temperature = temperature
        self.max_len = MODEL_CONFIG["MAX_LEN"]
        self.model = GPT(config)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def _sample_batch(self, prop, batch_size):
        context = torch.tensor(tokenize(START_CONTEXT), dtype=torch.long).repeat(batch_size, 1)
        prop = prop.repeat(batch_size, 1)
        tokens = sample(
            model=self.model,
            x=context,
            steps=self.max_len,
            temperature=self.temperature,
            sample_=True,
            top_k=None,
            prop=prop,
        )
        return [untokenize(t) for t in tokens.cpu().numpy().tolist()]

    def generate(self, smiles, n_samples, batch_size, max_raw):
        """Sample up to ``n_samples`` valid, unique molecules sharing the pharmacophore of ``smiles``.

        The reference compound itself is never returned. Sampling stops once ``n_samples``
        molecules are collected or ``max_raw`` raw SMILES have been drawn.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        prop = torch.tensor([get_pharma_fp(mol)], dtype=torch.int)
        seen = {_canonical(mol)}
        found = []
        drawn = 0
        while len(found) < n_samples and drawn < max_raw:
            for raw in self._sample_batch(prop, batch_size):
                gen = Chem.MolFromSmiles(raw) if raw else None
                if gen is None:
                    continue
                canon = _canonical(gen)
                if canon in seen:
                    continue
                seen.add(canon)
                found.append(Chem.MolToSmiles(gen))
                if len(found) == n_samples:
                    break
            drawn += batch_size
        return found

"""Pharmacophore-conditioned sampling with the pretrained 72-bit TransPharmer model."""
import sys
from types import SimpleNamespace

import torch
from output_checks import OutputFilter
from rdkit import Chem, RDLogger

from .model import GPT, sample
from .pharm import get_pharma_fp
from .tokens import PAD_TOKEN, STOI, tokenize, untokenize

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


class Generator:
    def __init__(self, checkpoint_path, temperature=TEMPERATURE):
        config = SimpleNamespace(DEVICE="cpu", MODEL=SimpleNamespace(**MODEL_CONFIG))
        self.temperature = temperature
        self.max_len = MODEL_CONFIG["MAX_LEN"]
        self.model = GPT(config)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def _sample_batch(self, prop, batch_size, use_cache):
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
            use_cache=use_cache,
            # padding doubles as end-of-molecule, so a row that emits it is done and can leave
            # the batch; untokenize() strips the padding that fills its remaining positions
            eos_id=STOI[PAD_TOKEN],
        )
        return [untokenize(t) for t in tokens.cpu().numpy().tolist()]

    def generate(self, smiles, n_samples, batch_size, max_raw, use_cache):
        """Sample up to ``n_samples`` valid, unique molecules for the pharmacophore of ``smiles``.

        Every raw sample is re-parsed and must be a single component with an InChIKey. The
        input is never returned: any output with the input's first InChIKey block is dropped,
        so stereoisomers of the input count as the input. Repeats are dropped on the first
        InChIKey block too, so stereoisomers of one output count as repeats. Sampling stops
        once ``n_samples`` molecules are collected or ``max_raw`` raw SMILES have been drawn.
        If sampling fails part-way, the molecules found so far are returned and the error
        goes to stderr.

        Raises
        ------
        ValueError
            If ``smiles`` is empty, unparsable, has several components or has no InChIKey.
        """
        checks = OutputFilter(smiles)
        if not checks.ok:
            raise ValueError(
                "input is empty, unparsable, multi-component or has no InChIKey"
            )
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        prop = torch.tensor([get_pharma_fp(mol)], dtype=torch.int)
        found = []
        drawn = 0
        try:
            while len(found) < n_samples and drawn < max_raw:
                for raw in self._sample_batch(prop, batch_size, use_cache=use_cache):
                    kept = checks.accept(raw)
                    if kept is None:
                        continue
                    found.append(kept)
                    if len(found) == n_samples:
                        break
                drawn += batch_size
        except Exception as exc:
            print(
                "[ERROR] sampling stopped for %r after %d raw samples: %s: %s"
                % (smiles, drawn, type(exc).__name__, exc),
                file=sys.stderr,
            )
        print("[INFO] %s drawn=%d %s" % (smiles, drawn, checks.summary()))
        return found

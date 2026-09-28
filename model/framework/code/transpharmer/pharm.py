# 72-bit 2D pharmacophore fingerprint used to condition TransPharmer.
# Adapted from https://github.com/iipharma/transpharmer-repo (MIT licence), utils/pharm/misc.py
import os
from functools import lru_cache

from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

FDEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BaseFeatures.fdef")


@lru_cache(maxsize=1)
def _sig_factory():
    """Two-point pharmacophores with two topological distance bins (0-3 and 3-8 bonds)."""
    fdef = ChemicalFeatures.BuildFeatureFactory(FDEF_PATH)
    factory = SigFactory(fdef, minPointCount=2, maxPointCount=2)
    factory.SetBins([(0, 3), (3, 8)])
    factory.Init()
    return factory


def get_pharma_fp(mol):
    """Return the 72-bit pharmacophore fingerprint of an RDKit molecule as a list of 0/1."""
    fp = Generate.Gen2DFingerprint(mol, _sig_factory(), dMat=None)
    return [int(b) for b in fp]

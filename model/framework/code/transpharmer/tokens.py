# SMILES vocabulary and (un)tokenisation of TransPharmer.
# Adapted from https://github.com/iipharma/transpharmer-repo (MIT licence), dataset.py
import re

PAD_TOKEN = "<"

TOKENS = [
    "#", "%10", "%11", "%12", "(", ")", "-", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    PAD_TOKEN, "=", "B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S",
    "[B-]", "[BH-]", "[BH2-]", "[BH3-]", "[B]", "[C+]", "[C-]", "[CH+]", "[CH-]", "[CH2+]",
    "[CH2]", "[CH]", "[F+]", "[H]", "[I+]", "[IH2]", "[IH]", "[N+]", "[N-]", "[NH+]", "[NH-]",
    "[NH2+]", "[NH3+]", "[N]", "[O+]", "[O-]", "[OH+]", "[O]", "[P+]", "[PH+]", "[PH2+]",
    "[PH]", "[S+]", "[S-]", "[SH+]", "[SH]", "[Se+]", "[SeH+]", "[SeH]", "[Se]", "[Si-]",
    "[SiH-]", "[SiH2]", "[SiH]", "[Si]", "[b-]", "[bH-]", "[c+]", "[c-]", "[cH+]", "[cH-]",
    "[n+]", "[n-]", "[nH+]", "[nH]", "[o+]", "[s+]", "[sH+]", "[se+]", "[se]",
    "b", "c", "n", "o", "p", "s",
]
VOCAB_SIZE = len(TOKENS)
STOI = {t: i for i, t in enumerate(TOKENS)}
ITOS = {i: t for i, t in enumerate(TOKENS)}

_PATTERN = (
    r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
_REGEX = re.compile(_PATTERN)


def tokenize(smiles):
    """Convert a SMILES string into a list of vocabulary indices."""
    return [STOI[s] for s in _REGEX.findall(smiles)]


def untokenize(tokens):
    """Convert vocabulary indices back into a SMILES string (padding is dropped)."""
    smiles = "".join(ITOS[int(t)] for t in tokens)
    return smiles.replace(PAD_TOKEN, "")

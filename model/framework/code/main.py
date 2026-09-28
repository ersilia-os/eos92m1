# imports
import os
import sys

from ersilia_pack_utils.core import read_smiles, write_out

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

from transpharmer import Generator

# pretrained 72-bit pharmacophore-conditioned checkpoint (fetched with eosvc)
CHECKPOINT_PATH = os.path.join(root, "..", "..", "checkpoints", "guacamol_pc_72bit.pt")
# The reference model has no KV cache, so each raw SMILES costs a full 100-step
# forward pass (~1s/molecule on CPU with default torch threading). Empirically,
# a single 200-molecule batch typically yields >100 valid, unique, non-reference
# molecules, so MAX_RAW allows up to 3 batches (~10-15 min worst case per input)
# before padding remaining slots with empty strings.
NUM_SAMPLES = 100  # molecules returned per input; must match run_columns.csv
BATCH_SIZE = 200  # raw SMILES drawn per sampling round
MAX_RAW = 600  # raw SMILES drawn per input before giving up on filling all slots

# load model once
generator = Generator(CHECKPOINT_PATH)


# my model
def my_model(smiles_list):
    outputs = []
    for smi in smiles_list:
        try:
            samples = generator.generate(smi, NUM_SAMPLES, BATCH_SIZE, MAX_RAW)
        except Exception:
            samples = []
        outputs.append(samples + [""] * (NUM_SAMPLES - len(samples)))
    return outputs


# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = my_model(smiles_list)

# check input and output have the same length
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

header = [f"smi_{str(i).zfill(2)}" for i in range(NUM_SAMPLES)]

# write output in a .csv file
write_out(outputs, header, output_file, str)

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

# pretrained 72-bit pharmacophore-conditioned checkpoint (26 MB, stored in this repository)
CHECKPOINT_PATH = os.path.join(root, "..", "..", "checkpoints", "guacamol_pc_72bit.pt")
# Each raw SMILES is sampled with a KV cache and a batch leaves as rows finish. MAX_RAW caps the
# raw SMILES drawn per input (3 batches); inputs that do not fill all slots are padded with empty
# strings. The number of molecules kept is reported per input in the run log.
NUM_SAMPLES = 100  # molecules returned per input; must match run_columns.csv
BATCH_SIZE = 200  # raw SMILES drawn per sampling round
MAX_RAW = 600  # raw SMILES drawn per input before giving up on filling all slots

# load model once
generator = Generator(CHECKPOINT_PATH)


# my model
def my_model(smiles_list, cache=True):
    outputs = []
    for smi in smiles_list:
        try:
            samples = generator.generate(
                smi, NUM_SAMPLES, BATCH_SIZE, MAX_RAW, use_cache=cache
            )
        except Exception as exc:
            print(
                "[ERROR] %r: %s: %s" % (smi, type(exc).__name__, exc), file=sys.stderr
            )
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
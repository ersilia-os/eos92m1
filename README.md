# TransPharmer Pharmacophore-Informed Molecule Generation

Generates new molecules that share the topological pharmacophore of an input reference compound while often changing its scaffold, useful for scaffold hopping. A GPT-style SMILES decoder prompted with a 72-bit pharmacophore fingerprint was trained on about 1.27 million GuacaMol (ChEMBL-derived) molecules. Designs seeded with a PLK1 inhibitor were synthesised, and three of four were sub-micromolar, the best at 5.1 nM. Sampling is stochastic, only about 18% of raw samples were valid and unique in that run, and outputs often keep moieties of the reference.

This model was incorporated on 2026-09-24.Last packaged on 2026-09-28.

## Information
### Identifiers
- **Ersilia Identifier:** `eos92m1`
- **Slug:** `transpharmer`

### Domain
- **Task:** `Sampling`
- **Subtask:** `Generation`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Compound generation`, `Chemical language model`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `100`
- **Output Consistency:** `Variable`
- **Interpretation:** Up to 100 generated molecules as SMILES that share the input compounds topological pharmacophore.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| smi_00 | string |  | Generated compound index 0 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_01 | string |  | Generated compound index 1 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_02 | string |  | Generated compound index 2 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_03 | string |  | Generated compound index 3 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_04 | string |  | Generated compound index 4 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_05 | string |  | Generated compound index 5 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_06 | string |  | Generated compound index 6 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_07 | string |  | Generated compound index 7 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_08 | string |  | Generated compound index 8 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |
| smi_09 | string |  | Generated compound index 9 sharing the input's 72-bit pharmacophore fingerprint (empty if fewer than 100 were found) |

_10 of 100 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos92m1](https://hub.docker.com/r/ersiliaos/eos92m1)
- **Docker Architecture:** `AMD64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos92m1.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos92m1.zip)

### Resource Consumption
- **Model Size (Mb):** `26`
- **Environment Size (Mb):** `1170`
- **Image Size (Mb):** `1235.49`

**Computational Performance (seconds):**
- 10 inputs: `740.84`
- 100 inputs: `-1`
- 10000 inputs: `-1`

### References
- **Source Code**: [https://github.com/iipharma/transpharmer-repo](https://github.com/iipharma/transpharmer-repo)
- **Publication**: [https://doi.org/10.1038/s41467-025-56349-0](https://doi.org/10.1038/s41467-025-56349-0)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2025`
- **Ersilia Contributor:** [arnaucoma24](https://github.com/arnaucoma24)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos92m1
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos92m1
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!

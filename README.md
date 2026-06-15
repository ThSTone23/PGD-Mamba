# PGD-Mamba

## Installation

Clone the repository:

```bash
git clone git@github.com:ThSTone23/PGD-Mamba.git
cd PGD-Mamba
```

Create the environment from the environment.yml:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate mamba_recon_env
```

Install causal convolution and mamba packages:

```bash
cd causal-conv1d
python setup.py install

cd ../mamba
python setup.py install
```

Install the CLIP dependency:

```bash
pip install open_clip_torch
```

## Dataset

Download the IXI and FastMRI datasets from the official websites:

- IXI: https://brain-development.org/ixi-dataset/
- FastMRI: https://fastmri.org/

Place the processed datasets in the datasets folder inside code:

```text
code/datasets/ixi/
code/datasets/fastmri/
```

## Run Commands

```bash
cd code
python train.py --exp pgd_mamba_fastmri_4x --dataset fastmri --model mamba_unrolled --patch_size 2 --batch_size 2 --gpu_id 0 --max_iterations 10000 --labeled_num 100 --acceleration 4 --use_fourier 1 --window_size 0 --opts MODEL.USE_CLIP_PRIOR True
```

## Citation

You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.

```bibtex
@InProceedings{
    author    = {},
    title     = {},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
```

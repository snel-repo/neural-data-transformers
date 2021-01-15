"""
posthoc script to create a directory for "best lfve"
"""

from typing import List, Union
from os import path
import json
import argparse
import ray, yaml, shutil
from ray import tune
import torch

from tune_models import tuneNDT

from defaults import DEFAULT_CONFIG_DIR
from src.config.default import flatten

PBT_HOME = path.expanduser('~/ray_results/ndt/gridsearch')
OVERWRITE = True
PBT_METRIC = 'smth_masked_loss'
BEST_MODEL_METRIC = 'best_masked_loss'
LOGGED_COLUMNS = ['smth_masked_loss', 'masked_loss', 'r2', 'unmasked_loss']

DEFAULT_HP_DICT = {
    'TRAIN.WEIGHT_DECAY': tune.loguniform(1e-8, 1e-3),
    'TRAIN.MASK_RATIO': tune.uniform(0.1, 0.4)
}

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config", "-e",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument('--eval-only', '-ev', dest='eval_only', action='store_true')
    parser.add_argument('--no-eval-only', '-nev', dest='eval_only', action='store_false')
    parser.set_defaults(eval_only=False)

    parser.add_argument(
        "--name", "-n",
        type=str,
        default="",
        help="defaults to exp filename"
    )

    parser.add_argument(
        "--gpus-per-worker", "-g",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--cpus-per-worker", "-c",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=-1,
        help="-1 indicates -- use max possible workers on machine (assuming 0.5 GPUs per trial)"
    )

    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=20,
        help="samples for random search"
    )

    parser.add_argument(
        "--seed", "-d",
        type=int,
        default=-1,
        help="seed for config"
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    launch_search(**vars(args))

def build_hp_dict(raw_json: dict):
    hp_dict = {}
    for key in raw_json:
        info: dict = raw_json[key]
        sample_fn = info.get("sample_fn", "uniform")
        assert hasattr(tune, sample_fn)
        if sample_fn == "choice":
            hp_dict[key] = tune.choice(info['opts'])
        else:
            assert "low" in info, "high" in info
            sample_fn = getattr(tune, sample_fn)
            hp_dict[key] = sample_fn(info['low'], info['high'])
    return hp_dict

def launch_search(exp_config: Union[List[str], str], name: str, workers: int, gpus_per_worker: float, cpus_per_worker: float, eval_only: bool, samples: int, seed: int) -> None:
    # ---------- PBT I/O CONFIGURATION ----------
    # the directory to save PBT runs (usually '~/ray_results')

    if len(path.split(exp_config)[0]) > 0:
        CFG_PATH = exp_config
    else:
        CFG_PATH = path.join(DEFAULT_CONFIG_DIR, exp_config)
    variant_name = path.split(CFG_PATH)[1].split('.')[0]
    if seed > 0:
        variant_name = f"{variant_name}-s{seed}"
    if name == "":
        name = variant_name
    pbt_dir = path.join(PBT_HOME, name)
    # the name of this PBT run (run will be stored at `pbt_dir`)

    # ---------------------------------------------
    # * No train step
    # load the results dataframe for this run
    df = tune.Analysis(
        pbt_dir
    ).dataframe()
    df = df[df.logdir.apply(lambda path: not 'best_model' in path)]

    lfves = []
    for logdir in df.logdir:
        ckpt = torch.load(path.join(logdir, f'ckpts/{variant_name}.lfve.pth'), map_location='cpu')
        lfves.append(ckpt['best_unmasked_val']['value'])
    df['best_unmasked_val'] = lfves
    best_model_logdir = df.loc[df['best_unmasked_val'].idxmin()].logdir
    best_model_dest = path.join(pbt_dir, 'best_model_unmasked')
    if path.exists(best_model_dest):
        shutil.rmtree(best_model_dest)
    shutil.copytree(best_model_logdir, best_model_dest)

if __name__ == "__main__":
    main()
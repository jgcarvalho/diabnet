import os
from typing import Dict, Any
from datetime import date

# import toml
from sys import argv
from torch.utils.data import random_split
from diabnet import data
from diabnet.train import train

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CATALOG = {
    "positive": os.path.join(
        DATA_DIR, "visits_sp_unique_train_positivo_1000_random_0.csv"
    ),
    "random": os.path.join(
        DATA_DIR, "visits_sp_unique_train_positivo_0_random_1000.csv"
    ),
    "negative": os.path.join(
        DATA_DIR, "visits_sp_unique_train_positivo_0_negative_1000.csv"
    ),
    "shuffled-snps": os.path.join(
        DATA_DIR, "visits_sp_unique_train_shuffled_snps_positivo_1000_random_0.csv"
    ),
    "shuffled-labels": os.path.join(
        DATA_DIR, "visits_sp_unique_train_shuffled_labels_positivo_1000_random_0.csv"
    ),
    "shuffled-ages": os.path.join(
        DATA_DIR, "visits_sp_unique_train_shuffled_ages_positivo_1000_random_0.csv"
    ),
    "shuffled-parents": os.path.join(
        DATA_DIR, "visits_sp_unique_train_shuffled_parents_positivo_1000_random_0.csv"
    ),
    "families": [
        os.path.join(
            DATA_DIR,
            f"visits_sp_unique_train_famid_{fam_id}_positivo_1000_random_0.csv",
        )
        for fam_id in [0, 10, 14, 1, 30, 32, 33, 3, 43, 7]
    ],
}


def net(
    fn_dataset: str,
    fn_out_prefix: str,
    fn_log: str,
    params: Dict[str, Dict[str, Any]],
    epochs: int,
    n_ensemble: int,
) -> None:
    with open(fn_log, "w") as logfile:
        logfile.write(f"PARAMETERS {params}\n")
        feat_names = data.get_feature_names(
            fn_dataset, use_sex=True, use_parents_diagnosis=True
        )

        dataset = data.DiabDataset(
            fn_dataset,
            feat_names,
            label_name="T2D",
            soft_label=True,
            soft_label_baseline=params["soft-label-baseline"],
            soft_label_topline=params["soft-label-topline"],
            soft_label_baseline_slope=params["soft-label-baseline-slope"],
        )

        # 10% from training_set to use as validation
        len_trainset = int(0.9 * len(dataset))

        for i in range(n_ensemble):
            print(f"Model {i:03}")
            logfile.write(f"Model {i:03}\n")
            trainset, valset = random_split(
                dataset, [len_trainset, len(dataset) - len_trainset]
            )

            fn_out = f"{fn_out_prefix}-{i:03}.pth"

            train(
                params,
                trainset,
                valset,
                epochs,
                fn_out,
                logfile,
                device="cuda",
                is_trial=False,
            )


def train_from(config: Dict[str, Dict[str, Any]]) -> None:
    print(f'Title {config["title"]}')
    d = date.today().isoformat()
    slot = {}

    fn = f'model-{slot}-{config["params"]["hidden-neurons"]}-{config["params"]["optimizer"]}-{config["params"]["lc-layer"]}-{d}'

    if config["datasets"]["positive"]:
        dataset = "data/visits_sp_unique_train_positivo_1000_random_0.csv"
        fn_out = "results/models/" + fn.format("positive")
        fn_log = "results/logs/" + fn.format("positive")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["random"]:
        dataset = "data/visits_sp_unique_train_positivo_0_random_1000.csv"
        fn_out = "results/models/" + fn.format("random")
        fn_log = "results/logs/" + fn.format("random")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["negative"]:
        dataset = "data/visits_sp_unique_train_positivo_0_negative_1000.csv"
        fn_out = "results/models/" + fn.format("negative")
        fn_log = "results/logs/" + fn.format("negative")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["shuffled-snps"]:
        dataset = "data/visits_sp_unique_train_shuffled_snps_positivo_1000_random_0.csv"
        fn_out = "results/models/" + fn.format("shuffled-snps")
        fn_log = "results/logs/" + fn.format("shuffled-snps")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["shuffled-labels"]:
        dataset = (
            "data/visits_sp_unique_train_shuffled_labels_positivo_1000_random_0.csv"
        )
        fn_out = "results/models/" + fn.format("shuffled-labels")
        fn_log = "results/logs/" + fn.format("shuffled-labels")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["shuffled-ages"]:
        dataset = "data/visits_sp_unique_train_shuffled_ages_positivo_1000_random_0.csv"
        fn_out = "results/models/" + fn.format("shuffled-ages")
        fn_log = "results/logs/" + fn.format("shuffled-ages")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["shuffled-parents"]:
        dataset = (
            "data/visits_sp_unique_train_shuffled_parents_positivo_1000_random_0.csv"
        )
        fn_out = "results/models/" + fn.format("shuffled-parents")
        fn_log = "results/logs/" + fn.format("shuffled-parents")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["families"]:

        for fam_id in [0, 10, 14, 1, 30, 32, 33, 3, 43, 7]:
            dataset = (
                f"data/visits_sp_unique_train_famid_{fam_id}_positivo_1000_random_0.csv"
            )
            fn_out = f'results/models/model-positive-famid-{fam_id}-{config["params"]["hidden-neurons"]}-{config["params"]["optimizer"]}-{config["params"]["lc-layer"]}-{d}'
            fn_log = f'results/logs/model-positive-famid-{fam_id}-{config["params"]["hidden-neurons"]}-{config["params"]["optimizer"]}-{config["params"]["lc-layer"]}-{d}'
            net(
                dataset,
                fn_out,
                fn_log,
                config["params"],
                config["run"]["epochs"],
                config["run"]["ensemble"],
            )


def main() -> None:
    """Argument parser for training DiabNet with a configuration file."""
    from diabnet import __name__, __version__
    import argparse
    import toml

    # Overrides method in HelpFormatter
    class CapitalisedHelpFormatter(argparse.HelpFormatter):
        def add_usage(self, usage, actions, groups, prefix=None):
            if prefix is None:
                prefix = "Usage: "
            return super(CapitalisedHelpFormatter, self).add_usage(
                usage, actions, groups, prefix
            )

    # argparse
    parser = argparse.ArgumentParser(
        prog="DiabNet",
        description="A Neural Network to predict type 2 diabetes (T2D).",
        formatter_class=CapitalisedHelpFormatter,
        add_help=True,
    )

    # Change parser titles
    parser._positionals.title = "Positional arguments"
    parser._optionals.title = "Optional arguments"

    # Positional arguments
    parser.add_argument(
        "config", help="Path to a configuration file.", default=None, type=str
    )

    # Optional arguments
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__name__} v{__version__}",
        help="Show DiabNet version and exit.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Train DiabNet from file
    train_from(toml.load(args.config))


if __name__ == "__main__":
    main()

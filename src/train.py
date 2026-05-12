import argparse
import os

import torch
from omegaconf import OmegaConf

from src.dataset.ukb_dataset import ImageFeatureDataset

from src.dataset.utils import compute_label_dims
from src.generative_model.stylegan import StyleGAN2Model
from src.generative_model.trainer import create_trainer
from src.utils.utils import get_labels, load_yaml_config, make_exp_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--train_config",
        type=str,
        help="name of yaml config file",
        default="configs/configs_train/test.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(config_filename=args.train_config)
    experiment_folder = make_exp_folder(config)
    config = OmegaConf.create(config)
    labels = get_labels(config)

    shared_kwargs = dict(
        image_dir=config.data.image_dir,
        feature_path=config.data.feature_path,
        index_col=config.data.index_col,
        target_size=config.data.image_size,
        column_names=config.data.column_names if config.data.column_names != "all" else "all",
        img_extension=config.data.img_extension,
        subfolder_search=config.data.subfolder_search,
        fullpath_in_index=config.data.fullpath_in_index,
        split_ratios=OmegaConf.to_container(config.data.split_ratios, resolve=True),
        split_seed=config.data.split_seed,
        stratify_on=config.data.stratify_on if config.data.stratify_on else None,
        n_quantiles=config.data.n_quantiles,
    )

    train_set = ImageFeatureDataset(**shared_kwargs, split="train")
    val_set   = ImageFeatureDataset(**shared_kwargs, split="val")

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        config.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=True,
    )

    if config.gpus == -1:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = len(config.gpus)

    lambda_gp = (
        0.0002
        * (config.data.image_size**2)
        / (config.data.batch_size * num_gpus)
    )

    cond_dims = compute_label_dims(train_set, config.data.conditional_labels)
    class_dims = compute_label_dims(train_set, config.data.classifier_labels)

    print("\nClassifier label dimensions:")
    for label, dim in zip(config.data.classifier_labels, class_dims):
        task = "regression" if dim == 1 else "classification"
        print(f"  {label}: dim={dim} -> {task}")


    cond_distributions = []
    if len(cond_dims) > 0:
        if all(dim > 1 for dim in cond_dims):
            for i, dim in enumerate(cond_dims):
                _, counts = torch.unique(train_set.features[:, i], return_counts=True)
                cond_distributions.append(
                    torch.distributions.Categorical(probs=counts.float() / counts.sum())
                )

        elif all(dim == 1 for dim in cond_dims):
            for i, dim in enumerate(cond_dims):
                values = train_set.features[:, i].float()
                mean = values.mean()
                std = values.std().clamp_min(1e-6)
                cond_distributions.append(torch.distributions.Normal(mean, std))
        else:
            cond_distributions = None
    else:
        cond_distributions = None


    trainer, checkpoint_callback = create_trainer(config, experiment_folder)
    model = StyleGAN2Model(
        config,
        experiment_folder,
        lambda_gp,
        cond_dims,
        class_dims,
        cond_distributions,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=config.resume,
    )

    with open(os.path.join(experiment_folder, "best_ckpt.txt"), "w") as text_file:
        text_file.write(checkpoint_callback.best_model_path)
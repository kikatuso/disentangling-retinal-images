import glob
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ImageFeatureDataset(Dataset):
    """Dataset for images paired with continuous metadata features.

    A drop-in replacement for EyePACS. Returns a dict with 'image' and 'labels'
    keys, where 'labels' is a normalized continuous feature vector.

    Splits are performed internally using a seeded stratified split on quantile-
    binned labels, so no external split files are needed.

    Attributes:
        image_dir: Root directory containing images.
        feature_path: Path to CSV file with continuous metadata features.
        index_col: Column in the CSV to use as the index (matched against image stems).
        split: One of {'train', 'val', 'test'}.
        split_ratios: Dict with keys 'train', 'val', 'test' that sum to 1.0.
            E.g. {'train': 0.7, 'val': 0.15, 'test': 0.15}
        split_seed: Random seed for reproducible splits.
        stratify_on: Column name to stratify splits on. The column will be binned
            into quantiles for stratification. If None, no stratification is used.
        n_quantiles: Number of quantile bins for stratification.
        img_extension: Image file extension to glob for.
        subfolder_search: If True, search recursively through subdirectories.
        fullpath_in_index: If True, match full image paths against CSV index
            instead of stems.
        target_size: If set, images are resized with aspect-ratio-preserving
            padding to this square size.
        transform: Optional transform applied to the image tensor after loading.
        column_names: List of feature column names to use, or 'all' for every column.
    """

    def __init__(
        self,
        image_dir: str,
        feature_path: str,
        index_col: str,
        split: str = "train",
        split_ratios: Optional[dict] = None,
        split_seed: int = 42,
        stratify_on: Optional[str] = None,
        n_quantiles: int = 4,
        img_extension: str = ".png",
        subfolder_search: bool = False,
        fullpath_in_index: bool = False,
        target_size: Optional[int] = None,
        transform=None,
        column_names: Union[List[str], str] = "all",
    ):
        super().__init__()
        self.target_size = target_size
        self.transform = transform

        assert split in ("train", "val", "test"), \
            f"split must be one of 'train', 'val', 'test', got '{split}'"

        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, \
            "split_ratios must sum to 1.0"

        # --- Gather image paths ---
        if subfolder_search:
            all_image_paths = sorted(
                glob.glob(f"{image_dir}/**/*{img_extension}", recursive=True)
            )
        else:
            all_image_paths = sorted(glob.glob(f"{image_dir}/*{img_extension}"))

        # --- Load and validate features ---
        features = pd.read_csv(feature_path, index_col=index_col)
        if column_names != "all":
            assert set(column_names).issubset(set(features.columns)), \
                "Some specified column names are not in the features CSV."
            # Keep stratify column even if not in column_names
            cols_to_load = list(column_names)
            if stratify_on is not None and stratify_on not in cols_to_load:
                cols_to_load.append(stratify_on)
            features = features[cols_to_load]
        features = features[~features.index.duplicated(keep="last")]

        # --- Align images and features ---
        if fullpath_in_index:
            common = sorted(set(all_image_paths) & set(features.index))
            features = features.loc[common]
        else:
            image_map = {Path(p).stem: p for p in all_image_paths}
            common = sorted(set(image_map) & set(features.index))
            all_image_paths = [image_map[k] for k in common]
            features = features.loc[common]

        # --- Stratified seeded split ---
        indices = np.arange(len(common))
        stratify_labels = None
        if stratify_on is not None:
            col = features[stratify_on].to_numpy(dtype=float)
            # Replace NaN/-1 with median before binning
            col = np.where((np.isnan(col)) | (col == -1), np.nanmedian(col), col)
            quantile_edges = np.nanpercentile(
                col, np.linspace(0, 100, n_quantiles + 1)
            )
            # Deduplicate edges (can happen with skewed distributions)
            quantile_edges = np.unique(quantile_edges)
            stratify_labels = np.digitize(col, quantile_edges[1:-1])

        val_rel_size = split_ratios["val"] / (split_ratios["val"] + split_ratios["test"])

        train_idx, valtest_idx = train_test_split(
            indices,
            test_size=(split_ratios["val"] + split_ratios["test"]),
            random_state=split_seed,
            stratify=stratify_labels,
        )
        val_idx, test_idx = train_test_split(
            valtest_idx,
            test_size=(1.0 - val_rel_size),
            random_state=split_seed,
            stratify=stratify_labels[valtest_idx] if stratify_labels is not None else None,
        )

        split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        chosen = split_indices[split]

        if fullpath_in_index:
            self.image_paths = [common[i] for i in chosen]
        else:
            self.image_paths = [all_image_paths[i] for i in chosen]
        features = features.iloc[chosen]

        # Drop stratify column if it wasn't in the original column_names request
        if (
            stratify_on is not None
            and column_names != "all"
            and stratify_on not in column_names
        ):
            features = features.drop(columns=[stratify_on])

        print(
            f"[{split}] {len(self.image_paths)} samples "
            f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
        )

        # --- Normalize features ---
        features = features.replace(-1, np.nan)
        features = np.log1p(features).clip(lower=0)
        self.feature_median = features.median()
        self.feature_iqr = features.quantile(0.75) - features.quantile(0.25)
        features = features.fillna(self.feature_median)
        features = (features - self.feature_median) / self.feature_iqr.replace(0, 1)

        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.feature_names = features.columns.tolist()
        self.feature_median = torch.tensor(
            self.feature_median.values, dtype=torch.float32
        )
        self.feature_iqr = torch.tensor(self.feature_iqr.values, dtype=torch.float32)

    @property
    def label_dim(self) -> int:
        return self.features.shape[1]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _open_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return (
            torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        )

    def _resize_with_padding(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = TF.resize(img, [new_h, new_w])
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top, left = pad_h // 2, pad_w // 2
        img = TF.pad(img, [left, top, pad_w - left, pad_h - top])
        return img

    def __getitem__(self, idx: int) -> dict:
        image = self._open_image(self.image_paths[idx])
        if self.target_size is not None:
            image = self._resize_with_padding(image)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "labels": self.features[idx],
        }
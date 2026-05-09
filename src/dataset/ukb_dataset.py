import glob
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def build_pca_stratify_labels(
    features: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    n_components: int = 3,
    n_quantiles: int = 4,
    seed: int = 42,
) -> Optional[np.ndarray]:
    if columns is None:
        df = features
    elif isinstance(columns, str):
        df = features[[columns]]
    else:
        df = features[columns]

    if df.shape[0] < 2 or df.shape[1] == 0:
        return None

    X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.where(X == -1, np.nan, X)

    for j in range(X.shape[1]):
        col = X[:, j]
        median = np.nanmedian(col) if np.any(~np.isnan(col)) else 0.0
        col[np.isnan(col)] = median
        X[:, j] = col

    X = StandardScaler().fit_transform(X)

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    if n_comp < 1:
        return None

    pcs = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    bin_columns = []
    for j in range(n_comp):
        raw = pcs[:, j]
        edges = np.unique(np.percentile(raw, np.linspace(0, 100, n_quantiles + 1)))
        if len(edges) <= 2:
            bins = np.zeros(len(raw), dtype=np.int64)
        else:
            bins = np.digitize(raw, edges[1:-1])
        bin_columns.append(bins)

    combined = np.zeros(len(df), dtype=np.int64)
    for power, bins in enumerate(bin_columns):
        combined += bins * (n_quantiles ** power)

    _, labels = np.unique(combined, return_inverse=True)

    if np.min(np.bincount(labels)) < 2:
        return None

    return labels


def _safe_split_indices(
    indices: np.ndarray,
    labels: Optional[np.ndarray],
    split_ratios: dict,
    seed: int,
):
    val_rel_size = split_ratios["val"] / (split_ratios["val"] + split_ratios["test"])

    train_labels = labels
    if train_labels is not None and np.min(np.bincount(train_labels)) < 2:
        train_labels = None

    train_idx, valtest_idx = train_test_split(
        indices,
        test_size=(split_ratios["val"] + split_ratios["test"]),
        random_state=seed,
        stratify=train_labels,
    )

    valtest_labels = None
    if labels is not None:
        valtest_labels = labels[valtest_idx]
        if np.min(np.bincount(valtest_labels)) < 2:
            valtest_labels = None

    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=(1.0 - val_rel_size),
        random_state=seed,
        stratify=valtest_labels,
    )

    return train_idx, val_idx, test_idx


class ImageFeatureDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        feature_path: str,
        index_col: str,
        split: str = "train",
        split_ratios: Optional[dict] = None,
        split_seed: int = 42,
        stratify_on: Optional[Union[str, List[str]]] = None,
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

        assert split in ("train", "val", "test"), (
            f"split must be one of 'train', 'val', 'test', got '{split}'"
        )

        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, (
            "split_ratios must sum to 1.0"
        )

        if isinstance(stratify_on, str):
            stratify_on = [stratify_on]

        if subfolder_search:
            all_image_paths = sorted(
                glob.glob(f"{image_dir}/**/*{img_extension}", recursive=True)
            )
        else:
            all_image_paths = sorted(glob.glob(f"{image_dir}/*{img_extension}"))

        features = pd.read_csv(feature_path, index_col=index_col)
        features = features[~features.index.duplicated(keep="last")]

        if column_names != "all":
            assert set(column_names).issubset(set(features.columns)), (
                "Some specified column names are not in the features CSV."
            )
            cols_to_load = list(column_names)
            if stratify_on is not None:
                for col in stratify_on:
                    if col not in cols_to_load:
                        cols_to_load.append(col)
            features = features[cols_to_load]

        if fullpath_in_index:
            common = sorted(set(all_image_paths) & set(features.index))
            features = features.loc[common]
        else:
            image_map = {Path(p).stem: p for p in all_image_paths}
            common = sorted(set(image_map) & set(features.index))
            all_image_paths = [image_map[k] for k in common]
            features = features.loc[common]


        indices = np.arange(len(common))

        stratify_labels = None
        if stratify_on is not None:
            stratify_labels = build_pca_stratify_labels(
                features=features,
                columns=stratify_on,
                n_components=3,
                n_quantiles=n_quantiles,
                seed=split_seed,
            )

        train_idx, val_idx, test_idx = _safe_split_indices(
            indices=indices,
            labels=stratify_labels,
            split_ratios=split_ratios,
            seed=split_seed,
        )

        split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        chosen = split_indices[split]

        if fullpath_in_index:
            self.image_paths = [common[i] for i in chosen]
        else:
            self.image_paths = [all_image_paths[i] for i in chosen]

        features = features.iloc[chosen]

        if stratify_on is not None and column_names != "all":
            extra = [c for c in stratify_on if c not in column_names]
            if extra:
                features = features.drop(columns=extra)

        print(
            f"[{split}] {len(self.image_paths)} samples "
            f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
        )

        features = features.replace(-1, np.nan)
        features = np.log1p(features).clip(lower=0)

        feature_median = features.median()
        feature_iqr = features.quantile(0.75) - features.quantile(0.25)

        features = features.fillna(feature_median)
        features = (features - feature_median) / feature_iqr.replace(0, 1)

        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.feature_names = features.columns.tolist()
        self.feature_median = torch.tensor(
            feature_median.values, dtype=torch.float32
        )
        self.feature_iqr = torch.tensor(feature_iqr.values, dtype=torch.float32)
        self._num_classes = {name: 1 for name in self.feature_names}

    @property
    def label_dim(self) -> int:
        return self.features.shape[1]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _open_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

    def _resize_with_padding(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = TF.resize(img, [new_h, new_w],antialias=True)
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top, left = pad_h // 2, pad_w // 2
        return TF.pad(img, [left, top, pad_w - left, pad_h - top])

    def __getitem__(self, idx: int) -> dict:
        image = self._open_image(self.image_paths[idx])
        if self.target_size is not None:
            image = self._resize_with_padding(image)
        if self.transform is not None:
            image = self.transform(image)
        image = image * 2 - 1  # [0,1] -> [-1,1] to match GAN expectation
        return {
            "image": image,
            "labels": self.features[idx],
        }
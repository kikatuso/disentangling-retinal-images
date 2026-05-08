import math
import os
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
from pytorch_lightning.utilities import rank_zero_only

from src.generative_model.augment import AugmentPipe
from src.generative_model.classifier import (Classifier, LinearClassifier,
                                             LinearRegressor, MLPRegressor)
from src.generative_model.discriminator import Discriminator
from src.generative_model.generator import Generator
from src.generative_model.loss import (PathLengthPenalty,
                                       compute_gradient_penalty,
                                       distance_correlation)
from src.utils.metrics import SimpleMetric

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class StyleGAN2Model(pl.LightningModule):
    """StyleGAN2 pytorch lightning module.

    Extends the StyleGAN architecture with GAN inversion and independent subspace
    learning. Subspace heads support both classification (c_shape > 1) and
    regression (c_shape == 1) targets, selected per-subspace automatically.

    Attributes:
        config: Configuration file.
        experiment_folder: Path to experiment folder.
        lambda_gp: Lambda R1 gradient penalty.
        cond_dims: Labels for conditional GAN training.
        class_dims: Dimension of subspace classification/regression problems.
            A value of 1 means a regression target; >1 means classification.
        cond_distributions: List of distributions of conditional labels.
    """

    def __init__(
        self,
        config: dict,
        experiment_folder: str,
        lambda_gp: float,
        cond_dims: Optional[List[int]] = [],
        class_dims: Optional[List[int]] = [],
        cond_distributions=None,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.experiment_folder = experiment_folder
        self.lambda_gp = lambda_gp

        self.cond_dims = cond_dims
        self.cond_dim = sum(self.cond_dims)
        self.class_dims = class_dims
        self.class_dim = sum(self.class_dims)

        self.cond_distributions = cond_distributions

        # Determine per-subspace task: regression (dim==1) or classification (dim>1)
        if self.class_dim > 0:
            self.subspace_is_regression = [d == 1 for d in self.class_dims]
            class_label_dims = list(zip(self.config.data.classifier_labels, self.class_dims))
        else:
            self.subspace_is_regression = []
            class_label_dims = 0

        print(
            f"dim of cond. labels: {self.cond_dim}, "
            f"dims subspace heads {class_label_dims}, "
            f"regression mask: {self.subspace_is_regression}"
        )

        self.G = Generator(
            config.latent_dim,
            self.cond_dim,
            config.latent_dim,
            config.classifier.subspace_dims,
            config.seperate_mapping_networks,
            config.num_mapping_layers,
            config.data.image_size,
            3,
            synthesis_layer=config.generator,
        )
        self.D = Discriminator(
            self.cond_dim,
            config.data.image_size,
            3,
            w_num_layers=config.num_mapping_layers,
            latent_dim=config.latent_dim,
        )

        # Core GAN metrics
        self.metric_D_real = SimpleMetric()
        self.metric_D_fake = SimpleMetric()
        self.metric_D = SimpleMetric()
        self.metric_rGP = SimpleMetric()
        self.metric_G = SimpleMetric()
        self.metric_rPLP = SimpleMetric()
        self.metric_aug_p = SimpleMetric()

        metric_E_w_fake = {}
        metric_E_feature_real = {}
        metric_overall_loss = {}

        if self.class_dim > 0:
            number_subspaces = len(class_dims)
            self.subspace_dims = config.classifier.subspace_dims
            metric_subspace_cs_loss = {}
            metric_subspace = {}
            metric_regressors = {}   # MSE per regression subspace
            metric_classifiers = {}  # accuracy/jaccard per classification subspace
            metric_dCor = {}
            Cs = {}

            for i in range(number_subspaces):
                if self.subspace_is_regression[i]:
                    # Regression head: maps subspace -> scalar
                    if config.classifier.linear:
                        Cs[str(i)] = LinearRegressor(w_shape=self.subspace_dims[i])
                    else:
                        Cs[str(i)] = MLPRegressor(
                            hidden_layers=config.classifier.hidden_layers or 1,
                            w_shape=self.subspace_dims[i],
                        )
                else:
                    # Classification head
                    if config.classifier.linear:
                        Cs[str(i)] = LinearClassifier(
                            w_shape=self.subspace_dims[i],
                            c_shape=self.class_dims[i],
                        )
                    else:
                        Cs[str(i)] = Classifier(
                            hidden_layers=config.classifier.hidden_layers,
                            w_shape=self.subspace_dims[i],
                            c_shape=self.class_dims[i],
                        )

        for state in ["trn", "val", "test"]:
            metric_E_w_fake[state] = SimpleMetric()
            metric_E_feature_real[state] = SimpleMetric()
            metric_overall_loss[state] = SimpleMetric()
            if self.class_dim > 0:
                metric_subspace_cs_loss[state] = SimpleMetric()
                metric_subspace[state] = SimpleMetric()
                metric_regressors[state] = torch.nn.ModuleDict({})
                metric_classifiers[state] = torch.nn.ModuleDict({})
                metric_dCor[state] = torch.nn.ModuleDict({})

                for i in range(number_subspaces):
                    if self.subspace_is_regression[i]:
                        # Track MSE for regression subspaces
                        metric_regressors[state][str(i)] = torchmetrics.MeanSquaredError()
                    else:
                        # Track accuracy + jaccard for classification subspaces
                        num_classes = self.class_dims[i]
                        metrics = torchmetrics.MetricCollection(
                            [
                                torchmetrics.classification.MulticlassAccuracy(
                                    num_classes, average="micro"
                                ),
                                torchmetrics.classification.MulticlassJaccardIndex(
                                    num_classes, average="macro"
                                ),
                            ]
                        )
                        metric_classifiers[state][str(i)] = metrics.clone(
                            prefix=f"{state}_c_{i+1}_"
                        )

                for i in range(number_subspaces + 1):
                    for j in range(i):
                        metric_dCor[state][f"{j}{i}"] = SimpleMetric()

        self.metric_E_w_fake = torch.nn.ModuleDict(metric_E_w_fake)
        self.metric_E_feature_real = torch.nn.ModuleDict(metric_E_feature_real)
        self.metric_overall_loss = torch.nn.ModuleDict(metric_overall_loss)

        if self.class_dim > 0:
            self.metric_subspace_cs_loss = torch.nn.ModuleDict(metric_subspace_cs_loss)
            self.metric_subspace = torch.nn.ModuleDict(metric_subspace)
            self.Cs = torch.nn.ModuleDict(Cs)
            self.metric_regressors = torch.nn.ModuleDict(metric_regressors)
            self.metric_classifiers = torch.nn.ModuleDict(metric_classifiers)
            self.metric_dCor = torch.nn.ModuleDict(metric_dCor)

            if config.buffer_size is not None:
                self.W_train = torch.randn(
                    size=(config.buffer_size, config.data.batch_size, config.latent_dim)
                )
                self.W_val = torch.randn(
                    size=(config.buffer_size, config.data.batch_size, config.latent_dim)
                )
                self.distance_correlation_weight = 1 + config.buffer_size
            else:
                self.distance_correlation_weight = 1

        self.augment_pipe = AugmentPipe(
            config.ada_start_p,
            config.ada_target,
            config.ada_interval,
            config.ada_fixed,
            config.data.batch_size,
        )
        self.grid_z = torch.randn(config.num_vis_images, self.config.latent_dim)
        if self.cond_dim > 0:
            self.grid_c = self.get_cond_labels(shape=config.num_vis_images)

        self.automatic_optimization = False
        self.path_length_penalty = PathLengthPenalty(0.01, 2)
        self.aug = False if config.ada_start_p < 0 else True

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        generator_parameters = list(self.G.parameters())
        discriminator_parameters = list(self.D.parameters())
        if self.class_dim > 0:
            for i in range(len(self.class_dims)):
                generator_parameters += list(self.Cs[str(i)].parameters())
                discriminator_parameters += list(self.Cs[str(i)].parameters())

        g_opt = torch.optim.Adam(
            generator_parameters, lr=self.config.lr_g, betas=(0.0, 0.99), eps=1e-8
        )
        d_opt = torch.optim.Adam(
            discriminator_parameters, lr=self.config.lr_d, betas=(0.0, 0.99), eps=1e-8
        )
        return g_opt, d_opt

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------

    def get_mapped_latent(self, z1, z2, c, style_mixing_prob):
        style_mixing = False
        wz1 = None
        if c is not None:
            wz1 = self.G.wz_mapping(z1)
            wc = self.G.wc_mapping(c)
            z1 = torch.cat([wz1, wc], dim=1)
        if torch.rand(()).item() < style_mixing_prob:
            style_mixing = True
            cross_over_point = int(torch.rand(()).item() * self.G.w_mapping.num_ws)
            if c is not None:
                wz2 = self.G.wz_mapping(z2)
                z2 = torch.cat([wz2, wc], dim=1)
            w1 = self.G.w_mapping(z1)[:, :cross_over_point, :]
            w2 = self.G.w_mapping(z2, skip_w_avg_update=True)[:, cross_over_point:, :]
            return torch.cat((w1, w2), dim=1), wz1, style_mixing
        else:
            return self.G.w_mapping(z1), wz1, style_mixing

    def get_cond_labels(self, shape=None):
        if self.cond_dim == 0:
            return None

        n = shape if shape is not None else self.config.data.batch_size
        parts = []

        for dist, dim in zip(self.cond_distributions, self.cond_dims):
            if dim == 1:
                parts.append(dist.sample((n,)).float().unsqueeze(1))
            else:
                y = dist.sample((n,))
                parts.append(self.onehot_labels(y, dim))

        return torch.cat(parts, dim=1).to(self.device)

    def onehot_labels(self, labels, num_classes):
        return torch.eye(num_classes)[[labels]].to(self.device)

    def get_random_latent(self):
        batch_size = self.config.data.batch_size
        z1 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        return z1, z2

    def get_latent(self):
        z1, z2 = self.get_random_latent()
        batch_cond_labels_fake = self.get_cond_labels()
        w_fake, wz_fake, style_mixing = self.get_mapped_latent(
            z1, z2, batch_cond_labels_fake, 0.5
        )
        return w_fake, wz_fake, style_mixing, batch_cond_labels_fake

    def forward(self):
        w_fake, _, _, _ = self.get_latent()
        return self.G.synthesis(w_fake)

    # ------------------------------------------------------------------
    # Training / validation / test steps  (unchanged from original except
    # calls to _shared_subspaces_eval_step which is updated below)
    # ------------------------------------------------------------------

    def format_cond_labels(self, labels):
        parts = []
        for i, dim in enumerate(self.cond_dims):
            col = labels[:, i]
            if dim == 1:
                parts.append(col.float().unsqueeze(1))
            else:
                parts.append(self.onehot_labels(col, dim))
        return torch.cat(parts, dim=1).to(self.device)


    def training_step(self, batch, batch_idx):
        overall_loss = 0.0
        batch_image_real = batch["image"]
        if self.cond_dim > 0:    
            batch_cond_labels_real = self.format_cond_labels(batch["labels"][:, :len(self.cond_dims)])
        else:
            batch_cond_labels_real = None

        g_opt, d_opt = self.optimizers()

        w_fake, wz_fake, style_mixing, batch_cond_labels_fake = self.get_latent()
        batch_image_fake = self.G.synthesis(w_fake)

        # 1. Discriminator
        d_logits_fake, w_fake_hat, _ = self.D(
            self.augment_pipe(batch_image_fake.detach()), batch_cond_labels_fake
        )
        d_logits_real, w_real_hat, d_features_real = self.D(
            self.augment_pipe(batch_image_real), batch_cond_labels_real
        )
        self.augment_pipe.accumulate_real_sign(d_logits_real.sign().detach())

        d_loss_fake = torch.nn.functional.softplus(d_logits_fake).mean()
        d_loss_real = torch.nn.functional.softplus(-d_logits_real).mean()
        d_loss = (d_loss_fake + d_loss_real) / 2.0

        # 1.1 Encoder loss
        if not style_mixing:
            if self.cond_dim > 0:
                enc_loss_fake = torch.nn.functional.mse_loss(wz_fake.detach(), w_fake_hat)
            else:
                enc_loss_fake = torch.nn.functional.mse_loss(
                    w_fake[:, 0, :].detach(), w_fake_hat
                )
            if self.config.cycle_consistency:
                batch_image_real_hat = self.G.synthesis(
                    w_real_hat.unsqueeze(1).repeat([1, self.G.num_ws, 1])
                )
                _, _, d_features_real_hat = self.D(
                    self.augment_pipe(batch_image_real_hat.detach()), batch_cond_labels_real
                )
                enc_loss_real = torch.nn.functional.mse_loss(d_features_real, d_features_real_hat)

            d_loss = (
                d_loss
                + self.config.lambda_enc_fake * enc_loss_fake
                + self.config.lambda_enc_real * enc_loss_real
            )

        # 1.2 Buffer
        if (self.class_dim > 0) and (self.config.buffer_size is not None):
            W_train = torch.cat(
                [
                    w_real_hat,
                    self.W_train.clone().view(-1, w_real_hat.shape[1]).to(self.device),
                ],
                dim=0,
            )
            self.W_train[batch_idx % self.config.buffer_size, :, :] = w_real_hat.detach()
        else:
            W_train = None

        # 1.3 Subspace losses
        if self.class_dim > 0:
            subspaces_loss = self._shared_subspaces_eval_step(
                batch, batch_idx, state="trn", w_real_hat=w_real_hat, W=W_train, log=True
            )
            d_loss = d_loss + subspaces_loss

        # 1.4 R1 regularization
        if (batch_idx + 1) % self.config.lazy_gradient_penalty_interval == 0:
            batch_image_real.requires_grad_(True)
            if self.aug:
                d_logits_real, _, _ = self.D(
                    self.augment_pipe(batch_image_real, disable_grid_sampling=True),
                    batch_cond_labels_real,
                )
            else:
                d_logits_real, _, _ = self.D(batch_image_real, batch_cond_labels_real)
            gp = compute_gradient_penalty(batch_image_real, d_logits_real)
            self.metric_rGP.update(gp.detach())
            gp_loss = self.lambda_gp / 2 * gp * self.config.lazy_gradient_penalty_interval
            d_loss = d_loss + gp_loss

        self.metric_D_real.update(d_loss_real.detach())
        self.metric_D_fake.update(d_loss_fake.detach())
        self.metric_D.update(d_loss.detach())
        overall_loss = overall_loss + d_loss.detach()

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # 2. Generator
        d_logits_fake, w_fake_hat, _ = self.D(
            self.augment_pipe(batch_image_fake), batch_cond_labels_fake
        )
        g_loss = torch.nn.functional.softplus(-d_logits_fake).mean()
        self.metric_G.update(g_loss.detach())

        if not style_mixing:
            if self.cond_dim > 0:
                enc_loss_fake = torch.nn.functional.mse_loss(wz_fake.detach(), w_fake_hat)
            else:
                enc_loss_fake = torch.nn.functional.mse_loss(
                    w_fake[:, 0, :].detach(), w_fake_hat
                )
            if self.config.cycle_consistency:
                _, w_real_hat, d_features_real = self.D(
                    self.augment_pipe(batch_image_real), batch_cond_labels_real
                )
                batch_image_real_hat = self.G.synthesis(
                    w_real_hat.detach().unsqueeze(1).repeat([1, self.G.num_ws, 1])
                )
                _, _, d_features_real_hat = self.D(
                    self.augment_pipe(batch_image_real_hat), batch_cond_labels_real
                )
                enc_loss_real = torch.nn.functional.mse_loss(d_features_real, d_features_real_hat)

            g_loss = (
                g_loss
                + self.config.lambda_enc_fake * enc_loss_fake
                + self.config.lambda_enc_real * enc_loss_real
            )
            self.metric_E_w_fake["trn"].update(enc_loss_fake.detach())
            self.metric_E_feature_real["trn"].update(enc_loss_real.detach())

        if self.class_dim > 0:
            subspaces_loss = self._shared_subspaces_eval_step(
                batch, batch_idx, state="trn", w_real_hat=w_real_hat.detach(), log=False
            )
            g_loss = g_loss + subspaces_loss

        if (batch_idx * (self.current_epoch + 1)) > self.config.lazy_path_penalty_after and (
            batch_idx + 1
        ) % self.config.lazy_path_penalty_interval == 0:
            plp = self.path_length_penalty(batch_image_fake, w_fake)
            if not torch.isnan(plp):
                plp_loss = (
                    self.config.lambda_plp * plp * self.config.lazy_path_penalty_interval
                )
                g_loss = g_loss + plp_loss
            self.metric_rPLP.update(plp.detach())

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        overall_loss = overall_loss + g_loss.detach()
        self.metric_overall_loss["trn"].update(overall_loss)
        self.execute_ada_heuristics()

    def validation_step(self, batch, batch_idx):
        val_overall_loss = 0.0
        encoder_loss = self._shared_encoder_eval_step(batch, batch_idx, state="val")
        val_overall_loss = val_overall_loss + encoder_loss
        if self.class_dim > 0:
            subspace_loss = self._shared_subspaces_eval_step(
                batch, batch_idx, state="val", log=True
            )
            val_overall_loss = val_overall_loss + subspace_loss
        self.metric_overall_loss["val"].update(val_overall_loss)

    def test_step(self, batch, batch_idx):
        test_overall_loss = 0.0
        encoder_loss = self._shared_encoder_eval_step(batch, batch_idx, state="test")
        test_overall_loss = test_overall_loss + encoder_loss
        if self.class_dim > 0:
            subspace_loss = self._shared_subspaces_eval_step(
                batch, batch_idx, state="test", log=True
            )
            test_overall_loss = test_overall_loss + subspace_loss
        self.metric_overall_loss["test"].update(test_overall_loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch_image_real = batch["image"] if isinstance(batch, dict) else batch
        _, w_real_hat, _ = self.D(batch_image_real, None)
        return w_real_hat

    # ------------------------------------------------------------------
    # Shared encoder eval step (unchanged)
    # ------------------------------------------------------------------

    def _shared_encoder_eval_step(self, batch, batch_idx, state: str = "val"):
        if self.cond_dim > 0:
            batch_labels_real = self.format_cond_labels(batch["labels"][:, : len(self.cond_dims)])

            w_fake, wz_fake, _, batch_labels_fake = self.get_latent()
        else:
            batch_labels_real = None
            batch_labels_fake = None
            w_fake, wz_fake, _, _ = self.get_latent()

        batch_image_fake = self.G.synthesis(w_fake)
        _, w_hat_fake, _ = self.D(batch_image_fake, batch_labels_fake)

        if self.cond_dim > 0:
            w_fake_loss = (wz_fake - w_hat_fake).square().mean()
        else:
            w_fake_loss = (w_fake[:, 0, :] - w_hat_fake).square().mean()

        batch_image_real = batch["image"]
        _, w_real_hat, d_feature_real = self.D(batch_image_real, batch_labels_real)
        batch_image_real_hat = self.G.wz_to_image(wz=w_real_hat, c=batch_labels_real)
        _, _, d_feature_real_hat = self.D(batch_image_real_hat, batch_labels_real)
        feature_real_error = torch.nn.functional.mse_loss(d_feature_real, d_feature_real_hat)

        self.metric_E_w_fake[state].update(w_fake_loss.detach())
        self.metric_E_feature_real[state].update(feature_real_error.detach())

        encoder_loss = (
            self.config.lambda_enc_fake * w_fake_loss.detach()
            + self.config.lambda_enc_real * feature_real_error.detach()
        )
        return encoder_loss

    # ------------------------------------------------------------------
    # Subspace eval step — updated for regression
    # ------------------------------------------------------------------

    def _shared_subspaces_eval_step(
        self,
        batch,
        batch_idx,
        w_real_hat=None,
        W=None,
        state: str = "trn",
        log: bool = False,
    ):
        """Optimize subspace heads (regression or classification) and minimise
        distance correlation between subspaces.

        For regression subspaces (class_dim == 1) MSE loss is used.
        For classification subspaces (class_dim > 1) cross-entropy is used.
        Labels are expected as a flat float tensor of shape (batch, total_labels)
        where each column corresponds to one subspace target (continuous for
        regression, integer-encoded for classification).
        """
        subspace_cs_loss = 0
        batch_image, batch_labels = batch["image"], batch["labels"]

        if w_real_hat is None:
            _, w_real_hat, _ = self.D(batch_image, None)
            if (state == "val") and (self.config.buffer_size is not None):
                W = torch.cat(
                    [
                        w_real_hat,
                        self.W_val.clone().view(-1, w_real_hat.shape[1]).to(self.device),
                    ],
                    dim=0,
                )
                self.W_val[batch_idx % self.config.buffer_size, :, :] = w_real_hat.detach()

        start = len(self.cond_dims)

        for i in range(len(self.class_dims)):
            # Extract the i-th subspace from the encoder output
            subspace = w_real_hat[
                :, sum(self.subspace_dims[:i]): sum(self.subspace_dims[: i + 1])
            ]
            y_hat = self.Cs[str(i)](subspace)  # (batch,) for regression; (batch, C) for clf

            if self.subspace_is_regression[i]:
                # Continuous target: shape (batch,) after squeeze
                target = batch_labels[:, start + i].float()
                c_loss = torch.nn.functional.mse_loss(y_hat, target)
                if log:
                    self.metric_regressors[state][str(i)].update(y_hat.detach(), target.detach())
            else:
                # Categorical target: shape (batch,) long
                target = batch_labels[:, start + i].long()
                c_loss = torch.nn.functional.cross_entropy(input=y_hat, target=target)
                if log:
                    self.metric_classifiers[state][str(i)].update(y_hat, target)

            subspace_cs_loss = subspace_cs_loss + c_loss

        subspace_cs_loss = subspace_cs_loss / len(self.class_dims)

        # Distance correlation between all pairs of subspaces
        dCor_measures = []
        free_subspace = self.config.latent_dim - sum(self.subspace_dims)
        all_subspace_dims = self.subspace_dims + [free_subspace]
        for i in range(len(self.class_dims) + 1):
            for j in range(i):
                s1, e1 = sum(all_subspace_dims[:j]), sum(all_subspace_dims[: j + 1])
                s2, e2 = sum(all_subspace_dims[:i]), sum(all_subspace_dims[: i + 1])
                W1 = W[:, s1:e1] if W is not None else w_real_hat[:, s1:e1]
                W2 = W[:, s2:e2] if W is not None else w_real_hat[:, s2:e2]
                dCor_measure = distance_correlation(W1, W2)
                dCor_measures.append(dCor_measure)
                if log:
                    self.metric_dCor[state][f"{j}{i}"].update(dCor_measure.detach())

        dCor = torch.mean(torch.stack(dCor_measures))
        if state != "test":
            dCor_loss = (
                self.config.lambda_distance_correlation
                * self.distance_correlation_weight
                * dCor
            )
        else:
            dCor_loss = self.config.lambda_distance_correlation * dCor

        subspaces_loss = self.config.lambda_subspace_cs * subspace_cs_loss + dCor_loss

        if log:
            self.metric_subspace_cs_loss[state].update(subspace_cs_loss.detach())
            self.metric_subspace[state].update(subspaces_loss.detach())

        return subspaces_loss

    # ------------------------------------------------------------------
    # ADA
    # ------------------------------------------------------------------

    def execute_ada_heuristics(self):
        if self.aug:
            if (self.global_step + 1) % self.config.ada_interval == 0:
                self.augment_pipe.heuristic_update()
            self.metric_aug_p.update(self.augment_pipe.p.item())

    # ------------------------------------------------------------------
    # Epoch-end logging
    # ------------------------------------------------------------------

    def on_train_epoch_end(self):
        metric_dict = {
            "D_fake": self.metric_D_fake.compute(),
            "D_real": self.metric_D_real.compute(),
            "D": self.metric_D.compute(),
            "G": self.metric_G.compute(),
            "rGP": self.metric_rGP.compute(),
            "rPLP": self.metric_rPLP.compute(),
            "trn_E_feature_real": self.metric_E_feature_real["trn"].compute(),
            "trn_E_w_fake": self.metric_E_w_fake["trn"].compute(),
            "trn_overall_loss": self.metric_overall_loss["trn"].compute(),
            "step": float(self.current_epoch),
        }
        if self.aug:
            metric_dict["aug_p"] = self.metric_aug_p.compute()
        if self.class_dim > 0:
            self._extend_metric_dict_on_state_epoch_end(metric_dict, state="trn")

        self.log_dict(metric_dict, prog_bar=False, logger=True,
                      on_step=False, on_epoch=True, sync_dist=True)

        self.metric_D_fake.reset(); self.metric_D_real.reset()
        self.metric_D.reset(); self.metric_G.reset()
        self.metric_rGP.reset(); self.metric_rPLP.reset()
        self.metric_E_feature_real["trn"].reset()
        self.metric_E_w_fake["trn"].reset()
        self.metric_overall_loss["trn"].reset()
        if self.aug:
            self.metric_aug_p.reset()
        if self.class_dim > 0:
            self._reset_subspace_metrics(state="trn")

    def on_validation_epoch_end(self):
        metric_dict = {
            "val_E_feature_real": self.metric_E_feature_real["val"].compute(),
            "val_E_w_fake": self.metric_E_w_fake["val"].compute(),
            "val_overall_loss": self.metric_overall_loss["val"].compute(),
            "step": float(self.current_epoch),
        }
        if self.class_dim > 0:
            self._extend_metric_dict_on_state_epoch_end(metric_dict, state="val")

        odir_samples = os.path.join(self.experiment_folder, "images/")
        Path(odir_samples).mkdir(exist_ok=True, parents=False)
        self._export_fake_images("", odir_samples)
        self._export_real_images(odir_samples)

        self.log_dict(metric_dict, prog_bar=False, logger=True,
                      on_step=False, on_epoch=True, sync_dist=True)

        self.metric_E_feature_real["val"].reset()
        self.metric_E_w_fake["val"].reset()
        self.metric_overall_loss["val"].reset()
        if self.class_dim > 0:
            self._reset_subspace_metrics(state="val")

    def on_test_epoch_end(self):
        metric_dict = {
            "test_E_feature_real": self.metric_E_feature_real["test"].compute(),
            "test_E_w_fake": self.metric_E_w_fake["test"].compute(),
            "test_overall_loss": self.metric_overall_loss["test"].compute(),
            "step": float(self.current_epoch),
        }
        if self.class_dim > 0:
            self._extend_metric_dict_on_state_epoch_end(metric_dict, state="test")

        self.log_dict(metric_dict, prog_bar=False, logger=True,
                      on_step=False, on_epoch=True, sync_dist=True)

        self.metric_E_feature_real["test"].reset()
        self.metric_E_w_fake["test"].reset()
        self.metric_overall_loss["test"].reset()
        if self.class_dim > 0:
            self._reset_subspace_metrics(state="test")

    def _extend_metric_dict_on_state_epoch_end(self, metric_dict, state="trn"):
        for i in range(len(self.class_dims)):
            if self.subspace_is_regression[i]:
                metric_dict[f"{state}_reg_{i+1}_mse"] = (
                    self.metric_regressors[state][str(i)].compute()
                )
            else:
                metric_dict.update(self.metric_classifiers[state][str(i)].compute())
        for i in range(len(self.class_dims) + 1):
            for j in range(i):
                metric_dict[f"{state}_dCor_w{j+1}_w{i+1}"] = (
                    self.metric_dCor[state][f"{j}{i}"].compute()
                )
        metric_dict[f"{state}_subspace_cs_loss"] = (
            self.metric_subspace_cs_loss[state].compute()
        )
        metric_dict[f"{state}_subspace_loss"] = self.metric_subspace[state].compute()

    def _reset_subspace_metrics(self, state="trn"):
        for i in range(len(self.class_dims)):
            if self.subspace_is_regression[i]:
                self.metric_regressors[state][str(i)].reset()
            else:
                self.metric_classifiers[state][str(i)].reset()
        for i in range(len(self.class_dims) + 1):
            for j in range(i):
                self.metric_dCor[state][f"{j}{i}"].reset()
        self.metric_subspace_cs_loss[state].reset()
        self.metric_subspace[state].reset()

    # ------------------------------------------------------------------
    # Image export helpers (unchanged)
    # ------------------------------------------------------------------

    @rank_zero_only
    def _export_fake_images(self, prefix, output_dir_vis):
        vis_generated_images = []
        if self.cond_dim > 0:
            labels = self.grid_c.to("cuda").split(self.config.data.batch_size)
        for iter_idx, latent in enumerate(self.grid_z.split(self.config.data.batch_size)):
            latent = latent.to(self.device)
            fake = (
                self.G(latent, labels[iter_idx], noise_mode="const").cpu()
                if self.cond_dim > 0
                else self.G(latent, None, noise_mode="const").cpu()
            )
            if iter_idx < self.config.num_vis_images // self.config.data.batch_size:
                vis_generated_images.append(fake)
        torch.cuda.empty_cache()
        vis_generated_images = torch.cat(vis_generated_images, dim=0)
        torchvision.utils.save_image(
            vis_generated_images,
            Path(output_dir_vis) / f"{prefix}{self.current_epoch}.png",
            nrow=int(math.sqrt(vis_generated_images.shape[0])),
            value_range=(-1, 1),
            normalize=True,
        )

    @rank_zero_only
    def _export_real_images(self, output_dir_vis):
        vis_reals = []
        vis_recons = []
        for iter_idx, batch in enumerate(self.trainer.val_dataloaders):
            batch_image_real = batch["image"]
            if iter_idx < self.config.num_vis_images // self.config.data.batch_size:
                _, w_real_hat, _ = self.D(batch_image_real.to(self.device), None)
                labels = None
                if self.cond_dim > 0:
                    labels = self.format_cond_labels(batch["labels"][:, : len(self.cond_dims)])
                recons = self.G.wz_to_image(wz=w_real_hat, c=labels)
                if self.current_epoch == (self.config.val_check_interval - 1):
                    vis_reals.append(batch_image_real)
                vis_recons.append(recons)
            if (iter_idx * self.config.data.batch_size) >= self.grid_z.shape[0]:
                break
            elif iter_idx >= self.config.num_vis_images // self.config.data.batch_size:
                break

        torch.cuda.empty_cache()
        if self.current_epoch == (self.config.val_check_interval - 1):
            vis_reals = torch.cat(vis_reals, dim=0)
            torchvision.utils.save_image(
                vis_reals,
                os.path.join(output_dir_vis, "reals.png"),
                nrow=int(math.sqrt(vis_reals.shape[0])),
                value_range=(-1, 1),
                normalize=True,
            )
        vis_recons = torch.cat(vis_recons, dim=0)
        torchvision.utils.save_image(
            vis_recons,
            os.path.join(output_dir_vis, f"recon_{self.current_epoch}.png"),
            nrow=int(math.sqrt(vis_recons.shape[0])),
            value_range=(-1, 1),
            normalize=True,
        )
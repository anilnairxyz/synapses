"""synapses — representation learning & mechanistic interpretability explorations."""

from synapses.classifier import SmallCNN, classify, load_classifier
from synapses.data import (
    CIFAR10_CLASSES,
    binarize,
    cifar10_eval_loaders,
    cifar10_ssl_loader,
    fixed_binarized_loader,
    mnist_loaders,
)
from synapses.device import device_report, get_device
from synapses.extract import class_means, encode_latents, encode_moments
from synapses.losses import VAELoss, vae_loss
from synapses.metrics import active_units, evaluate_vae, kl_per_dim, linear_probe
from synapses.run import RunLogger, cosine_warmup_lr, list_checkpoints, set_seed
from synapses.ssl_augment import TwoViewTransform, eval_transform, ssl_transform
from synapses.ssl_backbone import CifarResNet18
from synapses.ssl_eval import (
    embed_dataset,
    evaluate_representation,
    knn_accuracy,
    participation_ratio,
    project_2d,
)
from synapses.ssl_losses import nt_xent_loss, simsiam_loss
from synapses.vae import MLPVAE, load_vae
from synapses.viz import (
    bar_by_class,
    image_grid,
    kl_spectrum_bar,
    latent_grid_coords,
    latent_scatter,
    line_plot,
    manifold_grid,
    rate_distortion_plot,
)

__all__ = [
    "get_device",
    "device_report",
    "MLPVAE",
    "load_vae",
    "vae_loss",
    "VAELoss",
    "mnist_loaders",
    "binarize",
    "set_seed",
    "RunLogger",
    "cosine_warmup_lr",
    "list_checkpoints",
    "encode_latents",
    "encode_moments",
    "class_means",
    "fixed_binarized_loader",
    "active_units",
    "kl_per_dim",
    "linear_probe",
    "evaluate_vae",
    "latent_scatter",
    "latent_grid_coords",
    "manifold_grid",
    "image_grid",
    "bar_by_class",
    "line_plot",
    "kl_spectrum_bar",
    "rate_distortion_plot",
    "SmallCNN",
    "load_classifier",
    "classify",
    "CIFAR10_CLASSES",
    "cifar10_ssl_loader",
    "cifar10_eval_loaders",
    "CifarResNet18",
    "ssl_transform",
    "eval_transform",
    "TwoViewTransform",
    "embed_dataset",
    "knn_accuracy",
    "evaluate_representation",
    "participation_ratio",
    "project_2d",
    "nt_xent_loss",
    "simsiam_loss",
]

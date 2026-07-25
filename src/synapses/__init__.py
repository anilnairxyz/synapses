"""synapses — representation learning & mechanistic interpretability explorations."""

from synapses.classifier import SmallCNN, classify, load_classifier
from synapses.data import binarize, fixed_binarized_loader, mnist_loaders
from synapses.device import device_report, get_device
from synapses.extract import class_means, encode_latents, encode_moments
from synapses.losses import VAELoss, vae_loss
from synapses.metrics import active_units, kl_per_dim, linear_probe
from synapses.run import RunLogger, set_seed
from synapses.vae import MLPVAE, load_vae
from synapses.viz import (
    bar_by_class,
    image_grid,
    kl_spectrum_bar,
    latent_grid_coords,
    latent_scatter,
    line_plot,
    manifold_grid,
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
    "encode_latents",
    "encode_moments",
    "class_means",
    "fixed_binarized_loader",
    "active_units",
    "kl_per_dim",
    "linear_probe",
    "latent_scatter",
    "latent_grid_coords",
    "manifold_grid",
    "image_grid",
    "bar_by_class",
    "line_plot",
    "kl_spectrum_bar",
    "SmallCNN",
    "load_classifier",
    "classify",
]

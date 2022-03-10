from .SingleImagePipeline import SingleImagePipeline
import numpy as np
import time

class ImagesPipeline():
  def __init__(
    self,
    images,
    names,
    resolutions,
    add_circle_padding=True,
    noise_only=True,
    verbose=True):

    assert images.ndim == 3 or images.ndim == 2, f"Images dimension either (N, x, y) or (x, y), but given {images.shape}"
    assert isinstance(names, str) or names.ndim == 1, f"Names dimension either (N, name) or (name), but given {names.shape}"
    assert isinstance(resolutions, int) or resolutions.ndim == 1, f"Resolution dimension either (N, res) or res, but given {resolutions.shape}"
    assert isinstance(add_circle_padding, bool) or add_circle_padding.ndim == 1, f"Circle padding dimension either (N, bool) or bool, but given {add_circle_padding.shape}"
    assert isinstance(noise_only, bool) or noise_only.ndim == 1, f"Nois only dimension either (N, bool) or bool, but given {noise_only.shape}"

    self.verbose = verbose
    self.pipelines = []

    if images.ndim == 3:
      self.N_pipelines = images.shape[0]
      self.verbose = verbose
      self.pipelines = []

      names = self._check_dim_enlarge(names, self.N_pipelines, dtype=str)
      resolutions = self._check_dim_enlarge(resolutions, self.N_pipelines, dtype=int)
      add_circle_padding = self._check_dim_enlarge(add_circle_padding, self.N_pipelines, dtype=bool)
      noise_only = self._check_dim_enlarge(noise_only, self.N_pipelines, dtype=bool)

      for i in range(self.N_pipelines):
        self.pipelines.append(SingleImagePipeline(images[i], names[i], resolutions[i], add_circle_padding[i], noise_only[i], verbose ))

    else:
      self.N_pipelines = 1
      self.pipelines.append(SingleImagePipeline(images, names, resolutions, add_circle_padding, noise_only, verbose))

  def run(self, pipeline_ids = None, samples=1024, angle_generation='uniform', double_angles=True,snr=25, k=9, reconstruction_angles='linear_spaced', log_wandb=False):
    if pipeline_ids is None:
      pipeline_ids = list(range(self.N_pipelines))

    m = len(pipeline_ids)
    samples = self._check_dim_enlarge(samples, m, int)
    angle_generation = self._check_dim_enlarge(angle_generation, m, str)
    double_angles = self._check_dim_enlarge(double_angles, m, bool)
    snr = self._check_dim_enlarge(snr, m, int)
    k = self._check_dim_enlarge(k, m, int)
    reconstruction_angles = self._check_dim_enlarge(reconstruction_angles, m, str)

    for i in pipeline_ids:
      t = time.time()
      # def run(self, snr=25, k=9, reconstruction_angles='linear_spaced', log_wandb=False, reset=False):
      self.pipelines[i].run(samples[i], angle_generation[i], double_angles[i], snr[i], k[i], reconstruction_angles[i], log_wandb=log_wandb)

  def _check_dim_enlarge(self, value, n, dtype):
    if isinstance(value, dtype):
        return np.repeat(value, n)
    return value

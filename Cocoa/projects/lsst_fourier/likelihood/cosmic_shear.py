from cobaya.likelihoods.lsst_fourier._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_lsst_fourier_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  def initialize(self):
    super(cosmic_shear,self).initialize(probe="xi")
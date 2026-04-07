from cobaya.likelihoods.lsst_fourier._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_lsst_fourier_interface as ci
import numpy as np

class combo_2x2pt(_cosmolike_prototype_base):
  def initialize(self):
    super(combo_2x2pt,self).initialize(probe="2x2pt")
#!/usr/bin/env python3.11
"""
This is a sample code for the paper
IVYSO: Image Vectorization Yielding Shape Ordering
Author: Mark Ho Law and Sung Ha Kang

Primary contact: Mark Ho Law (hlaw@gatech.edu)
"""

from lib.quantizer import quantizer
from lib.helper import save_data, load_data
from lib.shape_layer import shape_layer, shape_layer_factory
from lib.shape_layer_ordering import shape_layer_order
from lib.shape_layer_phase_builder import phase_builder, phase_factory
from lib.shape_layer_euler import euler
from lib.shape_layer_bezier_fitter import Bezier_fitter, Bezier_Wrapper, svg_helper
from lib.shape_layer_PCI import Partial_Convex_Init
from lib.new_helper import inverse_color_transform, get_boundary
from lib.mult_seg import Mult_Seg
from lib.dilator import Dilator

from skimage.measure import find_contours
from time import time
import argparse

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

if __name__ == "__main__":
  """
  Parse arguments
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_path", type = str, help = "Input Image Path")
  parser.add_argument("--output_path", type = str, help = "Output folder")
  ## General Parameters
  parser.add_argument("--K", type = int, help = "number of colors in K mean quantization step")
  parser.add_argument("--noisy_threshold", type = int, help = "connected regions with pixels smaller than this number will be classified as noisy layers", default = 5)

  ## Parameters for grouping quantization
  #parser.add_argument("--seg_switch", type = bool, help = "do grouping quantization or not. Suggest True for real images, and False for cartoon images", default = True)
  parser.add_argument('--seg_on', dest='seg_switch', action='store_true')
  parser.add_argument('--seg_off', dest='seg_switch', action='store_false')
  parser.set_defaults(seg_switch=True)
  parser.add_argument("--mu", type = float, help = "the weight in Mulitphase Segmentation. The smaller, the more phases", default = 1.)
  parser.add_argument("--max_phases", type = int, help = "number of maximum phases allowed in Multiphase Segmentation", default = 5)
  parser.add_argument("--seg_max_iter", type = int, help = "maximum number of iterations in Multiplyase Segmentation", default = 2)

  ## Parameters for ordering
  parser.add_argument("--delta", type = float, help = "ordering threshold. must be between 0 and 1", default = 0.05)

  ## Parameters for inpainting
  parser.add_argument("--euler_max_iter", type = int, help = "maximum number of iterations for each shape layer's inpainting", default = 100)
  parser.add_argument("--eps", type = float, help = "the epsilon in the double-well potential model", default = 5.)
  parser.add_argument("--a", type = float, help = "weight on arc length term", default = .1)
  parser.add_argument("--b", type = float, help = "weight on curvature term", default = 1.)
  parser.add_argument("--r", type = float, help = "level set to extract", default = .1)

  ## Parameters for Bezier Curve fitting
  parser.add_argument("--b_eps", type = float, help = "fitting error tolerance", default = 1.)
  parser.add_argument("--b_radius", type = float, help = "threshold to be identified as local curvature extrema", default = .8)

  args = parser.parse_args()
  path = args.output_path


  ################ COLOR QUANTIZATION ############### 
  print("Quantizing")
  pic                 = cv.imread(args.image_path)
  num_color           = args.K
  Q                   = quantizer(pic, num_color, random_state=8964, denoise = False, use_mode_filter = False)
  pic, colors, layers = Q.quantize()
  Q.check_out(path, pic, colors, layers)

  ################# GROUPING QUANTIZATION ############### 
  pic = load_data(join(path, "pic"))
  layers = load_data(join(path, "layers"))
  colors = load_data(join(path, "colors"))
  if args.seg_switch:
    M = Mult_Seg(pic, mu = args.mu, max_phase = args.max_phases, max_iter = args.seg_max_iter)
    phase = M.new_segment(allow_max_phase=True, display = False)
    M.save(join(path, "coarse_seg_result"))
  else:
    phase = None

  ############### SHAPE LAYERS CONSTRUCTION ############### 
  pic = load_data(join(path, "pic"))
  layers = load_data(join(path, "layers"))
  colors = load_data(join(path, "colors"))
  if args.seg_switch:
    phase = load_data(join(path, "coarse_seg_result"))
  shape_layer_params = {"threshold": args.noisy_threshold} 

  S = shape_layer_factory(pic, layers, colors, shape_layer_params = shape_layer_params,
                          coarse_seg_result = phase, use_seg_result = args.seg_switch, auto_denoise = False) # auto denoise was True
  S.check_out(join(path, "shape_layers"), noisy_layer_path = join(path, "noisy_layers"))

  ############### LAYER ORDERING ############### 
  D = load_data(join(path, "shape_layers"))
  params = {"threshold": args.delta}
  S = shape_layer_order(D, params = params)
  print("Ordering...")
  S.order()
  S.check_out(join(path, "shape_layers"))

  ################ PHASE CONSTRUCTION ############### 
  print("Building phases...")
  D = load_data(join(path, "shape_layers"))
  params = {"r": 5, "thre": .5, "area_threshold": 30}
  PF = phase_factory(D, params = params)

  PF.build_all_phases()
  PF.save(join(path, "shape_layers"))

  ################## SHAPE CONVEXIFICATION ############### 
  #print("Convexifying")
  D = load_data(join(path, "shape_layers"))
  param = {"max_iter": args.euler_max_iter, "tol": 1e-9, "eps": args.eps, "a": args.a, "b": args.b, "area_threshold": 30., "mu": 1e0, "footprint": 4}
  bezier_prep_param = {"r": args.r}
  E       = euler(D, params = param, bezier_prep_param = bezier_prep_param)
  #ret     = E.parallel_solve()
  E.parallel_solve(solve_parallel = True, use_partial_convex_init=False)
  E.save(join(path, "shape_layers"))

  ################# MAKE SVG WITH NEW BEZIER FITTER ############### 
  D = load_data(join(path, "shape_layers"))
  noisy_D = load_data(join(path, "noisy_layers"))
  print("Fitting")
  for key in D["shape_layers"].keys():
    D["shape_layers"][key].reset_bezier_param(.1)
  BW = Bezier_Wrapper(D, k = 6, circle_radius = args.b_radius, eps = args.b_eps, use_weighted = False)
  BW.fit_all_colors()
  ret = BW.check_out()

  name = join(path, "ivyso_output.svg")
  pic = load_data(join(path, "pic"))
  h, w, _ = pic.shape
  S = svg_helper(name, ret, h, w)
  S.write_svg()


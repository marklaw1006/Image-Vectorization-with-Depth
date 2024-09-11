#!/usr/bin/env python3.9

# External library
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter, binary_fill_holes
import multiprocessing as mp
from itertools import groupby
from numpy_indexed import indices
from skimage.morphology import area_closing
from scipy.sparse import csr_matrix, isspmatrix, csc_matrix
from skimage import measure
from numpy_indexed import indices
from skimage.measure import label
from skimage.morphology import erosion, square, dilation, disk
import time
from tqdm import tqdm
import numba
from numba import njit, jit
from numba_progress import ProgressBar

# Internal library
from .helper import W, W_prime, W_prime_prime, gradient, laplacian, save_data, is_clockwise
from .new_helper import get_boundary, compute_curvature_on_closed_curve, smart_convert_sparse
from .shape_layer import wild_card_int, self_color

level_str              = "level"
counter_boundary_str   = "counter_orientation"
clockwise_boundary_str = "clock_orientation"
hole_filled_str        = "hole_filled"
image_str              = "image"
inpainted_im_str       = "inpainted_im"
color_str              = "color"
phase_str              = "phase"
supp_str               = "supp"
shape_layers_str       = "shape_layers"
grid_graph_str         = "grid_graph"
order_graph_str        = "order_graph"
mutual_bd_graph_str    = "mutual_bd_graph"

key_type = numba.types.UniTuple(numba.types.int64, 2)
val_type = numba.types.float64[:,:]

before, after = 0, 1

####################### AUXILIARY FUNCITONS ####################### 
#region def normalise(v: np.ndarray) -> np.ndarray:
def normalise(v: np.ndarray) -> np.ndarray:
  return v / np.linalg.norm(v)
#endregion

#region def turn_right(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> bool:
def turn_right(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> bool:
  """
  Auxiliary function for <_modified_graham_scan>. 
  Tell whether <x1> -> <x2> -> <x3> turns right.
  """
  v1, v2 = x2 - x1, x3 - x2
  return v2[0] * v1[1] - v2[1] * v1[0] > 0
#endregion

#region def split(bd: np.ndarray, sub_bd: np.ndarray, thre: float = 0.01) -> list[np.ndarray]:
def split(bd: np.ndarray, sub_bd: np.ndarray, thre: float = 0.01) -> list[np.ndarray]:
  assert(np.ndim(bd) == 2 and np.ndim(sub_bd) == 2)
  n    = len(bd)
  thre = int(n * thre)
  ind  = indices(bd, sub_bd, axis = 0)

  mask = np.zeros(n)
  mask[ind] = 1

  bd_flag = False
  if np.abs(np.max(ind) - (np.min(ind) + n)) < thre:
    mask[np.max(ind) : ] = 1
    mask[ : np.min(ind)] = 1
    bd_flag = True

  mask = area_closing(mask, area_threshold = thre)

  ## split into lists
  ret = [list(g) for k, g in groupby(range(n), lambda i: mask[i] == 1) if k]
  if bd_flag and len(ret) > 1:
    ret[-1] += ret[0]
    ret.pop(0)
  ret = [bd[r, :] for r in ret]
  return ret
#endregion

#region def get_query_pts(p: np.ndarray, h: int, w: int) -> list[np.ndarray]:
def get_query_pts(p: np.ndarray, h: int, w: int) -> list[np.ndarray]:
  p_up        = ((p[0] - 1) % h, p[1])
  p_down      = ((p[0] + 1) % h, p[1])
  p_left      = (p[0], (p[1] - 1) % w)
  p_right     = (p[0], (p[1] + 1) % w)
  p_top_left  = ((p[0] - 1) % h, (p[1] - 1) % w)
  p_top_right = ((p[0] - 1) % h, (p[1] + 1) % w)
  p_bot_left  = ((p[0] + 1) % h, (p[1] - 1) % w)
  p_bot_right = ((p[0] + 1) % h, (p[1] + 1) % w)

  return list(map(tuple, [p_up, p_down, p_left, p_right, p_top_left, p_top_right, p_bot_left, p_bot_right]))
#endregion

#region def insert_pt_to_bd(bd: np.ndarray, p: np.ndarray, pt_on_bd: list[np.ndarray], return_index: bool = False) -> np.ndarray or (np.ndarray, int):
def insert_pt_to_bd(bd: np.ndarray, p: np.ndarray, pt_on_bd: list[np.ndarray], return_index: bool = False) -> np.ndarray or (np.ndarray, int):
  """
  Insert <p> to <bd>.
  Insert <p> after the first point(regardless of the length of <pt_on_bd>)
  """
  pt_on_bd = np.array(pt_on_bd)
  ind      = np.min(indices(bd, pt_on_bd, axis = 0))
  if return_index:
    return np.insert(bd, ind, p, axis = 0), ind
  else:
    return np.insert(bd, ind, p, axis = 0)
#endregion

#region def new_W_prime_prime(u: np.ndarray) -> np.ndarray:
def new_W_prime_prime(u: np.ndarray) -> np.ndarray:
  if np.ndim(u) == 2:
    u = u.flatten()
  
  ret = np.ones(u.shape) * 2
  ret[np.logical_and(-.5 < u, u < .5)] = -2
  return ret
#endregion

#region def get_zero_contour(im: np.ndarray) -> list[np.ndarray]:
def get_zero_contour(im: np.ndarray) -> list[np.ndarray]:
  if np.min(im) < 0:
    offset = 0
  else:
    offset = 0.5
  contours = measure.find_contours(im, offset)
  return [c[::-1, :] for c in contours]
#endregion

#region def use_sparse(im: np.ndarray) -> bool:
def use_sparse(im: np.ndarray) -> bool:
  """
  Rough estimate of whether should use sparse representation
  """
  n = np.count_nonzero(im)
  return 3 * n < im.shape[0] * im.shape[1]
#endregion

#region def find_contig_segs(arr: list[bool] or np.ndarray) -> list:
def find_contig_segs(arr: list[bool] or np.ndarray, min_len: int = 3) -> list:
  """
  Given a boolean-like 1D array <arr>, find the contiguous 1's in it.
  """
  if isinstance(arr, list):
    arr = np.array(arr)
  arr = arr.astype(bool)

  i = 0
  ret = list()
  for k, g in groupby(arr):
    l = len(list(g))
    if k and l > min_len:
      ret.append((i, i + l)) # append index of closed-open interval
    i += l

  return ret
#endregion

#region def get_points_within_r(pt: np.ndarray, r: float) -> np.ndarray:
def get_points_within_r(pt: np.ndarray, r: float) -> np.ndarray:
  """
  Return points that are within <r> distance from <pt>
  """
  r_int = np.int32(np.ceil(r))
  candidate_pts = [[pt[0] + x, pt[1] + y] 
                    for x in range(-r_int, r_int) for y in range(-r_int, r_int)
                    if pt[0] + x >= 0 and pt[1] + y >= 0]
  candidate_pts = np.array(candidate_pts)
  candidate_pts = candidate_pts[np.linalg.norm(candidate_pts - pt, axis = 1) <= r, :]
  return candidate_pts
#endregion

#region def get_size(shape_layers: dict) -> (int, int):
def get_size(shape_layers: dict) -> (int, int):
  """
  return size of shape layers as a dictionary
  """
  S = next(iter(shape_layers.values()))
  return S.h, S.w
#endregion

#region def get_zero_contour(im: np.ndarray) -> list[np.ndarray]:
def get_zero_contour(im: np.ndarray) -> list[np.ndarray]:
  contours = measure.find_contours(im, 0)
  return contours[::-1, :]
#endregion

#region def get_contour(im: np.ndarray) -> list[np.ndarray]:
def get_contour(im: np.ndarray) -> list[np.ndarray]:
  contours = measure.find_contours(im, 0)
  return contours
#endregion

#region def compute_fft(x):
@jit(nopython = False)
def compute_fft(x):
  y = np.zeros_like(x, dtype = np.complex128)
  with numba.objmode(y = 'complex128[:, :]'):
    y = np.fft.fft2(x)
  return y
#endregion

#region def compute_ifft(x):
@jit(nopython = False)
def compute_ifft(x):
  y = np.zeros_like(x, dtype = np.complex128)
  with numba.objmode(y = 'complex128[:, :]'):
    y = np.fft.ifft2(x)
  return y
#endregion


class euler():
  #region def __init__(self, layer_info: dict, params = None, max_threads = mp.cpu_count() - 1):
  def __init__(self, D : dict, params = None, max_threads = mp.cpu_count() - 1, bezier_prep_param: dict = None):
    self.shape_layers    = D[shape_layers_str]
    self.graph           = D["grid_graph"]
    self.order_graph     = D["order_graph"]
    self.mutual_bd_graph = D["mutual_bd_graph"]
    self.bezier_prep_param = bezier_prep_param

    self.max_level = max([self.shape_layers[key].get_level() for key in self.shape_layers.keys()])

    # noisy pixels
    self.wild_node = list(filter(lambda n : self.graph.nodes[n][self_color][0] == wild_card_int, self.graph))
    
    self.h, self.w = get_size(self.shape_layers)
    self.max_threads = max_threads

    self.a = 1
    self.mu = 1e5 # fidelity. Constant. Large
    self.eta = 1e10

    self.max_iter = 100
    self.tol      = 1e-3

    self.area_threshold = .1 # object with area smaller than this threshold will be solved by modified Graham scan

    self.eps = 5  # epsilon for diffusion
    self.b = 1e2 # curvature weight
    self.c = 1.   # stability in solving Euler-Elastica.

    self.base_eps = 5.
    self.eps_multiplier = 2
    self.level_space = 3

    self.base_b = 1.
    self.b_multiplier = 1e1

    self.footprint = disk(2)

    self._load_param(params)

    self.coeff_matrix = self.get_coeff_matrix()

    ### put key in descending depth order
    #self.key_depth = list(filter(lambda key : not self.shape_layers[key].check_is_noise(), self.shape_layers.keys()))
    #self.depth = np.array([self.shape_layers[key].get_level() for key in self.key_depth])
    #ind = np.argsort(self.depth)[::-1]
    #self.depth = self.depth[ind]
    #self.key_depth = [self.key_depth[i] for i in ind]
    #self.cumsum_im = np.concatenate([self.shape_layers[key].get_dense_layer_im()[:,:,np.newaxis] for key in self.key_depth], axis = 2)
    #self.cumsum_im = np.cumsum(self.cumsum_im, axis = 2)

    #self.wild_image = np.zeros((self.h, self.w))
    #for coord in self.wild_node:
    #  self.wild_image[coord[0], coord[1]] = 1

    #fig, ax = plt.subplots(2, len(self.key_depth), sharex = True, sharey = True)
    #for k in range(len(self.key_depth)):
    #  ax[0, k].imshow(self.shape_layers[self.key_depth[k]].get_dense_layer_im())
    #  ax[1, k].imshow(self.cumsum_im[:,:,k])
    #plt.show()
    #breakpoint()


  #endregion

  ########################################## AUXILIARY FUNCITONS ########################################## 
  #region def _get_outer_boundary(self, im: np.ndarray) -> np.ndarray: 
  def _get_outer_boundary(self, im: np.ndarray) -> np.ndarray: 
    bd = get_boundary(im)
    bd = list(filter(is_clockwise, bd))
    assert(len(bd) == 1)
    return bd[0]
  #endregion

  ########################################## INITIALISATION ########################################## 
  #region def _load_param(self, params) -> None:
  def _load_param(self, params) -> None:
    if params is None:
      return
    assert(isinstance(params, dict))
    c_equal_b = False
    for key, val in params.items():
      if key == "max_iter":
        self.max_iter = val
      elif key == "a":
        self.a = val
      elif key == "l":
        self.l = val
      elif key == "tol":
        self.tol = val
      elif key == "mu":
        self.mu = val
      elif key == "eta":
        self.eta = val
      elif key == "area_threshold":
        self.area_threshold = val
      elif key == "eps":
        self.eps = val
      elif key == "b":
        self.b = val
      elif key == "c":
        self.c = val
      elif key == "c_equal_b":
        c_equal_b = val
      elif key == "footprint":
        if isinstance(val, np.ndarray):
          self.footprint = val
        elif isinstance(val, int):
          self.footprint = disk(val)
        else:
          raise TypeError("Unknown parameter for footprint")
      else:
        raise ValueError("Unknown Parameter")
    
    if c_equal_b:
      self.c = self.b
  #endregion

  #region def initialisation(self, i: tuple) -> (np.ndarray, np.ndarray):
  def initialisation(self, i: tuple) -> (np.ndarray, np.ndarray):
    u = self.components[i][image_str].astype(float)
    u[ u >= 1 ] = 1
    u[ u <= 0 ] = -1
    v = np.zeros((self.h, self.w))
    return u, v
  #endregion

  #region def get_coeff_matrix(self) -> np.ndarray:
  def get_coeff_matrix(self) -> np.ndarray:
    """
    Get coefficient matrix 2 - cos(z^{1}_{i}) - cos(z^{2}_{j})
    Output:
          2D numpy array
    """
    I = np.matmul(np.expand_dims(2 * np.pi * np.arange(self.h) / self.h, 1), np.ones((1, self.w)))
    J = np.matmul(np.ones((self.h, 1)), np.expand_dims(2 * np.pi * np.arange(self.w) / self.w, 0))
    return 2.0 - np.cos(I) - np.cos(J)
  #endregion

  #region def _clean_up_boundary(self, pts: np.ndarray) -> np.ndarray:
  def _clean_up_boundary(self, pts: np.ndarray) -> np.ndarray:
    """
    Often times there would be the case that the boundary can go like a -> b -> c -> b -> a -> d, which is so unfavorable to findint commond boundary.
    Clean up the boundary so that they will go like a -> b -> c -> d. 
    """
    # build a dictionary to keep track which points is visited
    nodes = list(map(tuple, pts))

    D = {n : False for n in nodes}

    # Pick a good point that has exactly one degree in and one degree out
    # That would be a point that appears only once in <pts>
    _, c = np.unique(pts, axis = 0, return_counts = True)
    ind = np.argwhere(c == 1)[0]

    ret = list()
    ptr = int(ind)
    while True:
      if not D[nodes[ptr]]:
        ret.append(pts[ptr])
        D[nodes[ptr]] = True
      ptr = int((ptr + 1) % len(pts))
      if ptr == ind:
        break

    assert(all(D.values())) # all nodes visited
    return np.array(ret)
  #endregion

  #region def _get_observable_boundary(self, i: tuple, reg_area: np.ndarray, fid_area: np.ndarray, 
  def _get_observable_boundary(self, i: tuple, reg_area: np.ndarray, fid_area: np.ndarray, 
                               reg_bd: np.ndarray, fid_bd: np.ndarray, thre = 1.5) -> np.ndarray:
    """
    Auxiliary function for <curve_interp_init>
    """
    ind    = np.where(np.min(cdist(fid_bd, reg_bd), axis = 1) <= thre)[0]
    curves = self.__get_oriented_curve(ind, fid_bd)
    return curves
  #endregion

  ########################################## NEW PARTIALLY CONVEX INITIALISATION ########################################## 
  #region def partial_convex_init(self, i: tuple) -> (np.ndarray, np.ndarray):
  def partial_convex_init(self, i: tuple) -> (np.ndarray, np.ndarray):
    return self.PCI.partial_convex_init(i)
  #endregion

  ########################################## GETTING AREA ########################################## 
  #region def get_reg_area(self, i: tuple) -> np.ndarray:
  def get_reg_area(self, i: tuple) -> np.ndarray:
    """
    Given a node, get the region that it can do curvature and arc length minimization.
    Basically the regions of any node at higher level.
    Return a binary matrix.
    """
    ################################### OLD ################################### 
    #thre = 1.5
    #level = self.shape_layers[i].get_level()
    #image = self.shape_layers[i].get_dense_layer_im()
    #shape_layers_on_top = filter(lambda key : self.shape_layers[key].get_level() >= level 
    #                              and not self.shape_layers[key].check_is_noise(), self.shape_layers.keys()) 

    #for key in shape_layers_on_top:
    #  np.maximum(image, self.shape_layers[key].get_dense_layer_im(), out = image)
    
    #image_coord = np.column_stack(np.where(image))
    #wild_coord  = np.array(list(map(list, self.wild_node)))
    #try:
    #  ind = np.where(np.min(cdist(wild_coord, image_coord), axis = 1) <= thre)[0]
    #  for j in ind:
    #    image[self.wild_node[j]] = 1
    #except ValueError:
    #  pass

    ################################### NEW ################################### 
    #ind = self.key_depth.index(i)
    #image = self.cumsum_im[:,:,ind]

    ##reg = dilation(image, footprint = [(np.ones((3, 1)), 1), (np.ones((1, 3)), 1)])
    #reg = dilation(image, footprint = np.ones((2,2)))
    #np.multiply(reg, self.wild_image, out = reg)

    #np.maximum(image, reg, out = reg)

    #test_im = self.shape_layers[i].get_dense_layer_im()
    #reg = label(reg)
    #ind = np.max(np.multiply(test_im, reg))
    #return reg == ind

    return self.shape_layers[i].get_reg_area()

  #endregion

  #region def _remove_disconnected_region(self, i: tuple, im: np.ndarray) -> np.ndarray:
  def _remove_disconnected_region(self, i: tuple, im: np.ndarray) -> np.ndarray:
    """
    When computing regularization area, there are a lot of disconnected regions.
    Remove them
    """
    ori_im = self.shape_layers[i].get_dense_layer_im()

    im = label(im)
    ind = np.max(np.multiply(ori_im, im))
    im[im != ind] = 0
    im[im != 0] = 1
    return im
  #endregion
  
  #region def get_fid_area(self, i: tuple) -> np.ndarray:
  def get_fid_area(self, i: tuple) -> np.ndarray:
    """
    Return the fidelity area of i. Make noisy pixels in/around the shape to be included in fid_area.
    """
    fid_area = self.shape_layers[i].get_dense_layer_im()
    return fid_area.astype(float)
  #endregion
  
  #region def get_weight(self, i: tuple, spacing = 3, sigma = 5, p = 10) -> np.ndarray:
  def get_weight(self, i: tuple, spacing = 3, sigma = 5, p = 10) -> np.ndarray:
    """
    This function computes the curvature term's weighting on different pixel.
    Higer curvature means higher weight.
    Output:
          ret: 2D numpy array.
    """
    fid_area = self.components[i][image_str]
    bd       = self._get_outer_boundary(fid_area)
    kappa    = np.abs(compute_curvature_on_closed_curve(bd, spacing = spacing))
    kappa    = np.power(np.maximum(kappa, 1), p)
    ret      = np.ones((self.h, self.w))
    for ptr, p in enumerate(bd):
      ret[p[0], p[1]] = kappa[ptr]
    
    ret = gaussian_filter(ret, sigma)
    return ret
  #endregion

  #region def showcase(self, area_threshold: float = 0) -> None:
  def showcase(self, area_threshold: float = 0) -> None:
    candidate_keys = filter(lambda key : np.sum(self.shape_layers[key].get_dense_layer_im()) >= area_threshold, self.shape_layers.keys())
    for key in candidate_keys:
      print(key)
      im           = self.shape_layers[key].get_dense_layer_im()
      reg_area     = self.get_reg_area(key)
      euler_output = self.shape_layers[key].get_euler_output()

      names = ["im", "reg_area", "euler_output"]
      images = [im, reg_area, euler_output]

      total_num = sum([i is not None for i in images])

      fig, ax = plt.subplots(1, total_num, sharex = True, sharey = True)

      for ptr in range(total_num):
        ax[ptr].imshow(images[ptr])
        ax[ptr].set_title(names[ptr])
      plt.show()
  #endregion

  ########################################## SOLVING ########################################## 
  #region def parallel_euler(self) -> list:
  def parallel_euler(self) -> list:
    args = sorted([(k, self.components[k][level_str]) for k in self.components.keys()], 
                  key = lambda x : x[1], reverse = True)
    assert(len(args) == len(self.b_seq))

    args = [a[0] for a in args]
    args = list(zip(args, self.b_seq, self.c_seq, self.eps_seq))

    with mp.Pool(processes = self.max_threads) as p:
      ret = p.map(self.euler_solve, args)

    ret = list(ret)

    #fig, ax = plt.subplots(len(ret), 2, sharex = True, sharey = True, figsize = (10, 12))
    fig, ax = plt.subplots(len(ret), 3, figsize = (10, 12))
    for ptr, r in enumerate(ret):
      ax[ptr, 0].imshow(r[1])
      ax[ptr, 0].set_title("image" + str(r[0]))

      im = r[1]
      im[im <= 0] = 0
      im = np.abs(im - self.components[r[0]][image_str])

      ax[ptr, 1].imshow(im)
      ax[ptr, 1].set_title("abs difference")
      ax[ptr, 2].plot(r[2])
    plt.show()
    breakpoint()

    return ret
  #endregion

  #region def euler_solve(self, pack: tuple, debug_plot: bool)-> (tuple, np.ndarray, list):
  def euler_solve(self, pack: tuple, debug_plot: bool)-> (tuple, np.ndarray, list):
    """
    <i> is a key in <self.components>
    """
    #i, b, c, eps = pack[0], pack[1], pack[2], pack[3]
    i, b, c, eps, init = pack[0], self.b, self.c, self.eps, pack[4]

    ## new phase builder
    phase, supp = self.shape_layers[i].get_phase(), self.shape_layers[i].get_supp()
    if isspmatrix(phase):
      phase = phase.toarray()

    if isspmatrix(supp):
      supp = supp.toarray()
    
    reg_area = self.get_reg_area(i)
    fid_area = self.get_fid_area(i)
    reg_area[reg_area <= 0] = 0
    fid_area[fid_area <= 0] = 0

    ## dilate regularization area
    #dilation(reg_area, self.footprint, out = reg_area)
    #reg_area = self.reg_area_adaptive_dilation(i)

    ## Instead of Omega\Disks, try R\Disks
    #old_supp = supp.copy()
    np.multiply(supp, reg_area, out = supp)

    #newly added
    np.multiply(supp, phase, out = phase)
    ## coefficient matrix to solve for v, since different layers may use different b
    coeff_matrix_v = 4 * b * self.coeff_matrix + self.a + c
    coeff_matrix_u = 2 * np.power(eps, 2) * self.coeff_matrix + c

    ## Initialisation
    #u = phase.copy()

    #u = -self.shape_layers[i].get_partial_convex_init()
    u = -self.shape_layers[i].get_dense_layer_im()

    v = np.zeros_like(u)

    reg_area, fid_area, u = list(map(lambda x : x.astype(float), [reg_area, fid_area, u]))
    num_iter = 0
    energy = list()

    idx_reg = 1 - reg_area > 0
    idx_fid = fid_area > 0
    while num_iter < self.max_iter:
      #################### Solve v #################### 
      v = self.solve_v_phase(u, v, coeff_matrix_v, b, c, eps, phase, supp)
      v[v <= -1] = -1 
      v[v >  1] = 1

      #################### Solve u #################### 
      u = self.solve_u(u, v, coeff_matrix_u, c, eps)
      u[u <= -1] = -1 
      u[u >  1] = 1

      ## hard constraints
      #u[idx_reg] = 0
      u[idx_reg] = 1
      u[idx_fid] = -1 # 0
      
      num_iter += 1

      #energy.append(self.calculate_energy(b, v, reg_area, fid_area, eps))

    #u = self.new_component_fusion(u, fid_area, phase, supp, init, reg_area)
    u = self.new_component_fusion(u, fid_area, phase, supp, fid_area, reg_area)

    #u = self.new_component_fusion(u, fid_area, phase, supp, fid_area)
    #np.multiply(reg_area, u, out = u)

    #u = self.clean_up_cmps(u, idx_fid)

    #_, ax = plt.subplots(1, 3, sharex = True, sharey = True)
    #ax[0].imshow(u)
    #ax[1].imshow(reg_area)
    #ax[2].imshow(fid_area)
    #plt.show()
    #breakpoint()


    #np.maximum(u, init, out = u)
    np.maximum(u, fid_area, out = u)

    if debug_plot:
      fig, ax = plt.subplots(1, 6, sharex = True, sharey = True)
      ax[0].imshow(phase)
      ax[0].set_title("phase")
      ax[1].imshow(supp)
      ax[1].set_title("supp")
      ax[2].imshow(reg_area)
      ax[2].set_title("reg_area")
      ax[3].imshow(fid_area)
      ax[3].set_title("fid_area")
      ax[4].imshow(u)
      ax[4].set_title("result")
      ax[5].imshow(init)
      ax[5].set_title("init")
      plt.show()

    u[u <= 0] = 0
    return (i, u, energy)
  #endregion

  #region def solve_v_phase(self, u: np.ndarray, v: np.ndarray, reg_area: np.ndarray,  
  def solve_v_phase(self, u: np.ndarray, v: np.ndarray, coeff_matrix_v: np.ndarray, 
              b: float, c: float, eps: float, phase: np.ndarray, supp: np.ndarray) -> np.ndarray:
    RHS = 2 * self.mu * np.multiply(supp, u - phase)
    RHS -= (b / np.power(eps, 2)) * np.multiply(v, W_prime_prime(u))
    RHS += c * v
    RHS  = np.fft.fft2(RHS)

    v = np.real(np.fft.ifft2(np.divide(RHS, coeff_matrix_v)))
    return v
  #endregion

  #region def solve_u(self, u: np.ndarray, v: np.ndarray, coeff_matrix_u: np.ndarray, c: float, eps: float) -> np.ndarray:
  def solve_u(self, u: np.ndarray, v: np.ndarray, coeff_matrix_u: np.ndarray, c: float, eps: float) -> np.ndarray:
    ## extra conditioning.
    coe_u     = 2 * np.power(u, 2) # 2u^2
    max_coe_u = np.max(coe_u) 
    uRHS      = np.multiply(max_coe_u - coe_u + 2, u) - eps * v + c * u
    uRHS      = np.fft.fft2(uRHS)
    u         = np.real(np.fft.ifft2(np.divide(uRHS, coeff_matrix_u + max_coe_u)))

    ## solve according to lit
    #u = np.real(np.fft.ifft2(np.fft.fft2((W_prime(u) / (2 * eps) + v) / (4 * eps)) / (-2 * self.coeff_matrix + c) ))
    return u
  #endregion

  #region def calculate_energy(self, b: float, u: np.ndarray, reg_area: np.ndarray, fid_area: np.ndarray, eps: float) -> float:
  def calculate_energy(self, b: float, u: np.ndarray, reg_area: np.ndarray, fid_area: np.ndarray, eps: float) -> float:
    grad_u      = gradient(u)
    L           = laplacian(u)
    norm_grad_u = np.sum(np.power(grad_u[0], 2) + np.power(grad_u[1], 2))

    ret  = self.a * (norm_grad_u * eps * .5 + np.sum(W(u)) / (2 * eps)) 
    ret += (b / eps) * np.sum(np.power(L * eps  - W_prime(u) / (2 * eps) , 2))
    ret += self.mu * np.sum(np.multiply(fid_area, np.power(u - 1, 2)))
    ret += self.mu * np.sum(np.multiply(1 - reg_area, np.power(u + 1, 2)))
    return ret
  #endregion

  #region def component_fusion(self, u: np.ndarray, fid_area: np.ndarray, 
  def component_fusion(self, u: np.ndarray, fid_area: np.ndarray, 
                       phase: np.ndarray, supp: np.ndarray, init: np.ndarray, reg_area: np.ndarray) -> np.ndarray:
    """
    Given the elastica phase, return the occluded shape.
    Input:
          u: np.ndarray. Output of the Euler Elastica Model
          fid_area: np.ndarray. Fidelity area.
          supp: np.ndarray. support of the corner bases
    """
    init[init < 0] = 0
    elastica_phase = np.sign(np.multiply(1 - init, u))

    neg_region = (elastica_phase < 0).astype(float)
    neg_region[init > 0] = 0.

    #return np.bitwise_or(init.astype(bool), neg_region.astype(bool)).astype(float)
    ret = np.bitwise_or(init.astype(bool), neg_region.astype(bool)).astype(float)

    return np.bitwise_or(ret, np.multiply(phase == -1, reg_area))
  #endregion

  #region def new_component_fusion(self, u: np.ndarray, fid_area: np.ndarray, 
  def new_component_fusion(self, u: np.ndarray, fid_area: np.ndarray, 
                       phase: np.ndarray, supp: np.ndarray, init: np.ndarray, reg_area) -> np.ndarray:
    """
    Given the elastica phase, return the occluded shape.
    Input:
          u: np.ndarray. Output of the Euler Elastica Model
          fid_area: np.ndarray. Fidelity area.
          supp: np.ndarray. support of the corner bases
    """
    init[init < 0] = 0

    temp_elastica = np.multiply(1 - init, u) # (Omega \ F) intersect u
    temp_elastica[temp_elastica >= 0] = 0 # (Omega \ F) intersect {u < 0}
    temp_elastica *= -1  # consider only the negative value, make them positive
    init += temp_elastica # init union ((Omega \ F) intersect {u < 0})

    init += np.multiply(phase == -1, reg_area) # newly added
    init[init > 1] = 1.
    return init
  #endregion

  #region def clean_up_cmps(self, u: np.ndarray, idx_fid: np.ndarray) -> np.ndarray:
  def clean_up_cmps(self, u: np.ndarray, idx_fid: np.ndarray) -> np.ndarray:
    """
    Because of the periodic boundary enforced in the Euler-Elastica Model,
    components touching a boundary of the domain may diffuse to the opposite boundary.
    Clean this up. Get rid of the region(s) that are not connected(in graph sense) to the fidelity region.
    """
    label_im, num = label(u.astype(int), connectivity = 1, return_num = True)

    if num == 1: # only one connected component -> do nothing
      return u

    cm_im = np.bitwise_and(idx_fid, label_im.astype(bool)).astype(int)
    cmp_num = np.max(np.multiply(cm_im, label_im))
    u[label_im != cmp_num] = 0.
    return u
  #endregion

  #region def reg_area_adaptive_dilation(self, i: tuple) -> np.ndarray:
  def reg_area_adaptive_dilation(self, i: tuple) -> np.ndarray:
    """
    Dilate the reg area according to the size of shape layer <i>
    """
    return self.get_reg_area(i)
    ratio     = np.sum(self.shape_layers[i].get_dense_layer_im()) / (self.h * self.w)
    footprint = np.ceil(self.footprint * ratio)
    return dilation(self.get_reg_area(i), footprint)
  #endregion

  #region def prepare_numba_data(self, args: list[tuple], use_partial_convex_init: bool) -> list[numba.typed.Dict]:
  def prepare_numba_data(self, args: list[tuple], use_partial_convex_init: bool) -> list[numba.typed.Dict]:
    """
    Get ready for numba parallel algorithm
    """
    print("Preparing Data for Parallel Algorithm...")
    ## create dictionaries for numba
    phase_dict = numba.typed.Dict.empty(key_type, val_type)
    supp_dict = numba.typed.Dict.empty(key_type, val_type)
    init_dict = numba.typed.Dict.empty(key_type, val_type)
    reg_dict = numba.typed.Dict.empty(key_type, val_type)
    fid_dict = numba.typed.Dict.empty(key_type, val_type)

    ## fill in dicionaries
    for key in tqdm(args):
      phase, supp = self.shape_layers[key].get_phase(), self.shape_layers[key].get_supp()
      init = self.shape_layers[key].get_partial_convex_init() if use_partial_convex_init else self.shape_layers[key].get_dense_layer_im()
      if isspmatrix(phase):
        phase = phase.toarray()
      if isspmatrix(supp):
        supp = supp.toarray()
      if isspmatrix(init):
        init = init.toarray()
      init_dict[key]  = init.astype(np.float64)
      fid_dict[key]   = np.maximum(self.get_fid_area(key).astype(np.float64), 0)
      reg_dict[key]   = np.maximum(self.get_reg_area(key).astype(np.float64), 0)
      supp_dict[key]  = np.multiply(reg_dict[key], supp)
      phase_dict[key] = np.multiply(supp_dict[key], phase)
    
    return [phase_dict, supp_dict, init_dict, reg_dict, fid_dict]
  #endregion

  #region def numba_parallel_solve(phase_dict: numba.typed.Dict, supp_dict: numba.typed.Dict, 
  @staticmethod
  @jit(nopython = False, parallel = True)
  def numba_parallel_solve(phase_dict: numba.typed.Dict, supp_dict: numba.typed.Dict, 
                           init_dict: numba.typed.Dict, reg_dict: numba.typed.Dict, fid_dict: numba.typed.Dict,
                           a: float, b: float, c: float, eps: float, mu: float,
                           max_iter: int, coeff_matrix: np.ndarray, progress) -> None:
    coeff_matrix_v = 4 * b * coeff_matrix + a + c
    coeff_matrix_u = 2 * np.power(eps, 2) * coeff_matrix + c

    keys = numba.typed.List()
    for key in phase_dict.keys():
      keys.append(key)
    
    ret = numba.typed.Dict.empty(key_type, val_type)
    n = len(keys)
    for ptr in numba.prange(n):
      init, phase, supp  = init_dict[keys[ptr]], phase_dict[keys[ptr]], supp_dict[keys[ptr]]
      reg_area, fid_area = reg_dict[keys[ptr]], fid_dict[keys[ptr]]
      u = -init.astype(np.float64)
      v = np.zeros_like(u).astype(np.float64)
      idx_reg = 1 - reg_area > 0
      idx_fid = fid_area > 0
      for _ in range(max_iter):
        ############# solve v subproblem ############# 
        RHS = 2 * mu * np.multiply(supp, u - phase)
        RHS -= (b / np.power(eps, 2)) * np.multiply(v, 12 * np.power(u, 2) - 4)
        RHS += c * v
        RHS = compute_fft(RHS)
        RHS = np.divide(RHS, coeff_matrix_v)
        RHS = compute_ifft(RHS)
        v = np.real(RHS)
        v = np.maximum(v, -1)
        v = np.minimum(v, 1)

        ############# solve u subproblem ############# 
        coe_u = 2 * np.power(u, 2)
        max_coe_u = np.max(coe_u)
        uRHS = np.multiply(max_coe_u - coe_u + 2, u) - eps * v + c * u
        uRHS = compute_fft(uRHS)
        uRHS = np.divide(uRHS, coeff_matrix_u + max_coe_u)
        uRHS = compute_ifft(uRHS)
        u = np.real(uRHS)
        u = np.maximum(u, -1)
        u = np.minimum(u, 1)

        u += idx_reg * 2.
        u = np.minimum(u, 1)
        u += idx_fid * (-2.)
        u = np.maximum(u, -1)

      
      ############# Component fusion (see new_component_fusion) ############# 
      init = np.maximum(init, 0)
      temp_elastica = np.multiply(1 - init, u)
      temp_elastica = np.minimum(temp_elastica, 0)
      temp_elastica *= -1
      init += temp_elastica
      init += np.multiply(phase == -1, reg_area)
      init = np.minimum(init, 1)
      init = np.maximum(init, fid_area)
      init = np.maximum(init, 0)
      ret[keys[ptr]] = init 
      progress.update(1)
    
    return ret
  #endregion

  ########################################## MIXED SOLVING ########################################## 
  #region def parallel_solve(self, solve_parallel: bool = False) -> list:
  def parallel_solve(self, solve_parallel: bool = True, use_partial_convex_init: bool = True, debug_plot: bool = False) -> list:
    main_keys = list(filter(lambda key : not self.shape_layers[key].check_is_noise(), self.shape_layers.keys()))
    args = sorted(main_keys, 
                  key     = lambda k : self.shape_layers[k].get_level(), 
                  reverse = True)

    # don't do euler elastica on small shape layers
    #small_args = list(filter(lambda key : np.sum(self.shape_layers[key].get_partial_convex_init() > 0) < self.area_threshold, args))
    small_args = list(filter(lambda key : np.sum(self.shape_layers[key].get_dense_layer_im() > 0) < self.area_threshold, args))
    for a in small_args:
      args.remove(a)

    print("Solving big shape layers")
    if solve_parallel:
      preped_data = self.prepare_numba_data(args, use_partial_convex_init)
      start = time.time()
      print("Begin solving...")
      with ProgressBar(total = len(args)) as progress:
        result = self.numba_parallel_solve(*preped_data, self.a, self.b, self.c, 
                                          self.eps, self.mu, self.max_iter, self.coeff_matrix,
                                          progress)
      end = time.time() - start
      print(f"parallel Euler used: {end}")
      for key, val in result.items():
        self.shape_layers[key].set_euler_output(val)
        D = self._prepare_for_bezier(key, val)
        self.shape_layers[key].prep_for_bezier(D, bezier_prep_param = self.bezier_prep_param)
    else:
      for arg in tqdm(args):
        _, u, _ = self.mixed_solve(arg)
        D = self._prepare_for_bezier(arg, u)
        self.shape_layers[arg].set_euler_output(u)
        self.shape_layers[arg].prep_for_bezier(D, bezier_prep_param = self.bezier_prep_param)
    
    print("Solving small shape layers")
    for a in small_args:
      #ret.append((a, self.shape_layers[a].get_partial_convex_init()))
      #u = self.shape_layers[a].get_partial_convex_init()
      u = self.shape_layers[a].get_dense_layer_im()
      u[u==0] = -1
      D = self._prepare_for_bezier(a, u)
      self.shape_layers[a].set_euler_output(u)
      self.shape_layers[a].prep_for_bezier(D, bezier_prep_param = self.bezier_prep_param)
    
    if debug_plot:
      for key in self.shape_layers.keys():
        self.shape_layers[key].showcase()


  #endregion

  #region def _fill_holes(self, k: tuple) -> np.ndarray:
  def _fill_holes(self, k: tuple) -> np.ndarray:
    """
    For images with holes, remove them.
    Reconstruct the image using only clockwise oriented boundary.
    """
    if len(self.components[k][counter_boundary_str]) == 0:
      return self.components[k][image_str].copy()
    
    ct = [c[:, ::-1] for c in self.components[k][clockwise_boundary_str]]
    
    im = np.zeros((self.h, self.w))
    im = cv.fillPoly(im, pts = ct , color = 1.)

    return im

  #endregion

  #region def mixed_solve(self, k: tuple, debug_plot: bool = False) -> (tuple, np.ndarray, float):
  def mixed_solve(self, k: tuple, debug_plot: bool = False) -> (tuple, np.ndarray, float):
    """
    If the shape is small, use Partial convex init and CV to remove pixelized effect.
    If the shape is large, use Euler Elastica. Also need to determine the parameters.
    """
    im = self.shape_layers[k].get_partial_convex_init()
    if isspmatrix(im):
      im = im.toarray()
    im[im < 0] = 0
    im = binary_fill_holes(im).astype(float)
    im[im == 0] = -1
    b, c, eps     = self._determine_param(k)
    k, im, energy = self.euler_solve((k, b, c, eps, im), debug_plot = debug_plot)

    im[im <= 0] = 0

    return k, im, energy
  #endregion

  #region def _determine_param(self, k: tuple) -> (float, float, float):
  def _determine_param(self, k: tuple) -> (float, float, float):
    """
    Determine parameters for Euler Elastica Method.
    b: curvature weight. 
    c: regularization parameter (actually this shouldn't be changed)
    eps: trade off between diffusion and double well potential. When shape k goes toward the bottom, this should be larger for stronger diffusion.
    """
    ## mountain, 
    #level = self.components[k][level_str]
    level = self.shape_layers[k].get_level()
    c = 1. 
    b = 10 ** (2 * level)
    eps = min(.01, 10 ** (-(5 - level)))

    return b, c, eps
  #endregion

  ########################################## CHECK OUT ########################################## 
  #region def prepare_for_bezier(self, ret: list) -> dict:
  def prepare_for_bezier(self, ret: list) -> dict:
    max_level = max([self.shape_layers[key].get_level() for key in self.shape_layers.keys()])
    output = dict()
    for r in ret:
      i, image = r[0], r[1]

      #contours = find_contours(image, level = 0.)
      contours = get_boundary(image)
      #contours = get_zero_contour(image)

      output[i] = dict()
      if np.all(image == 1):
        output[i].update({"is_bg": True})
      else:
        output[i].update({"is_bg": False})
        segs = [s[:, ::-1] for s in contours]
        output[i].update({"segs": segs})
      output[i].update({"level": max_level - self.shape_layers[i].get_level()})
      output[i].update({"color": i[0]})
      if use_sparse(image):
        image = csc_matrix(image)
      output[i].update({"image": image})

    return output
  #endregion

  #region def _prepare_for_bezier(self, i: tuple, im: csr_matrix or np.ndarray) -> dict:
  def _prepare_for_bezier(self, i: tuple, im: csr_matrix or np.ndarray) -> dict:
    if isspmatrix(im):
      im = im.toarray()

    is_bg = np.all(im == 1)
    level = self.max_level - self.shape_layers[i].get_level()
    if is_bg:
      return {"is_bg": is_bg, "level": level, "color": i[0]}

    segs = [s[:,::-1] for s in get_boundary(im)]

    return {"is_bg": is_bg,  
            "segs": segs, 
            "level": level, 
            "color": i[0], 
            "image": smart_convert_sparse(im)} 
  #endregion

  #region def check_out(self, path: str, D: dict) -> None:
  def check_out(self, path: str, D: dict) -> None:
    save_data(path, D)
  #endregion

  #region def save(self, path: str) -> None:
  def save(self, path: str) -> None:
    D = {shape_layers_str: self.shape_layers, 
         grid_graph_str: self.graph,
         order_graph_str: self.order_graph,
         mutual_bd_graph_str: self.mutual_bd_graph,
         }
    save_data(path, D)
  #endregion

  ########################################## SHAPE DIFFUSION ########################################## 
  #region def shape_diffusion(self, im: np.ndarray) -> np.ndarray:
  def shape_diffusion(self, im: np.ndarray) -> np.ndarray:
    neg_one_region = im < 0

    im[im < 0] = 0
    im_interior = erosion(im, square(3))

    im = self._simple_shape_diffusion(im, im_interior)

    im[neg_one_region] = -1
    return im
  #endregion

  #region def _simple_shape_diffusion(self, im: np.ndarray, im_interior: np.ndarray, 
  def _simple_shape_diffusion(self, im: np.ndarray, im_interior: np.ndarray, 
                             l: float = .2, nstep: int = 10, optDiff: np.ndarray = None) -> np.ndarray:
    """
    Do simple diffusion for more accurate normal computation. Correspond to <bakeShape>.
    """
    u = im.copy()
    if optDiff is None:
      optDiff = np.ones_like(im) 

    degIJ = np.pad(optDiff, 1, 'constant', constant_values = 0)
    degIJ = np.roll(degIJ, 1, axis = 0) + np.roll(degIJ, -1, axis = 0) \
            + np.roll(degIJ, 1, axis = 1) + np.roll(degIJ, -1, axis = 1)
    degIJ = degIJ[1:-1, 1:-1]

    for _ in range(nstep):
      np.multiply(u, optDiff, out = u)
      uext = np.pad(u, 1, 'constant', constant_values = 0)
      uext = np.roll(uext, 1, axis = 0) + np.roll(uext, -1, axis = 0) \
             + np.roll(uext, 1, axis = 1) + np.roll(uext, -1, axis = 1)
      uext = uext[1:-1, 1:-1]
      
      unew  = np.multiply((1 - l * degIJ), u)
      unew += l * uext

      u = np.multiply(1 - im_interior, unew) + im_interior * 1.
    return u
  #endregion

  ########################################## CHECK LEVEL ########################################## 
  #region def check_level(self) -> None:
  def check_level(self) -> None:
    display = [(self.components[key][image_str], self.components[key][level_str]) for key in self.components.keys()]

    display = sorted(display, key = lambda x : x[1], reverse = True) 

    for d in display:
      plt.figure(d[1])
      plt.imshow(d[0])
      plt.show()
  #endregion

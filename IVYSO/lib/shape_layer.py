#!/usr/bin/env python3.9

# External
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix
from scipy.ndimage import label, binary_fill_holes, generate_binary_structure, binary_dilation
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from skimage.measure import find_contours
from numpy_indexed import in_, indices
from skimage.morphology import convex_hull_image
from scipy.spatial.distance import cdist
import threading
from multiprocessing import cpu_count
from tqdm import tqdm

# Internal
from .new_helper import smart_convert_sparse, is_clockwise, get_boundary, color_transformation, save_data, compute_curvature_at_points, inverse_color_transform

self_color          = "self_color"
neighbor_color      = "neighbor_color"
is_noise_str        = "is_noise"
shape_layers_str    = "shape_layers"
grid_graph_str      = "grid_graph"
mutual_bd_graph_str = "mutual_bd_graph"
phase_str           = "phase"

wild_card_tuple = (300,300,300)
wild_card_int   = 300 * np.power(256, 2) + 300 * 256 + 300

class shape_layer():
  #region def __init__(self, color: tuple or int, layer_im: np.ndarray, params: dict = None) -> None:
  def __init__(self, color: tuple or int, layer_im: np.ndarray, params: dict = None) -> None:
    self.color    = color
    self.layer_im = layer_im
    self.params   = params

    self.h, self.w = self.layer_im.shape

    self.threshold = 30 # connected components with pixels less than this number will be seen as noise
    self.osculating_circle_radius_thre = 5
    self.same_color_layer = False

    self._load_params()

    self.curv_thre = 1 / self.osculating_circle_radius_thre

    ## Set up attributes
    self.id = None
    self.is_noise = np.count_nonzero(self.layer_im) <= self.threshold
    if not self.is_noise:
      self.set_up_attributes()
   
    ## store in sparse matrix format for memory efficiency
    self.layer_im = smart_convert_sparse(self.layer_im)
  #endregion

  #region def _load_params(self) -> None:
  def _load_params(self) -> None:
    if self.params is None:
      return 

    for key, val in self.params.items():
      if key == "threshold":
        self.threshold = val 
      elif key == "osculating_circle_radius_thre":
        self.osculating_circle_radius_thre = val
      elif key == "same_color_layer":
        self.same_color_layer = val
      else:
        raise KeyError(f"Unknown Parameter: {key}")
  #endregion

  #region def _find_boundary(self) -> (list, list):
  def _find_boundary(self) -> (list, list):
    """
    Get the boundary of the shape layer.
    The first entry contains all clockwise boundary, and the second contains counter-clockwise boundary.
    """
    boundary = get_boundary(self.layer_im)
    #boundary = list(map(self._clean_up_boundary, boundary))

    boundary    = list(zip(range(len(boundary)), boundary))
    clock_bd    = list(filter(lambda x : is_clockwise(x[1]), boundary))
    clock_ind   = {c[0] for c in clock_bd}
    clock_bd    = [b[1] for b in clock_bd]
    counter_bd  = [b[1] for b in boundary if b[0] not in clock_ind]

    return clock_bd, counter_bd
  #endregion

  #region def _clean_up_boundary(self, pts: np.ndarray) -> np.ndarray:#{{{
  def _clean_up_boundary(self, pts: np.ndarray) -> np.ndarray:#{{{
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
  #}}}
  #endregion

  #region def set_up_attributes(self) -> None:
  def set_up_attributes(self) -> None:
    self.clock_bd, self.counter_bd = self._find_boundary()
    self.holes_filled              = smart_convert_sparse(binary_fill_holes(self.layer_im))
    self.level                     = None
    self.phase                     = None
    self.supp                      = None
    self.inpaintable_pts           = None
    self.partial_convex_init       = None
    self.euler_output              = None
    self.bezier_prep               = None
    self.T_junc                    = None
    self.cnt                       = None
    self.high_curv_T_junc          = None
    self.convex_im                 = smart_convert_sparse(convex_hull_image(self.layer_im))
    self.reg_area                  = None
    #self.nnz                       = np.count_nonzero(self.layer_im)
  #endregion

  ############################ ACCESS ############################ 
  #region def get_layer_im(self) -> np.ndarray or csr_matrix:
  def get_layer_im(self) -> np.ndarray or csr_matrix:
    return self.layer_im
  #endregion

  #region def get_holes_filled(self) -> csr_matrix or np.ndarray:
  def get_holes_filled(self) -> csr_matrix or np.ndarray:
    return self.holes_filled
  #endregion

  #region def get_dense_layer_im(self) -> np.ndarray:
  def get_dense_layer_im(self) -> np.ndarray:
    if isspmatrix(self.layer_im):
      return self.layer_im.toarray()
    return self.layer_im
  #endregion

  #region def get_coord(self) -> np.ndarray:
  def get_coord(self) -> np.ndarray:
    """
    Return non-zero coordinate as an n by 2 np.narray
    """
    if isspmatrix(self.layer_im):
      return np.column_stack(self.layer_im.nonzero())
    elif isinstance(self.layer_im, np.ndarray):
      return np.argwhere(self.layer_im)
    else:
      raise TypeError("Unknown Type")
  #endregion

  #region def get_id(self) -> tuple:
  def get_id(self) -> tuple:
    return self.id
  #endregion

  #region def get_clock_bd(self) -> list():
  def get_clock_bd(self) -> list():
    return self.clock_bd
  #endregion

  #region def check_is_noise(self) -> bool:
  def check_is_noise(self) -> bool:
    return self.is_noise
  #endregion

  #region def get_level(self) -> int:
  def get_level(self) -> int:
    return self.level
  #endregion

  #region def get_inpaintable_pts(self) -> np.ndarray:
  def get_inpaintable_pts(self) -> np.ndarray:
    return self.inpaintable_pts
  #endregion

  #region def get_partial_convex_init(self) -> np.ndarray:
  def get_partial_convex_init(self) -> np.ndarray:
    ret = -np.ones((self.h, self.w))
    ind = self.partial_convex_init
    if isspmatrix(ind):
      ind = ind.toarray()
    ret[ind] = 1
    return ret
  #endregion

  #region def get_phase(self) -> np.ndarray:
  def get_phase(self) -> np.ndarray:
    ret = -np.ones((self.h, self.w))
    ind1, ind0 = self.phase[1], self.phase[0]
    if isspmatrix(ind1): ind1 = ind1.toarray()
    if isspmatrix(ind0): ind0 = ind0.toarray()
    ret[ind1] = 1.
    ret[ind0] = 0.
    return ret  
  #endregion

  #region def get_supp(self, return_dense: bool = True) -> np.ndarray:
  def get_supp(self, return_dense: bool = True) -> np.ndarray:
    if return_dense and isspmatrix(self.supp):
      return self.supp.toarray()
    return self.supp
  #endregion

  #region def check_is_bg(self) -> bool:
  def check_is_bg(self) -> bool:
    if self.bezier_prep["is_bg"] is not None:
      return self.bezier_prep["is_bg"]
    else:
      return np.logical_and.reduce(np.all(self.get_dense_layer_im[0,:]), 
                                   np.all(self.get_dense_layer_im[:,0]),
                                   np.all(self.get_dense_layer_im[-1,:]),
                                   np.all(self.get_dense_layer_im[:,-1]))

  #endregion

  #region def get_bezier_level(self) -> int:
  def get_bezier_level(self) -> int:
    return self.bezier_prep["level"]
  #endregion

  #region def get_euler_output(self) -> np.ndarray:
  def get_euler_output(self, return_dense: bool = True) -> np.ndarray:
    if return_dense and isspmatrix(self.euler_output):
      return self.euler_output.toarray()
    return self.euler_output
  #endregion

  #region def get_T_junc(self) -> np.ndarray:
  def get_T_junc(self) -> np.ndarray:
    return self.T_junc
  #endregion

  #region def get_cnt(self) -> list[np.ndarray]:
  def get_cnt(self) -> list[np.ndarray]:
    return self.cnt
  #endregion

  #region def get_landmark(self) -> np.ndarray:
  def get_landmark(self) -> np.ndarray:
    return self.landmark
  #endregion

  #region def get_convex_im(self) -> csr_matrix or np.ndarray:
  def get_convex_im(self) -> csr_matrix or np.ndarray:
    return self.convex_im
  #endregion

  #region def get_color(self) -> tuple or int:
  def get_color(self) -> tuple or int:
    return self.color
  #endregion

  #region def get_nnz(self) -> int:
  def get_nnz(self) -> int:
    #return self.nnz
    if isspmatrix(self.layer_im):
      return self.layer_im.getnnz()
    else:
      return np.sum(self.layer_im > 0)

  #endregion

  #region def get_reg_area(self, return_np: bool = True) -> np.ndarray or csr_matrix:
  def get_reg_area(self, return_np: bool = True) -> np.ndarray or csr_matrix:
    assert(self.reg_area is not None)
    if return_np and isspmatrix(self.reg_area):
      return self.reg_area.toarray()
    else:
      return self.reg_area
  #endregion

  ############################ MODIFICATION ############################ 
  #region def set_id(self, id: tuple) -> None:
  def set_id(self, id: tuple) -> None:
    self.id = id
  #endregion

  #region def set_level(self, level: int) -> None:
  def set_level(self, level: int) -> None:
    self.level = level
  #endregion

  #region def set_phase(self, phase: np.ndarray) -> None:
  def set_phase(self, phase: np.ndarray) -> None:
    phase1, phase0 = smart_convert_sparse(phase == 1, dtype = bool), smart_convert_sparse(phase == 0, dtype = bool)
    self.phase = {1 : phase1 , 0 : phase0}
  #endregion
 
  #region def set_supp(self, supp: np.ndarray) -> None:
  def set_supp(self, supp: np.ndarray) -> None:
    self.supp = smart_convert_sparse(supp)
  #endregion

  #region def set_inpaintable_pts(self, inpaintable_pts: np.ndarray) -> None:
  def set_inpaintable_pts(self, inpaintable_pts: np.ndarray) -> None:
    self.inpaintable_pts = inpaintable_pts
  #endregion

  #region def set_partial_convex_init(self, partial_convex_init: np.ndarray) -> None:
  def set_partial_convex_init(self, partial_convex_init: np.ndarray) -> None:
    #partial_convex_init[partial_convex_init == -1] = 0
    #self.partial_convex_init = smart_convert_sparse(partial_convex_init)
    self.partial_convex_init = smart_convert_sparse(partial_convex_init == 1, dtype = bool)
  #endregion

  #region def set_euler_output(self, euler_output: np.ndarray) -> None:
  def set_euler_output(self, euler_output: np.ndarray) -> None:
    array, _ = label(euler_output > 0)
    ind = np.max(np.multiply(self.get_dense_layer_im(), array))
    np.multiply(euler_output, array == ind, out = euler_output)
    self.euler_output = smart_convert_sparse(euler_output)
  #endregion

  #region def set_T_junc(self, T_junc: np.ndarray) -> None:
  def set_T_junc(self, T_junc: np.ndarray) -> None:
    self.T_junc = T_junc
  #endregion

  #region def set_high_curv_T_junc(self, possible_T_junc: list, spacing: int = 5) -> None:
  def set_high_curv_T_junc(self, possible_T_junc: list, spacing: int = 5) -> None:
    self.high_curv_T_junc = self.find_t_junc(possible_T_junc, spacing = spacing)
  #endregion

  #region def reset_bezier_param(self, l: float) -> None:
  def reset_bezier_param(self, l: float) -> None:
    """
    Recalculate the level set given a different <l>
    """
    self.prep_for_bezier(self.bezier_prep, r = l)
  #endregion

  #region def set_reg_area(self, reg_area: np.ndarray) -> None:
  def set_reg_area(self, reg_area: np.ndarray) -> None:
    self.reg_area = smart_convert_sparse(reg_area)
  #endregion

  ############################ COMPUTATION ############################ 
  #region def compute_level_set(self, r: float = .5, im_name: str = "euler_output") -> list[np.ndarray]:
  def compute_level_set(self, r: float = .5, im_name: str = "euler_output") -> list[np.ndarray]:
    """
    r: float. The level set value wanted. 
    """
    im = None
    if im_name == "euler_output":
      im = self.get_euler_output()
    elif im_name == "init":
      im = self.get_partial_convex_init()
    elif im_name == "layer_im":
      im = self.get_dense_layer_im()
    else:
      raise KeyError("Unknown image name")
    
    assert(im is not None)
    assert(0 <= r <= 1)

    ## padding for finding the boundary
    im = np.pad(im, 1, 'constant', constant_values = 0)

    cnt = find_contours(im, r)
    cnt = list(filter(lambda c : not is_clockwise(c), cnt)) # skimage.measure.find_contours return outside boundary as counterclockwise.

    cnt = [c - 1 for c in cnt] ## Adjust padding
    cnt = [c[::-1, :] for c in cnt]
    return cnt
  #endregion
  
  #region def prep_for_bezier(self, D: dict, r: float = .5, bezier_prep_param: dict = None, dist_thre: float = .6) -> None:
  def prep_for_bezier(self, D: dict, r: float = .5, bezier_prep_param: dict = None, dist_thre: float = .6) -> None:
    if bezier_prep_param is not None:
      assert(isinstance(bezier_prep_param, dict))
      for key, val in bezier_prep_param.items():
        if key == "r":
          r = val
        elif key == "dist_thre":
          dist_thre = val
        else:
          raise KeyError
    self.cnt = self.compute_level_set(r = r)
    self.bezier_prep = D
    self.landmark = self.find_high_curv_t_junc_to_keep(self.cnt, dist_thre) 
  #endregion

  #region def find_t_junc(self, t_juncs: list[tuple], spacing: int = 5) -> np.ndarray:
  def find_t_junc(self, t_juncs: list[tuple], spacing: int = 5) -> np.ndarray:
    """
    Given some possible T junctions in the whole pic, see which one belongs to this shape layer.
    Then, keep only those with high curvature.
    """
    my_t_junc = np.array(list(filter(lambda n : self.layer_im[n] > 0, t_juncs))).astype(np.float32)

    if len(my_t_junc) == 0:
      return np.array([])

    #plt.imshow(self.get_dense_layer_im())
    #plt.plot(my_t_junc[:,1], my_t_junc[:,0], 'ro')
    #plt.plot(self.get_clock_bd()[0][:,1], self.get_clock_bd()[0][:,0], 'b-')
    #plt.show()

    high_curv_t_junc = list()
    for bd in self.get_clock_bd():
      bd = bd.astype(np.float32)
      flag = in_(my_t_junc, bd)
      if any(flag):
        ind = indices(bd, my_t_junc[flag, :])
        for cur_spacing in range(spacing, 0, -1):
          try:
            curv = np.abs(np.array(compute_curvature_at_points(ind, bd, spacing = cur_spacing)))
          except (RuntimeWarning, IndexError):
            if cur_spacing == 1:
              raise ValueError("dividing zero")
            else:
              pass
        high_curv_t_junc.append(bd[ind[curv > self.curv_thre], : ])

    if len(high_curv_t_junc) > 0:
      high_curv_t_junc = np.vstack(high_curv_t_junc)

    return high_curv_t_junc
  #endregion

  #region def find_high_curv_t_junc_to_keep(self, cnt: list[np.ndarray], dist_thre: float) -> np.ndarray:
  def find_high_curv_t_junc_to_keep(self, cnt: list[np.ndarray], dist_thre: float) -> np.ndarray:
    """
    At the beginning when cooking up shape layers, we found some high curvature T junctions. Call them <T>
    Now after the Euler Elastica step is done, we have a level set.
    If the points in <T> are "sufficiently" close to some points on the level set,
    then we keep these points in the level set as some hard landmark constraint.
    """
    if self.high_curv_T_junc is None or len(self.high_curv_T_junc) == 0:
      return []
    landmarks = list()
    for c in cnt:
      for p in self.high_curv_T_junc:
        ind = np.sum(np.power(c - p, 2), axis = 1)
        if ind.min() > dist_thre:
          continue
        ind = np.flatnonzero(ind == ind.min())
        ind = ind[np.argmax(np.abs(compute_curvature_at_points(ind, c)))]
        landmarks.append(c[ind, :])
   
    if len(landmarks) > 0:
      landmarks = np.vstack(landmarks)
    

    return landmarks
  #endregion

  ############################ VISUALIZATION ############################ 
  #region def showcase(self) -> None:
  def showcase(self) -> None:
    flags = [self.layer_im is not None, 
             self.phase is not None, 
             self.supp is not None, 
             self.partial_convex_init is not None, 
             self.euler_output is not None]
    names = ["original", "phase", "supp", "partial convex init", "euler output"]
    ims = [self.get_dense_layer_im(), 
           self.get_phase(),  
           self.get_supp(), 
           self.get_partial_convex_init(), 
           self.get_euler_output()]
    n = sum(flags)

    fig, ax = plt.subplots(1, n, sharex = True, sharey = True)

    ptr = 0
    for flag, name, im in zip(flags, names, ims):
      if not flag:
        continue
      ax[ptr].imshow(im)
      ax[ptr].set_title(name)
      if name == "euler output":
        for c in self.cnt:
          plt.plot(c[:,1], c[:,0], 'ro')
        #if self.high_curv_T_junc is not None and len(self.high_curv_T_junc) > 0: 
        #  plt.plot(self.high_curv_T_junc[:,1], self.high_curv_T_junc[:,0], 'bo')
        if self.landmark is not None and len(self.landmark) > 0: 
          plt.plot(self.landmark[:,1], self.landmark[:,0], 'bo')
      ptr += 1
    plt.show()
  #endregion



class shape_layer_factory():
  #region def __init__(self, pic: np.ndarray, layers: dict, colors: np.ndarray, structure: int = 1,
  def __init__(self, pic: np.ndarray, layers: dict, colors: np.ndarray, structure: int = 1,
               shape_layer_params: dict = None, factory_params: dict = None, 
               coarse_seg_result: dict = None, use_seg_result: bool = False, 
               auto_denoise: bool = False, auto_run: bool = True, use_parallel: bool = True) -> None:
    self.pic                = pic
    self.h, self.w          = self.pic.shape[0], self.pic.shape[1]
    self.layers             = layers
    self.colors             = colors
    self.shape_layer_params = shape_layer_params
    self.factory_params     = factory_params
    self.structure          = structure
    self.coarse_seg_result  = coarse_seg_result
    self.use_seg_result     = use_seg_result
    self.auto_denoise       = auto_denoise 
    self.auto_run           = auto_run
    self.use_parallel       = use_parallel 
    self.same_color_layer   = False # if true, all shapes with same color are considered one single shape layer

    self.threshold = 30 # connected components with pixels less than this number will be seen as noise
    self.closedness = 1
    self.single_region_threshold = 3 # only regions with area strictly larger than this value would be connected. 

    self._load_params()

    print("Constructing Shape Layers...")
    self.construct_shape_layers()

    if self.use_seg_result:
      assert(self.coarse_seg_result is not None)
      self.aggregate_shape_layers()

    if self.auto_denoise:
      old_threshold = self.threshold
      self.threshold = self._determine_threshold(plot_hist = False)
      if self.threshold != old_threshold:
        self.reset_shape_layers(self.threshold)
        print(f"auto detect threshold: {self.threshold}")

    print("Creating Grid Graph...")
    self.grid_graph, self.mutual_bd_graph = self.create_graphs()
  #endregion

  #region def _load_params(self) -> None:
  def _load_params(self) -> None:
    if self.structure == 2:
      self.structure = generate_binary_structure(2, 2)
    elif self.structure == 1:
      self.structure = generate_binary_structure(2, 1)

    if self.factory_params is None:
      return 

    for key, val in self.factory_params.items():
      if key == "threshold":
        self.threshold = val
      elif key == "structure":
        self.structure = val
        assert(isinstance(self.structure, int) and 1 <= self.structure <= 2)
        if self.structure == 2:
          self.structure = generate_binary_structure(2, 2)
        elif self.structure == 1:
          self.structure = generate_binary_structure(2, 1)
      elif key == "closedness":
        self.closedness = val
        assert(isinstance(self.closedness, int) and self.closedness > 0)
      elif key == "single_region_threshold":
        self.single_region_threshold = val
      elif key == "same_color_layer":
        self.same_color_layer = val
      else:
        raise KeyError("Unknown Parameter")

  #endregion

  #region def construct_shape_layers(self) -> dict:
  def construct_shape_layers(self) -> dict:
    self.shape_layers = dict()
    if not self.use_parallel:
      for color, im in self.layers.items():
        L = self.construct_one_color_shape_layers(color, im)
        color_id = color_transformation(color)

        for ptr, S in enumerate(L): 
          S.set_id((color_id, ptr))
          self.shape_layers.update({(color_id, ptr) : S})
    else:
      num_threads = max(cpu_count() - 1, 1)
      colors = list(self.layers.keys())
      ims = list(self.layers.values())
      batch_size = np.ceil(len(colors) / num_threads).astype(int)
      divided_colors = [colors[ptr * batch_size : min((ptr + 1) * batch_size, len(colors))] for ptr in range(num_threads)]
      divided_ims = [ims[ptr * batch_size : min((ptr + 1) * batch_size, len(colors))] for ptr in range(num_threads)]
      all_threads = [threading.Thread(target = self.parallel_construct_one_color_shape_layers_wrapper, args = (colors, ims,) )
                     for colors, ims in zip(divided_colors, divided_ims)]
      [thread.start() for thread in all_threads]
      [thread.join() for thread in all_threads]

  #endregion

  #region def parallel_construct_one_color_shape_layers_wrapper(self, colors: list, ims: list) -> None:
  def parallel_construct_one_color_shape_layers_wrapper(self, colors: list, ims: list) -> None:
    for color, im in zip(colors, ims):
      L = self.construct_one_color_shape_layers(color, im)
      color_id = color_transformation(color)

      for ptr, S in enumerate(L):
        S.set_id((color_id, ptr))
        self.shape_layers.update({(color_id, ptr): S})
  #endregion
  
  #region def construct_one_color_shape_layers(self, color: tuple or int, im: np.ndarray or csr_matrix) -> list[shape_layer]:
  def construct_one_color_shape_layers(self, color: tuple or int, im: np.ndarray or csr_matrix) -> list[shape_layer]:
    if isspmatrix(im):
      im = im.toarray()
    
    assert(isinstance(im, np.ndarray))

    if not self.same_color_layer:
      labeled_array, num_label = label(im, structure = self.structure)

      if self.closedness > 1 and num_label > 1:
        labeled_array, num_label = self._group_close_regions(labeled_array, num_label)

      if num_label > 1:
        return [shape_layer(color, (labeled_array == l).astype(float), params = self.shape_layer_params) for l in range(1, num_label + 1)]
      else:
        return [shape_layer(color, (labeled_array == 1).astype(float), params = self.shape_layer_params)]
    else:
      return [shape_layer(color, im, params = self.shape_layer_params)]
  #endregion

  #region def _group_close_regions(self, labeled_array: np.ndarray, num_label: int) -> (np.ndarray, int):
  def _group_close_regions(self, labeled_array: np.ndarray, num_label: int) -> (np.ndarray, int):
    """
    For disconnected pieces in the picture of the same color, 
    group them as one shape layer if self.closedness > 1.
    That is, if two regions R1 and R2, their l-inf distance dist(R1, R2) <= self.closedness, 
    make them as one single reigon.
    Output new <labeled_array> and <num_label>.
    """
    G = nx.Graph()
    G.add_nodes_from(range(1, num_label + 1))

    coord = {ptr: np.argwhere(labeled_array == ptr) for ptr in range(1, num_label + 1)}
    for v1, v2 in combinations(range(1, num_label + 1), 2):
      if coord[v1].shape[0] <= self.single_region_threshold or coord[v2].shape[0] <= self.single_region_threshold:
        continue
      D = cdist(coord[v1], coord[v2], 'chebyshev')
      if np.any(D <= self.closedness):
        G.add_edge(v1, v2)

    new_im = np.zeros_like(labeled_array)
    for ptr, cc in enumerate(nx.connected_components(G)):
      for c in cc:
        new_im[labeled_array == c] = ptr + 1 
    
    return new_im.astype(int), np.max(new_im)
  #endregion
  
  #region def parallel_create_grpahs(self) -> (nx.classes.graph.Graph, nx.classes.graph.Graph):
  def parallel_create_grpahs(self) -> (nx.classes.graph.Graph, nx.classes.graph.Graph):
    assert(self.shape_layers is not None)
    self.grid_graph = nx.grid_graph((self.w, self.h))
    self.wild_max_ptr = 0

    ################### set each node's self color ################### 
    self.temp_store_colors = dict()
    self.temp_store_attri = dict()

    shape_layers = list(self.shape_layers.values())
    num_thread = np.max(cpu_count() - 1, 1)
    batch_size = np.ceil(shape_layers / num_thread).astype(int)
    divided_shape_layers = [shape_layers[ptr * batch_size : min((ptr + 1) * batch_size, len(shape_layers))]
                            for ptr in range(num_thread)]
    all_threads = [threading.Thread(target = self._set_node_color_wrapper, args = (shape_layer_list,))
                   for shape_layer_list in divided_shape_layers]
    [thread.start() for thread in all_threads]
    [thread.join() for thread in all_threads]

    nx.set_node_attributes(self.grid_graph, self.temp_store_attri)
  
    ################### contract graph ################### 
    mutual_bd_graph = self._create_mutual_bd_graph(self.grid_graph, self.temp_store_colors)

    ################### set neighboring colors ################### 
    graph = self._set_neighboring_colors(self.grid_graph, self.temp_store_attri)

    ################### find potential T-junctions ################### 
    self.assign_t_junc_nodes(graph)

    del self.temp_store_attri, self.temp_store_colors
    return self.grid_graph, mutual_bd_graph
  #endregion

  #region def _set_node_color_wrapper(self, shape_layers: list) -> None:
  def _set_node_color_wrapper(self, shape_layers: list) -> None:
    for S in shape_layers:
      coord = list(map(tuple, S.get_coord()))
      if S.check_is_noise():
        for c in coord:
          self.temp_store_attri.update({c : {self_color: (wild_card_int, self.wild_max_ptr)}})
        self.colors.update({(wild_card_int, self.wild_max_ptr) : coord})
        self.wild_max_ptr += 1
      else:
        for c in coord:
          self.temp_store_attri.update({c : {self_color: S.get_id()}})
        self.temp_store_colors.update({S.get_id() : coord})
  #endregion

  #region def create_graphs(self) -> nx.classes.graph.Graph:
  def create_graphs(self) -> nx.classes.graph.Graph:
    """
    Only return <mutual_bd_graph>. Seems to be the only graph that is needed for layer ordering.
    """
    assert(self.shape_layers is not None)
    graph = nx.grid_graph((self.w, self.h))
    wild_max_ptr = 0

    ################### set each node's self color ################### 
    print("Setting each node's self color")
    colors = dict()
    attri = dict()
    for key, S in tqdm(self.shape_layers.items()):
      coord = list(map(tuple, S.get_coord()))
      if S.check_is_noise():
        for c in coord:
          attri.update({c : {self_color: (wild_card_int, wild_max_ptr)}})
        colors.update({(wild_card_int, wild_max_ptr) : coord})
        wild_max_ptr += 1
      else:
        for c in coord:
          attri.update({c : {self_color: S.get_id()}})
        colors.update({S.get_id() : coord})
    
    nx.set_node_attributes(graph, attri)

    ################### contract graph ################### 
    print("Generating mutual bd graph")
    mutual_bd_graph = self._create_mutual_bd_graph(graph, colors)

    ################### set neighboring colors ################### 
    print("Setting neighboring colors")
    graph = self._set_neighboring_colors(graph, attri)

    ################### find potential T-junctions ################### 
    #print("Finding potential T-junctions(useless, delete later)")
    #self.assign_t_junc_nodes(graph)

    return graph, mutual_bd_graph
  #endregion

  #region def _create_mutual_bd_graph(self, graph: nx.classes.graph.Graph, colors: dict) -> nx.classes.graph.Graph:
  def _create_mutual_bd_graph(self, graph: nx.classes.graph.Graph, colors: dict) -> nx.classes.graph.Graph:
    mutual_bd_graph = nx.Graph()
    mutual_bd_graph.add_nodes_from(colors.keys())

    for key, nodes in tqdm(colors.items()):
      neighbor_color = set([graph.nodes[neigh][self_color] 
                            for node in nodes 
                            for neigh in graph.neighbors(node)])
      neighbor_color = set(filter(lambda c : c != key, neighbor_color))
      for nc in neighbor_color:
        mutual_bd_graph.add_edge(key, nc)

    noisy_nodes = list(filter(lambda n : n[0] == wild_card_int, mutual_bd_graph.nodes()))
   
    new_edges = list()
    for n in noisy_nodes:
      neighbors  = list(mutual_bd_graph.neighbors(n))
      new_edges += [(n1, n2) for n1, n2 in combinations(neighbors, 2)]
    
    new_edges = filter(lambda x : x[0][0] != wild_card_int and x[1][0] != wild_card_int, new_edges)

    mutual_bd_graph.remove_nodes_from(noisy_nodes)
    mutual_bd_graph.add_edges_from(new_edges)

    return mutual_bd_graph
  #endregion

  #region def _set_neighboring_colors(self, graph: nx.classes.graph.Graph, attri: dict) -> nx.classes.graph.Graph:
  def _set_neighboring_colors(self, graph: nx.classes.graph.Graph, attri: dict) -> nx.classes.graph.Graph:
    ################### contract wild nodes connected components ################### 
    wild_nodes    = list(filter(lambda n : attri[n][self_color][0] == wild_card_int, attri.keys()))
    wild_subgraph = graph.subgraph(wild_nodes)
    wild_cc       = list(nx.connected_components(wild_subgraph))

    contracted_graph, super_wild_nodes = self.__contract_all_wild_nodes(graph, wild_cc)

    ################### set all wild nodes' neighbor color ################### 
    wild_nodes_neighbor_color = self._set_wild_nodes_neighbor_color(contracted_graph, super_wild_nodes)
    nx.set_node_attributes(contracted_graph, wild_nodes_neighbor_color, neighbor_color)

    ################### set all non-wild nodes' neighbor color ################### 
    non_wild_nodes_neighbor_color = self._set_all_nodes_neighbor_color(contracted_graph, super_wild_nodes)
    nx.set_node_attributes(contracted_graph, non_wild_nodes_neighbor_color, neighbor_color)

    ################### update all nodes' neighboring colors ################### 
    neighbor_color_dict = self._unpack_contracted_graph(contracted_graph, super_wild_nodes)
    nx.set_node_attributes(graph, neighbor_color_dict, neighbor_color)

    return graph
  #endregion

  #region def __contract_all_wild_nodes(self, graph: nx.classes.graph.Graph, wild_cc: list) -> (nx.classes.graph.Graph(), list):
  def __contract_all_wild_nodes(self, graph: nx.classes.graph.Graph, wild_cc: list) -> (nx.classes.graph.Graph(), list):
    """
    Auxiliary function for <graph_initialization>
    """
    contracted_graph = graph.copy()
    super_wild_nodes = list()
    for c in wild_cc:
      c = list(c)
      super_wild_nodes.append(c[0])
      contracted_graph = self.__merge_nodes(contracted_graph, c)

    return contracted_graph, super_wild_nodes
  #endregion

  #region def _set_wild_nodes_neighbor_color(self, contracted_graph: nx.classes.graph.Graph, super_wild_nodes: list) -> dict:
  def _set_wild_nodes_neighbor_color(self, contracted_graph: nx.classes.graph.Graph, super_wild_nodes: list) -> dict:
    """
    Auxiliary function for <graph_initialization>.
    Set wild nodes' neighboring color. 
    Return a dictionary keyed by each wild node in <super_wild_nodes>, values is a list of neighboring colors.
    """
    ret = {}
    for node in super_wild_nodes:
      neighbor_colors = [contracted_graph.nodes[neighbor][self_color] 
                         for neighbor in contracted_graph.neighbors(node)
                         if contracted_graph.nodes[neighbor][self_color][0] != wild_card_int] 
      ret.update({node: set(neighbor_colors)})

    return ret
  #endregion

  #region def __merge_nodes(self, G: nx.classes.graph.Graph, nodes: list) -> nx.classes.graph.Graph:
  def __merge_nodes(self, G: nx.classes.graph.Graph, nodes: list) -> nx.classes.graph.Graph:
    """
    Contract <nodes> in <G>
    """
    assert(len(nodes) > 0)
    super_node = nodes[0]
    G.add_node(super_node, children = nodes)

    if len(nodes) == 1:
      nx.contracted_nodes(G, super_node, nodes[0], self_loops = False, copy = False)
      return G

    for node in nodes:
      nx.contracted_nodes(G, super_node, node, self_loops = False, copy = False)

    return G
  #endregion

  #region def _set_all_nodes_neighbor_color(self, contracted_graph: nx.classes.graph.Graph, super_wild_nodes: list or set) -> dict:
  def _set_all_nodes_neighbor_color(self, contracted_graph: nx.classes.graph.Graph, super_wild_nodes: list or set) -> dict:
    """
    Auxiliary function for <graph_initialization>.
    Set all non-wild node's neighboring color
    """
    if isinstance(super_wild_nodes, list):
      super_wild_nodes = set(super_wild_nodes)
    assert(isinstance(super_wild_nodes, set))

    ret = dict()
    for node in contracted_graph.nodes:
      if node in super_wild_nodes:
        continue

      neighboring_color = list()
      for neighbor in contracted_graph.neighbors(node):
        if neighbor not in super_wild_nodes:
          neighboring_color.append(contracted_graph.nodes[neighbor][self_color])
        else:
          neighboring_color += list(contracted_graph.nodes[neighbor][neighbor_color])
      ret.update({node: set(neighboring_color)})
    return ret
  #endregion

  #region def _unpack_contracted_graph(self, contracted_graph: nx.classes.graph.Graph, super_wild_nodes: list) -> dict:
  def _unpack_contracted_graph(self, contracted_graph: nx.classes.graph.Graph, super_wild_nodes: list) -> dict:
    """
    Auxiliary function for <graph_initialization>
    Convert the information stored in the contracted graph to a dictionary for updating the grid graph
    """
    neighbor_color_dict = {node: contracted_graph.nodes[node][neighbor_color] for node in contracted_graph.nodes}
    neighbor_color_dict.update({n: contracted_graph.nodes[w][neighbor_color] 
                                for w in super_wild_nodes 
                                for n in contracted_graph.nodes[w]["contraction"][w]["children"]})
    return neighbor_color_dict
  #endregion

  #region def assign_t_junc_nodes(self, graph: nx.classes.graph.Graph, debug_plot: bool = False) -> None:
  def assign_t_junc_nodes(self, graph: nx.classes.graph.Graph, debug_plot: bool = False) -> None:
    """
    Find nodes that are connected to 3 different colors(excluding noise)
    """
    non_noisy_nodes = filter(lambda n : graph.nodes[n][self_color][0] != wild_card_int, graph.nodes())
    ret = list(filter(lambda n : len(graph.nodes[n][neighbor_color]) >= 3, non_noisy_nodes))

    for key in filter(lambda key : not self.shape_layers[key].check_is_noise(), self.shape_layers.keys()):
      self.shape_layers[key].set_high_curv_T_junc(ret, spacing = 3)

    if debug_plot:
      pts = np.array(ret)
      plt.imshow(self.pic)
      plt.plot(pts[:,1], pts[:,0], 'ro')
      plt.show()
      breakpoint()

    return ret
  #endregion

  #region def check_out(self, path: str) -> None:
  def check_out(self, path: str, noisy_layer_path: str = None ) -> None:
    key_to_remove = list()
    for key in self.shape_layers.keys():
      if self.shape_layers[key].check_is_noise():
        key_to_remove.append(key)

    if noisy_layer_path is not None:
      noisy_D = {key: self.shape_layers[key] for key in key_to_remove}
      save_data(noisy_layer_path, noisy_D)
    
    for key in key_to_remove:
      self.shape_layers.pop(key)

    D = {shape_layers_str: self.shape_layers, 
         mutual_bd_graph_str: self.mutual_bd_graph,
         grid_graph_str: self.grid_graph}
    save_data(path, D)
  #endregion

  #region def _determine_threshold(self, plot_hist: bool = False) -> int:
  def _determine_threshold(self, plot_hist: bool = False) -> int:
    """
    Determine denoise threshold by looking at the histogram of shape layers' size.
    Try the first local minimum.
    """
    sizes = np.array([val.get_nnz() for val in self.shape_layers.values()])
    hist, counts = np.unique(sizes, return_counts = True)
    ind = np.argsort(hist)
    hist, counts = hist[ind], counts[ind]

    threshold_ind = self._find_local_minimum(counts)
    if threshold_ind is not None:
      threshold = hist[threshold_ind]
    else:
      threshold = self.threshold

    if plot_hist:
      print(f"threshold: {threshold}")
      show_ind = hist <= 20
      coord = np.column_stack((hist[show_ind], counts[show_ind]))
      plt.plot(coord[:,0], coord[:,1], 'b-')
      plt.show()
      plt.clf()

    return threshold
  #endregion

  #region def _find_local_minimum(self, freq: list[float] or np.ndarray[float]) -> int:
  def _find_local_minimum(self, freq: list[float] or np.ndarray[float]) -> int:
    if len(freq) < 3:
      return None  # No local minimum possible with less than 3 elements

    for i in range(1, len(freq) - 1):
      if freq[i] < freq[i - 1] and freq[i] < freq[i + 1]:
        return i

    return None  # No local minimum found
  #endregion
  
  #region def reset_shape_layers(self, threshold: float) -> None:
  def reset_shape_layers(self, threshold: float) -> None:
    for key in self.shape_layers.keys():
      if self.shape_layers[key].get_nnz() <= threshold and not self.shape_layers[key].check_is_noise():
        ## smaller than threshold but not identified as noise previously
        ## make it noise
        self.shape_layers[key].is_noise = True
      
      if self.shape_layers[key].get_nnz() > threshold and self.shape_layers[key].check_is_noise():
        ## larger than threshold but identified as noise.
        ## construct attributes.
        self.shape_layers[key].set_up_attributes()
  #endregion

  ########################################## PROCESS COARSE SEG RESULT ########################################## 
  #region def _determine_phase_color(self, phase: np.ndarray) -> list[tuple]:
  def _determine_phase_color(self, phase: np.ndarray) -> list[tuple]:
    """
    Determine mean color of each connected component in <phase>
    Return list of tuples. Each tuple is of length 3.
    The first is color, the second is a binary image indicating the connected components.
    The third is the color in <self.colors> closest to the first element.
    """
    max_phase = np.max(phase)
    pic = np.reshape(self.pic, (self.h * self.w, 3))

    ret = list()
    for cur_phase in range(max_phase + 1): 
      cur_phase = phase == cur_phase
      labeled_im, num = label(cur_phase)
      for ptr in range(1, num + 1):
        im  = labeled_im == ptr
        ind = np.argwhere(im)
        ind = np.ravel_multi_index((ind[:,0], ind[:,1]), (self.h, self.w))

        #color = np.mean(pic[ind, :], axis = 0)
        colors, counts = np.unique(pic[ind, :], axis = 0, return_counts = True)
        color = colors[np.argmax(counts), :]

        closest_color = np.argmin(np.linalg.norm(self.colors - color, axis = 1))
        closest_color = self.colors[closest_color]
        ret.append((tuple(color), im, tuple(closest_color)))
    
    return ret
  #endregion

  #region def aggregate_shape_layers(self) -> dict:
  def aggregate_shape_layers(self) -> dict:
    if isinstance(self.coarse_seg_result, dict):
      phase = self.coarse_seg_result[phase_str]
    else:
      phase = self.coarse_seg_result
    sep_phases = self._determine_phase_color(phase)

    tbd = list()
    for key, sh_layer in self.shape_layers.items():
      im = sh_layer.get_dense_layer_im()
      color = inverse_color_transform(key[0])
      candidate_phases = filter(lambda s : s[2] == color, sep_phases)

      # check if it is "completely" embedded in one candidate phase
      # if it is, delete it from <self.shape_layers>
      for c in candidate_phases:
        c_im = c[1]
        if np.sum(np.multiply(im, c_im)) == np.sum(im):
          tbd.append(key)
          break
    
    ## add <sep_phases> to <self.shape_layers>
    for s in sep_phases:
      color = color_transformation(s[2])
      color_shape_layers = list(filter(lambda n : n[0] == color, self.shape_layers.keys()))
      max_ptr = max(color_shape_layers, key = lambda c : c[1])

      new_shape_layer = shape_layer(color, s[1], params = self.shape_layer_params)
      new_shape_layer.set_id((color, max_ptr[1] + 1))
      self.shape_layers.update({(color, max_ptr[1] + 1) : new_shape_layer})

    for key in tbd:
      self.shape_layers.pop(key)
  #endregion



class debug_util():
  """
  For debug use only.
  """
  def __init__(self, shape_layers: dict, pic: np.ndarray = None) -> None:
    self.shape_layers = shape_layers
    self.pic = pic
  
  def see_denoised_pic(self) -> None:
    denoised_pic = np.zeros_like(self.pic)
    check = np.zeros((self.pic.shape[0], self.pic.shape[1]))
    special_keys = [(1716257, 178),(10225449, 810)]
    for key in special_keys:
      val = self.shape_layers[key]
      if val.check_is_noise():
        continue
      color = val.get_color()
      if isinstance(color, int):
        color = inverse_color_transform(color)
      if not isinstance(color, np.ndarray):
        color = np.array(color)
      coord = np.argwhere(val.get_layer_im())
      check[coord[:,0], coord[:,1]] += 1
      denoised_pic[coord[:,0], coord[:,1], :] = color

    for key, val in self.shape_layers.items():
      if key == (10225449, 810) or key == (1716257, 178):
        continue
      if val.check_is_noise():
        continue
      color = val.get_color()
      if isinstance(color, int):
        color = inverse_color_transform(color)
      if not isinstance(color, np.ndarray):
        color = np.array(color)
      coord = np.argwhere(val.get_layer_im())
      check[coord[:,0], coord[:,1]] += 1
      denoised_pic[coord[:,0], coord[:,1], :] = color
    
    fig, ax = plt.subplots(1, 3, sharex = True, sharey = True)
    ax[0].imshow(self.pic)
    ax[0].set_title("original")
    ax[1].imshow(denoised_pic)
    ax[1].set_title("denoised")
    ax[2].imshow(check)
    plt.show()

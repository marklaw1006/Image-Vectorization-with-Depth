#!/usr/bin/env python3.9

# EXTERNAL
import numpy as np
from numpy_indexed import in_, indices
import matplotlib.pyplot as plt
from skimage.morphology import erosion, square 
from itertools import groupby
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes, label
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, isspmatrix
from networkx import topological_sort
from skimage.morphology import dilation
from time import time
import threading
import multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm
from random import shuffle
import warnings

#INTERNAL
from .shape_layer import shape_layer, shape_layer_factory, neighbor_color
from .new_helper import get_boundary, save_data
from .shape_layer_PCI import Partial_Convex_Init

counter_boundary_str   = "counter_orientation"
clockwise_boundary_str = "clock_orientation"
image_str              = "image"
order_graph_str        = "order_graph"
components_str         = "components"
level_str              = "level"
grid_graph_str         = "grid_graph"
self_color             = "self_color"
neighbor_color         = "neighbor_color"
inpaintable_pts_str    = "inpaintable_pts"
mutual_bd_graph_str    = "mutual_bd_graph"
new_pic_str            = "new_pic"
phase_str              = "phase"
supp_str               = "supp"
shape_layers_str       = "shape_layers"

wild_card_int   = 300 * np.power(256, 2) + 300 * 256 + 300

#region def get_points_within_r(pt: np.ndarray, r: float, shape: tuple) -> np.ndarray:
def get_points_within_r(pt: np.ndarray, r: float, shape: tuple) -> np.ndarray:
  """
  Return points that are within <r> distance from <pt>
  """
  r_int = np.int32(np.ceil(r))
  candidate_pts = [[pt[0] + x, pt[1] + y] 
                    for x in range(-r_int, r_int) for y in range(-r_int, r_int)
                    if pt[0] + x >= 0 and pt[1] + y >= 0 and
                       pt[0] + x < shape[0] and pt[1] + y < shape[1]]
  candidate_pts = np.array(candidate_pts)
  candidate_pts = candidate_pts[np.linalg.norm(candidate_pts - pt, axis = 1) <= r, :]
  return candidate_pts
#endregion

#region def find_contig_segs(arr: list[bool] or np.ndarray) -> list:
def find_contig_segs(arr: list[bool] or np.ndarray, min_len: int = 3) -> list:
  """
  Given a boolean-like 1D array <arr>, find the contiguous 1's in it.
  Include finding cyclicly contiguous 1's
  """
  if isinstance(arr, np.ndarray):
    arr = arr.tolist()
  n = len(arr)
  arr += arr # for finding cyclicly

  i = 0
  ret = list()
  for k, g in groupby(arr):
    l = len(list(g))
    if k and l > min_len:
      ret.append((i, i + l)) # append index of closed-open interval
      if i <= n - 1 and i + l > n - 1:
        ret.pop(0)
        break
    i += l
    if i > n - 1:
      break
  return ret
#endregion

def new_find_contig_segs(arr: list[bool] or np.ndarray, min_len: int = 3) -> list[tuple]:
  if isinstance(arr, np.ndarray):
    arr = arr.tolist()
  n = len(arr)

  start, end = None, None

  ret = list()
  #for i in range(2 * n):
  #  if arr[i % n]:
  #    if start is None:
  #      start = i % n
  #    end = i % n 
  #  elif start is not None:
  #    if (end - start + 1) >= min_len:
  #      ret.append((start, end))
  #    start, end = None, None

  flag = False
  for i in range(2 * n):
    if arr[i % n] and not flag:
      start = i % n
      flag = True
    elif arr[i % n] and flag:
      end = i 
    elif not arr[i % n]:
      flag = False
    
    if (not flag) and (start is not None) and (end is not None):
      if (end - start + 1) >= min_len:
        ret.append((start, end))
      start, end = None, None
  return ret



class phase_factory():
  """
  Coordinate the phase building process.
  """
  def __init__(self, D: dict, params: dict = None) -> None:
    self.D               = D
    self.shape_layers    = self.D[shape_layers_str]
    self.grid_graph      = self.D[grid_graph_str]
    self.order_graph     = self.D[order_graph_str]
    self.mutual_bd_graph = self.D[mutual_bd_graph_str]
    self.params          = params
    self.use_convex_init = False
    self.allow_multiply_connected = False

    self._load_params()

    self.level_order = list(filter(lambda n : not self.shape_layers[n].check_is_noise(), self.shape_layers.keys()))
    self.level_order = sorted(self.level_order, key = lambda n : self.shape_layers[n].get_level(), reverse = False)

    self.level_order = self.level_order[::-1]

    temp_s = next(iter(self.shape_layers.values()))
    self.h, self.w = temp_s.h, temp_s.w
    del temp_s

    self.area_threshold = 1
    self.neigh_dist_thre = 10

    print("preparing for finding regularization area")
    end = time()
    # noisy pixels
    self.wild_node = list(filter(lambda n : self.grid_graph.nodes[n][self_color][0] == wild_card_int, self.grid_graph))
    ## put key in descending depth order
    self.key_depth = list(filter(lambda key : not self.shape_layers[key].check_is_noise(), self.shape_layers.keys()))
    self.depth = np.array([self.shape_layers[key].get_level() for key in self.key_depth])
    ind = np.argsort(self.depth)[::-1]
    self.depth = self.depth[ind]
    self.key_depth = [self.key_depth[i] for i in ind]
    self.cumsum_im = np.concatenate([self.shape_layers[key].get_dense_layer_im()[:,:,np.newaxis] for key in self.key_depth], axis = 2)
    self.cumsum_im = np.cumsum(self.cumsum_im, axis = 2)

    self.wild_image = np.zeros((self.h, self.w))
    for coord in self.wild_node:
      self.wild_image[coord[0], coord[1]] = 1
    end = time() - end
    print(f"preparation finished. Used {end} seconds")

    print("setting regularization area")
    end = time()
    self.set_regularization_area(force_serial=True)
    end = time() - end
    print(f"Finding regularization area used {end} seconds")

    self.PCI = Partial_Convex_Init(self.h, self.w,
                                   self.shape_layers, 
                                   self.grid_graph, 
                                   self.mutual_bd_graph, 
                                   self.order_graph)
                                  

    print("finding inpaintable pts")
    end = time()
    self.find_inpaintable_pts(force_serial=True)
    end = time() - end
    print(f"finding inpaintable pts used {end} seconds")

    #for key in self.shape_layers.keys():
    #  pts = self.shape_layers[key].get_inpaintable_pts()
    #  fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
    #  ax[0].imshow(self.shape_layers[key].get_dense_layer_im())
    #  try:
    #    if len(pts)> 0:
    #      ax[0].plot(pts[:,1], pts[:,0], 'ro')
    #  except TypeError:
    #    plt.show()
    #    breakpoint()
    #  ax[1].imshow(self.shape_layers[key].get_reg_area())
    #  plt.show()
    #breakpoint()
  
  #region def _load_params(self) -> None:
  def _load_params(self) -> None:
    if self.params is None:
      return
    
    for key, val in self.params.items():
      if key == "r":
        pass
      elif key == "min_len":
        pass
      elif key == "ucutOff":
        pass
      elif key == "random_init":
        pass
      elif key == "thre":
        pass
      elif key == "area_threshold":
        self.area_threshold = val
      elif key == "use_convex_init":
        self.use_convex_init = val
      elif key == "allow_multiply_connected":
        self.allow_multiply_connected = val
      elif key == "ratio":
        continue
      else:
        raise KeyError(f"Unknown Parameter: {key}")
  #endregion

  #region def find_inpaintable_pts(self) -> None:
  def find_inpaintable_pts(self, force_serial: bool = False, parallel_num_thre: int = 50) -> None:
    keys = list(filter(lambda key : not self.shape_layers[key].check_is_noise() 
                                    and self.shape_layers[key].get_nnz() >= self.area_threshold, self.shape_layers.keys()))
    if force_serial or len(keys) < parallel_num_thre:
      print("[Finding inpaintable points] computing serially")
      for key in tqdm(self.shape_layers.keys()):
        #self.shape_layers[key].set_inpaintable_pts(self.get_inpaintable_points(key))
        self.shape_layers[key].set_inpaintable_pts(self.new_get_inpaintable_points(key))
      
      #for key in self.shape_layers.keys():
      #  plt.imshow(self.shape_layers[key].get_dense_layer_im())
      #  pts = self.shape_layers[key].get_inpaintable_pts()
      #  if len(pts) > 0:
      #    plt.plot(pts[:,1], pts[:,0], 'ro')
      #  plt.show()

    elif not force_serial and len(keys) >= parallel_num_thre:
      print("[Finding inpaintable points] computing parallelly")
      shuffle(keys)
      num_threads = max(cpu_count() - 1, 1)
      batch_size = np.ceil(len(keys) / num_threads).astype(int)
      divided_keys = [keys[ptr * batch_size : min((ptr + 1) * batch_size, len(keys))] for ptr in range(num_threads)]
      ret_dict = [dict() for i in range(num_threads)]
      all_threads = [threading.Thread(target = self.get_inpaintable_points_for_parallel, args = (divided_keys[ptr], ret_dict[ptr])) for ptr in range(len(divided_keys))]
      #all_threads = [threading.Thread(target = self.get_inpaintable_points_for_parallel, args = (divided_keys[ptr],)) for ptr in range(len(divided_keys))]
      [thread.start() for thread in all_threads]
      [thread.join() for thread in all_threads]
      for r in ret_dict:
        for key, val in r.items():
          self.shape_layers[key].set_inpaintable_pts(val)
  #endregion

  #region def set_regularization_area(self) -> None:
  def set_regularization_area(self, force_serial: bool = False, parallel_num_thre: int = 100) -> None:
    """
    Find regularization area for all shape layers
    Auto run parallely if too many shape layers.
    """
    keys = list(filter(lambda k : not self.shape_layers[k].check_is_noise(), self.shape_layers.keys()))
      
    if force_serial or len(keys) < parallel_num_thre:
      print("[Finding regularization area] computing serially")
      self._compute_reg_area_on_keys(keys)
    elif not force_serial and len(keys) >= parallel_num_thre:
      print("[Finding regularization area] computing parallely")
      num_threads = max(cpu_count() - 1, 1)
      batch_size = np.ceil(len(keys) / num_threads).astype(int)
      divided_keys = [keys[ptr * batch_size : min((ptr + 1) * batch_size, len(keys))] for ptr in range(num_threads)]

      ## mp manager 
      with mp.Manager() as manager:
        shared_key_depth = manager.list(self.key_depth)
        shared_cumum_im = manager.Array('d', self.cumsum_im.flatten())
        shared_wild_image = manager.Array('d', self.wild_image.flatten())
        shared_shape_layers = manager.dict()
        shared_shape_layers.update(self.shape_layers)

        args = [(key, shared_key_depth, shared_cumum_im, shared_wild_image, shared_shape_layers) for key in divided_keys]
        processes = [mp.Process(target = self._compute_reg_area_on_keys, args = (k, )) for k in args]
        [p.start() for p in processes]
        [p.join() for p in processes]
        #with mp.Pool(processes = num_threads) as pool:
        #  num = 0
        #  for r in pool.imap_unordered(self._compute_reg_area_on_keys, args):
        #    num += len(r)
        #  print(num, len(keys))

      ## mp process
      #processes = [mp.Process(target = self._compute_reg_area_on_keys, args = (k, )) for k in divided_keys]
      #[p.start() for p in processes]
      #[p.join() for p in processes]

      ## threading
      #all_threads = [threading.Thread(target = self._compute_reg_area_on_keys, args = (lst,)) for lst in divided_keys]
      #[thread.start() for thread in all_threads]
      #[thread.join() for thread in all_threads]
  #endregion

  #region def _compute_reg_area_on_keys(self, keys: list) -> None:
  def _compute_reg_area_on_keys(self, keys: list) -> None:
    for i in keys:
      ind = self.key_depth.index(i)
      image = self.cumsum_im[:,:,ind]

      reg = dilation(image, footprint = np.ones((2,2)))
      np.multiply(reg, self.wild_image, out = reg)

      np.maximum(image, reg, out = reg)

      test_im = self.shape_layers[i].get_dense_layer_im()
      reg, _ = label(reg)
      ind = np.max(np.multiply(test_im, reg))
      reg = reg == ind
      reg = binary_closing(reg).astype(float)
      self.shape_layers[i].set_reg_area(reg)
  #endregion

  #region def _compute_reg_area_on_keys_mp(self, keys: list) -> None:
  def _compute_reg_area_on_keys_mp(self, i) -> None:
    ind = self.key_depth.index(i)
    image = self.cumsum_im[:,:,ind]

    reg = dilation(image, footprint = np.ones((2,2)))
    np.multiply(reg, self.wild_image, out = reg)

    np.maximum(image, reg, out = reg)

    test_im = self.shape_layers[i].get_dense_layer_im()
    reg, _ = label(reg)
    ind = np.max(np.multiply(test_im, reg))
    reg = reg == ind
    reg = binary_closing(reg).astype(float)
    return reg
  #endregion

  #region def get_inpaintable_points_for_parallel(self, L: list[tuple]) -> None:
  def get_inpaintable_points_for_parallel(self, L: list[tuple], ret: dict) -> None:
    """
    Given <i>-th shape layer, find the points that are removable.
    Those would be the points that are adjacent to shapes that are on top of <i> shape layer
    """
    for i in tqdm(L):
      assert(i in self.shape_layers.keys())
      cmp_on_top = self._get_cmp_on_top(i)

      i_pts = self.shape_layers[i].get_clock_bd()[0]
      i_pts = list(map(tuple, i_pts))
      D = {tuple(p): False for p in i_pts}
      for j in cmp_on_top:
        for p in self.__is_neighbor_to(i_pts, j):
          D[p] = True
      pts = [list(key) for key, val in D.items() if val]
      pts = np.array(pts)
      if len(pts) == 0:
        ret.update({i: []})
        continue
      ind = np.argsort(indices(i_pts, pts, axis = 0))
      pts = pts[ind, :]
      ret.update({i: pts})
      #self.shape_layers[i].set_inpaintable_pts(pts)
  #endregion

  #region def new_get_inpaintable_points(self, i: tuple) -> np.ndarray:
  def new_get_inpaintable_points(self, i: tuple) -> np.ndarray:
    cmp_on_top = set(self._get_cmp_on_top(i))

    if not self.allow_multiply_connected:
      i_pts = self.shape_layers[i].get_clock_bd()[0].astype(int)
    else:
      i_pts = self.shape_layers[i].get_clock_bd()
      if len(i_pts) > 1:
        i_pts = np.concatenate(i_pts, axis = 0)
      else:
        i_pts = i_pts[0]
      i_pts = i_pts.astype(int)

    ret = list()
    neighbors = {p : self.grid_graph.nodes[p][neighbor_color] for p in map(tuple,i_pts)}

    for p, color in neighbors.items():
      if len(color.intersection(cmp_on_top)) > 0:
        ret.append(p)
    
    ret = np.array(ret)
    if len(ret) == 0:
      return ret
    ind = np.argsort(indices(i_pts, ret, axis = 0))
    return ret[ind, :]
  #endregion

  #region def get_inpaintable_points(self, i: tuple) -> np.ndarray:
  def get_inpaintable_points(self, i: tuple) -> np.ndarray:
    """
    Given <i>-th shape layer, find the points that are removable.
    Those would be the points that are adjacent to shapes that are on top of <i> shape layer
    """
    assert(i in self.shape_layers.keys())
    cmp_on_top = self._get_cmp_on_top(i)

    i_pts = self.shape_layers[i].get_clock_bd()[0]
    i_pts = list(map(tuple, i_pts))
    D = {tuple(p): False for p in i_pts}
    for j in cmp_on_top:
      try:
        if np.sum(self.shape_layers[i].get_convex_im().multiply(self.shape_layers[j].get_dense_layer_im())) == 0:
          continue
      except ValueError:
        print(type(self.shape_layers[i].get_convex_im()))
        breakpoint()
      for p in self.__is_neighbor_to(i_pts, j):
        D[p] = True
    pts = [list(key) for key, val in D.items() if val]
    pts = np.array(pts)
    if len(pts) == 0:
      return pts
    ind = np.argsort(indices(i_pts, pts, axis = 0))
    pts = pts[ind, :]

    return pts
  #endregion

  #region def _get_cmp_on_top(self, i: tuple) -> list[tuple]:
  def _get_cmp_on_top(self, i: tuple) -> list[tuple]:
    """
    Return the keys of shape layers that are on top of <i>
    """
    #return(list(filter(lambda key : self.components[key][level_str] > self.components[i][level_str], self.components.keys())))
    #return list(filter(lambda key : not self.shape_layers[key].check_is_noise() 
    #                                and self.shape_layers[key].get_level() > self.shape_layers[i].get_level(), 
    #                                self.shape_layers.keys()))
    return self.level_order[ : self.level_order.index(i)]
  #endregion

  #region def __is_neighbor_to(self, i_pts: list or np.ndarray, j: tuple, return_numpy: bool = False) -> list or np.ndarray:
  def __is_neighbor_to(self, i_pts: list or np.ndarray, j: tuple, return_numpy: bool = False) -> list or np.ndarray:
    """
    Auxiliary function for <_get_removable_pts>.
    """
    if isinstance(i_pts, np.ndarray): i_pts = list(map(tuple, i_pts.astype(int)))

    ret = list()
    for p in i_pts:
      nb = list(self.grid_graph.neighbors(p))
      ## is it directly neighbor to j?
      if any([self.grid_graph.nodes[n][self_color] == j for n in nb]):
        ret.append(p)
        continue

      ## is it neighbor to a noisy pixel that is neighbor to j?
      nb = list(filter(lambda x : self.grid_graph.nodes[x][self_color][0] == wild_card_int, nb))
      if any([j in self.grid_graph.nodes[n][neighbor_color] for n in nb]):
        ret.append(p)
        continue
      
      ## is it close enough to the closest point in j?
      #if np.min(np.linalg.norm(np.argwhere(self.shape_layers[j].get_dense_layer_im()) - np.array(p), axis = 1)) < self.neigh_dist_thre:
      #  ret.append(p)
      #  continue


    if return_numpy:
      return np.array(list(map(list, ret)))
    return ret
  #endregion

  #region def save(self, path: str) -> None:
  def save(self, path: str, keep_noise: bool = False) -> None:
    key_to_remove = list()
    if not keep_noise:
      for key in self.shape_layers.keys():
        if self.shape_layers[key].check_is_noise():
          key_to_remove.append(key)
    
    for key in key_to_remove:
      self.shape_layers.pop(key)

    D = {shape_layers_str: self.shape_layers, 
         grid_graph_str: self.grid_graph,
         order_graph_str: self.order_graph,
         mutual_bd_graph_str: self.mutual_bd_graph,
         }

    save_data(path, D)
  #endregion

  #region def build_phase(self, i: tuple, debug_plot: bool) -> (np.ndarray, np.ndarray, np.ndarray):
  def build_phase(self, i: tuple, debug_plot: bool) -> (np.ndarray, np.ndarray, np.ndarray):
    assert(i in self.shape_layers.keys())

    convex_init_im, convex_init_bd = None, None
    inpaintable_pts = self.shape_layers[i].get_inpaintable_pts()
    if self.use_convex_init:
      convex_init_im, _ = self.PCI.new_partial_convex_init(i)
      #convex_init_im, _ = self.PCI.partial_convex_init(i)

      temp_convex_init_im = np.maximum(convex_init_im, 0.)
      temp_convex_init_im = binary_fill_holes(temp_convex_init_im)

      convex_init_bd = get_boundary(temp_convex_init_im)[0]

      temp_convex_init_im = temp_convex_init_im.astype(float)
      temp_convex_init_im[temp_convex_init_im == 0] = -1
      convex_init_im = temp_convex_init_im

    P = phase_builder(self.shape_layers[i], inpaintable_pts, self.PCI.get_reg_area(i), 
                      convex_init_im = convex_init_im, convex_init_bd = convex_init_bd, 
                      params = self.params)
    phase, supp = P.new_get_phases()

    if debug_plot:
      print(i)
      fig, ax = plt.subplots(1, 7, sharex = True, sharey = True)
      if convex_init_im is not None:
        ax[0].imshow(convex_init_im)
        ax[0].set_title("convex init")
      else:
        ax[0].imshow(self.shape_layers[i].get_dense_layer_im())
        ax[0].set_title("original")
      ax[1].imshow(self.shape_layers[i].get_reg_area())
      ax[1].set_title("reg area")
      ax[2].imshow(phase)
      ax[2].set_title("phase")
      ax[3].imshow(supp)
      ax[3].set_title("support")
      ax[4].imshow(np.multiply(phase, self.shape_layers[i].get_reg_area()))
      ax[4].set_title("phase * reg")
      ax[5].imshow(np.multiply(supp, self.shape_layers[i].get_reg_area()))
      ax[5].set_title("support * reg")
      ax[6].imshow(self.shape_layers[i].get_dense_layer_im())
      ax[6].set_title("fid area")
      plt.show()

    return phase, supp, convex_init_im
  #endregion

  #region def build_all_phases(self, debug_plot: bool = False) -> None:
  def build_all_phases(self, debug_plot: bool = False) -> None:
    for key in self.shape_layers.keys():
      if self.shape_layers[key].check_is_noise():
        continue
      phase, supp, partial_convex_init, = self.build_phase(key, debug_plot)
      self.shape_layers[key].set_phase(phase)
      self.shape_layers[key].set_supp(supp)
      if self.use_convex_init:
        self.shape_layers[key].set_partial_convex_init(partial_convex_init)

      #self.D[components_str][key].update({phase_str: phase, supp_str: supp})
  #endregion

  #region def parallel_build_all_phases(self) -> None:
  def parallel_build_all_phases(self) -> None:
    keys = list(filter(lambda s : not self.shape_layers[s].check_is_noise, self.shape_layers.keys()))
    num_thread = max(cpu_count() - 1, 1)
    batch_size = np.ceil(len(keys) / num_thread).astype(int)
    divided_keys = [keys[ ptr * batch_size : min( (ptr + 1) * batch_size , len(keys))] for ptr in range(num_thread)]
    ret_dict = [dict() for _ in range(num_thread)]
    all_threads = [threading.Thread(target = self._build_phase_parallel_wrapper, args = (divided_keys[ptr], ret_dict[ptr], )) for ptr in range(len(divided_keys))]
    [thread.start() for thread in all_threads]
    [thread.join() for thread in all_threads]
    for r in ret_dict:
      for key, val in r:
        self.shape_layers[key].set_phase(val[0])
        self.shape_layers[key].set_supp(val[1])
        if self.use_convex_init:
          self.shape_layers[key].set_partial_convex_init(val[2])
  #endregion
  
  #region def _build_phase_parallel_wrapper(self, keys: list) -> None:
  def _build_phase_parallel_wrapper(self, keys: list, ret: dict) -> None:
    for key in keys: 
      phase, supp, partial_convex_init = self.build_phase(key)
      #self.shape_layers[key].set_phase(phase)
      #self.shape_layers[key].set_supp(supp)
      #if self.use_convex_init:
      #  self.shape_layers[key].set_partial_convex_init(partial_convex_init)
      ret.update({key: (phase, supp, partial_convex_init)})
  #endregion
      

class phase_builder():
  def __init__(self, shape_layer: shape_layer, inpaintable_pts: np.ndarray, 
               reg_area: np.ndarray or csr_matrix, convex_init_im: np.ndarray = None,
               convex_init_bd: np.ndarray = None, params: dict = None) -> None:
    self.shape_layer     = shape_layer
    self.convex_init_im  = convex_init_im
    self.fid_bd          = self.shape_layer.get_clock_bd()
    self.bd_dtype        = self.fid_bd[0].dtype
    self.reg_area = self.filter_reg_area(reg_area)
    self.use_convex_init = convex_init_im is not None and convex_init_bd is not None
    self.allow_multiply_connected = False

    if convex_init_bd is not None:
      self.convex_init_bd  = convex_init_bd.astype(self.bd_dtype)
    self.inpaintable_pts = inpaintable_pts.astype(self.bd_dtype)

    #default parameters
    self.r = 6.
    self.min_len = 0
    self.ucutOff = 1 / 12
    self.use_random_init = False
    self.thre = .5
    self.area_threshold = 1
    self.ratio = .95 # geometric ratio applied to calculating tangent vectors.
    self.use_concave_corners = False

    self._load_parameters(params)
  
  #region def _load_parameters(self, params: dict) -> None:
  def _load_parameters(self, params: dict) -> None:
    if params is None:
      return 
    
    for key, val in params.items():
      if key == "r":
        self.r = val
      elif key == "min_len":
        self.min_len = val
      elif key == "ucutOff":
        self.ucutOff = val
      elif key == "random_init":
        self.use_random_init = val
        assert(isinstance(self.use_random_init, bool))
      elif key == "thre":
        self.thre = val
      elif key == "ratio":
        self.ratio = val
      elif key == "area_threshold":
        self.area_threshold = val
      elif key == "use_concave_corners":
        self.use_concave_corners = val
        if self.use_concave_corners:
          warnings.warn("Computing phase on concave corners is buggy.")
      elif key == "use_convex_init":
        self.use_convex_init = val
      elif key == "allow_multiply_connected":
        self.allow_multiply_connected = val
      else:
        raise KeyError(f"Unknown parameter: {key}")
    
    if self.allow_multiply_connected:
      self.fid_bd = [bd.astype(self.bd_dtype) for bd in self.shape_layer.get_clock_bd()]
  #endregion
  
  #region def filter_reg_area(self, reg_area: np.ndarray) -> np.ndarray:
  def filter_reg_area(self, reg_area: np.ndarray) -> np.ndarray:
    """
    Since regularization area may be disconnected, only pick the one that is connected to the original shape layer.
    """
    labeled_im, num = label(reg_area)

    ret = None
    for ptr in range(1, num + 1):
      cur_im = labeled_im == ptr
      if np.any(np.multiply(cur_im, self.shape_layer.get_dense_layer_im())):
        ret = cur_im
    return ret
  #endregion

  #region def new_get_phases(self, use_convex_init: bool = True) -> (np.ndarray, np.ndarray):
  def new_get_phases(self) -> (np.ndarray, np.ndarray):
    """
    Unlike <get_phases>, instead of computing the normal(which appears to be prone to bug), 
    we find the convex corners and take the (weighted) tangent vectors to calculate the phase.
    Support calculation remains the same.
    """
    if self.use_convex_init: 
      im = self.convex_init_im.copy()
      bd = self.convex_init_bd
    else:
      im = self.shape_layer.get_dense_layer_im()
      bd = self.fid_bd
      im[im == 0] = -1

    #im_interior = erosion(im, square(3))
    #im_diff     = self.simple_shape_diffusion(im, im_interior) # slightly diffused image

    convex_pts = self.find_convex_corners(im, min_len = self.min_len)

    if self.use_concave_corners:
      concave_pts = self.find_concave_corners()
    else:
      concave_pts = list()

    # find supp first
    supp = self._new_get_supp(im, convex_pts, concave_pts, self.reg_area)

    # compute the phase
    phase = self._new_compute_phase(im, convex_pts, bd, concave_pts)

    return phase, supp
  #endregion

  #region def _new_get_supp(self, im: np.ndarray, convex_pts: np.ndarray, concave_pts :np.ndarray, reg_area: np.ndarray) -> np.ndarray:
  def _new_get_supp(self, im: np.ndarray, convex_pts: np.ndarray, concave_pts: np.ndarray, reg_area: np.ndarray) -> np.ndarray:
    supp = np.zeros_like(im)

    # convex points 
    points = [get_points_within_r(p, self.r, supp.shape) for p in convex_pts]

    for pts in points:
      for p in pts:
        supp[p[0], p[1]] = 1
    
    ## concave points
    if self.use_concave_corners:
      points = [get_points_within_r(p, self.r, supp.shape) for p in concave_pts]

      for pts in points:
        for p in pts:
          supp[p[0], p[1]] = 1

    supp[im > 0] = 0

    return supp
  #endregion

  #region def _new_compute_phase(self, im: np.ndarray, convex_pts: np.ndarray, bd: list[np.ndarray], concave_pts: np.ndarray) -> np.ndarray:
  def _new_compute_phase(self, im: np.ndarray, convex_pts: np.ndarray, bd: list[np.ndarray], concave_pts: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(im)

    for pt in convex_pts:
      prev_pts, post_pts       = self._get_prev_post_points(bd, pt, self.r)
      # compute vectors
      prev_vecs, post_vecs     = prev_pts - pt, post_pts - pt
      prev_weight, post_weight = np.array([self.ratio ** l for l in range(len(prev_vecs))]), np.array([self.ratio ** l for l in range(len(post_vecs))])
      prev_vecs, post_vecs     = np.multiply(prev_vecs, prev_weight[:, np.newaxis]), np.multiply(post_vecs, post_weight[:, np.newaxis])
      prev_vecs, post_vecs     = np.sum(prev_vecs, axis = 0), np.sum(post_vecs, axis = 0) 
      prev_vecs, post_vecs     = -prev_vecs / (1e-4 + np.linalg.norm(prev_vecs)), -post_vecs / (1e-4 + np.linalg.norm(post_vecs))

      # get points within r
      neighbor_pts = get_points_within_r(pt, self.r, phase.shape)
      # filter out points already in <im>
      neighbor_pts = np.array(list(filter(lambda p : im[p[0], p[1]] == -1, neighbor_pts)))
      # find the points that are between <prev_vecs> and <post_vecs>
      neighbor_pts -= pt
        
      prev_sign = np.sign(neighbor_pts[:,0] * prev_vecs[1] - neighbor_pts[:,1] * prev_vecs[0])
      post_sign = np.sign(post_vecs[0] * neighbor_pts[:,1] - post_vecs[1] * neighbor_pts[:,0])
      ind = np.logical_and(prev_sign == 1, post_sign == 1)

      pos_pts = neighbor_pts[ind, :] + pt
      ind = [not i for i in ind]
      neg_pts = neighbor_pts[ind, :] + pt

      for p in pos_pts:
        phase[p[0], p[1]] = 1

      for p in neg_pts:
        phase[p[0], p[1]] = -1

    if self.use_concave_corners:
      reg_bd = get_boundary(self.reg_area)[0]
      for pt in concave_pts:
        neighbor_pts = get_points_within_r(pt, self.r, phase.shape)

        prev_pts, post_pts       = self._get_prev_post_points(reg_bd, pt, self.r)
        # compute vectors
        prev_vecs, post_vecs     = prev_pts - pt, post_pts - pt
        prev_weight, post_weight = np.array([self.ratio ** l for l in range(len(prev_vecs))]), np.array([self.ratio ** l for l in range(len(post_vecs))])
        prev_vecs, post_vecs     = np.multiply(prev_vecs, prev_weight[:, np.newaxis]), np.multiply(post_vecs, post_weight[:, np.newaxis])
        prev_vecs, post_vecs     = np.sum(prev_vecs, axis = 0), np.sum(post_vecs, axis = 0) 
        prev_vecs, post_vecs     = -prev_vecs / np.linalg.norm(prev_vecs), -post_vecs / np.linalg.norm(post_vecs)

        reg1, reg2 = list(), list()
        for p in neighbor_pts:
          if self.reg_area[p[0], p[1]] < 1:
            # points not in reg area
            phase[p[0], p[1]] = 0
          
          v = p - pt
          if self.reg_area[p[0], p[1]] == 1 and np.sign(prev_vecs[0] * v[1] - prev_vecs[1] * v[0]) == -1:
            reg1.append(p)

          if self.reg_area[p[0], p[1]] == 1 and np.sign(post_vecs[0] * v[1] - post_vecs[1] * v[0]) == 1:
            reg2.append(p)

        reg1, reg2 = np.array(reg1), np.array(reg2)

        center1, center2 = np.mean(reg1, axis = 0), np.mean(reg2, axis = 0)

        dist1, dist2 = np.mean(np.linalg.norm(bd - center1, axis = 1)), np.mean(np.linalg.norm(bd - center2, axis = 1))

        if dist1 > dist2:
          for p in reg1:
            phase[p[0], p[1]] = 0
        else:
          for p in reg2:
            phase[p[0], p[1]] = 0

    return phase
  #endregion

  #region def get_phases(self, use_convex_init: bool = True):
  def get_phases(self, use_convex_init: bool = True):
    if use_convex_init: 
      im = self.convex_init_im.copy()
      bd = self.convex_init_bd
    else:
      im = self.shape_layer[image_str]
      bd = self.fid_bd
    im_interior = erosion(im, square(3))
    im_diff     = self.simple_shape_diffusion(im, im_interior) # slightly diffused image

    convex_pts = self.find_convex_corners(im, min_len = self.min_len)

    prev_normal, post_normal = self.compute_normal_at_convex_corners(im_diff, bd, convex_pts)

    package = list(zip(convex_pts, prev_normal, post_normal))

    phase, supp = self.build_phases(package, self.r, self.convex_init_im)
    return phase, supp
  #endregion

  #region def find_convex_corners(self, im: np.ndarray, min_len: int) -> np.ndarray:
  def find_convex_corners(self, im: np.ndarray, min_len: int, 
                          simple_closing: bool = True, footprint: int = 3) -> np.ndarray:
    """
    Find the convex corners for defining corner phase and support.
    Those are the points which are immediately before and after the inpaintable points
    along the fidelity boundary.
    """
    if len(self.inpaintable_pts) == 0:
      return np.array([])

    ret = list()
    for bd in self.fid_bd:
      #flag = in_(self.fid_bd[0], self.inpaintable_pts, axis = 0)
      flag = in_(bd, self.inpaintable_pts, axis = 0)
      if simple_closing:
        flag = binary_closing(flag, np.ones(footprint))
      segs = new_find_contig_segs(flag, min_len = min_len)

      ## get each segment's begining and ending point.
      test_pts = list()
      for s in segs:
        for ptr in range(2):
          ind = s[ptr]

          #### Some how this needs to be commented out for illusory
          #if ptr == 1:
          #  ## because <find_contig_segs> returns closed-open intervals.
          #  ind -= 1

          ind %= len(bd)
          #test_pts.append(self.fid_bd[0][ind, :])
          test_pts.append(bd[ind, :])
      test_pts = np.array(test_pts)
      if len(test_pts) == 0:
        return np.array([])

      ## test_pts are still on <self.fid_bd>. Project onto <self.convex_init_bd>.
      if self.use_convex_init:
        ind = np.argmin(cdist(test_pts, self.convex_init_bd), axis = 1)
        
        test_pts = self.convex_init_bd[ind, :]
        ind = list(map(lambda p : self._is_convex(p, im, self.r), test_pts))
        #return test_pts[ind, :]
        ret.append(test_pts)
      else:
        #return test_pts
        ret.append(test_pts)

    #plt.imshow(im)
    #if len(self.inpaintable_pts) > 0:
    #  plt.plot(self.inpaintable_pts[:,1], self.inpaintable_pts[:,0], 'bo')
    #for r in ret:
    #  if len(r) > 0:
    #    plt.plot(r[:,1], r[:,0], 'ro')
    #plt.show()

    return np.concatenate(ret, axis = 0)

  #endregion

  #region def find_concave_corners(self, shift: int = 5, angle_thre: float = np.pi / 6) -> np.ndarray:
  def find_concave_corners(self, shift: int = 5, angle_thre: float = np.pi / 6) -> np.ndarray:
    """
    Find the concave corners on the boundary of regularization region
    """
    if isspmatrix(self.reg_area):
      reg_area = self.reg_area.toarray()
    else:
      reg_area = self.reg_area

    bd = get_boundary(reg_area)[0]
    if len(bd) <= shift:
      shift = 1

    prev_tan = bd - np.roll(bd, -shift, axis = 0)
    prev_tan = np.divide(prev_tan, np.linalg.norm(prev_tan, axis = 1)[:, np.newaxis])
    post_tan = np.roll(bd, shift, axis = 0) - bd
    post_tan = np.divide(post_tan, np.linalg.norm(post_tan, axis = 1)[:, np.newaxis])

    cross_product = np.multiply(post_tan[:,0], prev_tan[:,1]) - np.multiply(post_tan[:,1], prev_tan[:,0])
    flag = cross_product < 0

    np.abs(cross_product, out = cross_product)
    np.minimum(cross_product, 1, out = cross_product)

    angle = np.arcsin(cross_product)
    ind = np.logical_and(flag, angle > angle_thre)

    ind = self._filter_concave_corner(ind, bd, reg_area)

    return bd[ind, :]
  #endregion

  #region def _filter_concave_corner(self, ind: list[bool], bd: np.ndarray, reg_area: np.ndarray, min_len: int = 3) -> list or np.ndarray:
  def _filter_concave_corner(self, ind: list[bool], bd: np.ndarray, reg_area: np.ndarray, min_len: int = 3) -> list or np.ndarray:
    """
    Since concave corners may be clustered together, find the one that "is the most significant"
    Find by computing the ratio of area.
    """
    ret = list()
    for start_ind, end_ind in find_contig_segs(ind, min_len = min_len):
      max_area_ratio = -np.inf
      max_ind = None
      for ptr in range(start_ind, end_ind):
        ptr %= len(bd)
        neighbor_points = get_points_within_r(bd[ptr, :], self.r, reg_area.shape)

        cur_ratio = 0
        for p in neighbor_points:
          if reg_area[p[0],p[1]] > 0:
            cur_ratio += 1
        cur_ratio /= len(neighbor_points)

        if cur_ratio > max_area_ratio:
          max_area_ratio = cur_ratio
          max_ind = ptr
      ret.append(max_ind)
    return ret
  #endregion

  #region def _is_convex(self, pt: np.ndarray, im: np.ndarray, r: float, 
  def _is_convex(self, pt: np.ndarray, im: np.ndarray, r: float, return_ratio: bool = False) -> bool:
    """
    Given a binary image <im>, test if <pt> is a convex corner.
    """
    neigh_pts = get_points_within_r(pt, r, im.shape).astype(np.int32)
    in_im_pts = list(filter(lambda p : im[p[0], p[1]] == 1 , neigh_pts))

    r = (len(in_im_pts) / len(neigh_pts))
    if not return_ratio:
      return  r < self.thre - self.ucutOff and r > self.ucutOff
    else:
      return  (r < self.thre - self.ucutOff and r > self.ucutOff), r
  #endregion

  #region def simple_shape_diffusion(self, im: np.ndarray, im_interior: np.ndarray, 
  def simple_shape_diffusion(self, im: np.ndarray, im_interior: np.ndarray, 
                             l: float = .2, nstep: int = 4, optDiff: np.ndarray = None) -> np.ndarray:
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

  #region def compute_normal(self, im: np.ndarray, pts: np.ndarray) -> np.ndarray:
  def compute_normal(self, im: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Compute normal on <pts> in <im>.
    Input: 
          im: np.ndarray. input image
          pts: n by 2 np.ndarray. coordinates of inquired points.
    Output:
          vec: n by 2 np.ndarray. Read along rows.
    """
    pts = pts.astype(np.int32)
    x, y = pts[:,0], pts[:,1]
    im_ext = np.pad(im, 1, 'constant', constant_values = 0)
    grad_x = .5 * (im_ext[ x + 1 , y ] - im_ext[ x - 1 , y])
    grad_y = .5 * (im_ext[ x , y + 1 ] - im_ext[ x , y - 1])

    norm = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2) + 1e-5)
    outer_normal_x = - np.divide(grad_x, norm)
    outer_normal_y = - np.divide(grad_y, norm)
    vec = np.column_stack((outer_normal_x, outer_normal_y))
    return vec
  #endregion

  #region def compute_normal_at_convex_corners(self, im: np.ndarray, convex_corners: np.ndarray, 
  def compute_normal_at_convex_corners(self, im: np.ndarray, bd: np.ndarray, convex_corners: np.ndarray, 
                                       weight_decay: float = 4. ) -> (np.ndarray, np.ndarray):
    """
    <self.compute_normal> computes normal at specified points.
    This function calculates previous and post normal vector with weight decay at convex corners.
    Input:
          im: np.ndarray. Should be a slightly diffused one
          convex_corners: n by 2 np.ndarray. 
          weight_decay: flaot
    Output:
          prev_normal: n by 2 np.ndarray
          post_normal: n by 2 np.ndarray
    """
    prev_normal, post_normal = list(), list()
    for p in convex_corners:
      prev_pts, post_pts = self._get_prev_post_points(bd, p, self.r)

      prev_normals = self.compute_normal(im, prev_pts)
      post_normals = self.compute_normal(im, post_pts)

      prev_vec = self._compute_weighted_normal(p, prev_pts, prev_normals, weight_decay, self.r)
      post_vec = self._compute_weighted_normal(p, post_pts, post_normals, weight_decay, self.r)

      prev_normal.append(prev_vec)
      post_normal.append(post_vec)

    return np.array(prev_normal), np.array(post_normal)
  #endregion
  
  #region def _get_prev_post_points(self, bd: list[np.ndarray], pt: np.ndarray, r: float) -> (np.ndarray, np.ndarray):
  def _get_prev_post_points(self, bd: list[np.ndarray], pt: np.ndarray, r: float) -> (np.ndarray, np.ndarray):
    """
    Auxiliary function for <compute_normal_at_convex_corners>.
    Given boundary <bd>, find the point before and after <pt> that are within <r>.
    Input:
          bd: n by 2 np.ndarray. Boundary with an orientation.(Changed to a list of bd, for when allowing multiply connected regions to be one shape layer)
          pt: coordinate of a point. Should be in <bd>
          r: distance
    Output:
          prev_pts: m by 2 np.ndarray. Points before <pt> that are within distance <r>. Return in the order
                       from the closest(top) to the furthest(bottom).
          post_pts: k by 2 np.ndarray. Points after <pt> that are within distance <r>. Return in the order
                       from the closest(top) to the furthest(bottom).
    """
    temp_bd = bd.copy()
    for bd in temp_bd:
      if not any(in_(bd, [pt], axis = 0)):
        continue
      ind = indices(bd, [pt], axis = 0)[0]
      n = len(bd)
      prev_pts, post_pts = list(), list()

      ## prev_pts first
      ptr = ind - 1
      num = 0
      while np.linalg.norm(bd[ptr, :] - pt) <= r:
        prev_pts.append(bd[ptr, :])
        ptr -= 1
        num += 1
        if num >= min(len(bd) - 1 , 20): 
          break
      
      ## post_pts
      ptr = (ind + 1) % n
      num = 0
      while np.linalg.norm(bd[ptr, :] - pt) <= r:
        post_pts.append(bd[ptr, :])
        ptr = (ptr + 1) % n
        num += 1
        if num >= min(len(bd) - 1 , 20): 
          break

      prev_pts, post_pts = np.vstack(prev_pts), np.vstack(post_pts)
      
    return prev_pts, post_pts
  #endregion
  
  #region def _compute_weighted_normal(self, pt: np.ndarray, pts: np.ndarray, 
  def _compute_weighted_normal(self, pt: np.ndarray, pts: np.ndarray, 
                               normal: np.ndarray, weight_decay: float, r: float) -> np.ndarray:
    """
    Auxiliary function for <compute_normal_at_convex_corners>.
    Given the inquired point <pt>, and its prev/post points <pts> and normal vectors <normal>, compute
    the weighted normal vector with weight decay <weight_decay>.
    Input:
          pt: 1 by 2 np.ndarray. The inquired point.
          pts: m by 2 np.ndarray. The points before/after <pt>. They should be from closest(top) to furthers(bottom)
          normal: m by 2 np.ndarray. normal vectors. correspond to points in <pts>.
          weight_decay: float. weight decay in Gaussian
          r: float. For weight decay.
    Output:
          vec: 1 by 2 np.ndarray. vector
    """
    center = np.mean(pts, axis = 0)
    dist = np.sum(np.power(pts - center, 2), axis = 1)

    weight = np.exp(4 * (- weight_decay * dist / (r ** 2)))

    vec = np.sum(np.multiply(normal, weight[:,np.newaxis]), axis = 0) / np.sum(weight)

    ## normalization
    vec /= np.linalg.norm(vec)
    return vec
  #endregion

  #region def build_phases(self, package: list[tuple], r: float, init_im: np.ndarray) -> (np.ndarray, np.ndarray):
  def build_phases(self, package: list[tuple], r: float, init_im: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Given the points and their pre/post normal vectors, build the corner base and support
    Input:
          package: list of corners and pre/post normal vectors
          r: float. Corner disk's radius.
          init_im: np.ndarray. Initialization. Binary image.
    Output: 
          phase: self.h by self.w np.ndarray.
          supp: self.h by self.w np.ndarray.
    """
    phase, supp = np.zeros_like(self.convex_init_im), np.zeros_like(self.convex_init_im)
    if np.sum(self.shape_layer.get_dense_layer_im() > 0) < self.area_threshold:
      phase -= 1
      phase[init_im == 1] = 0
      return phase, supp

    for p in package:
      cur_corner, prev_normal, post_normal = p[0], p[1], p[2]
      cur_corner = cur_corner.astype(np.int32)

      points = get_points_within_r(cur_corner, r, init_im.shape)

      pos_points = list(filter(lambda p : np.dot(p - cur_corner, prev_normal) >= 0 
                               and np.dot(p - cur_corner, post_normal) >= 0, points))

      neg_points = list(filter(lambda p : not(np.dot(p - cur_corner, prev_normal) >= 0 
                               and np.dot(p - cur_corner, post_normal) >= 0), points))

      pos_points, neg_points = np.array(pos_points).astype(np.int32), np.array(neg_points).astype(np.int32)

      phase[pos_points[:,0], pos_points[:,1]] = 1.
      phase[neg_points[:,0], neg_points[:,1]] = -1.

      points = points.astype(np.int32)

      supp[points[:,0], points[:,1]] = 1.
    
    phase[init_im > 0] = 0.

    if self.use_random_init:
      np.random.seed(8964)
      rand_init = 2 * (np.random.rand(*self.convex_init_im.shape) - .5)
      ind = np.logical_and(init_im == -1, phase == 0)
      phase[phase < 1] = -1
      phase[init_im == 1] = 0
      phase[ind] = rand_init[ind]
    else:
      phase[phase < 1] = -1
      phase[init_im == 1] = 0
    
    supp[init_im == 1] = 0
    
    return phase, supp
  #endregion 

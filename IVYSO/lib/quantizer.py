#!/usr/bin/env python3.9

# external
from os.path import join
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.morphology import remove_small_holes
from scipy.spatial.distance import cdist
from scipy.ndimage import label
import networkx as nx
from PIL import ImageFilter, Image

# internal
from .new_helper import image_transformation, color_transformation, inverse_color_transform, image_inverse_color_transform, smart_convert_sparse
from .helper import save_data, load_data

methods = ['kmeans', 'hist', 'chromo', 'reg_kmeans']
self_color = "self_color"
is_noise_str = "is_noise"

## breakpoint

##################################################################### 
######################## AUXILIARY FUNCTIONS ######################## 
##################################################################### 
#region def batch_arg_cdist(XA: np.ndarray, XB: np.ndarray, batch: int = 500) -> np.ndarray:
def batch_arg_cdist(XA: np.ndarray, XB: np.ndarray, batch: int = 500) -> np.ndarray:
  """
  Auxiliary function to  <__spherical_kmean>
  Compute <cdist> argmin along axis = 0 in batch.
  Use great circle distance.
  XA must be observations, and XB must be centers.
  Assume len(XB) << batch, so no need to do batch on <XB>.
  """
  assert(np.ndim(XA) == 2 and np.ndim(XB) == 2)
  assert(XA.shape[1] == XB.shape[1])
  assert(XA.shape[0] >= XB.shape[0])
  ret = list()
  for ptr in range((len(XA) // batch) + 1):
    cur_XA_mat = XA[ptr * batch : min((ptr + 1) * batch , XA.shape[0]), :]
    ret.append(np.argmin(cdist(cur_XA_mat, XB, lambda x, y : np.arccos(np.dot(x, y))), axis = 1))

  return np.concatenate(ret)
#endregion

#region def update_centers(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
def update_centers(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
  """
  Auxiliary function for <__spherical_kmean>
  Update centers.
  """
  ll              = np.sort(np.unique(labels).astype(int))
  centers         = np.vstack([np.mean(X[labels == l], axis = 0) for l in ll])
  norm            = np.linalg.norm(centers, axis = 1)
  norm[norm == 0] = 1.
  norm            = np.column_stack([norm] * 3)
  centers         = np.divide(centers, norm)
  return centers
#endregion

def hex2rgb(hex: str) -> tuple[int]:
  hex = hex.lstrip('#')
  return tuple(int(hex[i:i+2],16) for i in [0,2,4])

class quantizer():
  def __init__(self, pic: np.ndarray, num_color: int, method = methods[0],
               bgr2rgb = True, random_state = 721, threshold = 40, chromaticity: bool = False, denoise = True,
               denoise_threshold: int = 30, denoise_dist_thre: float = 3., l: float = 0.02, already_quantized: bool = False,
               color_palette: list[str] = None, use_color_palette: bool = False, use_mode_filter: bool = True,
               mode_filter_times: int = 5):
    self.method = method
    assert(self.method in methods)
    #pic = cv.bilateralFilter(pic, 30, 50, 50)
    self.pic               = pic
    self.h, self.w         = self.pic.shape[:2]
    self.num_color         = num_color
    self.threshold         = threshold
    self.chromaticity      = chromaticity
    self.denoise_threshold = denoise_threshold
    self.denoise_dist_thre = denoise_dist_thre # only noisy nodes that are within this threshold from all three/four regions would be reassigned.
    self.random_state      = random_state
    self.l                 = l
    self.already_quantized = already_quantized # is the pic already quantized?
    self.denoise           = denoise
    self.color_palette     = color_palette
    self.use_color_palette = use_color_palette
    self.use_mode_filter   = use_mode_filter
    self.mode_filter_times = mode_filter_times

    if self.use_mode_filter:
      filtered_im = Image.fromarray(np.uint8(self.pic)).convert('RGB')
      filtered_im = filtered_im.filter(ImageFilter.ModeFilter)
      for _ in range(self.mode_filter_times - 1):
        filtered_im = filtered_im.filter(ImageFilter.ModeFilter)
      #filtered_im = np.asarray(filtered_im).astype(np.uint8)
      self.pic = np.asarray(filtered_im).astype(np.uint8)

    if self.use_color_palette and self.color_palette is None:
      raise ValueError("empty color palette")

    if bgr2rgb: self.pic = cv.cvtColor(self.pic, cv.COLOR_BGR2RGB)

  #region def quantize(self, debug_plot: bool = False):
  def quantize(self, debug_plot: bool = False):
    if self.already_quantized:
      #pic = np.reshape(self.pic, (self.h * self.w, 3))
      #pic, colors,layers = self.get_info()
      #if self.denoise:
      #  pic, colors = self._new_denoise_pic(pic, colors)
      #return pic, colors, layers
      pic = np.reshape(self.pic, (self.h * self.w, 3))
      colors = np.unique(pic, axis = 0)
      if self.denoise:
        pic, colors = self._new_denoise_pic(pic, colors)
      pic = np.reshape(pic, (self.h, self.w, 3))
      layers = self._quantize_layers(pic, colors)
      return pic, colors, layers
    elif self.use_color_palette:
      pic, colors, layers = self.get_info_with_color_palette()
      return pic, colors, layers
    elif self.method == methods[0]:
      pic, colors, layers = self.kmean_quantize(debug_plot)
    elif self.method == methods[1]: 
      pic, colors, layers = self.imhist_quantize()
    elif self.method == methods[2]: 
      pic, colors, layers = self.chromo_quantize()
    elif self.method == methods[3]: 
      pic, colors, layers = self.reg_kmean_quantize()

    return pic, colors, layers
  #endregion

  ########################### DENOISING ########################### 
  #region def _denoise_pic(self, input_pic: np.ndarray, colors: list[tuple], debug_plot = False) -> np.ndarray:
  def _denoise_pic(self, input_pic: np.ndarray, colors: list[tuple], debug_plot = False) -> np.ndarray:
    """
    Simple Denoising: for any small connected components with pixels less than <self.denoise_threshold>, get rid of them
    """
    if debug_plot:
      fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
      ax[0].imshow(input_pic)

    pic = image_transformation(input_pic).copy()
    while True:
      flag = False
      for color in map(color_transformation, colors):
        cur_bin_pic = pic == color
        new_bin_pic = remove_small_holes(cur_bin_pic, area_threshold = self.denoise_threshold)
        chg_pic = np.abs(new_bin_pic ^ cur_bin_pic)
        if np.any(chg_pic):
          pic[np.where(chg_pic)] = color
          flag = True
          break

      if not flag:
        ## if the currect picture does not change anymore, break
        break

    if debug_plot:
      ax[1].imshow(image_inverse_color_transform(pic))
      plt.show()

    return image_inverse_color_transform(pic)
  #endregion

  #region def _new_denoise_pic(self, input_pic: np.ndarray, colors: np.ndarray, debug_plot: bool = False) -> np.ndarray:
  def _new_denoise_pic(self, input_pic: np.ndarray, colors: np.ndarray, debug_plot: bool = False) -> np.ndarray:
    """
    For noisy regions that are close to T-junctions, assign them to the closest color in the sense of chromaticity.
    Return the new pic.
    """
    pic = image_transformation(input_pic)
    colors_dict = {color_transformation(c) : c for c in colors}
    colors = [color_transformation(c) for c in colors]

    grid_graph = self.__create_grid_graph(pic, colors)

    nodes = self.__find_nodes_to_reassign(grid_graph)

    denoised_pic = self.__reassign_nodes(input_pic, grid_graph, nodes, colors_dict, use_chromaticity = True)
    colors = denoised_pic.reshape((-1, denoised_pic.shape[2]))
    colors = np.unique(colors, axis = 0).astype(int)
    return denoised_pic, colors
  #endregion
   
  #region def __create_grid_graph(self, pic: np.ndarray, colors: list[int]) -> nx.classes.graph.Graph:
  def __create_grid_graph(self, pic: np.ndarray, colors: list[int]) -> nx.classes.graph.Graph:
    """
    Return a grid graph with each node in the form of (transformed_color, identifier).
    Each node has two attribute: <self_color> and <is_noise_str>
    """
    grid_graph = nx.grid_graph((self.w, self.h))

    node_color_dict = dict()
    for color in colors:
      labeled_array, num_labels = label(pic == color)
      for l in range(1, num_labels + 1):
        flag = np.sum(labeled_array == l) < self.denoise_threshold
        node_color_dict.update({c: {self_color: (color, l), is_noise_str: flag}
                                    for c in map(tuple, np.argwhere(labeled_array == l))})
    
    nx.set_node_attributes(grid_graph, node_color_dict)
    return grid_graph
  #endregion
  
  #region def __find_nodes_to_reassign(self, grid_graph: nx.classes.graph.Graph, debug_plot: bool = False) -> list[tuple]:
  def __find_nodes_to_reassign(self, grid_graph: nx.classes.graph.Graph, debug_plot: bool = False) -> list[tuple]:
    """
    Return the nodes that need their color reassigned.
    """
    noisy_nodes = filter(lambda n : grid_graph.nodes[n][is_noise_str], grid_graph.nodes)
    noisy_graph = nx.subgraph(grid_graph, noisy_nodes)

    ret = list()
    for c in nx.connected_components(noisy_graph):
      cmp_neighbor_color = set()
      # add all neighbor colors
      for n in c:
        cmp_neighbor_color = cmp_neighbor_color.union({grid_graph.nodes[neigh][self_color] for neigh in grid_graph.neighbors(n)})
      # take away all noisy pixels' colors
      cmp_neighbor_color = cmp_neighbor_color.difference({grid_graph.nodes[n][self_color] for n in c})
      if len(cmp_neighbor_color) >= 3:
        #ret += list(c)
        ret += self.__filter_noisy_nodes_within_dist(cmp_neighbor_color, list(c), grid_graph)

    if debug_plot:
      plt.clf()
      _, ax = plt.subplots(1, 2, sharex = True, sharey = True)
      ax[0].set_title("All noisy pixels")
      ax[1].set_title("Noisy pixels to be reassigned")
      all_noisy_pixels = np.array(list(noisy_graph.nodes))
      im = np.zeros((self.h, self.w))
      for p in all_noisy_pixels:
        im[p[0], p[1]] = 1
      ax[0].imshow(im)

      im = np.zeros((self.h, self.w))
      for p in ret:

        im[p[0], p[1]] = 1
      ax[1].imshow(im)
      plt.show()
      breakpoint()

    return ret
  #endregion

  #region def __filter_noisy_nodes_within_dist(self, cmp_neighbor_color: set, c: list, grid_graph: nx.classes.graph.Graph) -> list:
  def __filter_noisy_nodes_within_dist(self, cmp_neighbor_color: set, c: list, grid_graph: nx.classes.graph.Graph) -> list:
    ## find all nodes with <self_color> in <cmp_neighbor_color>
    nodes_dict = {c : np.array(list(filter(lambda n : grid_graph.nodes[n][self_color] == c, grid_graph.nodes))) for c in cmp_neighbor_color}

    npc = np.array(c)
    flags = [True] * len(c)
    for key in nodes_dict.keys():
      flags = [x and y for x, y in zip(flags, list(np.min(cdist(npc, nodes_dict[key]), axis = 1) < self.denoise_dist_thre))]
    
    return [c[ptr] for ptr in range(len(flags)) if flags[ptr]]
  #endregion
  
  #region def __reassign_nodes(self, pic: np.ndarray, grid_graph: nx.classes.graph.Graph, nodes: list[tuple], colors_dict: dict,
  def __reassign_nodes(self, pic: np.ndarray, grid_graph: nx.classes.graph.Graph, nodes: list[tuple], colors_dict: dict,
                             use_chromaticity: bool = True, debug_plot: bool = False) -> np.ndarray:
    pic = image_transformation(pic)
    for n in nodes:
      possible_colors = [grid_graph.nodes[neigh][self_color][0] 
                         for neigh in grid_graph.neighbors(n) 
                         if not grid_graph.nodes[neigh][is_noise_str]]
      new_color = self.__assign_color(grid_graph.nodes[n][self_color][0], possible_colors, colors_dict, use_chromaticity)
      pic[n[0], n[1]] = new_color
    pic = image_inverse_color_transform(pic)
    return pic
  #endregion

  #region def __assign_color(self, ori_color: int or tuple[int], possible_colors: list[int] or list[tuple], colors_dict: dict, use_chromaticity: bool):
  def __assign_color(self, ori_color: int or tuple[int], possible_colors: list[int] or list[tuple], colors_dict: dict, use_chromaticity: bool):
    ori_color = colors_dict[ori_color]

    save_possible_colors = possible_colors.copy() 

    for ptr in range(len(possible_colors)):
      if isinstance(possible_colors[ptr], int):
        # maintain to be in RGB space
        possible_colors[ptr] = colors_dict[possible_colors[ptr]]
      if isinstance(save_possible_colors, tuple):
        # maintain to be transformed color
        save_possible_colors[ptr] = color_transformation(save_possible_colors[ptr])
    
    possible_colors = np.array(possible_colors)

    if use_chromaticity:
      ori_color = ori_color / np.sqrt(np.sum(np.power(ori_color, 2)))
      try:
        possible_colors = possible_colors / np.linalg.norm(possible_colors, axis = 1, ord = 2)[:, np.newaxis]
      except np.AxisError:
        return color_transformation(ori_color)

    min_ind = np.argmin(np.linalg.norm(possible_colors - ori_color, axis = 1, ord = 2))
    new_color = save_possible_colors[min_ind]
    return new_color
  #endregion

  ########################### PROCESS ALREADY QUANTIZED IMAGE ########################### 
  #region def get_info(self) -> (np.ndarray, np.ndarray, dict):
  def get_info(self) -> (np.ndarray, np.ndarray, dict):
    pic = self.pic.copy()
    pic = np.reshape(pic, (self.h * self.w, 3))
    colors, counts = np.unique(pic, axis = 0, return_counts = True)
    order = np.argsort(counts)[::-1]
    colors = colors[order, :]
    colors = colors[: self.num_color, :]

    dist = np.linalg.norm(pic[:, np.newaxis] - colors, axis = 2)
    ind = np.argmin(dist, axis = 1)
    pic = colors[ind]
    pic = np.reshape(pic, (self.h, self.w, 3))

    return pic, colors, self._quantize_layers(pic, colors)
  #endregion

  #region def get_info_with_color_palette(self) -> (np.ndarray, np.ndarray, dict):
  def get_info_with_color_palette(self) -> (np.ndarray, np.ndarray, dict):
    # convert hex to rgb
    colors = np.array([hex2rgb(hex) for hex in self.color_palette])
    pic = self.pic.copy()
    pic = pic.reshape((self.h * self.w, 3))

    ind = np.argmin(cdist(pic, colors), axis = 1)
    pic = colors[ind]
    pic = pic.reshape((self.h, self.w, 3))

    return pic, colors, self._quantize_layers(pic, colors)
  #endregion

  ########################### K MEAN ########################### 
  #region def kmean_quantize(self, debug_plot: bool) -> (np.ndarray, np.ndarray, dict):
  def kmean_quantize(self, debug_plot: bool) -> (np.ndarray, np.ndarray, dict):
    if self.already_quantized:
      colors = self._get_color()
      pic = self.pic
    else:
      pic, colors = self._raw_kmean_quantize(self.num_color, debug_plot = True)

    if debug_plot:
      _, ax = plt.subplots(1, 2, sharex = True, sharey = True)
      ax[0].imshow(pic)
      ax[0].set_title("before denoising")

    if self.denoise:
      pic, colors = self._new_denoise_pic(pic, colors)

    if debug_plot:
      ax[1].imshow(pic)
      ax[1].set_title("after denoising")
      plt.show()
      breakpoint()
    layers      = self._quantize_layers(pic, colors)
    return pic, colors, layers
  #endregion

  #region def _get_color(self) -> np.ndarray:
  def _get_color(self) -> np.ndarray:
    image_array = np.reshape(self.pic, (self.h * self.w, 3))
    image_colors = np.unique(image_array, axis = 0)
    return image_colors
  #endregion

  #region def _kmean_quantize(self, num_color, debug_plot = False) -> (np.ndarray, np.ndarray):
  def _kmean_quantize(self, num_color, debug_plot = False) -> (np.ndarray, np.ndarray):
    image_array = np.reshape(self.pic, (self.h * self.w, 3))
    # Use K-Means to do the clustering
    #kmeans      = KMeans(n_clusters = num_color, random_state = self.random_state).fit(image_array)
    kmeans      = MiniBatchKMeans(n_clusters = num_color, random_state = self.random_state).fit(image_array)

    # Find the top "num_color" colors
    image_colors, counts = np.unique(image_array, return_counts = True, axis = 0)
    colors               = image_colors[np.argsort(counts, kind = 'stable')[- num_color :]]

    # Check that each K-Means clusters is "close enough" to one of the "num_color" colors
    # Also, "correct" K-means centers
    #center = []
    #for center in kmeans.cluster_centers_:
    #  min_dist    = np.inf
    #  temp_center = center
    #  for color in colors:
    #    if np.linalg.norm(center - color) < min_dist:
    #      min_dist    = np.linalg.norm(center - color)
    #      temp_center = color
    #  if min_dist > self.threshold: 
    #    print(min_dist)
    #    raise("Error in kmean")
    #  center.append(temp_center)
    #center = np.array(corrected_center)

    center = np.array(kmeans.cluster_centers_)
    
    color         = kmeans.predict(image_array)
    quantized_pic = center[color].reshape(self.h, self.w, -1)

    if debug_plot:
      plt.imshow(quantized_pic)
      plt.show()

    return quantized_pic, colors
  #endregion

  #region def _raw_kmean_quantize(self, num_color, debug_plot = False) -> (np.ndarray, np.ndarray):
  def _raw_kmean_quantize(self, num_color, debug_plot = False) -> (np.ndarray, np.ndarray):
    image_array = np.reshape(self.pic, (self.h * self.w, 3)).astype(float)
    if self.chromaticity:
      norm            = np.linalg.norm(image_array, axis = 1)
      norm[norm == 0] = 1
      im = np.divide(image_array, norm[:, np.newaxis])
    else:
      im = image_array
      #if self.use_mode_filter:
      #  filtered_im = Image.fromarray(np.uint8(im)).convert('RGB')
      #  for _ in range(self.mode_filter_times):
      #    filtered_im = filtered_im.filter(ImageFilter.ModeFilter)
      #  filtered_im = np.asarray(filtered_im).astype(int)

    # Use K-Means to do the clustering
    kmeans = KMeans(n_clusters = num_color, random_state = self.random_state, n_init = 10).fit(im)
    color  = kmeans.predict(im)
    new_im = kmeans.cluster_centers_[color]

    if self.chromaticity:
      _, ind = np.unique(new_im, axis = 0, return_inverse = True)
      color = list()
      for i in range(np.max(ind) + 1):
        flag = ind == i
        new_color = np.mean(image_array[flag, :], axis = 0).astype(int)
        new_im[flag, 0] = new_color[0]
        new_im[flag, 1] = new_color[1]
        new_im[flag, 2] = new_color[2]
        color.append(new_color)
      color = np.vstack(color).astype(int)
    new_im = np.reshape(new_im, (self.h, self.w, 3)).astype(int)

    #if self.use_mode_filter:
    #  if debug_plot:
    #    fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
    #    ax[0].imshow(new_im)
    #    ax[0].set_title("before applying mode filter")
    #  old_num_color = len(np.unique(new_im.reshape((self.h * self.w, 3)), axis = 0))
    #  filtered_im = Image.fromarray(np.uint8(new_im)).convert('RGB')
    #  for _ in range(self.mode_filter_times):
    #    filtered_im = filtered_im.filter(ImageFilter.ModeFilter)
    #  new_im = np.asarray(filtered_im).astype(int)
    #  color = np.unique(new_im.reshape((self.h * self.w, 3)), axis = 0)
    #  if debug_plot:
    #    ax[1].imshow(new_im)
    #    ax[1].set_title("after applying mode filter")
    #    plt.show()
    #    print(f"before applying mode filtering, number of color: {old_num_color}. After, {len(color)}")
    #    breakpoint()
    #  return new_im, color
    
    #return new_im, kmeans.cluster_centers_.astype(int)
    if self.chromaticity:
      return new_im, color.astype(int)
    else:
      return new_im, kmeans.cluster_centers_.astype(int)

  #endregion

  #region def _quantize_layers(self, pic: np.ndarray, colors: list) -> dict:
  def _quantize_layers(self, pic: np.ndarray, colors: list) -> dict:
    transformed_pic = image_transformation(pic)
    layers = {tuple(c): (transformed_pic == color_transformation(c)).astype(int) for c in colors}
    return layers
  #endregion

  ########################### HISTOGRAM ########################### 
  #region def imhist_quantize(self) -> (np.ndarray, np.ndarray, dict):
  def imhist_quantize(self) -> (np.ndarray, np.ndarray, dict):
    pic, colors = self._imhist_quantize(self.num_color)
    pic         = self._denoise_pic(pic, colors)
    layers      = self._quantize_layers(pic, colors)
    return pic, colors, layers
  #endregion

  #region def _imhist_quantize(self, num_color: int, debug_plot = False) -> (np.ndarray, np.ndarray):
  def _imhist_quantize(self, num_color: int, debug_plot = False) -> (np.ndarray, np.ndarray):
    pic            = image_transformation(self.pic)
    colors, counts = np.unique(pic.flatten(), return_counts = True)

    vals = sorted(list(zip(colors, counts)), reverse = True, key = lambda x : x[1])
    vals = vals[:num_color]

    color_pic = np.ones((self.h, self.w, num_color))

    for ptr in range(color_pic.shape[2]):
      color_pic[:,:,ptr] *= vals[ptr][0]

    pic = np.tile(np.expand_dims(pic, 2), (1, 1, num_color))

    ind = np.argmin(np.abs(pic - color_pic), axis = 2)

    ret = np.zeros((self.h, self.w))

    for ptr, v in enumerate(vals):
      ret[ind == ptr] = v[0]

    ret = image_inverse_color_transform(ret).astype(np.uint8)

    colors = [inverse_color_transform(v[0]) for v in vals]

    return ret, colors
  #endregion
  
  #region def _get_bin_color(self, bins: list, do_inv = False) -> list:
  def _get_bin_color(self, bins: list, do_inv = False) -> list:
    """
    Given bins from histogram, turn them into proper colors
    """
    n    = len(bins)
    bins = [.5 * (bins[i] - bins[i - 1]) for i in range(n - 1, 0, -1)]
    bins = list(map(int, bins))
    if do_inv:
      return list(map(inverse_color_transform, bins))
    else:
      return bins
  #endregion

  ########################### CHROMO ########################### 
  #region def chromo_quantize(self) -> (np.ndarray, np.ndarray, dict):
  def chromo_quantize(self) -> (np.ndarray, np.ndarray, dict):
    pic, colors = self._chromo_quantize(self.num_color)
    pic         = self._denoise_pic(pic, colors)
    layers      = self._quantize_layers(pic, colors)
    return pic, colors, layers
  #endregion
    
  #region def _chromo_quantize(self, num_color: int, debug_plot: bool = False) -> (np.ndarray, np.ndarray):
  def _chromo_quantize(self, num_color: int, debug_plot: bool = False) -> (np.ndarray, np.ndarray):
    clusters, centers = self.__spherical_kmean(self.pic, num_color)
    centers           = centers.astype(int)
    new_im            = np.reshape(centers[clusters], (self.h, self.w, 3))

    if debug_plot:
      plt.imshow(new_im)
      plt.show()

    return new_im, centers
  #endregion

  #region def __spherical_kmean(self, X: np.ndarray, num_cluster: int, seed: int = 8964, max_iter: int = 721) -> (np.ndarray, np.ndarray):
  def __spherical_kmean(self, X: np.ndarray, num_cluster: int, seed: int = 8964, max_iter: int = 721) -> (np.ndarray, np.ndarray):
    """
    Do K means on unit sphere using great circle distance.
    """
    np.random.seed(seed)
    ## Project everything onto the sphere
    if np.ndim(X) == 3 and X.shape[2] == 3: # 3D image
      X = np.reshape(X, (self.h * self.w, 3))

    assert(X.shape == (self.h * self.w , 3))
    ori_X = X.copy()

    norm = np.linalg.norm(X, axis = 1)
    norm[norm == 0] = 1
    norm = np.column_stack([norm] * 3)
    X    = np.divide(X, norm)
    del norm

    ## Initialise K means centers on the first octant of unit sphere 
    centers = np.abs(np.random.rand(num_cluster, 3))
    norm    = np.linalg.norm(centers, axis = 1) + 1e-5
    norm    = np.column_stack([norm] * 3)
    centers = np.divide(centers, norm)
    centers[-1] = np.array([0,0,0])

    last_cluster = None
    num_iter = 0
    ## K mean
    while True:
      print(f"iter:     {num_iter}")
      cur_cluster = batch_arg_cdist(X, centers)
      centers     = update_centers(X, cur_cluster, centers)

      num_iter += 1

      if last_cluster is not None and np.all(last_cluster == cur_cluster):
        break
      
      if num_iter >= max_iter:
        break

      last_cluster = cur_cluster

    ## Recompute centers in <ori_X>
    ll      = np.sort(np.unique(cur_cluster))
    centers = np.vstack([np.mean(ori_X[cur_cluster == l], axis = 0) for l in ll])

    return cur_cluster, centers
  #endregion

  ########################### REG K MEAN ########################### 
  #region def reg_kmean_quantize(self) -> (np.ndarray, np.ndarray, dict): 
  def reg_kmean_quantize(self) -> (np.ndarray, np.ndarray, dict): 
    pic, colors = self._reg_kmean_quantize(self.l)
    pic         = self._denoise_pic(pic, colors)
    layers      = self._quantize_layers(pic, colors)
    return pic, colors, layers
  #endregion

  #region def _reg_kmean_quantize(self, l: float) -> (np.ndarray, np.ndarray):
  def _reg_kmean_quantize(self, l: float) -> (np.ndarray, np.ndarray):
    X = self.__make_observations(self.pic)

    colors = np.array([np.mean(X, axis = 0)]) # To access pixel i's color, do colors[labels[i]]
    num    = np.array([X.shape[0]])           # number of pixels, to access pixel i's number, do num[labels[i]]
    labels = np.zeros(X.shape[0], dtype = int)
    label_set = [0]

    max_iter = 1
    for itr in range(max_iter):
      for ptr in range(X.shape[0]):
        print("iter:     %d / %d" % (ptr, X.shape[0]))
        cur_label = labels[ptr]

        chg = np.inf
        ni_term_1 = 1 / (num[cur_label] * (num[cur_label] - 1)) # 1 / (n_i * (n_i - 1))
        ni_term_2 = ni_term_1 * np.power(num[cur_label], 2)  # n_i / (n_i - 1)
        di_ci = np.sum(np.power(colors[cur_label] - X[ptr], 2))
        #if len(label_set) > 1:
        if cur_label != 0:
          ## Try existing labels
          ind = np.argwhere(labels != cur_label)[0]
          nl_term_1 = 1 / np.multiply(num[ind], num[ind] + 1)
          nl_term_2 = np.multiply(nl_term_1, np.power(num[ind], 2))

          di_cl = np.power(np.linalg.norm(colors[ind] - X[ptr], axis = 1), 2)

          chg = l * (ni_term_1 - nl_term_1) + np.multiply(di_cl, nl_term_2) - di_ci * ni_term_2

          k   = np.argmin(chg)
          chg = chg[k]
          if k >= cur_label: k += 1

        ## Try new label
        new_label_chg  = (ni_term_1 + 1) * l
        new_label_chg -= di_ci * ni_term_2

        if min(chg, new_label_chg) > 0:
          continue

        if chg <= new_label_chg:
          ## use one of the existing label
          colors[k]          = (colors[k] * num[k] - X[ptr]) / (num[k] - 1)
          num[k]            -= 1
          colors[cur_label]  = (colors[cur_label] * num[cur_label] + X[ptr]) / (num[cur_label] + 1)
          num[cur_label]    += 1
          labels[cur_label]  = k
        else:
          ## make new label
          colors[cur_label]  = (colors[cur_label] * num[cur_label] - X[ptr]) / (num[cur_label] - 1)
          num[cur_label]    -= 1
          new_label          = max(label_set) + 1
          colors             = np.append(colors, [X[ptr]], axis = 0)
          num                = np.append(num, 1)
          labels[cur_label]  = new_label
          label_set.append(new_label)

    pic = colors[labels]

    pic = np.concatenate([np.expand_dims(np.reshape(pic[:,ptr], (self.h, self.w)), 2) 
                          for ptr in range(3)], axis = 2)

    return pic, colors
  #endregion

  #region def __make_observations(self, X: np.ndarray) -> np.ndarray:
  def __make_observations(self, X: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for <_reg_kmean_quantize>
    For a color picture, view as a num_pixels by 3 matrix.
    Stack by column
    """
    return np.column_stack([np.reshape(X[:,:,ptr] , (-1, 1)) for ptr in range(X.shape[2])])
  #endregion

  ########################### COMBINE WITH SEGMENTATION ########################### 
  def match_seg_result(self, pic: np.ndarray, colors: np.ndarray, layers: dict, seg_result: dict):
    phase = seg_result["phase"]

    fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
    ax[0].imshow(pic)
    ax[1].imshow(phase)
    plt.show()


  ########################### CHECK OUT ########################### 
  #region def check_out(self, path: str, pic: np.ndarray, colors: np.ndarray, layers: dict) -> None:
  def check_out(self, path: str, pic: np.ndarray, colors: np.ndarray, layers: dict) -> None:
    for key, val in layers.items():
      layers[key] = smart_convert_sparse(val)

    save_data(join(path, "layers"), layers)
    save_data(join(path, "colors"), colors)
    save_data(join(path, "pic"), pic)
  #endregion

#if __name__ == "__main__":
#  path = join("real_data", "matisse_2")
#  pic  = join(path, "resized_matisse_2_quantized.png")
#
#  pic                 = cv.imread(pic)
#  pic = cv.bilateralFilter(pic, 15, 10, 10)
#  num_color           = 15
#  #Q                   = quantizer(pic, num_color, chromaticity = False, random_state = 721, 
#  #                                denoise_threshold = 20, already_quantized = False)
#  #pic, colors, layers = Q.quantize()
#
#  #plt.imshow(pic)
#  #plt.show()
#
#  (h, w) = pic.shape[:2]
#  pic = cv.cvtColor(pic, cv.COLOR_BGR2LAB)
#  pic = pic.reshape((h * w, 3))
#
#  clt = MiniBatchKMeans(num_color)
#  labels = clt.fit_predict(pic)
#  quant = clt.cluster_centers_.astype("uint8")[labels]
#  quant = quant.reshape((h, w, 3))
#  pic = pic.reshape((h, w, 3))
#
#  quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
#  pic = cv.cvtColor(pic, cv.COLOR_LAB2BGR)
#  quant = cv.cvtColor(quant, cv.COLOR_BGR2RGB)
#  pic = cv.cvtColor(pic, cv.COLOR_BGR2RGB)
#
#  fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
#  ax[0].imshow(pic)
#  ax[1].imshow(quant)
#  plt.show()

if __name__ == "__main__":
  path = join("real_data", "fruit5")
  pic = join(path, "small_fruit.jpg")

  pic = cv.imread(pic)
  num_color = 15

  Q = quantizer(pic, num_color, random_state = 721, denoise_threshold = 3)
  #pic, colors, layers = Q.quantize()

  #Q.check_out(path, pic, colors, layers)

  pic = load_data(join(path, "pic"))
  colors = load_data(join(path, "colors"))
  layers = load_data(join(path, "layers"))

  seg_result = load_data(join(path, "quantized_seg_result"))

  Q.match_seg_result(pic, colors, layers, seg_result)

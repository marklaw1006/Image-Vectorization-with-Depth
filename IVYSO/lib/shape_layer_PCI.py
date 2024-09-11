#!/usr/bin/env python3.9

import numpy as np
import networkx as nx
import cv2 as cv
from scipy.spatial import ConvexHull
from numpy_indexed import indices
from skimage.morphology import area_closing
from skimage.morphology import dilation
from itertools import groupby
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.sparse import isspmatrix_csc
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

from .shape_layer import  wild_card_int, self_color, neighbor_color
from .helper import is_counter_clockwise, is_clockwise
from .new_helper import get_boundary

level_str              = "level"
counter_boundary_str   = "counter_orientation"
clockwise_boundary_str = "clock_orientation"
hole_filled_str        = "hole_filled"
image_str              = "image"

class Partial_Convex_Init():
  def __init__(self, h: int, w: int, shape_layers: dict, 
                     grid_graph: nx.classes.graph.Graph(), 
                     mutual_bd_graph: nx.classes.graph.Graph(), 
                     order_graph: nx.classes.digraph.DiGraph()):
    self.h, self.w       = h, w
    self.shape_layers    = shape_layers 
    self.grid_graph      = grid_graph
    self.order_graph     = order_graph
    self.mutual_bd_graph = mutual_bd_graph

    self.wild_node = list(filter(lambda n : self.grid_graph.nodes[n][self_color][0] == wild_card_int, 
                                 self.grid_graph))

    ## put key in descending depth order(just read from shape layers)
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

  #region def partial_convex_init(self, i: tuple) -> (np.ndarray, np.ndarray):
  def partial_convex_init(self, i: tuple) -> (np.ndarray, np.ndarray):
    """
    Give a paritally convex initialisation. Modifed graham scan.
    """
    cmp_on_top = list(self.order_graph.predecessors(i))
    cmp_on_top = list(filter(lambda key : self.shape_layers[key].get_level() > self.shape_layers[i].get_level(), 
                             self.shape_layers.keys()))
    removable_pts = self._get_removable_pts(i, cmp_on_top)

    bd = self._modified_graham_scan(self.shape_layers[i].get_clock_bd()[0], removable_pts)
    im = np.zeros((self.h, self.w))
    cv.fillPoly(im, pts = [bd], color = 1)

    im = np.multiply(im, self.get_reg_area(i))

    im[im == 0] = -1
    return im.astype(float), np.zeros((self.h, self.w))
  #endregion

  #region def new_partial_convex_init(self, i: tuple) -> (np.ndarray, np.ndarray):
  def new_partial_convex_init(self, i: tuple) -> (np.ndarray, np.ndarray):
    """
    Give a paritally convex initialisation. Modifed graham scan.
    """
    cmp_on_top = list(self.order_graph.predecessors(i))
    cmp_on_top = list(filter(lambda key : self.shape_layers[key].get_level() > self.shape_layers[i].get_level(), 
                             self.shape_layers.keys()))
    removable_pts = self._new_get_removable_pts(i, cmp_on_top)

    im = np.zeros((self.h, self.w))
    for b in self.shape_layers[i].get_clock_bd():
      bd = self._modified_graham_scan(b, removable_pts)
      cv.fillPoly(im, pts = [bd], color = 1)

    im = np.multiply(im, self.get_reg_area(i))

    im[im == 0] = -1
    return im.astype(float), np.zeros((self.h, self.w))
  #endregion

  #region def _modified_graham_scan(self, bd: np.ndarray, removable_pts: dict) -> np.ndarray:
  def _modified_graham_scan(self, bd: np.ndarray, removable_pts: dict) -> np.ndarray:
    """
    Auxiliary function for <partial_convex_init>.
    modified graham scan. 
    """
    bd = bd[:, ::-1]
    if not any([val for val in removable_pts.values()]):
      return bd
    ret = list()

    points = list(map(tuple, bd))
    points.sort(key=lambda x: [x[0],x[1]])
    start = points.pop(0)
    ret.append(start)
    points.sort(key = lambda p: (self.get_slope(p, start), -p[1], p[0]))

    for pt in points:
      ret.append(pt)
      while len(ret) > 2 and self.get_cross_product(ret[-3], ret[-2], ret[-1]) < 0 and removable_pts[(ret[-2][1], ret[-2][0])]:
        ret.pop(-2)

    ret = np.array(list(map(list, ret)))
    return ret
  #endregion

  #region def get_slope(self, p1: np.ndarray, p2: np.ndarray) -> float:
  def get_slope(self, p1: np.ndarray, p2: np.ndarray) -> float:
    if p1[0] == p2[0]:
      return float('inf')
    else:
      return 1.0*(p1[1]-p2[1])/(p1[0]-p2[0])
  #endregion

  #region def get_cross_product(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
  def get_cross_product(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:#{{{
    return ((p2[0] - p1[0])*(p3[1] - p1[1])) - ((p2[1] - p1[1])*(p3[0] - p1[0]))
  #endregion

  #region def _get_removable_pts(self, i: tuple, cmp_on_top: list) -> dict:
  def _get_removable_pts(self, i: tuple, cmp_on_top: list) -> dict:
    #i_pts = self.shape_layers[i].get_clock_bd()[0]
    #i_pts = list(map(tuple, i_pts))
    #D = {tuple(p): False for p in i_pts}
    #for j in cmp_on_top:
    #  for p in self.__is_neighbor_to(i_pts, j):
    #    D[p] = True

    #return D
    i_pts = list(map(tuple, self.shape_layers[i].get_clock_bd()[0]))
    D = {tuple(p): False for p in i_pts}
    for p in map(tuple, self.shape_layers[i].get_inpaintable_pts()):
      D[p] = True
    return D
  #endregion

  def _new_get_removable_pts(self, i: tuple, cmp_on_top: list) -> dict:
    #D = dict()
    #for i_pts in self.shape_layers[i].get_clock_bd():
    #  i_pts = list(map(tuple, i_pts))
    #  D.update({tuple(p): False for p in i_pts})
    #  for j in cmp_on_top:
    #    for p in self.__is_neighbor_to(i_pts, j):
    #      D[p] = True

    #return D
    i_pts = list(map(tuple, self.shape_layers[i].get_clock_bd()[0]))
    D = {tuple(p): False for p in i_pts}
    for p in map(tuple, self.shape_layers[i].get_inpaintable_pts()):
      D[p] = True
    return D

  #region def get_reg_area(self, i: tuple) -> np.ndarray:
  def get_reg_area(self, i: tuple) -> np.ndarray:
    """
    Given a node, get the region that it can do curvature and arc length minimization.
    Basically the regions of any node at higher level.
    Return a binary matrix.
    """
    #thre = 1.5
    #level = self.shape_layers[i].get_level()

    #image = self.shape_layers[i].get_dense_layer_im()
    #                          
    #for key in filter(lambda key : self.shape_layers[key].get_level() >= level, self.shape_layers.keys()):
    #  np.maximum(image, self.shape_layers[key].get_dense_layer_im(), out = image)
    
    #image_coord = self.shape_layers[i].get_coord()
    #wild_coord  = np.array(list(map(list, self.wild_node)))
    #try:
    #  ind = np.where(np.min(cdist(wild_coord, image_coord), axis = 1) <= thre)[0]
    #  for j in ind:
    #    image[self.wild_node[j]] = 1
    #except ValueError:
    #  pass
    ##image = binary_fill_holes(image)

    #image = self._remove_disconnected_region(i, image)
    #return image

    #################################### NEW ################################### 
    #ind = self.key_depth.index(i)
    #image = self.cumsum_im[:,:,ind]

    ##reg = image + self.wild_image
    ##reg = label(reg, connectivity = 2)
    ##ind = np.max(np.multiply(image, reg))
    ##reg = (reg == ind).astype(float)
    ##return reg 

    #reg = dilation(image, footprint = np.ones((2,2)))
    #np.multiply(reg, self.wild_image, out = reg)
    #np.maximum(image, reg, out = reg)

    #test_im = self.shape_layers[i].get_dense_layer_im()
    #reg = label(reg)
    #ind = np.max(np.multiply(test_im, reg))
    #return reg == ind

    return self.shape_layers[i].get_reg_area()

  #endregion

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

    if return_numpy:
      return np.array(list(map(list, ret)))
    return ret
  #endregion

  #region def _get_all_predecessors(self, i: tuple) -> list:
  def _get_all_predecessors(self, i: tuple) -> list:
    """
    Auxiliary function for <partial_convex_init>.
    Get all components that are above <i> and has an edge to it.
    """
    ret = list(self.order_graph.predecessors(i))
    next_add = ret.copy()
    while True:
      next_add = list(map(list, map(self.order_graph.predecessors, next_add)))  # list of list
      next_add = [l for lst in next_add for l in lst]
      ret += next_add
      if len(next_add) == 0:
        break
    ret = list(set(ret))
    return list(filter(lambda r : (r, i) in self.mutual_bd_graph.edges(), ret))
  #endregion

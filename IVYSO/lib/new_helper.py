#!/usr/bin/env python3.9
import numpy as np
from scipy.interpolate import interp1d
import cv2 as cv
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sys import getsizeof

import warnings
#warnings.filterwarnings("error")

from .helper import *

max_c = 256

def batch_cdist(XA: np.ndarray, XB: np.ndarray, batch: int = 500) -> np.ndarray:#{{{
  """
  Same function as <scipy.distance.spatial.cdist>. But with batch optimizaton
  """
  assert(np.ndim(XA) == 2 and np.ndim(XB) == 2)
  assert(XA.shape[1] == XB.shape[1])
  XA_num_batch = (len(XA) // batch) + 1
  XB_num_batch = (len(XB) // batch) + 1

  ret = list()
  for A_ptr in range(XA_num_batch):
    cur_A_size = min((A_ptr + 1) * batch , XA.shape[0]) - A_ptr * batch
    cur_XA_mat = XA[A_ptr * batch : min((A_ptr + 1) * batch , XA.shape[0]), :]
    cur_dist = np.ones(cur_A_size) * np.inf
    for B_ptr in range(XB_num_batch):
      cur_XB_mat = XB[B_ptr * batch : min((B_ptr + 1) * batch , XB.shape[0]), :]
      dist       = np.min(cdist(cur_XA_mat, cur_XB_mat), axis = 1)
      cur_dist   = np.minimum(cur_dist, dist)
    ret.append(cur_dist)

  return np.concatenate(ret)
#}}}

def image_transformation(pic: np.ndarray) -> np.ndarray:#{{{
  """
  Turn the image into 1D representation
  """
  assert(pic.shape[2] == 3)
  ret = pic[:,:,0] * np.power(max_c, 2) + pic[:,:,1] * max_c + pic[:,:,2]
  return ret.astype(int)
#}}}

def color_transformation(color: tuple or list or np.ndarray) -> int:#{{{
  """
  Turn <color> into single channel representation
  """
  return int(color[0] * np.power(max_c, 2) + color[1] * max_c + color[2])
#}}}

def inverse_color_transform(color: int) -> tuple:#{{{
  """
  Inverse of <color_transformation>
  """
  c1 = color // np.power(max_c, 2)
  color -= c1 * np.power(max_c, 2)
  c2 = color // max_c
  color -= c2 * max_c
  c3 = color
  return (c1, c2, c3)
#}}}

def image_inverse_color_transform(pic: np.ndarray) -> np.ndarray:#{{{
  pic = pic.astype(int)
  ret = list()

  ret.append(pic // np.power(max_c, 2))
  pic -= ret[0] * np.power(max_c, 2)
  ret.append(pic // max_c)
  pic -= ret[1] * max_c
  ret.append(pic)
  ret = list(map(lambda x : np.expand_dims(x, 2), ret))
  return np.concatenate(ret, axis = 2)
#}}}

def compute_curvature_on_open_curve(curve: np.ndarray, spacing = 1) -> np.ndarray:
  assert(np.ndim(curve) == 2 and curve.shape[1] == 2)
  in_curve = curve[spacing : -spacing, :]

  curve1 = curve[: -2 * spacing, :] - in_curve
  curve2 = curve[2 * spacing :, :] - in_curve

  det = np.multiply(curve1[:,0], curve2[:,1]) - np.multiply(curve1[:,1], curve2[:,0])

  n1 = np.linalg.norm(curve1, axis = 1)
  n2 = np.linalg.norm(curve2, axis = 1)
  n3 = np.linalg.norm(curve1 - curve2, axis = 1)

  n = np.multiply(n1, n2, n3)

  return -2 * np.divide(det, n)

def integrate_curvature_on_open_curve(curve: np.ndarray, spacing = 1) -> np.ndarray:#{{{
  assert(np.ndim(curve) == 2 and curve.shape[1] == 2)
  in_curve = curve[spacing : -spacing, :]

  curve1 = curve[: -2 * spacing, :] - in_curve
  curve2 = curve[2 * spacing :, :] - in_curve

  det = np.multiply(curve1[:,0], curve2[:,1]) - np.multiply(curve1[:,1], curve2[:,0])

  n1 = np.linalg.norm(curve1, axis = 1)
  n2 = np.linalg.norm(curve2, axis = 1)
  n3 = np.linalg.norm(curve1 - curve2, axis = 1)

  #n = np.multiply(n2, n3) # multiply n1
  n = np.multiply(n1, n3) # multiply n2
  #n = np.multiply(n1, n2) # multiply n3

  return -2 * np.sum(np.multiply(np.divide(det, n), n2))
#}}}

def new_integrate_curvature_on_open_curve(curve: np.ndarray, spacing = 1) -> np.ndarray:#{{{
  assert(np.ndim(curve) == 2 and curve.shape[1] == 2)
  in_curve = curve[spacing : -spacing, :]

  curve1 = curve[: -2 * spacing, :] - in_curve
  curve2 = curve[2 * spacing :, :] - in_curve

  det = np.multiply(curve1[:,0], curve2[:,1]) - np.multiply(curve1[:,1], curve2[:,0])

  n1 = np.linalg.norm(curve1, axis = 1)
  n2 = np.linalg.norm(curve2, axis = 1)
  n3 = np.linalg.norm(curve1 - curve2, axis = 1)

  #n = np.multiply(n2, n3) # multiply n1
  n = np.multiply(n1, n3) # multiply n2
  #n = np.multiply(n1, n2) # multiply n3

  return np.sum(np.divide(det, n)) * -2
#}}}


def compute_curvature_at_points(ind: list, contour: np.ndarray, spacing = 3) -> list:
  ret = list()
  for i in ind:
    c1 = contour[i - spacing, :] - contour[i, :]
    c2 = contour[(i + spacing) % len(contour), :] - contour[i, :]

    det = c1[0] * c2[1] - c1[1] * c2[0]
    n1 = np.linalg.norm(c1)
    n2 = np.linalg.norm(c2)
    n3 = np.linalg.norm(c1 - c2)

    if n3 == 0:
      ret.append(np.inf)
      continue
    
    with warnings.catch_warnings():
      warnings.filterwarnings('error')
      try:
        curv = -2 * det / (n1 * n2 * n3)
      except Warning:
        #import matplotlib.pyplot as plt
        #plt.plot(contour[:,1], contour[:,0], 'b-')
        #plt.plot(contour[i, 1], contour[i, 0], 'ro')
        #plt.plot(contour[i - spacing, 1], contour[i - spacing, 0], 'go')
        #plt.plot(contour[i + spacing, 1], contour[i + spacing, 0], 'mo')
        #plt.show()
        #breakpoint()
        curv = np.inf

    ret.append(curv)
  return ret


################### For <decide_where_wild_nodes_go> in <new_layering> ###################
def compute_curvature_on_closed_curve(curve: np.ndarray, spacing = 1) -> np.ndarray:
  assert(np.ndim(curve) == 2 and curve.shape[1] == 2)
  
  curve1 = np.roll(curve,  spacing, axis = 0) - curve
  curve2 = np.roll(curve, -spacing, axis = 0) - curve

  det = np.multiply(curve1[:,0], curve2[:,1]) - np.multiply(curve1[:,1], curve2[:,0])

  n1 = np.linalg.norm(curve1, axis = 1)
  n2 = np.linalg.norm(curve2, axis = 1)
  n3 = np.linalg.norm(curve1 - curve2, axis = 1)

  n = np.multiply(n1, n2, n3)

  flag = n == 0

  det[flag != 1] = -2 * np.divide(det[flag != 1], n[flag != 1])
  det[flag] = 1e9

  return det


def arc_length_parameterization(curve: np.ndarray):
  assert(np.ndim(curve) == 2 and curve.shape[1] == 2)

  c = curve - np.roll(curve, 1, axis = 1)

  l = np.linalg.norm(c, axis = 1)

  l[0] = 0.
  l = np.cumsum(l)
  l /= l[-1]
  return l

def _compute_bd_length(pts: np.ndarray):
  assert(np.ndim(pts) == 2 and pts.shape[1] == 2)
  return np.sum(np.linalg.norm(pts - np.roll(pts, 1, axis = 0)))

def compute_max_change(old_image: np.ndarray, new_image: np.ndarray):
  old_bd = get_boundary(old_image)
  old_P = sum(list(map(_compute_bd_length, old_bd)))
  old_A = np.sum(old_image > 0)
  new_bd = get_boundary(new_image)
  P = sum(list(map(_compute_bd_length, new_bd)))
  A = np.sum(new_image > 0)
  return np.abs((P / A) - (old_P / old_A))
  
################### End For <decide_where_wild_nodes_go> in <new_layering> ###################

def get_boundary(image: np.ndarray) -> list[np.ndarray]:
  temp_im = image.copy()
  temp_im= temp_im.astype(np.uint8)
  bd, _ = cv.findContours(temp_im, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
  bd = list(map(lambda b : b.squeeze(1), bd))
  bd = [np.column_stack((b[:,1], b[:,0])) for b in bd]
  return bd


def stitching(segs: list[np.ndarray])  -> list[np.ndarray]:
  """
  Some times the resulted images can have pixels diagonally touching each other. 
  This results in multiple clockwise boundaries.
  Stitch them to make them one.
  """
  ## must be no loops. just return.
  if len(segs) <= 1:
    return segs

  thre = 1 # a number slightly larger than sqrt(2)
  ## group those that should be stitched up.
  G = nx.Graph()
  G.add_nodes_from(range(len(segs)))

  id_segs = list(zip(range(len(segs)), segs))

  for (ptr1, segs1), (ptr2, segs2) in combinations(id_segs, 2):
    if ptr1 > ptr2:
      ptr1, ptr2 = ptr2, ptr1
      segs1, segs2 = segs2, segs1
    mat = cdist(segs1, segs2)
    ind = np.argwhere(mat < thre)
    if len(ind) > 0:
      G.add_edge(ptr1, ptr2)
  
  S = [G.subgraph(c).copy() for c in nx.connected_components(G)] # subgraph of each connected components

  ret = list()

  for s in S:
    result_path = None
    for ptr1, ptr2 in s.edges():
      if ptr1 > ptr2:
        ptr1, ptr2 = ptr2, ptr1

      if result_path is None:
        result_path = segs[ptr1]
      
      mat = cdist(result_path, segs[ptr2])
      ind = np.argwhere(mat < thre)
      insert_at, rotate_by = ind[np.argmin(ind[:,0]), :]
      result_path = np.insert(result_path, 
                              (insert_at + 1) % len(result_path), 
                              np.roll(segs[ptr2], -rotate_by, axis = 0), axis = 0)
    ret.append(result_path)

  return ret

#region def smart_convert_sparse(mat: np.ndarray) -> np.ndarray or csr_matrix:
def smart_convert_sparse(mat: np.ndarray, dtype = float) -> np.ndarray or csr_matrix:
  sparse_matrix = csr_matrix(mat, dtype = dtype)

  dense_memory = mat.nbytes
  sparse_memory = getsizeof(sparse_matrix)

  if dense_memory > sparse_memory:
    return sparse_matrix
  else:
    return mat
#endregion


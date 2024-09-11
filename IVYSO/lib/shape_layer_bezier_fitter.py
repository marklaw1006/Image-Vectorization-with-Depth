#!/usr/bin/env python3.9

# External library
import numpy as np
import pickle
import networkx as nx
from numpy_indexed import indices
import matplotlib.pyplot as plt
from itertools import cycle, islice
from warnings import warn, filterwarnings
import svgwrite
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.spatial._qhull import QhullError
from scipy.ndimage import gaussian_filter1d

#filterwarnings('error')

# Internal library
from .helper import save_data
from .helper import _make_clockwise
from .new_helper import inverse_color_transform, color_transformation
from .new_helper import compute_curvature_on_closed_curve

coeff_str      = "coeff"
start_ind_str  = "start_ind"
end_ind_str    = "end_ind"
contour_str    = "contour"
order_str      = "order"
layer_str      = "layer"
color_str      = "color"
is_bg_str      = "is_bg"
curve_str      = "curve"
book_str       = "book"
ret_str        = "ret"
is_ellipse_str = "is_ellipse"
is_circle_str  = "is_circle"
shape_layers_str = "shape_layers"

## constants for fitting bezier curves
linear    = 1
quadratic = 2
cubic     = 3
other     = -1

# breakpoint()

############################## AUXILIARY FUNCTIONS ############################## 
#region def cyclic_indexing(A: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
def cyclic_indexing(A: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
  """
  cyclic indexing. BOTH ENDS INCLUSIVE.
  """
  end_idx += 1
  if start_idx < 0: start_idx += A.shape[0]
  if end_idx < 0: end_idx += A.shape[0]

  if start_idx < end_idx:
    return A[start_idx:end_idx, ...]
  else:
    return np.concatenate((A[start_idx:, ...], A[:end_idx, ...]), axis=0)
#endregion

####################################################################################### 
############################## FOR FITTING BEZIER CURVES ############################## 
####################################################################################### 

class Bezier_fitter():
  def __init__(self, k = 5, circle_radius = 1.5, eps = .9, 
               ahead = 6, use_mid_pt: bool = False, use_weighted: bool = True, use_landmark: bool = True):
    self.k             = k             # <k> for shifting. Both are for finding curvature extrema
    self.circle_radius = circle_radius # at a point, if the radius of circle(1 / curvature) is smaller than this value, then it is a curvature extremum.
    self.eps           = eps
    self.ahead         = ahead
    self.use_weighted  = use_weighted
    self.use_mid_pt    = use_mid_pt
    self.use_landmark  = use_landmark

  ################### CURVATURE EXTREMA DETECTION ################### 
  #region def _find_curv_ext(self, contour: np.ndarray, debug_plot = False) -> np.ndarray:
  def _find_curv_ext(self, contour: np.ndarray, debug_plot = False) -> np.ndarray:
    """
    From <contour>, find the curvature extrema and return their indices.
    """
    contour = _make_clockwise(contour)
    curvature = compute_curvature_on_closed_curve(contour, spacing = self.k)
    radii     = np.array([1 / np.abs(c) if c != 0 else np.inf for c in curvature])
    ind = np.where(radii < self.circle_radius)[0].astype(int)
    ind = self.__clean_curv_ext(contour, ind, radii).astype(int)
    ind = np.sort(ind)

    if debug_plot:
      plt.figure(str(np.random.rand()))
      plt.plot(contour[:,0], contour[:,1], 'b-')
      plt.plot(contour[ind, 0], contour[ind, 1], 'ro')
      plt.show()

    return ind
  #endregion

  #region def __clean_curv_ext(self, contour: np.ndarray, ind: np.ndarray, radii: np.ndarray) -> np.ndarray:
  def __clean_curv_ext(self, contour: np.ndarray, ind: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for <_find_curv_ext>.
    Sometimes the curvature extrema are too close to each other. Clean out those that are consecutive to each other.
    """
    v = list(map(tuple, contour))
    v = list(zip(v, range(len(v))))
    G = nx.Graph()
    G.add_nodes_from(v)
    edges = [(v[ptr], v[(ptr + 1) % len(v)]) for ptr in range(len(v))]
    G.add_edges_from(edges)

    curv_ext = [v[i] for i in ind]
    G        = G.subgraph(curv_ext)
    cc       = nx.connected_components(G)

    ret = list()
    for c in cc:
      r     = [(radii[i[1]], i[1]) for i in c]
      max_r = min(r, key = lambda x : x[0])
      ret.append(max_r[1])
    return np.array(ret)
  #endregion

  ################### POLYNOMIAL FITTING ################### 
  #region def fitting(self, contours: list, landmark: np.ndarray, debug_plot: bool = False) -> list:
  def fitting(self, contours: list, landmark: np.ndarray, debug_plot: bool = False) -> list:
    """
    Fit Bezier curves to the contours. Allowed error(Hausdorff distance) to be <eps>
    Steps:
    1. Find the curvature extrema.
    2. Fit a third order degree polynomial to points between 2 curvature extrema
    3. While the error between these 2 curvature extrema is larger than <eps>, put an additional knot at the point with highest error.
    4. Repeat (3) until error is smaller than <eps>
    """
    ## Pre-Process contours
    ind      = [np.sort(np.unique(contour, return_index = True, axis = 0)[1]) for contour in contours]
    contours = [ct[i, :] for ct, i in zip(contours, ind)]

    ## Step 2
    package = list(zip(contours, map(self._find_curv_ext, contours)))
    ret = list()

    ## quick fix
    use_weighted = self.use_weighted
    use_mid_pt   = self.use_mid_pt
    use_landmark = self.use_landmark

    for p in package:
      ret.append(self._fit_single_contour(p[0], p[1], self.eps, landmark, ahead = self.ahead, 
                                          use_mid_pt = use_mid_pt, use_weighted = use_weighted, 
                                          use_landmark = use_landmark, debug_plot = debug_plot))

    return ret
  #endregion

  #region def _fit_single_contour(self, contour: np.ndarray, ind: np.ndarray, 
  def _fit_single_contour(self, contour: np.ndarray, ind: np.ndarray, 
                          eps: float, landmark: np.ndarray, ahead = 6, use_mid_pt = False, 
                          use_weighted = True, use_landmark: bool = True, debug_plot = False) -> list:
    """
    Auxiliary function for <fitting>. Fit a single contour with curvature extrema indices <ind>.
    Greedy Algorithm:
    At ind[ptr], find a third order polynomial to fit ind[ptr + 4].
    If error too large, try ind[ptr + 3]. And likewise until ind[ptr + 1]
    If error is still too large, insert the max error pt into ind and repeat
    <ahead>: int. How many curvature extrema ahead you would like to begin fitting?
    <use_mid_pt>: boolean. When True, use the mid pts between consecutive curvature extrema to do fitting.
    """
    if len(contour) == 1: # small noise 
      pt = contour[0]
      contour = np.array([[ pt[0] - 0.5, pt[1] - 0.5], 
                          [ pt[0] + 0.5, pt[1] - 0.5],
                          [ pt[0] + 0.5, pt[1] + 0.5],
                          [ pt[0] + 0.5, pt[1] - 0.5]])

    if use_landmark and len(landmark) > 0:
      try:
        ind = np.concatenate([ind, indices(contour, landmark)])
      except KeyError:
        pass
      #ind = indices(contour, landmark)
      ind = np.unique(ind)
      ind = np.sort(ind)

    if len(ind) == 0:
      ind = np.array([0,0])            # pick the first point as the starting point.
    else:
      ind = np.append(ind, ind[0])

    if use_mid_pt: ind = self.__find_mid_pts(contour, ind)

    ori_kappa = compute_curvature_on_closed_curve(contour, spacing = self.k).astype(float)
    any_curv_ext_flag = self._any_curvature_ext(ori_kappa)
    same_sign_flag = np.all(ori_kappa >= 0) or np.all(ori_kappa <= 0)
    if use_weighted:
      sigma = 5
      kappa = self.__kappa_denoising(ori_kappa.copy(), sigma, num_iter = 1)

    ret = list()
    ## some curvature extrema
    cur_ptr = 0
    while cur_ptr != len(ind) - 1:
      ahead_ptr = min(cur_ptr + ahead, len(ind) - 1)
      while True:
        if use_weighted:
          coeff, order = self.__fit_weighted_segment(contour, ind[cur_ptr], ind[ahead_ptr], kappa = kappa)
          err, err_ind, err_ind_ct = self._find_weighted_max_error(contour, ind[cur_ptr], ind[ahead_ptr], coeff, order, kappa)
        else:
          coeff, order = self.__fit_segment(contour, ind[cur_ptr], ind[ahead_ptr])
          err, err_ind, err_ind_ct = self._find_max_error(contour, ind[cur_ptr], ind[ahead_ptr], coeff, order)
        if err < eps:
          ## fitting successful 
          ret.append(self.__build_dict(coeff, contour, ind[cur_ptr], ind[ahead_ptr], order))
          cur_ptr = ahead_ptr
          break
        else:
          ## Try with closer point.
          if ahead_ptr > cur_ptr + 1:
            ahead_ptr -= 1 
          else:
            ind = np.insert(ind, ahead_ptr, err_ind_ct)

    if debug_plot:
      plt.plot(contour[:,0], contour[:,1], 'bo')
      x = np.linspace(0, 1, 100)
      A = self.__form_parameterization_matrix(x)
      A_1 = self.__form_parameterization_matrix_2(x)
      A = np.column_stack((A_1[:, 0], A, A_1[:,1]))
      for r in ret:
        if r[order_str] == quadratic:
          A = np.column_stack((np.power(1 - x, 2), 2 * np.multiply(1 - x, x), np.power(x, 2)))
        if r[order_str] == linear:
          A = np.column_stack((1 - x, x))
        if r[order_str] == cubic:
          A = self.__form_parameterization_matrix(x)
          A_1 = self.__form_parameterization_matrix_2(x)
          A = np.column_stack((A_1[:, 0], A, A_1[:,1]))

        pts = A @ r[coeff_str]
          
        plt.plot(pts[:,0], pts[:,1], 'r-')
      plt.show()
      breakpoint()

    return ret
  #endregion

  #region def __find_mid_pts(self, contour: np.ndarray, ind: np.ndarray or list[int]) -> np.ndarray:
  def __find_mid_pts(self, contour: np.ndarray, ind: np.ndarray or list[int]) -> np.ndarray:
    """
    Given a <contour> and indices of curvature extrema <ind>, find the indicies of the mid pts between consecutive curvature extrema.
    Return the indices of the mid pts.
    """
    if len(ind) == 2 and ind[0] == ind[1]:
      # no curvautre extremum detected
      return ind

    #segs = [self.__get_pts(contour, ind[ptr], ind[(ptr + 1) % len(ind)]) for ptr in range(len(ind))] # seems don't need to do (ptr + 1) % len(ind)
    segs = [self.__get_pts(contour, ind[ptr], ind[ptr + 1]) for ptr in range(len(ind) - 1)]
    segs = list(filter(lambda x : len(x) > 0, segs))

    mid_pts = np.array(list(map(self.___find_seg_mid_pt, segs)))

    ret = indices(contour, mid_pts, axis = 0)
    ret = np.append(ret, ret[0])

    return ret
  #endregion

  #region def ___find_seg_mid_pt(self, seg: np.ndarray) -> np.ndarray:
  def ___find_seg_mid_pt(self, seg: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for <__find_mid_pts>.
    Given a segment, return the point closest to the middle one.
    """
    assert(len(seg) > 0)
    arc_length  = np.cumsum(np.linalg.norm(np.diff(seg, axis = 0), axis = 1))
    arc_length /= arc_length[-1]

    return seg[np.argmin(np.abs(arc_length - .5))]
  #endregion

  #region def __build_dict(self, coeff: np.ndarray, contour: np.ndarray, start_ind: int, end_ind: int, order: int, is_ellipse: bool = False, is_circle: bool = False) -> dict: 
  def __build_dict(self, coeff: np.ndarray, contour: np.ndarray, 
                   start_ind: int, end_ind: int, order: int, 
                   is_ellipse: bool = False, is_circle: bool = False) -> dict: 
    """
    Auxiliary function for <_fit_single_contour>
    """
    return {coeff_str: coeff, start_ind_str: start_ind, 
            end_ind_str: end_ind, contour_str: contour, order_str: order,
            is_ellipse_str: is_ellipse, is_circle_str: is_circle}
  #endregion

  #region def __fit_segment(self, contour: np.ndarray, start_ind: int, end_ind: int) -> (np.ndarray, int):
  def __fit_segment(self, contour: np.ndarray, start_ind: int, end_ind: int)-> (np.ndarray, int):
    """
    Auxiliary function for <_fit_single_contour>
    Given points in <contour> within the interval [start_ind : end_ind], fit a cubic Bezier curve to tthem.
    Equivalent to solving a constrained optimization problem:
    B(t) = (1-t)^3 P_0 + 3(1-t)^2t P_1 + 3(1-t)t^2 P_2 + t^3 P_3 
    Since both endpoints are fixed, P_0 and P_3 are determined.
    Return a matrix that is [P_0 P_1 P_2 P_3]^T
    """
    pts    = self.__get_pts(contour, start_ind, end_ind)
    _, idx = np.unique(pts, axis = 0, return_index = True)
    pts    = pts[np.sort(idx),   : ]
    n      = len(pts)
    order  = None
    x      = self.__arc_length_parameterization(pts)
    P_0, P_3 = contour[start_ind, :], contour[end_ind, :]

    if n == 3:
      ## fit quadratic bezier curve
      order = quadratic
      A1    = np.power(1 - x, 2)
      A2    = 2 * np.multiply(1 - x, x)
      A3    = np.power(x, 2)
      b     = pts - np.column_stack((A1, A3)) @ np.vstack((P_0, P_3))
      coeff = np.linalg.lstsq(np.expand_dims(A2, 1), b, rcond = None)
      ret   = np.vstack((P_0, coeff[0], P_3))
      return ret, order

    if n == 2: 
      ## fit linear line
      order = linear
      ret = np.vstack((P_0, P_3))
      return ret, order
    
    assert(n > 3)
    order = cubic
    A = self.__form_parameterization_matrix(x)

    ## form the parametrization matrix for P_0, P_3
    A_1 = self.__form_parameterization_matrix_2(x)
    b = pts - A_1 @ np.vstack((P_0, P_3))

    coeff = np.linalg.lstsq(A, b, rcond = None)
    ret   = np.vstack((P_0, coeff[0], P_3))
    return ret, order
  #endregion

  #region def __fit__weighted_segment(self, contour: np.ndarray, start_ind: int, end_ind: int, kappa = None: np.ndarray) -> (np.ndarray, int):
  def __fit_weighted_segment(self, contour: np.ndarray, start_ind: int, end_ind: int, 
                             kappa: np.ndarray = None) -> (np.ndarray, int):
    """
    Auxiliary function for <_fit_single_contour>
    Given points in <contour> within the interval [start_ind : end_ind], fit a cubic Bezier curve to tthem.
    Equivalent to solving a constrained optimization problem:
    B(t) = (1-t)^3 P_0 + 3(1-t)^2t P_1 + 3(1-t)t^2 P_2 + t^3 P_3 
    Since both endpoints are fixed, P_0 and P_3 are determined.
    Return a matrix that is [P_0 P_1 P_2 P_3]^T
    """
    pts    = self.__get_pts(contour, start_ind, end_ind)
    _, idx = np.unique(pts, axis = 0, return_index = True)
    pts    = pts[np.sort(idx),:]
    n      = len(pts)
    order  = None
    x      = self.__arc_length_parameterization(pts)
    P_0, P_3 = contour[start_ind, :], contour[end_ind, :]
    kappa  = np.sqrt(cyclic_indexing(kappa, start_ind, end_ind))

    if n == 3:
      ## fit quadratic bezier curve
      order = quadratic
      A1    = np.power(1 - x, 2)
      A2    = 2 * np.multiply(1 - x, x)
      A3    = np.power(x, 2)
      b     = pts - np.column_stack((A1, A3)) @ np.vstack((P_0, P_3))
      coeff = np.linalg.lstsq(np.expand_dims(A2 * kappa, 1), b * kappa[:, np.newaxis], rcond = None)
      ret   = np.vstack((P_0, coeff[0], P_3))
      return ret, order 

    if n == 2: 
      ## fit linear line
      order = linear
      ret = np.vstack((P_0, P_3))
      return ret, order
    
    assert(n > 3)
    order = cubic
    A = self.__form_parameterization_matrix(x)
    if self.use_weighted:
      A *= kappa[:, np.newaxis]

    ## form the parametrization matrix for P_0, P_3
    A_1 = self.__form_parameterization_matrix_2(x)
    b   = pts - A_1 @ np.vstack((P_0, P_3))
    if self.use_weighted:
      b *= kappa[:, np.newaxis]

    coeff = np.linalg.lstsq(A, b, rcond = None)
    ret   = np.vstack((P_0, coeff[0], P_3))
    return ret, order
  #endregion

  #region def _find_max_error(self, contour: np.ndarray, start_ind: int, end_ind: int, coeff: np.ndarray, order: int, num = 100) -> (float, int, int):
  def _find_max_error(self, contour: np.ndarray, start_ind: int, end_ind: int, 
                      coeff: np.ndarray, order: int, num = 100) -> (float, int, int):
    """
    Auxiliary function for <__find_err_ind>
    Given the coefficients, compute the Hausdorff distance between the found polynomial and the points <contour[start_ind : end_ind + 1, :]>
    First entry of return is the error value, the second is the index in <contour> where the maximum error happens.
    """
    assert(order in [linear, quadratic, cubic])
    pts              = self.__get_pts(contour, start_ind, end_ind)
    x                = np.linspace(0, 1, num = num)
    if order == cubic:
      A   = self.__form_parameterization_matrix(x)
      A_1 = self.__form_parameterization_matrix_2(x)
      A   = np.column_stack((A_1[:,0], A, A_1[:,1]))

    if order == quadratic:
      A = np.column_stack((np.power(1 - x, 2), 2 * np.multiply(1 - x, x), np.power(x, 2)))

    if order == linear:
      A = np.column_stack((1 - x, x))

    interpolated_pts = A @ coeff
    D                = np.min(cdist(pts, interpolated_pts), axis = 1)
    ind              = np.argmax(D)
    ind_in_contour   = indices(contour, np.array([pts[ind, :]]), axis = 0)
    return D[ind], ind, ind_in_contour[0]
  #endregion

  #region def _find_weighted_max_error(self, contour: np.ndarray, start_ind: int, end_ind: int, coeff: np.ndarray, order: int, num = 100) -> (float, int, int):
  def _find_weighted_max_error(self, contour: np.ndarray, start_ind: int, end_ind: int, 
                               coeff: np.ndarray, order: int, kappa: np.ndarray, num: int = 100) -> (float, int, int):
    """
    Auxiliary function for <__find_err_ind>
    Given the coefficients, compute the Hausdorff distance between the found polynomial and the points <contour[start_ind : end_ind + 1, :]>
    First entry of return is the error value, the second is the index in <contour> where the maximum error happens.
    """
    assert(order in [linear, quadratic, cubic])
    pts              = self.__get_pts(contour, start_ind, end_ind)
    x                = np.linspace(0, 1, num = num)
    kappa            = cyclic_indexing(kappa, start_ind, end_ind)
    if order == cubic:
      A   = self.__form_parameterization_matrix(x)
      A_1 = self.__form_parameterization_matrix_2(x)
      A   = np.column_stack((A_1[:,0], A, A_1[:,1]))

    if order == quadratic:
      A = np.column_stack((np.power(1 - x, 2), 2 * np.multiply(1 - x, x), np.power(x, 2)))

    if order == linear:
      A = np.column_stack((1 - x, x))

    interpolated_pts = A @ coeff
    dist = np.min(cdist(pts, interpolated_pts), axis = 1)
    D    = np.multiply(kappa, dist)
    ind              = np.argmax(D)
    ind_in_contour   = indices(contour, np.array([pts[ind, :]]), axis = 0)
    return D[ind], ind, ind_in_contour[0]
  #endregion

  #region def __arc_length_parameterization(self, pts: np.ndarray) -> np.ndarray:
  def __arc_length_parameterization(self, pts: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for multiple functions.
    Given points <pts>, find the arc length parameterization and normalize it.
    """
    x = np.cumsum(np.sqrt(np.sum(np.power(np.diff(pts, axis = 0), 2), axis = 1)))
    return np.insert(x, 0, 0) / x[-1]
  #endregion
  
  #region def __get_pts(self, contour: np.ndarray, start_ind: int, end_ind: int) -> np.ndarray:
  def __get_pts(self, contour: np.ndarray, start_ind: int, end_ind: int) -> np.ndarray:
    """
    Auxiliary function for multiple functions.
    Get the points given the <start_ind> and <end_ind>. Both ends inclusive.
    """
    if start_ind == end_ind:
      return np.roll(contour, -start_ind, axis = 0)
    end = (end_ind + 1) % len(contour)
    if start_ind < end:
      pts = contour[start_ind : end, :]
    elif end <= start_ind: 
      pts = np.vstack((contour[start_ind: , :], contour[: end, :]))

    return pts
  #endregion

  #region def __form_parameterization_matrix(self, x: np.ndarray) -> np.ndarray:
  def __form_parameterization_matrix(self, x: np.ndarray) -> np.ndarray:
    return np.column_stack((3 * np.multiply(np.power(1 - x, 2), x) , 
                            3 * np.multiply(np.power(x, 2), 1 - x)))
  #endregion

  #region def __form_parameterization_matrix_2(self, x: np.ndarray) -> np.ndarray:
  def __form_parameterization_matrix_2(self, x: np.ndarray) -> np.ndarray:
    return np.column_stack((np.power(1 - x, 3), np.power(x, 3)))
  #endregion

  #region def _split(self, bd: np.ndarray) -> list[np.ndarray]:
  def _split(self, bd: np.ndarray) -> list[np.ndarray]:
    """
    Split a boundary if there is duplicated points except at the beginning and the end.
    """
    G = nx.DiGraph()
    edges = [(tuple(bd[ptr]), tuple(bd[ptr + 1])) for ptr in range(len(bd) - 1)]
    G.add_edges_from(edges)

    assert(all([n[1] <= 2 for n in G.in_degree(G.nodes)])) # all nodes almost intersected twice.

    dup_nodes = [t[0] for t in filter(lambda x : x[1] > 1, G.in_degree(G.nodes()))]
    if len(dup_nodes) == 0:
      return [bd]

    ret = list()
    for d in dup_nodes:
      ind = np.sort(np.argwhere(np.linalg.norm(bd - np.array(d), axis = 1) == 0)).flatten()
      if len(ind) != 2:
        continue
      new_bd = bd[ind[0] : ind[1] + 1]
      new_bd = new_bd[::-1]
      if self.__check_area(new_bd): ret.append(new_bd)
      bd = np.delete(bd, list(range(ind[0], ind[1])), axis = 0)

    ret.append(bd)
    return ret
  #endregion

  #region def __check_area(self, bd: np.ndarray) -> bool:
  def __check_area(self, bd: np.ndarray) -> bool:
    """
    Check if the area bounded by <bd> is positive
    """
    try:
      Delaunay(np.unique(bd, axis = 0))
      return True
    except QhullError:
      return False
  #endregion

  #region def __kappa_denoising(self, kappa: np.ndarray) -> np.ndarray:
  def __kappa_denoising(self, kappa: np.ndarray, sigma: int, num_iter: int = 1) -> np.ndarray:
    """
    This function denoises kappa by simple Gaussian filter.
    """
    np.abs(kappa, out = kappa)

    for _ in range(num_iter):
      kappa += .1
      gaussian_filter1d(kappa, sigma, output = kappa, mode = 'wrap')
      kappa /= np.max(kappa) # to keep the max to be 1
    return kappa
  #endregion

  #region def _any_curvature_ext(self, kappa: np.ndarray) -> bool:
  def _any_curvature_ext(self, kappa: np.ndarray) -> bool:
    if np.max(np.abs(kappa)) > 1 / self.circle_radius:
      return True
    return False
  #endregion

  #region def _fit_ellipse(self, contour: np.ndarray) -> np.ndarray:
  def _fit_ellipse(self, contour: np.ndarray) -> np.ndarray:
    """
    Given contour, try to fit an ellipse.
    Return A, B, C, D, E, F where the equation of ellipse is
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    """
    assert(len(contour) >= 3)
    if len(contour) < 6: # not enough point to determine an ellipse < 6: # not enough point to determine an ellipse < 6: # not enough point to determine an ellipse < 6: # not enough point to determine an ellipse
      ## try a circle if it still has more than three points
      if len(contour) >= 3:
        rand_ind = np.sort(np.random.choice(len(contour), size = 3, replace = False))
        pts = contour[rand_ind, :]

        RHS = -np.sum(np.power(pts, 2), axis = 1)
        LHS = np.column_stack([-2 * pts, -np.ones(3)])

        ret, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond = None)
        ret[-1] = np.sqrt(ret[0] ** 2 + ret[1] ** 2 - ret[-1])
        return ret
    
    ## uniformly get 6 points
    space = len(contour) // 6
    pts = np.vstack([contour[space * i] for i in range(6)])

    ## build matrix
    mat = np.column_stack([np.power(pts[:,0], 2), 
                           np.multiply(pts[:,0], pts[:,1]), 
                           np.power(pts[:,1],2), 
                           pts[:,0], 
                           pts[:,1], 
                           np.ones(len(pts))])
    
    S = np.dot(mat.T, mat)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2
    C[1,1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    return V[:,n]
    #endregion

  #region def _check_ellipse(self, coeff: np.ndarray, contour: np.ndarray) -> bool:
  def _check_ellipse(self, coeff: np.ndarray, contour: np.ndarray) -> bool:
    """
    Check if well fit
    """
    num = 100
    if len(coeff) == 6:
      pts = self.__parameterize_ellipse(coeff, num)
    elif len(coeff) == 3:
      pts = self.__parameterize_circle(coeff, num)
    else:
      raise ValueError("invalid coefficients")

    if np.all(np.min(cdist(contour, pts), axis = 0) < self.eps):
      return True
    else:
      return False
  #endregion 
  
  #region def __parameterize_ellipse(self, coeff: np.ndarray, num: int) -> np.ndarray:
  def __parameterize_ellipse(self, coeff: np.ndarray, num: int) -> np.ndarray:
    """
    Given the coefficietns of general equation of an ellipse,
    return parameterized points on it.
    """
    A, B, C, D, E, F = coeff

    a = A * (E ** 2) + C * (D ** 2) - B * D * E + (B ** 2 - 4 * A * C) * F
    a *= 2 * ((A + C) + np.sqrt((A - C) ** 2 + B ** 2))
    a = -np.sqrt(a)
    a /= B ** 2 - 4 * A * C

    b = A * (E ** 2) + C * (D ** 2) - B * D * E + (B ** 2 - 4 * A * C) * F
    b *= 2 * ((A + C) - np.sqrt((A - C) ** 2 + B ** 2))
    b = -np.sqrt(b)
    b /= B ** 2 - 4 * A * C

    x0 = (2 * C * D - B * E) / (B ** 2 - 4 * A * C)
    y0 = (2 * A * E - B * D) / (B ** 2 - 4 * A * C)

    if B != 0:
      theta = (C - A - np.sqrt((A - C) ** 2 + B ** 2) ) / B
      if theta != 0:
        theta = np.arctan(1 / theta)
      else:
        theta = np.pi / 2
    elif B == 0 and A < C:
      theta = 0
    elif B == 0 and A > C:
      theta = np.pi / 2
    
    angles = np.linspace(0, 2 * np.pi, num)
    pts = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
    pts[:,0] += x0
    pts[:,1] += y0

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (R @ pts.T).T
    #endregion

  #region def __parameterize_circle(self, coeff: np.ndarray, num: int) -> np.ndarray:
  def __parameterize_circle(self, coeff: np.ndarray, num: int) -> np.ndarray:
    """
    coeff[0] = x0, coeff[1] = y0, coeff[2] = radius
    """
    unit_circle = np.linspace(0, 2 * np.pi, num)
    pts = np.column_stack([coeff[2] * np.cos(unit_circle), coeff[2] * np.sin(unit_circle)])
    pts[:,0] += coeff[0]
    pts[:,1] += coeff[1]
    return pts
  #endregion

  #region def _find_inflection_points(self, kappa: np.ndarray) -> list:
  def _find_inflection_points(self, kappa: np.ndarray) -> list:
    """
    Find inflection points. Return index in <kappa>.
    """
    sgn = np.sign(kappa)
    ind = np.argwhere(sgn != 0)

    ret = list()
    for ptr in range(len(ind)):
      if sgn[ind[ptr]] != sgn[ind[(ptr + 1) % len(ind)]]:
        inflection_ind  = (ind[ptr] + ind[(ptr + 1) % len(ind)]) // 2
        inflection_ind %= len(sgn)
        ret.append(inflection_ind)

    return ret 
  #endregion

  ################### CHECKING OUT ################### 
  #region def check_out(self, ret: list, layer: int, color: tuple, is_bg: bool) -> dict:
  def check_out(self, ret: list, layer: int, color: tuple, is_bg: bool) -> dict:
    ret = {ret_str: ret, layer_str: layer, color_str: color, is_bg_str: is_bg}
    return ret
  #endregion

  #region def save(self, path:str, ret: dict) -> None:
  def save(self, path:str, ret: dict) -> None:
    save_data(path, ret)
  #endregion

######################################################################################## 
############################## FOR DEALING WITH ALL COLORS ############################# 
######################################################################################## 

class Bezier_Wrapper():
  def __init__(self, D: dict, **kwargs):
    self.D = D  # keys are transformed color  
    self.shape_layers = D[shape_layers_str]
    self.kwargs = kwargs

  #region def fit_all_colors(self, debug_plot: bool = False) -> None:
  def fit_all_colors(self, debug_plot: bool = False) -> None:
    """
    Return a dictionary keyed by color ((R, G, B), ind).
    """
    self.ret = list()
    B = Bezier_fitter(**self.kwargs)
    for key in self.shape_layers.keys():
      if self.shape_layers[key].check_is_noise():
        continue

      color = inverse_color_transform(key[0])
      print(f"doing {color}")
      try:
        if self.shape_layers[key].check_is_bg():
          book = {layer_str: self.shape_layers[key].get_bezier_level(), 
                  is_bg_str: True} 
          self.ret.append({color_str: (color, key[1]),
                          book_str : book})
        else:
          contours = self.shape_layers[key].get_cnt()
          #contours = [c[::-1, :] for c in contours]
          contours = [c[:,::-1] for c in contours]
          landmark = self.shape_layers[key].get_landmark()
          if landmark is None:
            landmark = []
          elif len(landmark) > 0:
            landmark = landmark[:,::-1]
          temp_ret = B.fitting(contours, landmark, debug_plot = debug_plot)
          temp_ret = B.check_out(temp_ret, self.shape_layers[key].get_bezier_level(), color, False)
          self.ret.append({color_str: (color, key[1]), 
                          book_str : temp_ret})
      except TypeError:
        #self.shape_layers[key].showcase()
        #breakpoint()
        continue
  #endregion

  #region def fit_one_color(self, color: tuple, debug_plot: bool = False) -> None:
  def fit_one_color(self, color: tuple, debug_plot: bool = False) -> None:
    B = Bezier_fitter(**self.kwargs)

    print(f"doing {color}")
    ret = None
    if self.shape_layers[color].check_is_bg():
      book = {layer_str: self.shape_layers[color].get_bezier_level(), 
              is_bg_str: True} 
      ret = {color_str: (color, color[1]), book_str : book}
    else:
      contours = self.shape_layers[color].get_cnt()
      landmark = self.shape_layers[color].get_landmark()
      temp_ret = B.fitting(contours, landmark, debug_plot = debug_plot)
      temp_ret = B.check_out(temp_ret, self.shape_layers[color].get_bezier_level(), color, False)
      ret = {color_str: (color, color[1]), book_str : temp_ret}

    return ret
  #endregion

  #region def check_out(self) -> list:
  def check_out(self) -> list:
    return self.ret
  #endregion

################################################################################### 
############################## FOR WRITING SVG FILES ############################## 
################################################################################### 

#region class svg_helper(): 
class svg_helper(): 
  #region def __init__(self, path: str, beziers: list, h: int, w: int):
  def __init__(self, path: str, beziers: list, h: int, w: int):
    """
    Input:
          beziers: keys: {layer: int} The k-th layer shapes.
                   keys of beziers[layer]: {"color", "curve", "is_bg"}
          h/w    : height and width of the input image
    """
    self.beziers   = beziers
    self.h, self.w = h, w
    self.path      = path
    self.S         = svgwrite.Drawing(self.path, (self.w, self.h), profile = 'tiny')

    ## Sort from bottom to top
    self.beziers = sorted(self.beziers, key = lambda b : b[book_str][layer_str], reverse = True)
  #endregion

  #region def write_svg(self) -> None:
  def write_svg(self, early_stop: tuple = None) -> None:
    for b in self.beziers:
      self.write_layer(b)
      if early_stop is not None and b[color_str] == early_stop:
        break
    self.S.save()

  #endregion

  #region def write_layer(self, b: dict) -> None:
  def write_layer(self, b: dict) -> None:
    color = b[color_str][0]
    if b[book_str][is_bg_str]:
      # if the element is background
      self.S.add(self.S.rect(insert = (0, 0), size = ('100%', '100%'), 
                 rx = None, ry = None, 
                 fill = 'rgb' + str(color)))
    else:
      ## for single connected region
      path = self._write_curve(b)
      if path:
        self.S.add(self.S.path(path, fill = "rgb" + str(color)))

      ## for multiple disconnected regions
      #paths = self._write_curves(b)
      #if paths:
      #  for path in paths:
      #    self.S.add(self.S.path(path, fill = "rgb" + str(color)))
  #endregion


  #region def _write_curves(self, b: dict) -> str:
  def _write_curves(self, b: dict) -> str:
    ret = list()
    curves = b[book_str][ret_str]

    for curve in curves:
      path = ""
      for ptr, c in enumerate(curve):
        if not c[is_ellipse_str] and not c[is_circle_str]:
          coeff, order = c[coeff_str], c[order_str]
          x, y = coeff[0,:]
          if ptr == 0:
            start_x, start_y = x, y
            path += f"M {x}, {y} "

          if order == cubic:
            x1, y1 = coeff[1, 0], coeff[1, 1]
            x2, y2 = coeff[2, 0], coeff[2, 1]
            x3, y3 = coeff[3, 0], coeff[3, 1]
            path += f"C {x1}, {y1} {x2}, {y2} {x3}, {y3} "
          elif order == quadratic:
            x1, y1 = coeff[1, 0], coeff[1, 1]
            x2, y2 = coeff[2, 0], coeff[2, 1]
            path += f"Q {x1}, {y1} {x2}, {y2} "
          elif order == linear:
            x1, y1 = coeff[1, 0], coeff[1, 1]
            path += f"L {x1}, {y1} "
          else:
            raise ValueError
        elif c[is_ellipse_str]:
          # TODO: complete here
          pass
        elif c[is_circle_str]:
          # TODO: complete here
          path += self.__write_circle(coeff)
      ret.append(path)
    return ret
  #endregion

  #region def _write_curve(self, b: dict) -> str:
  def _write_curve(self, b: dict) -> str:
    path   = ""
    curves = b[book_str][ret_str]

    for curve in curves:
      for ptr, c in enumerate(curve):
        if not c[is_ellipse_str] and not c[is_circle_str]:
          coeff, order = c[coeff_str], c[order_str]
          x, y = coeff[0,:]
          if ptr == 0:
            start_x, start_y = x, y
            path += f"M {x}, {y} "

          if order == cubic:
            x1, y1 = coeff[1, 0], coeff[1, 1]
            x2, y2 = coeff[2, 0], coeff[2, 1]
            x3, y3 = coeff[3, 0], coeff[3, 1]
            path += f"C {x1}, {y1} {x2}, {y2} {x3}, {y3} "
          elif order == quadratic:
            x1, y1 = coeff[1, 0], coeff[1, 1]
            x2, y2 = coeff[2, 0], coeff[2, 1]
            path += f"Q {x1}, {y1} {x2}, {y2} "
          elif order == linear:
            x1, y1 = coeff[1, 0], coeff[1, 1]
            path += f"L {x1}, {y1} "
          else:
            raise ValueError
        elif c[is_ellipse_str]:
          # TODO: complete here
          pass
        elif c[is_circle_str]:
          # TODO: complete here
          path += self.__write_circle(coeff)
    return path
  #endregion

  def __write_circle(self, coeff: np.ndarray) -> str:
    path += "circle cx=\"{coeff[0]}\" cy=\"{coeff[1]}\" r=\"{coeff[2]}\""

  def __write_ellipse(self, coeff: np.ndarray) -> str:
    pass

  def inspect_layers_in_order(self, path: str, D: dict) -> None:
    """
    Auxiliary function for paper writing
    """
    from os.path import join
    from cv2 import imwrite
    for ptr, b in enumerate(self.beziers):
      transformed_c = color_transformation(b[color_str][0])
      im = 255 * D['shape_layers'][(transformed_c, b[color_str][1])].get_dense_layer_im()
      im_name = join(path, str(ptr) + "_" + str(b[color_str]) + ".png")

      imwrite(im_name, im)

#endregion

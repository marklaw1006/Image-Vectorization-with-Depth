#!/usr/bin/env python3.9

from skimage.morphology import dilation, disk
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import isspmatrix
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage import measure
from numpy_indexed import intersection, difference
from itertools import product
import cv2 as cv


from .new_helper import get_boundary
from .helper import save_data

is_bg_str     = "is_bg"
segs_str      = "segs"
level_str     = "level"
color_str     = "color"
image_str     = "image"
clock_ori_str = "clock_orientation"
cmp_str       = "components"

tree_in_back = (6324919, 0)
tree_in_front = (4285336, 1)
house_front = (12179950, 0)

class Dilator():
  def __init__(self, D: dict, L: dict) -> None:
    """
    Input:
          D: Euler Elastica result
          L: Ordering result
    """
    self.D = D
    self.L = L
    assert(set(self.D.keys()) == set(self.L[cmp_str].keys()))

    #im = self.D[tree_in_back][image_str].toarray()
    #im += self.D[tree_in_front][image_str].toarray()

    #plt.imshow(im)
    #plt.show()
    #breakpoint()

    #for key in self.D.keys():
    #  print(key)

    #  im = self.D[key][image_str]
    #  if isspmatrix(im):
    #    im = im.toarray()
    #  
    #  plt.figure(str(key))
    #  plt.imshow(im)
    #  plt.show()
    
    fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
    im = self.D[house_front][image_str].toarray()
    ax[0].imshow(im)
    reg_area = self._get_reg_area(house_front)
    ax[1].imshow(reg_area)
    plt.show()
    breakpoint()

        
  def _find_inpainted_pts(self, key: tuple, thre: float = 1.5) -> np.ndarray:
    """
    Find the points on the boundary of <im> but not in that of original fidelity area
    """
    new_bd_segs = self.D[key][segs_str]
    ori_bd_segs = self.L[cmp_str][key][clock_ori_str]

    inpainted_pts = list()

    for new_seg, ori_seg in product(new_bd_segs, ori_bd_segs):
      new_seg = new_seg[:,::-1]
      if len(intersection(new_seg, ori_seg, axis = 0)) == 0:
        continue

      ## pts that are in new_seg but not in ori_seg
      pts = difference(new_seg, ori_seg, axis = 0)

      ## only points that are far enough from ori_seg are considered
      ind = np.where(np.min(cdist(pts, ori_seg), axis = 1) > thre)
      pts = np.squeeze(pts[ind, :], axis = 0)

      inpainted_pts.append(pts)

      #im = self.D[key][image_str]
      #if isspmatrix(im):
      #  im = im.toarray()
      #plt.imshow(im)
      #plt.plot(pts[:,1], pts[:,0], 'ro')
      #plt.show()
      #breakpoint()
    
    if len(inpainted_pts) == 0:
      return np.array([])
    else:
      return np.vstack(inpainted_pts)
  

  def _convolve_at_a_pt(self, im: np.ndarray, pt: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """
    Only for maximum convolution.
    """
    ret = np.zeros_like(im)

    center_x, center_y      = footprint.shape[1] // 2, footprint.shape[0] // 2
    left_len_x, right_len_x = center_x, footprint.shape[1] - center_x
    up_len_y, down_len_y    = footprint.shape[0] - center_y, center_y 

    start_x, end_x = max(pt[1] - left_len_x, 0), min(pt[1] - left_len_x + footprint.shape[1], im.shape[1])
    start_y, end_y = max(pt[0] - down_len_y, 0), min(pt[0] - down_len_y + footprint.shape[0], im.shape[0])

    max_val = np.max(np.multiply(im[start_y : end_y, start_x : end_x], footprint))

    im[start_y : end_y, start_x : end_x] = max_val
    return im
    


  def dilate_layer(self, key: tuple, footprint: tuple or np.ndarray) -> np.ndarray:
    """
    Dilate a Euler-Elastica'ed shape layer.
    Only apply kernel to <inpainted_points>
    """
    inpainted_pts = self._find_inpainted_pts(key)

    if len(inpainted_pts) == 0:
      if isspmatrix(self.D[key][image_str]):
        return self.D[key][image_str].toarray()
      else:
        return self.D[key][image_str]
    
    if isinstance(footprint, tuple):
      footprint = disk(footprint)

    assert(isinstance(footprint, np.ndarray))

    im = self.D[key][image_str]
    if isspmatrix(im):
      im = im.toarray()

    for pt in inpainted_pts:
      im = self._convolve_at_a_pt(im, pt, footprint)
    
    return im

  
  #region def dilate_all_layer(self, footprint = 1) -> None:
  def dilate_all_layer(self, footprint = 1) -> None:
    """
    Update each layer in <self.D>
    """
    footprint = disk(footprint)
    for key in self.D.keys():
      if self.D[key]["is_bg"]:
        # do nothing to background layer
        continue
    
      #im = self.D[key][image_str]
      #if isspmatrix(im):
      #  im = im.toarray()

      #im = self.dilate(key, im, footprint)

      im = self.dilate_layer(key, footprint)

      try:
        self.D[key][segs_str] = [s[:, ::-1] for s in get_boundary(im)]
      except cv.error:
        print(im)
        breakpoint()

  #endregion

  def dilate(self, key: tuple, im: np.ndarray, footprint: np.ndarray or tuple) -> np.ndarray:
    reg_area = self._get_reg_area(key)
    reg_area[reg_area <= 0] = 0

    dilation(im, footprint = footprint, out = im)

    np.multiply(im, reg_area, out = im)

    return im


  #region def __get_cmp_on_top(self, i: tuple) -> list[tuple]:
  def __get_cmp_on_top(self, i: tuple) -> list[tuple]:
    """
    Return the keys of shape layers that are on top of <i>
    """
    return (list(filter(lambda key : self.L[cmp_str][key][level_str] > self.L[cmp_str][i][level_str], self.L[cmp_str].keys())))
  #endregion

  def _get_reg_area(self, key: tuple) -> np.ndarray:
    im = self.L[cmp_str][key][image_str]
    if isspmatrix(im):
      im = im.toarray()
    
    for k in self.__get_cmp_on_top(key):
      if isspmatrix(self.L[cmp_str][k][image_str]):
        np.maximum(im, self.L[cmp_str][k][image_str].toarray(), out = im)
      else:
        np.maximum(im, self.L[cmp_str][k][image_str], out = im)

    return im



  #region def get_zero_contour(im: np.ndarray) -> list[np.ndarray]:
  def get_contour(self, im: np.ndarray) -> list[np.ndarray]:
    if np.min(im) < 0:
      offset = 0
    else:
      offset = 0.7
    contours = measure.find_contours(im, offset)
    return [c[:, ::-1] for c in contours]
  #endregion

  #region def dilate_by_depth(self) -> None:
  def dilate_by_depth(self) -> None:
    lv = .5 / self.max_depth 

    for key in self.D.keys():
      print(f"dilating {key}")
      if self.D[key]["is_bg"]:
        # do nothing to background layer
        continue

      im = self.D[key][image_str]
      if isspmatrix(im):
        im = im.toarray()
        old_im = im.copy()
      
      gaussian_filter(im, self.D[key][level_str] * lv, output = im)
      #self.D[key][segs_str] = [s[:, ::-1] for s in get_boundary(im)]

      #if len(segs) == 1 and len(segs[0]) == 1:
      #  if np.sum(np.abs(old_im - im)) < 1:
      #    print("no diff")
      #  fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
      #  ax[0].imshow(old_im)
      #  ax[1].imshow(im)
      #  plt.show()

      if np.sum(np.abs(old_im - im)) >= 1:
        self.D[key][segs_str] = [s[:, ::-1] for s in get_boundary(im)]
  #endregion

  def checkout(self, path: str) -> None:
    save_data(path, self.D)


if __name__  == "__main__":

  im = np.zeros((10,10))
  im[5,5] = 1

  new_im = gaussian_filter(im, 0.4)

  fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
  ax[0].imshow(im)
  ax[1].imshow(new_im)
  plt.show()

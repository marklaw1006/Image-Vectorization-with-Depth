#!/usr/bin/env python3.9
import networkx as nx
import numpy as np
from cv2 import filter2D

import pickle

def right_turn(p0: np.ndarray or tuple, p1: np.ndarray or tuple, p2: np.ndarray or tuple) -> bool:
  """
  Given 3 points, see if the path p0 -> p1 -> p2 includes a right turn
  """
  if (p2[1] - p0[1]) * (p1[0] - p0[0]) >= (p1[1] - p0[1]) * (p2[0] - p0[0]):
    return False
  return True

def is_counter_clockwise(pts: np.ndarray):
  area = 0

  m = len(pts)
  for ptr in range(m):
    x1, y1 = pts[ptr][0], pts[ptr][1]
    x2, y2 = pts[(ptr + 1) % m][0], pts[(ptr + 1) % m][1]
    area += (x2 - x1) * (y1 + y2)

  return area > 0

def is_clockwise(pts: np.ndarray) -> bool:
  return not is_counter_clockwise(pts)

def _make_counter_clockwise(pts):
  if is_clockwise(pts):
    return pts[::-1, :]
  return pts

def _make_clockwise(pts: np.ndarray) -> np.ndarray:
  if is_counter_clockwise(pts):
    return pts[::-1, :]
  return pts

def calculate_curvature(pts, shift_by = 1):
  tangent_vec = np.vstack((pts[shift_by:, :], pts[0 : shift_by,:])) - pts
  tangent_vec_norm = np.linalg.norm(tangent_vec, axis = 1)
  tangent_vec[:,0] = np.divide(tangent_vec[:,0], tangent_vec_norm)
  tangent_vec[:,1] = np.divide(tangent_vec[:,1], tangent_vec_norm)

  n = pts.shape[0]
  curv = []
  for ptr in range(n):
    norm1 = np.linalg.norm(pts[(ptr + shift_by) % n, :] - pts[ptr, :])
    norm2 = np.linalg.norm(tangent_vec[(ptr + shift_by) % n, :] - tangent_vec[ptr, :])
    if norm1 != 0:
      curv.append(norm2 / norm1)
    else:
      curv.append(0.0)
  return np.array(curv)

def find_curv_ext(curv, compare_pts = 10):
  ind = []
  n = len(curv)
  for ptr in range(n):
    flag = True
    for ptr2 in range(-compare_pts, compare_pts + 1):
      if ((ptr2 < 0 and curv[ptr] <= curv[ptr2]) 
         or (ptr > 0 and curv[ptr] <= curv[ptr2 % n])):
        flag = False
        break

    if flag:
      ind.append(ptr)

  return ind

def filter_curv_ext(pts, curv, delta = 7.0):
  """
  Input:
        pts: 2D numpy array. Unfiltered local curvature extrema.
             In counter-clockwise direction.
        curv: 2D numpy array. Approximated curvature. Correspond to <pts>.
        delta: float. <pts> within this distance will be regarded as one point
  Output:
        output_pt: 2D numpy array. Filtered local curvature extrema in counter-clockwise
                   orientation.
  """
  n = pts.shape[0]

  book = {}
  for ptr in range(n):
    book[tuple(pts[ptr])] = (curv[ptr], ptr) # <ptr> to keep track of the orientation

  G = nx.Graph()
  for ptr in range(n):
    dist = np.linalg.norm(pts[(ptr + 1) % n,:] - pts[ptr, :])
    G.add_node(tuple(pts[ptr, :]))
    if dist < delta:
      G.add_edge(tuple(pts[(ptr + 1) % n,:]), tuple(pts[ptr, :]), weight = dist)

  C = list(map(list, list(nx.connected_components(G))))
  for c in C:
    max_curv = book[c[0]][0]
    cur_pt   = c[0]
    for pt in c:
      if book[pt][0] > max_curv:
        cur_pt   = pt
        max_curv = book[pt][0]
    for pt in c:
      if pt != cur_pt:
        G.remove_node(pt)

  pts = [(p[0], p[1], book[p][1]) for p in G.nodes()]
  pts.sort(key = lambda x : x[2])
  pts = np.array([[p[0], p[1]] for p in pts])

  return pts

def old_W(u: np.ndarray) -> np.ndarray:
  return np.power(np.multiply(u, u - 1), 2)

def old_W_prime(u: np.ndarray) -> np.ndarray:
  return 4 * np.power(u, 3) - 6 * np.power(u, 2) + 2 * u

def old_W_prime_prime(u: np.ndarray) -> np.ndarray:
  return 12 * np.power(u, 2) - 12 * u + 2

def W(u):
  """
  The double well potential function $$W(u) = (u^{2} - 1)^{2}$$
  Input:
        u: 1D numpy array.
  Output:
        u: 1D numpy array.
  """
  if isinstance(u, list):
    u = np.array(u)

  u = np.power(u, 2) - 1
  return np.power(u, 2)

def W_prime(u):
  """
  First order derivative of the double well potential function $$W(u) = (u^{2} - 1)^{2}$$
  W^{prime)(u) = 4(u^{3} - u)
  Input:
        u: 1D numpy array.
  Output:
        u: 1D numpy array.

  """
  if isinstance(u, list):
    u = np.array(u)

  return 4 * (np.power(u,3) - u)

def W_prime_prime(u):
  """
  Second order derivative of the double well potential function $$W(u) = (u^{2} - 1)^{2}$$
  W^{prime prime)(u) = 12u^{2} - 4
  Input:
        u: 1D numpy array.
  Output:
        u: 1D numpy array.

  """
  if isinstance(u, list):
    u = np.array(u)
  return 12 * (np.power(u,2)) - 4

def x_forward_diff(u):
  """
  Forward difference along x direction, with periodic boundary condition
  Input:
        u: 2D numpy array
  Output:
        x: 2D numpy array
  """
  #return np.column_stack((u[:, 1 : ], u[:, 0])) - u
  return np.roll(u, 1, axis = 1) - u

def x_backward_diff(u):
  """
  Backward difference along x direction, with periodic boundary condition
  Input:
        u: 2D numpy array
  Output:
        x: 2D numpy array
  """
  #return np.column_stack((u[:, -1], u[:, : -1])) - u
  return np.roll(u, -1, axis = 1) - u

def y_forward_diff(u):
  """
  Forward difference along y direction, with periodic boundary condition
  Input:
        u: 2D numpy array
  Output:
        y: 2D numpy array
  """
  #return np.vstack((u[1 : ,:], u[0, :])) - u
  return np.roll(u, 1, axis = 0) - u

def y_backward_diff(u):
  """
  Backward difference along y direction, with periodic boundary condition
  Input:
        u: 2D numpy array
  Output:
        y: 2D numpy array
  """
  #return np.vstack((u[-1, :], u[ : -1, :])) - u
  return np.roll(u, -1, axis = 0) - u

def gradient(u):
  """
  This function calculates the gradient using forward difference with periodic boundary condition
  Input:
        u: 2D numpy array.
  Output:
        x: 2D numpy array.
        y: 2D numpy array.
  """
  return x_forward_diff(u), y_forward_diff(u)

def laplacian(u):
  """
  This function calculates the laplacian of u. The discretized laplacian is calculated by
  $$
  Delta u = partial_{x}^{-} partial_{x}^{+} u +  partial_{y}^{-} partial_{y}^{+} u
  $$
  Input:
        u: 2D numpy array.
  Output:
        L: 2D numpy array.
  """
  assert(np.ndim(u) == 2)
  return x_backward_diff(x_forward_diff(u)) + y_backward_diff(y_forward_diff(u))

def dnorm(x, mu, sd):
  return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size = 5, sigma = 5):
  kernel_1D = np.linspace(-(size // 2), size // 2, size)
  for i in range(size):
    kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
  kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

  kernel_2D *= 1.0 / kernel_2D.max()

  return 1.0 - kernel_2D + 1e-1

def save_data(path, result):
  output_name = path + ".data"
  handle = open(output_name, 'wb')
  pickle.dump(result, handle)
  handle.close()

def load_data(path):
  input_name = path + ".data"
  handle = open(input_name, 'rb')
  data = pickle.load(handle)
  handle.close()
  return data

def filter_quantized_image(layers, pic, colors):
  """
  Occasionally after Kmeans quantization, there are single disconneted pixels get misclassified.
  This function makes those pixel to be classified to the neigboring color
  Input:  
        layers: dictionary. keys: <colors>. values: list of numpy array
        pic   : 3D numpy array of shape (h, w, c).
        colors: list of tuples. keys of <layers>
  Output:
        layers: dictionary. keys: <colors>. values: list of numpy array
  """
  import matplotlib.pyplot as plt
  if np.ndim(pic) == 2:
    pic = np.expand_dims(pic, 2)

  h, w, _ = pic.shape

  pt_to_correct = []
  for c, v in layers.items():
    if len(v) == 0:
      continue

    coord = np.where(v != 0)

    c = np.array(c)


    for ptr in range(len(coord[0])):
      x, y = coord[0][ptr], coord[1][ptr]

      total_num = 4

      color = []
      if x != 0:
        # right
        if np.linalg.norm(pic[x - 1, y, :] - c) > 1e-2:
          total_num -= 1
          color.append(pic[x - 1, y, :])
      else:
        total_num -= 1

      if x != h - 1:
        # left
        if np.linalg.norm(pic[x + 1, y, :] - c) > 1e-2:
          total_num -= 1
          color.append(pic[x + 1, y, :])
      else:
        total_num -= 1

      if y != 0:
        # up
        if np.linalg.norm(pic[x, y - 1, :] - c) > 1e-2:
          total_num -= 1
          color.append(pic[x, y - 1, :])
      else:
        total_num -= 1

      if y != w - 1:
        # down
        if np.linalg.norm(pic[x, y + 1, :] - c) > 1e-2:
          total_num -= 1
          color.append(pic[x, y + 1, :])
      else:
        total_num -= 1

      if total_num <= 0:
        color        = np.array(color)
        color, count = np.unique(color, return_counts = True, axis = 0)
        color        = color[np.argmax(count), :]
        pic[x, y, :] = color

        #print(tuple(c), tuple(color.astype(int)))
        layers[tuple(color.astype(int))][x, y] = 1
        layers[tuple(c)][x, y] = 0

  return layers, pic

def dilate_image(image: np.ndarray, size = 3) -> np.ndarray:
  """
  Dilate the image so as to eliminate gaps
  Input:
        image: 2D numpy array
        size : int. size of the kernel. Assumed square
  """
  kernel = np.ones((size, size))
  kernel = kernel / np.sum(kernel)
  return filter2D(image, -1, kernel)
 

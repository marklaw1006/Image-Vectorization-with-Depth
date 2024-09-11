#!/usr/bin/env python3.9

#External
from os.path import join
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, bilateralFilter, COLOR_BGR2HSV
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.spatial.distance import cdist

#Internal
from .quantizer import quantizer
from .new_helper import save_data, load_data, image_transformation, color_transformation, smart_convert_sparse

class Mult_Seg():
  #region def __init__(self, pic: np.ndarray, mu: float = .3, max_phase: int = 6, max_iter: int = 3) -> None:
  def __init__(self, pic: np.ndarray, mu: float = .3, max_phase: int = 6, max_iter: int = 3) -> None:
    self.pic = pic
    self.h, self.w = self.pic.shape[:2]
    self.mu = mu
    self.max_phase = max_phase
    self.max_iter = max_iter

    self.is_rgb = np.ndim(self.pic) == 3
    self.is_gray = np.ndim(self.pic) == 2

    self.phase = np.zeros((self.h, self.w), dtype = int)
    self.length = np.zeros(self.max_phase, dtype = int)
    self.num_pixels = np.zeros(self.max_phase, dtype = int)
    self.num_pixels[0] = self.h * self.w
    if self.is_rgb:
      self.avg_colors = {phase: np.zeros(3) for phase in range(self.max_phase)}
    if self.is_gray:
      self.avg_colors = {phase: 0 for phase in range(self.max_phase)}
    self.avg_colors[0] = np.mean(self.pic, axis = (0, 1))
  #endregion
  
  #region def segment(self) -> None:
  def segment(self) -> None:
    x, y = np.arange(self.h), np.arange(self.w)
    x, y = np.meshgrid(x, y)
    coord = np.column_stack((x.flatten(), y.flatten()))
    del x, y

    cur_k = 0
    T, F = 0, 0

    ## make local variables
    pic        = self.pic.copy()
    phase      = self.phase.copy()
    length     = self.length.copy()
    num_pixels = self.num_pixels.copy()
    avg_colors = self.avg_colors.copy()
    max_phase  = self.max_phase
    max_iter   = self.max_iter
    mu         = self.mu
    is_gray    = self.is_gray
    is_rgb     = self.is_rgb
    h, w       = self.h, self.w

    for _ in range(max_iter):
      for x, y in tqdm(coord):
      #for x, y in coord:
        dE = 0
        l = phase[x, y]

        for j in range(min(max_phase, cur_k + 2)):
          if j == l:
            continue
          tj, tl = 0, 0
          if x > 0:
            tj -= 2 * (phase[x - 1, y] == j) - 1
            tl += 2 * (phase[x - 1, y] == l) - 1

          if y > 0:
            tj -= 2 * (phase[x, y - 1] == j) - 1
            tl += 2 * (phase[x, y - 1] == l) - 1

          if x < h - 1:
            tj -= 2 * (phase[x + 1, y] == j) - 1
            tl += 2 * (phase[x + 1, y] == l) - 1

          if y < w - 1:
            tj -= 2 * (phase[x, y + 1] == j) - 1
            tl += 2 * (phase[x, y + 1] == l) - 1
          
          dT = tj + tl
          #dF = ((tj / (num_pixels[j] + 1)) + 
          #      (tl / (num_pixels[l] - 1)) +  
          #      length[l] / ((num_pixels[l] - 1) * num_pixels[l]))
              
          dF = (tj / (num_pixels[j] + 1)) + (tl / (num_pixels[l] - 1))   

          if num_pixels[l] > 1:
            dF += length[l] / ((num_pixels[l] - 1) * num_pixels[l])
          if num_pixels[j] != 0:
            dF -= length[j] / ((num_pixels[j] + 1) * num_pixels[j] )
          
          if is_gray:
            j_color_diff_sq = np.linalg.norm(pic[x, y] - avg_colors[j]) ** 2
            l_color_diff_sq = np.linalg.norm(pic[x, y] - avg_colors[l]) ** 2
          elif is_rgb:
            j_color_diff_sq = np.linalg.norm(pic[x, y, :] - avg_colors[j]) / (np.sqrt(3))
            j_color_diff_sq **= 2
            l_color_diff_sq = np.linalg.norm(pic[x, y, :] - avg_colors[l]) / (np.sqrt(3))
            l_color_diff_sq **= 2
          j_num_pix_ratio = num_pixels[j] / (num_pixels[j] + 1)
          if num_pixels[l] > 1:
            l_num_pix_ratio = num_pixels[l] / (num_pixels[l] - 1)
          else:
            l_num_pix_ratio = 0
          dE_temp = (mu * (dF * T + F * dT + dF * dT) 
                     + j_color_diff_sq * j_num_pix_ratio 
                     - l_color_diff_sq * l_num_pix_ratio)

          if dE_temp < dE:
            dE   = dE_temp
            dF_c = dF
            h    = j
            dT_h = tj
            dT_l = tl

        if dE < 0:
          phase[x, y] = h
          if h == cur_k + 1:
            cur_k += 1

          if is_gray:
            l_color_diff = avg_colors[l] - pic[x, y] 
            h_color_diff = avg_colors[h] - pic[x, y] 
          elif is_rgb:
            l_color_diff = avg_colors[l] - pic[x, y, :] 
            h_color_diff = avg_colors[h] - pic[x, y, :] 
          avg_colors[l] = avg_colors[l] + (l_color_diff / (num_pixels[l] - 1))
          num_pixels[l] -= 1
          avg_colors[h] = avg_colors[h] - (h_color_diff / (num_pixels[h] + 1))
          num_pixels[h] += 1

          F         += dF_c
          T         += dT_h + dT_l
          length[l] += dT_l
          length[h] += dT_h

      plt.imshow(phase)    
      plt.draw()
      plt.pause(.01)

    self.phase = phase
    self.length = length  
    self.num_pixels = num_pixels
    self.avg_colors = avg_colors

    max_found_phase = np.max(self.phase) + 1
    print(f"Computed phase / Max allowed phase: {max_found_phase} / {max_phase}")
  #endregion

  #region def new_segment(self, allow_max_phase: bool = False) -> None:
  def new_segment(self, allow_max_phase: bool = False, stop_at: list[int] = None, display: bool = True) -> None:
    ## make local variables
    pic        = self.pic.copy()
    max_iter   = self.max_iter
    mu         = self.mu
    is_gray    = self.is_gray
    is_rgb     = self.is_rgb
    h, w       = self.h, self.w

    x, y = np.arange(self.h), np.arange(self.w)
    x, y = np.meshgrid(x, y)
    x, y = x.transpose(), y.transpose()
    coord = np.column_stack((x.flatten(), y.flatten()))
    del x, y

    # initialization
    phase      = np.zeros((h, w)) 
    phase      = np.pad(phase, ((1,1),(1,1)), 'constant', constant_values = -1)
    phase      = phase.astype(int)
    n = 100
    length, num_pixels = np.zeros(n), np.zeros(n)
    length[0] = 2 * (h + w)
    num_pixels[0] = h * w
    if is_rgb:
      avg_color = np.zeros((n, 3))
      avg_color[0, :] = np.mean(pic, axis = (0,1))
    if is_gray:
      avg_color = np.zeros(n)
      avg_color[0] = np.mean(pic)
    
    if stop_at is not None:
      ret_phases = list()

    for iter in range(max_iter):
      for x, y in tqdm(coord):
        x, y = x + 1, y + 1 # because of padding
        l    = phase[x, y]
        max_phase = np.max(phase)

        T_l = np.sum(length[ : max_phase + 1]) # total length when (x, y) in phase l.
        S_l = np.sum(np.divide(length[length > 0], num_pixels[num_pixels > 0]))
        #S_l = np.sum(np.divide(length[ : max_phase + 1], num_pixels[ : max_phase + 1]))

        dP_l = -4 + 2 * (float(phase[x - 1, y] == l) + 
                         float(phase[x + 1, y] == l) + 
                         float(phase[x, y - 1] == l) + 
                         float(phase[x, y + 1] == l))
        dn_l = num_pixels[l] - 1
        dE = np.zeros(max_phase + 2)
        K = max_phase + 2
        if allow_max_phase:
          K = min(K, self.max_phase)
        for j in range(K):
          j = int(j) 
          if j == l:
            continue

          dP_j = 4 - 2 * (float(phase[x - 1, y] == j) + 
                          float(phase[x + 1, y] == j) + 
                          float(phase[x, y - 1] == j) + 
                          float(phase[x, y + 1] == j))
          dn_j = num_pixels[j] + 1

          S_j = S_l - (length[l] / num_pixels[l])
          if num_pixels[j] > 0:
            S_j -= (length[j] / num_pixels[j])
          S_j += (dP_j / dn_j)
          #if dn_l > 0:
          S_j += (dP_l / dn_l)

          dS = S_j - S_l
          dT = dP_j + dP_l
          dE[j] = mu * (T_l * dS + S_j * dT)

          if is_rgb:
            dE[j] += (np.linalg.norm(pic[x - 1, y - 1, :] - avg_color[j, :]) ** 2 ) * (num_pixels[j] / (num_pixels[j] + 1))
            #if num_pixels[l] - 1 > 0: 
            dE[j] -= (np.linalg.norm(pic[x - 1, y - 1, :] - avg_color[l, :]) ** 2 ) * (num_pixels[l] / (num_pixels[l] - 1))
          if is_gray:
            dE[j] += ((pic[x - 1, y - 1] - avg_color[j]) ** 2) * (num_pixels[j] / (num_pixels[j] + 1))
            #if num_pixels[l] - 1 > 0: 
            dE[j] -= ((pic[x - 1, y - 1] - avg_color[l]) ** 2) * (num_pixels[l] / (num_pixels[l] - 1))

        if np.all(dE >= 0): 
          continue
      
        j = np.argmin(dE)
        phase[x, y] = j 
        length[l] += dP_l
        length[j] += 4 - 2 * (float(phase[x - 1, y] == j) + 
                              float(phase[x + 1, y] == j) + 
                              float(phase[x, y - 1] == j) + 
                              float(phase[x, y + 1] == j))

        if is_gray:
          avg_color[j] = (avg_color[j] * num_pixels[j] + pic[x - 1, y - 1]) / (num_pixels[j] + 1)
          #if num_pixels[l] - 1 > 0: 
          avg_color[l] = (avg_color[l] * num_pixels[l] - pic[x - 1, y - 1]) / (num_pixels[l] - 1)
        if is_rgb:
          avg_color[j, :] = (avg_color[j, :] * num_pixels[j] + pic[x - 1, y - 1, :]) / (num_pixels[j] + 1)
          #if num_pixels[l] - 1 > 0: 
          avg_color[l, :] = (avg_color[l, :] * num_pixels[l] - pic[x - 1, y - 1, :]) / (num_pixels[l] - 1)

        #if num_pixels[l] - 1 > 0: 
        num_pixels[l] -= 1
        num_pixels[j] += 1

      phases = np.unique(phase)
      num_phase = len(phases)
      print(f"iter: {iter}, num phases: {num_phase}")
      if stop_at is not None and iter in stop_at:
        ret_phases.append(phase[1:-1, 1:-1].copy())


      if display:
        plt.imshow(phase[1 : -1, 1 : -1])
        plt.draw()
        plt.pause(0.01)

    phases = np.unique(phase)
    num_phase = len(phases)
    print(f"Found phase: {num_phase}")
    self.phase = phase[1 : -1, 1 : -1]
    self.avg_colors = avg_color[phases, :]
    if display:
      plt.imshow(self.phase)
      plt.show()
    if stop_at is None:
      return phase[1 : -1, 1 : -1]
    else:
      return phase[1 : -1, 1 : -1], ret_phases
  #endregion

  #region def save(self, path: str) -> None:
  def save(self, path: str) -> None:
    D = {"phase": self.phase, 
         "length": self.length, 
         "num_pixels": self.num_pixels, 
         "avg_colors": self.avg_colors}
    save_data(path, D)
  #endregion

  #region def get_info(self, D: dict = None) -> (np.ndarray, np.ndarray, dict):
  def get_info(self, D: dict = None, do_correct_color: bool = True) -> (np.ndarray, np.ndarray, dict):
    if D is not None:
      phase = D["phase"]
      colors = D["avg_colors"]
    else:
      phase = self.phase
      colors = self.avg_colors
    
    if do_correct_color:
      pic, colors = self.correct_color(phase)
    else:
      pic = self.no_correct_color(phase, colors) 
      pic = pic.reshape((self.h, self.w, 3))
      colors = colors[:,::-1].astype(int)
    layers = self._quantize_layers(pic, colors)
    return pic, colors, layers
  #endregion
  
  #region def correct_color(self, phase: np.ndarray) -> np.ndarray:
  def correct_color(self, phase: np.ndarray) -> (np.ndarray, np.ndarray):
    max_phase = np.max(phase)
    pic = np.reshape(self.pic, (self.h * self.w, 3))

    ret = np.zeros_like(pic)
    for cur_phase in range(max_phase + 1): 
      cur_phase = phase == cur_phase
      labeled_im, num = label(cur_phase)
      for ptr in range(1, num + 1):
        ind = np.argwhere(labeled_im == ptr)
        ind = np.ravel_multi_index((ind[:,0], ind[:,1]), (self.h, self.w))
        color = np.mean(pic[ind, :], axis = 0)
        ret[ind, :] = color
    
    colors = np.unique(ret, axis = 0)
    colors = colors[:, ::-1]
    ret = np.reshape(ret, (self.h, self.w, 3))
    ret = ret[:,:,::-1]
    return ret, colors
  #endregion

  #region def no_correct_color(self, phase: np.ndarray, colors: np.ndarray) -> (np.ndarray, np.ndarray):
  def no_correct_color(self, phase: np.ndarray, colors: np.ndarray) -> (np.ndarray, np.ndarray):
    return colors[phase.flatten(), ::-1].astype(int)
  #endregion

  #region def _quantize_layers(self, pic: np.ndarray, colors: list) -> dict:
  def _quantize_layers(self, pic: np.ndarray, colors: list) -> dict:
    transformed_pic = image_transformation(pic)
    layers = {tuple(c): (transformed_pic == color_transformation(c)).astype(int) for c in colors}
    return layers
  #endregion

  #region def quantize_from_seg(self, seg: np.ndarray) -> (np.ndarray, np.ndarray, dict): 
  def quantize_from_seg(self, seg: np.ndarray) -> (np.ndarray, np.ndarray, dict): 
    phases = np.unique(seg)
    color = list()

    pic = self.pic.reshape((self.h * self.w, 3))

    for phase in phases:
      ind = np.argwhere(seg == phase)
      ind = np.ravel_multi_index((ind[:,0], ind[:,1]), (self.h, self.w))
      avg_color = np.mean(pic[ind, :], axis = 0)
      color.append(avg_color)
    color = np.array(color).astype(int)

    ind = np.argmin(cdist(pic, color), axis = 1)

    out = color[ind, :]
    out = out.reshape((self.h, self.w , 3))
    out = out[:,:,::-1]
    color = color[:,::-1]

    return out, color, self._quantize_layers(out, color)
  #endregion

  #region def _quantize_layers(self, pic: np.ndarray, colors: list) -> dict:
  def _quantize_layers(self, pic: np.ndarray, colors: list) -> dict:
    transformed_pic = image_transformation(pic)
    layers = {tuple(c): (transformed_pic == color_transformation(c)).astype(int) for c in colors}
    return layers
  #endregion
  
  #region def reset_mu(self, mu: float) -> None:
  def reset_mu(self, mu: float) -> None:
    self.mu = mu
  #endregion
  
  #region def reset_max_iter(self, max_iter: int) -> None:
  def reset_max_iter(self, max_iter: int) -> None:
    self.max_iter = max_iter
  #endregion

  #region def reset_pic(self, pic: np.ndarray) -> None:
  def reset_pic(self, pic: np.ndarray) -> None:
    self.pic = pic
  #endregion

  def reset_max_phase(self, phase: int) -> None:
    self.max_phase = phase

  #region def check_out(self, path: str, pic: np.ndarray, colors: np.ndarray, layers: dict) -> None:
  def check_out(self, path: str, pic: np.ndarray, colors: np.ndarray, layers: dict) -> None:
    for key, val in layers.items():
      layers[key] = smart_convert_sparse(val)

    save_data(join(path, "layers"), layers)
    save_data(join(path, "colors"), colors)
    save_data(join(path, "pic"), pic)
  #endregion


#if __name__ == "__main__":
#  path = join("real_data", "fruit5_MQ")
#  pic = imread(join(path, "small_fruit.jpg"))
#
#  ############################ Quantize with different mu(Fine) ########################### 
#  #small_mu = np.arange(1, 10) / 10
#  #big_mu = np.arange(10, 35, 5) / 10
#  #all_mu = np.concatenate((small_mu, big_mu))
#
#  #quantize_result = list()
#  #M = Mult_Seg(pic)
#  #Q = quantizer(pic, 0) # just for denoise
#  #for mu in all_mu:
#  #  print(f"mu: {mu}")
#  #  M.reset_mu(mu)
#  #  max_iter = 1 if mu < 1 else 3
#  #  M.reset_max_iter(max_iter)
#  #  phase = M.new_segment(display = False)
#  #  new_pic, colors, _ = M.quantize_from_seg(phase)
#  #  denoised_pic, colors = Q._new_denoise_pic(new_pic, colors)
#  #  ret = {"mu": mu, "phase": phase, "raw_pic": new_pic, "denoised_pic": denoised_pic}
#  #  quantize_result.append(ret)
#  
#  #save_data(join(path, "fine_seg_result"), quantize_result)
#
#  ############################ Inspect fine seg result ########################### 
#  #quantize_result = load_data(join(path, "fine_seg_result"))
#
#  #for result in quantize_result:
#  #  mu = result["mu"]
#  #  phase = result["phase"]
#  #  raw_pic = result["raw_pic"]
#  #  denoised_pic = result["denoised_pic"]
#
#  #  num_phases = len(np.unique(phase))
#  #  print(f"mu: {mu}, num of phases: {num_phases}")
#
#  #  fig, ax = plt.subplots(1, 3, sharex = True, sharey = True)
#  #  ax[0].imshow(phase)
#  #  ax[0].set_title("phase")
#  #  ax[1].imshow(raw_pic)
#  #  ax[1].set_title("raw pic")
#  #  ax[2].imshow(denoised_pic)
#  #  ax[2].set_title("denoised pic")
#  #  plt.show()
#
#  ############################ Quantize with different mu(Coarse) ########################### 
#  chosen_mu = .2
#  fine_seg_result = load_data(join(path, "fine_seg_result"))
#  key = np.argmin([np.abs(res["mu"] - chosen_mu) for res in fine_seg_result])
#
#  mu           = fine_seg_result[key]["mu"]
#  raw_pic      = fine_seg_result[key]["raw_pic"]
#  denoised_pic = fine_seg_result[key]["denoised_pic"]
#
#  M = Mult_Seg(denoised_pic, max_iter = 3, max_phase=5)
#
#  big_mu = np.linspace(10, 55, 10) / 10 
#  seg_result = list()
#  for cur_mu in big_mu:
#    M.reset_mu(cur_mu)
#    phase = M.new_segment(allow_max_phase=True, display = True)
#    ret = {"fine_seg_mu": mu, "cur_mu": cur_mu, "phase": phase}
#    seg_result.append(ret)
#  
#  name = str(int(mu * 10))
#  name = join(path, "coarse_seg_result_with_fine_mu_" + name)
#  save_data(name, seg_result)
#
#  ############################# Inspect coarse seg result ########################### 
#  #chosen_mu = .2
#  #chosen_mu = str(int(chosen_mu * 10))
#  #seg_result = load_data(join(path, "coarse_seg_result_with_fine_mu_" + chosen_mu))
#
#  #for result in seg_result:
#  #  fine_mu   = result["fine_seg_mu"]
#  #  coarse_mu = result["cur_mu"]
#  #  phase     = result["phase"]
#  #  num_phase = len(np.unique(phase))
#
#  #  print(f"fine_mu: {fine_mu}, coarse_mu: {coarse_mu}, num phases: {num_phase}")
#
#  #  plt.imshow(phase)
#  #  plt.show()

if __name__ == "__main__":
  path = join("real_data", "tree2")
  pic = load_data(join(path, "pic"))

  M = Mult_Seg(pic, mu = 1, max_iter = 3, max_phase=3)
  M.new_segment(allow_max_phase=True, display=True)

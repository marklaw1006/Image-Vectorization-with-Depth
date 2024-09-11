#!/usr/bin/env python3.9

## External
import networkx as nx
import numpy as np
from skimage.morphology import convex_hull_image
from itertools import combinations
from warnings import warn
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
from scipy.sparse import isspmatrix
from multiprocessing import cpu_count
import threading

## Internal
from .shape_layer import shape_layer_factory
from .new_helper import save_data, inverse_color_transform


shape_layers_str    = "shape_layers"
mutual_bd_graph_str = "mutual_bd_graph"
order_graph_str     = "order_graph"
grid_graph_str      = "grid_graph"

above   = 1
below   = -1
even    = 0 
no_info = -2

## for excel recording
no_mutual_pts   = "no mutual pts"
large_energy    = "energy larger than threshold"
small_energy    = "energy smaller than threshold"
touching_one_pt = "only touch at one point"
embedded        = "embedded"
barely_touching = "barely touching"
almost_st_line  = "almost straight line"

def generate_latex_graph(graph):
  latex_code = "\\begin{verbatim}\n"
  for node in graph.nodes():
    neighbors = list(graph.neighbors(node))
    latex_code += f"{node} -- {neighbors}\n"
  latex_code += "\\end{verbatim}"
  return latex_code

def generate_latex_digraph(digraph):
  latex_code = "\\begin{verbatim}\n"
  for source, target in digraph.edges():
    latex_code += f"{source} -> {target}\n"
  latex_code += "\\end{verbatim}"
  return latex_code



class shape_layer_order():
  def __init__(self, D: dict, params: dict = None) -> None:
    self.shape_layers = D[shape_layers_str]
    self.mutual_bd_graph = D[mutual_bd_graph_str]
    self.grid_graph = D[grid_graph_str]
    self.params = params

    temp_s = next(iter(self.shape_layers.values()))
    self.h, self.w = temp_s.h, temp_s.w
    del temp_s

    self.order_graph = None

    self.bg_key = self.find_bg()

    keys = list(filter(lambda key : not self.shape_layers[key].is_noise, self.shape_layers.keys()))

    self.sparse_layer_im = {key: self.shape_layers[key].get_layer_im() for key in keys}
    self.sparse_holes_filled = {key: self.shape_layers[key].get_holes_filled() for key in keys}
    self.sparse_convex_im = {key: self.shape_layers[key].get_convex_im() for key in keys}

    ## Parameters
    self.threshold = .1

    self._load_params()

  #region def _load_params(self) -> None:
  def _load_params(self) -> None:
    if self.params is None:
      return

    for key, val in self.params.items():
      if key == "threshold":
        self.threshold = val
      else:
        raise KeyError("Unknown Parameter")
  #endregion

  ############################# OREDERING ############################# 
  #region def order(self,  simple_order: bool = False, show_order_graph: bool = False) -> nx.classes.digraph.DiGraph:
  def order(self,  simple_order: bool = False, show_order_graph: bool = False) -> nx.classes.digraph.DiGraph:
    """
    If <simple_order> is true, only order shapes that share mutual boundary.
    """
    sparse_layer_im = self.sparse_layer_im.copy()
    sparse_holes_filled = self.sparse_holes_filled.copy()
    sparse_convex_im = self.sparse_convex_im.copy()
    threshold = self.threshold
    bg_key = self.bg_key
    ret = list()
    
    if simple_order:
      comb = self.mutual_bd_graph.edges()
    else:
      comb = combinations(self.shape_layers.keys(), 2)
      comb = list(filter(lambda c : c[0] != c[1], comb)) # filter same keys
      comb = list(filter(lambda c : not self.shape_layers[c[0]].check_is_noise() and not self.shape_layers[c[1]].check_is_noise(),
                        comb)) # filter one of them is noise

    for e in tqdm(comb):
      i, j = e[0], e[1]
      ret.append(self.order_ij(e[0], e[1]))
    # build order graph
    order_graph = nx.DiGraph()
    order_graph.add_nodes_from(self.shape_layers.keys())

    if self.bg_key is not None:
      edges = [(n, self.bg_key) for n in order_graph.nodes() if n != self.bg_key]
      order_graph.add_edges_from(edges)

    for r in ret:
      if r[2] == above:
        order_graph.add_edge(r[0], r[1], weight = r[3])

    self.order_graph = order_graph

    if show_order_graph:
      self._draw_graph(order_graph)

    return order_graph
  #endregion

  def _draw_graph(self, order_graph: nx.classes.digraph.DiGraph, output_latex_code: bool = False) -> None:
    dummy_graph = nx.DiGraph()
    dummy_edges = list()
    for e in order_graph.edges():
      out_node, in_node = e[0], e[1]
      new_out_node = (inverse_color_transform(out_node[0]), out_node[1])
      new_in_node  = (inverse_color_transform(in_node[0]), in_node[1])
      dummy_edges.append((new_out_node, new_in_node))

    dummy_graph.add_edges_from(dummy_edges)
    plt.clf()
    pos = nx.circular_layout(dummy_graph)
    nx.draw(dummy_graph, pos = pos, with_labels = True)
    plt.show()

    if output_latex_code:
      latex_code = nx.to_latex(dummy_graph)
      print(latex_code)
      breakpoint()

  #region def order_ij(self, i: tuple, j: tuple) -> (tuple, tuple, int):
  def order_ij(self, i: tuple, j: tuple) -> (tuple, tuple, int):
    if self.bg_key == i:
      return (j, i, above, np.inf)
    
    if self.bg_key == j:
      return (i, j, above, np.inf)

    #im_i, im_j     = self.shape_layers[i].get_dense_layer_im(), self.shape_layers[j].get_dense_layer_im()
    #im_i, im_j     = binary_fill_holes(im_i), binary_fill_holes(im_j)

    if self._new_check_embedded(i, j): 
      return (j, i, above, np.inf)

    if self._new_check_embedded(j, i): 
      return (i, j, above, -np.inf)

    diff = self._new_calculate_area_diff(i, j)
  
    if diff < -self.threshold:
      return (i, j, above, diff)
    elif diff > self.threshold:
      return (j, i, above, diff)
    else:
      return (i, j, even, diff)
  #endregion

  #region def _calculate_area_diff(self, im_i: np.ndarray, im_j: np.ndarray, conv_i: np.ndarray, conv_j: np.ndarray) -> float:
  def _calculate_area_diff(self, im_i: np.ndarray, im_j: np.ndarray, conv_i: np.ndarray, conv_j: np.ndarray) -> float:
    if isspmatrix(conv_i):
      ret1 = np.sum(conv_i.multiply(im_j))
    else:
      ret1 = np.sum(np.multiply(conv_i, im_j))
    ret1 /= np.sum(im_j)

    if isspmatrix(conv_j):
      ret2 = np.sum(conv_j.multiply(im_i))
    else:
      ret2 = np.sum(np.multiply(conv_j, im_i))
    ret2 /= np.sum(im_i)

    return ret1 - ret2
  #endregion

  #region def _new_calculate_area_diff(self, i: tuple, j: tuple) -> float:
  def _new_calculate_area_diff(self, i: tuple, j: tuple) -> float:
    if isspmatrix(self.sparse_convex_im[i]):
      ret1 = np.sum(self.sparse_convex_im[i].multiply(self.sparse_layer_im[j]).getnnz())
    else:
      if isspmatrix(self.sparse_layer_im[j]):
        ret1 = np.sum(self.sparse_layer_im[j].multiply(self.sparse_convex_im[i]))
      else:
        ret1 = np.sum(np.multiply(self.sparse_layer_im[j], self.sparse_convex_im[i]))
    
    ret1 /= np.sum(self.sparse_layer_im[j])

    if isspmatrix(self.sparse_convex_im[j]):
      ret2 = np.sum(self.sparse_convex_im[j].multiply(self.sparse_layer_im[i]).getnnz())
    else:
      if isspmatrix(self.sparse_layer_im[i]):
        ret2 = np.sum(self.sparse_layer_im[i].multiply(self.sparse_convex_im[j]))
      else:
        ret2 = np.sum(np.multiply(self.sparse_layer_im[i], self.sparse_convex_im[j]))
    
    ret2 /= np.sum(self.sparse_layer_im[i])

    return ret1 - ret2
  #endregion

  #region def _check_embedded(self, im_i: np.ndarray, im_j: np.ndarray) -> bool:
  def _check_embedded(self, im_i: np.ndarray, im_j: np.ndarray) -> bool:
    return np.sum(np.multiply(binary_fill_holes(im_i), im_j)) == np.sum(im_j)
  #endregion

  #region def _check_embedded(self, im_i: np.ndarray, im_j: np.ndarray) -> bool:
  def _new_check_embedded(self, i: tuple, j: tuple) -> bool:
    if isspmatrix(self.sparse_holes_filled[i]) and isspmatrix(self.sparse_layer_im[j]):
      return self.sparse_holes_filled[i].multiply(self.sparse_layer_im[j]).getnnz() == self.sparse_layer_im[j].getnnz()
    elif isspmatrix(self.sparse_holes_filled[i]) and not isspmatrix(self.sparse_layer_im[j]):
      return self.sparse_holes_filled[i].multiply(self.sparse_layer_im[j]).getnnz() == np.sum(self.sparse_layer_im[j])
    elif not isspmatrix(self.sparse_holes_filled[i]) and isspmatrix(self.sparse_layer_im[j]):
      return self.sparse_layer_im[j].multiply(self.sparse_holes_filled[i]).getnnz() == self.sparse_layer_im[j].getnnz()
    elif not isspmatrix(self.sparse_holes_filled[i]) and not isspmatrix(self.sparse_layer_im[j]):
      return np.sum(np.multiply(self.sparse_holes_filled[i], self.sparse_layer_im[j])) == np.sum(self.sparse_layer_im[j])

  #endregion

  #region def find_bg(self) -> tuple or None:
  def find_bg(self) -> tuple or None:
    """
    Find if any background shape layer
    If yes, return its key. Otherwise, return None
    """
    for i in self.shape_layers.keys():
      im = self.shape_layers[i].get_dense_layer_im()
      if all([np.all(im[0,:]), np.all(im[-1,:]), np.all(im[:, 0]), np.all(im[:, -1])]):
        return i
    return None
  #endregion
  
  ############################# PARALLEL OREDERING ############################# 
  def parallel_order(self) -> nx.classes.digraph.DiGraph:
    """
    Parallel implementation of ordering.
    Cannot produce excel table.
    """
    comb = list(combinations(self.shape_layers.keys(), 2))
    num_threads = np.ceil(max(cpu_count() - 1, 1) // 2).astype(int)
    batch_size = np.ceil(len(comb) / num_threads).astype(int)

    divided_comb = [comb[ptr * batch_size : min((ptr + 1) * batch_size, len(comb))] for ptr in range(num_threads)]
    self.ret = list()
    all_threads = [threading.Thread(target = self.parallel_order_wrapper, args = (c,)) for c in divided_comb]
    [thread.start() for thread in all_threads]
    [thread.join() for thread in all_threads]

    # build order graph
    order_graph = nx.DiGraph()
    order_graph.add_nodes_from(self.shape_layers.keys())

    if self.bg_key is not None:
      edges = [(n, self.bg_key) for n in order_graph.nodes() if n != self.bg_key]
      order_graph.add_edges_from(edges)

    for r in self.ret:
      if r[2] == above:
        order_graph.add_edge(r[0], r[1], weight = r[3])
    
    self.order_graph = order_graph

  def parallel_order_wrapper(self, comb: list) -> None:
    comb = list(filter(lambda c : c[0] != c[1], comb)) # filter same keys
    comb = list(filter(lambda c : not self.shape_layers[c[0]].check_is_noise() and not self.shape_layers[c[1]].check_is_noise(),
                       comb)) # filter one of them is noise
    
    ret = list()
    sparse_layer_im = self.sparse_layer_im.copy()
    sparse_holes_filled = self.sparse_holes_filled.copy()
    sparse_convex_im = self.sparse_convex_im.copy()
    threshold = self.threshold
    bg_key = self.bg_key
    for c in tqdm(comb): 
      i, j = c[0], c[1]
      ret.append(self._parallel_order_ij(i, j, sparse_layer_im[i], sparse_layer_im[j], 
                                         sparse_holes_filled[i], sparse_holes_filled[j],
                                         sparse_convex_im[i], sparse_convex_im[j], threshold, bg_key))
    self.ret += ret
  
  def _parallel_order_ij(self, i: tuple, j: tuple, i_sparse_layer_im, j_sparse_layer_im,
                         i_holes_filled, j_holes_filled, i_convex_im, j_convex_im, threshold, bg_key) -> (tuple, tuple, int):
    if bg_key == i:
      return (j, i, above, np.inf)

    if bg_key == j:
      return (i, j, above, np.inf)
    
    if self._parallel_check_embedded(i_holes_filled, j_sparse_layer_im):
      return (j, i, above, np.inf)

    if self._parallel_check_embedded(j_holes_filled, i_sparse_layer_im):
      return (i, j, above, -np.inf)
    
    diff = self._parallel_calculate_area_diff(i_convex_im, j_convex_im, i_sparse_layer_im, j_sparse_layer_im)
    if diff < -threshold:
      return (i, j, above, diff)
    elif diff > threshold:
      return (i, j, above, diff)
      return (j, i, above, diff)
    else:
      return (i, j, even, diff)

  def _parallel_check_embedded(self, i_holes_filled, j_layer_im) -> bool:
    if isspmatrix(i_holes_filled) and isspmatrix(j_layer_im):
      return i_holes_filled.multiply(j_layer_im).getnnz() == j_layer_im.getnnz()
    elif isspmatrix(i_holes_filled) and not isspmatrix(j_layer_im):
      return i_holes_filled.multiply(j_layer_im).getnnz() == np.sum(j_layer_im)
    elif not isspmatrix(i_holes_filled) and isspmatrix(j_layer_im):
      return j_layer_im.multiply(i_holes_filled).getnnz() == j_layer_im.getnnz()
    elif not isspmatrix(i_holes_filled) and not isspmatrix(j_layer_im):
      return np.sum(np.multiply(i_holes_filled, j_layer_im)) == np.sum(j_layer_im)
    
  def _parallel_calculate_area_diff(self, i_convex_im, j_convex_im, i_layer_im, j_layer_im) -> float:
    if isspmatrix(i_convex_im):
      ret1 = np.sum(i_convex_im.multiply(j_layer_im).getnnz())
    else:
      if isspmatrix(j_layer_im):
        ret1 = np.sum(j_layer_im.multiply(i_convex_im))
      else:
        ret1 = np.sum(np.multiply(j_layer_im, i_convex_im))
    
    ret1 /= np.sum(j_layer_im)

    if isspmatrix(j_convex_im):
      ret2 = np.sum(j_convex_im.multiply(i_layer_im).getnnz())
    else:
      if isspmatrix(i_layer_im):
        ret2 = np.sum(i_layer_im.multiply(j_convex_im))
      else:
        ret2 = np.sum(np.multiply(i_layer_im, j_convex_im))
    
    ret2 /= np.sum(i_layer_im)

    return ret1 - ret2

    
  ############################# PREP FOR CHECKING OUT ############################# 
  #region def _break_loops(self, G: nx.classes.digraph.DiGraph) -> nx.classes.digraph.DiGraph:
  def _break_loops(self, G: nx.classes.digraph.DiGraph) -> nx.classes.digraph.DiGraph:
    """
    Auxiliary function for <decide_levels>.
    For each cycle, break the one that has the lower weight
    """
    while True:
      #try:
      #  cycle  = nx.find_cycle(G)
      #  weight = [G.edges[c]["weight"] for c in cycle]
      #  edges  = list(zip(cycle, weight))
      #  break_edge = min(edges, key = lambda e : np.abs(e[1]))
      #  G.remove_edge(break_edge[0][0], break_edge[0][1])
      #except nx.exception.NetworkXNoCycle:
      #  return G
      try:
        cycle = nx.find_cycle(G)
        print("found cycle")
        break_edge = max([(e, self.__test_breaking(*e)) for e in cycle], key = lambda k : k[1])
        #break_edge = min([(e, self.__test_breaking(*e)) for e in cycle], key = lambda k : k[1])
        G.remove_edge(break_edge[0][0], break_edge[0][1])
      except nx.exception.NetworkXNoCycle:
        return G
  #endregion

  #region def __test_breaking(self, i: tuple, j: tuple) -> float:
  def __test_breaking(self, i: tuple, j: tuple) -> float:
    """
    Test if we should break an edge.
    Test by seeing how different would it be if we take conv_i union j and i union conv_j from the original image
    """
    ori_im = np.maximum(self.shape_layers[i].get_dense_layer_im(), self.shape_layers[j].get_dense_layer_im())
    ret = np.logical_xor(self.shape_layers[i].get_partial_convex_init() > 0, self.shape_layers[j].get_dense_layer_im()).astype(float)
    return np.sum(np.abs(ret - ori_im))
  #endregion

  #region def _decide_levels(self) -> nx.classes.digraph.DiGraph:
  def _decide_levels(self) -> nx.classes.digraph.DiGraph:
    """
    Given the digraph G, decide the levels of the components.
    The node at the bottom(i.e. sink) is at level 0.
    """
    G = nx.transitive_reduction(self._break_loops(self.order_graph))
    cur_level = 0
    ## start with sinks
    nodes = [n for n in G.nodes if G.out_degree(n) == 0]

    while len(nodes) > 0:
      cur_node = nodes.pop(0)
      self.shape_layers[cur_node].set_level(cur_level)
      cur_level += 1
      nodes += [n for n in G.predecessors(cur_node)]
    
    return G
  #endregion

  #region def _new_decide_levels(self) -> nx.classes.digraph.DiGraph:
  def _new_decide_levels(self) -> nx.classes.digraph.DiGraph:
    """
    Given the digraph G, decide the levels of the components.
    The node at the bottom(i.e. sink) is at level 0.
    """
    G = nx.transitive_reduction(self._break_loops(self.order_graph))

    L = list(reversed(list(nx.topological_sort(G))))

    for cur_level, l in enumerate(L):
      self.shape_layers[l].set_level(cur_level)
        
    return G
  #endregion

  #region def check_out(self, path: str, include_noise_shape_layers: bool = False) -> None:
  def check_out(self, path: str, include_noise_shape_layers: bool = False) -> None:
    if self.order_graph is None:
      warn("Not yet ordered....")

    self.order_graph = self._decide_levels()
    #self.order_graph = self._new_decide_levels()

    if not include_noise_shape_layers:
      key_to_remove = list()
      for key in self.shape_layers.keys():
        if self.shape_layers[key].check_is_noise():
          key_to_remove.append(key)
      for key in key_to_remove:
        self.shape_layers.pop(key)
    
    D = {shape_layers_str: self.shape_layers, mutual_bd_graph_str: self.mutual_bd_graph, grid_graph_str: self.grid_graph,
         order_graph_str: self.order_graph}
    
    save_data(path, D)
  #endregion

  ############################# VISUALIZATIOn ############################# 
  def show_order(self, save: bool = False, save_path: str = None):
    im = np.zeros((self.h, self.w))

    S = [val for val in self.shape_layers.values() if not val.check_is_noise()]

    S = sorted(S, key = lambda s : s.get_level())

    max_level = 0
    for s in S:
      if s.get_level() > max_level:
        max_level = s.get_level()
      im += s.get_dense_layer_im() * s.get_level() 
    plt.imshow(im)
    if save:
      assert(save_path is not None)
      plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
      save_path = join(save_path, "order.png")
      plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

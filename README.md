# Image Vectorization with Depth: convexified shape layers with depth ordering

Ho Law(GaTech, Atlanta), 
[Sung Ha Kang(GaTech, Atlanta)](https://kang.math.gatech.edu/index.html)

[[arXiv]](https://arxiv.org/abs/2409.06648)

Primary Contact: [[Ho Law](mailto:hlaw8@gatech.edu)]

<div align="center">
    <img src="pic/cat.gif" width="50%">
</div>

We propose new image vectorization with depth which considers depth ordering among shapes and use curvature-based inpainting for convexifying shapes in vectorization process. One advantage of our method is easy post-vectorization editing. See the picture above.

## Installation
```
conda create -n ivyso
conda activate ivyso
conda install -c anaconda python=3.8
pip install 'numpy==1.24.3'
pip install numpy_indexed
pip install 'numba==0.58.0'
pip install svgpathtools
pip install svgwrite
pip install scikit-image
pip install networkx
pip install tqdm
pip install threaded
pip install multiprocessing
pip install tqdm
pip install pillow
pip install scikit-learn
pip install numba-progress
```

## Run Experiments
```
conda activate ivyso
cd IVYSO
# PLease modify the parameters accordingly
python ivyso.py --image_path <image_path> --output_path <output_folder> --K <number of colors in K mean quantization step> --noisy_threshold <connected regions with pixels smaller than this number will be classified as noisy layers> --seg_on <do grouping quantization. Suggested for real images> --seg_off <do not do grouping quantization. Suggested for cartoon images> --mu <the weight in Mulitphase Segmentation. The smaller, the more phases> --max_phases <number of maximum phases allowed in Multiphase Segmentation> --seg_max_iter <maximum number of iterations in Multiplyase Segmentation> --delta <ordering threshold. must be between 0 and 1> --euler_max_iter <maximum number of iterations for each shape layer's inpainting> --eps <the epsilon in the double-well potential model> --a <weight on arc length term> --b <weight on curvature term> --r <level set to extract> --b_eps <fitting error tolerance> --b_radius <threshold to be identified as local curvature extrema>
# Here is an example
python ivyso.py --image_path demo_pic/burger.png --output_path demo_pic --K 20 --seg_off
```

## Reference
```
@article{law2024ivd,
title = {Image Vectorization with Depth: convexified shape layers with depth ordering},
author = {Law, Ho and Kang, Sung Ha},
year={2024}}
```
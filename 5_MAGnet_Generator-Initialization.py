#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/robgon-art/MAGnet/blob/main/5_MAGnet_Generate_Modern_Paintings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # **MAGNet: Modern Art Generator using Deep Neural Networks**
# ## How I used CLIP, SWAGAN, and a genetic algorithm to create modern paintings from text descriptions
#
# By Robert. A Gonsalves</br>
# ![sample images](https://raw.githubusercontent.com/robgon-art/MAGnet/main/MAGneg2_small.jpg)
#
# You can see my article here on [Medium](https://towardsdatascience.com/magnet-modern-art-generator-using-deep-neural-networks-57537457bb7).
#
# The source code and generated images are released under the [CC BY-SA license](https://creativecommons.org/licenses/by-sa/4.0/).</br>
# ![CC BY-NC-SA](https://licensebuttons.net/l/by-sa/3.0/88x31.png)
#
# ## Acknowledgements
# StyleGAN2 and SWAGAN implementations are by rosinality, https://github.com/rosinality/stylegan2-pytorch</br>
# Native ops are by orpatashnik, https://github.com/orpatashnik/StyleCLIP
#

# In[ ]:


#@title Initialize the System
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
get_ipython().system('pip install ftfy regex tqdm')
get_ipython().system('pip install git+https://github.com/openai/CLIP.git')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install gdown')
import torch
import warnings
warnings.filterwarnings('ignore')
import clip
from PIL import Image
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def get_top_N_semantic_similarity(similarity_list, N):
  results = zip(range(len(similarity_list)), similarity_list)
  results = sorted(results, key=lambda x: x[1],reverse = True)
  scores = []
  indices = []
  for index,score in results[:N]:
    scores.append(score)
    indices.append(index)
  return scores, indices
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
get_ipython().system('git clone https://github.com/robgon-art/MAGnet')
# MAGnet_250.pt
get_ipython().system('gdown --id 15-bJ9QyLGJqd97pAK9_O12cEQV9591o8')
import torch
import sys
from torchvision import utils
import sys
sys.path.insert(0, "/content/MAGnet")
from swagan import Generator
import matplotlib as mpl
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
input_resolution = 224
g_ema = Generator(size=512, style_dim=512, n_mlp=8, channel_multiplier=2).to("cuda")
checkpoint = torch.load("/content/MAGnet_250.pt")
g_ema.load_state_dict(checkpoint["g_ema"])

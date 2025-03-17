'''
@author    : wuutiing@outlook.com
@date      : 2025-03-16
@comments  : 

'''


import torch 
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "ToCa/flux-ToCa/src/"))

from flux.modules.cache_functions import force_init, cache_cutfresh, update_cache, cache_init, cal_type



class TokenCacheObj:
    def __init__(self, num_steps):
        timesteps = torch.linspace(1, 0, num_steps + 1).tolist()
        self.cache_dic, self.current = cache_init(timesteps)
        self.current['step']=0
        self.current['num_steps'] = len(timesteps)-1

    def cache_cutfresh(self, tokens):
        return cache_cutfresh(self.cache_dic, tokens, self.current)

    def update_cache(self, fresh_indices, fresh_tokens):
        return update_cache(fresh_indices, fresh_tokens, self.cache_dic, self.current)

    def force_init(self, tokens):
        return force_init(self.cache_dic, self.current, tokens)
        
    def cal_type(self):
        return cal_type(self.cache_dic, self.current)
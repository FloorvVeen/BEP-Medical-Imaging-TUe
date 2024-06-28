# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:19:05 2024

@author: 20202215
"""


class Model:
    def __init__(self, name, T1map, T2map, T2starmap, PDmap):
        self.name = name
        self.T1map = T1map
        self.T2map = T2map
        self.T2smap = T2starmap
        self.PDmap = PDmap 
        
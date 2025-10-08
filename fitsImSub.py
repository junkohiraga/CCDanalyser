#! /usr/bin/env python -i
# coding:utf-8
# created by J.S.Hiraga
# date 2022/06/3

# date 2022/06/09
# cube array re-arranged for 3d-fits data style
# 
# date 2022/09/29
# argv is include for input file name
# outpuf file : a lot of frames of 2dfits instead of 3d fits
#copy from readfrms6fits2d.py exchange n=2000 --> n=500
#
#date 2025/02/23
# radiation torelance experiment on 2023.9
# PNsensor experiment after damaged

import PIL as pil
#from my_functions_20190830 import *
from xaizalibs.CMOSanalyzerlib import *
import numpy as np
import matplotlib.pyplot as plt
import re, sys,os
#from ROOT import TH1F, TCanvas, TF1,gROOT, gStyle
import astropy.io.fits as ap

# global variables
default_rows = 256  # rows
default_chan = 128  # channels


lsArgv = sys.argv
if len(lsArgv) <=2:
    print('usage:readfrms.py [raw frame 2d-fits] [mean dark 2d-fits]')
    sys,exit(0)
    
raw = lsArgv[1]
mean_bg = lsArgv[2]

match = re.match(r'(.+)\.fits',lsArgv[1])
if match == None: exit()

print(raw)
basename=os.path.basename(lsArgv[1])
outfits=basename.replace(".fits", "")

Frames=np.zeros((default_rows,default_chan),dtype='float')
#Frames[:,:]=getArrFits("mean_BG_frame.fits",message=True)
Frames[:,:]=getArrFits(mean_bg,message=True)

Frames0=np.zeros((default_rows,default_chan),dtype='float')
#Frames0[:,:]=getArrFits("R20_102_32_230825_003_001_099991.fits",message=True)
Frames0[:,:]=getArrFits(raw,message=True)

sub=np.zeros((default_rows,default_chan),dtype='float')
sub=Frames0-Frames
saveAsFits(sub,outfits + '_sub.fits',message=True)

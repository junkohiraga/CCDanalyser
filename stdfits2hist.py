#! /usr/bin/env python3 
# coding:utf-8

import PIL as pil
#from my_functions_20190830 import *
from xaizalibs.CMOSanalyzerlib import *
import numpy as np
import matplotlib.pyplot as plt
import re, sys
from ROOT import TH1F, TCanvas, TF1,gROOT, gStyle
import astropy.io.fits as ap

lsArgv = sys.argv
if len(lsArgv) <= 1:
    print('usage:')
    print('stdfits2hist.py [fits_filename]')
    quit()

strFileName = lsArgv[1]
arr = ap.open(strFileName)
header = arr[0].header
image = arr[0].data
col_num = int(header.get('NAXIS1'))
row_num = int(header.get('NAXIS2'))

h1fe = TH1F("std_hist(BG)","std_hist", 1000, 0, 100)
for j in range(0,col_num,1):
  for i in range(0, row_num,1):
    h1fe.Fill(image[j,i])
#    print(image[j,i])

c1  = TCanvas("can", "histograms   ", 700, 400)
c1.SetLogy(1)
c1.SetGrid(1)

c1.SetLogy(0)
c1.SetGrid(1)
h1fe.Draw('E')
xmin=h1fe.GetMean()-10
xmax=h1fe.GetMean()+40
h1fe.GetXaxis().SetRangeUser(xmin,xmax)
h1fe.SetLineWidth(2)
h1fe.Draw()
sigma_mode=h1fe.GetMaximumBin()/10
spth=4*sigma_mode
area=h1fe.GetMaximum()
mean=h1fe.GetMean()
c1.SaveAs("std_hist.pdf")

print("split threshold = %f" %(spth))
f = open('threshold.txt','a')
f.write(str(spth))
f.close

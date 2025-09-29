#! /usr/bin/env python -i
# coding:utf-8
# original version by J.S.Hiraga
# 2022.06.01
# matplot lib does not work correctly
# ROOT module re-installed
# edited for root hist fitting 2022.06.14
# edited for 2023RadTorelance Exp 2025.02.25
# gain correction for non-split readout pnCCD 2025.06.08
# cp  ~/work/pnCCD/DarkCurrent_sato2023/DarkCalkuwahara/DarkCalibration/hist_2d_darkcurrent2.py
#2025.07.08 DETX/Y --> CCDX/Y
#2025.07.09 CCDX ではなく、camex_X/Y でないとCXCなどの場合転送回数が過剰になるので修正
#phasumに対しての補正ではなく、3x3画素の補正が必要なのでは？

#import PIL as pil
#from my_functions_20190830 import *
#from xaizalibs.CMOSanalyzerlib import *
import numpy as np
import matplotlib.pyplot as plt
import re, sys,os
from ROOT import TH1F, TCanvas, TF1,gROOT, gStyle,gPad,TH2F
import astropy.io.fits as ap
### added
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


lsArgv = sys.argv
if len(lsArgv) <= 2:
    print('usage:')
    print('evfitsb_cticor.py [fits_bintable_eventfile] [fits_bintable_ctilist]')
    sys.exit()
    
strFileName1 = lsArgv[1]
arr = ap.open(strFileName1)
header0 = arr[0].header
image0 = arr[0].data
header = arr[1].header
event_list = arr[1].data
org_cols = arr[1].columns

#Header information for CCD parameters readed
CCD_col = int(header.get('CCDCOL'))
CCD_row = int(header.get('CCDROW'))
CAMEX_num = int(header.get('CAMEXNUM'))
split_read = int(header.get('SP_READ'))
single_Mn = int(header.get('S_Mn'))
Mn_min = int(single_Mn*0.85)
Mn_max = int(single_Mn*1.07)
read_channel = int(header.get('READCH'))
transfer=int(CCD_row/(split_read+1))
gain_min = single_Mn+4000
gain_max = single_Mn+2500

event_th=int(float(header.get('EVENT_TH')))
split_th=int(float(header.get('SPLIT_TH')))

print(f"{strFileName1} is read.")

strFileName2 = lsArgv[2]
hdul = ap.open(strFileName2)
cti_list = hdul[1].data

cti_dict = {int(row[0]): row[1] for row in cti_list}
cti_off_dict = {int(row[0]): row[2] for row in cti_list}

camexx = event_list['CAMEX_X']
camexy = event_list['CAMEX_Y']
camexid = event_list['CAMEX_ID']
n_event = len(camexx)
#PHAsum,ph0-ph9 のそれぞれに対応するコラムのcti補正を施す。
ph_keys = ['PHAsum', 'ph0', 'ph1', 'ph2', 'ph3', 'ph4', 'ph5', 'ph6', 'ph7', 'ph8']
#ph_offsetX = [0, 0,-1,0,1,-1,1,-1,0,1] #vortexの定義に従ってなかった。。
#ph_offsetY = [0, 0,1,1,1,0,0,-1,-1,-1]
ph_offsetX = [0, 0, 0, 1,1, 1,0,-1,-1,-1]
ph_offsetY = [0, 0,1,1,0, -1, -1,-1,0,0,1]

new_cols = {}

#####イベント中心と周辺8画素それぞれの波高値についてCTI補正を行い、補正値を新たなコラムで追記する
for ph_key, dx, dy in zip(ph_keys, ph_offsetX, ph_offsetY):
    neighbor_x = camexx + dx
    neighbor_y = camexy + dy
    cti = np.array([cti_dict.get(int(x), 1.0) for x in neighbor_x])/np.array([cti_off_dict.get(int(x), 1.0) for x in neighbor_x])
    ph_raw = event_list[ph_key]
    ph_cor = ph_raw/(1+neighbor_y*cti)
    new_cols[ph_key + '_cti'] = ph_cor

print("PH of local maximum and adjacent pixels are corrected.")

new_dtype = event_list.dtype.descr + [(k,'f4') for k in new_cols.keys()]
new_event_list = np.empty(len(event_list),dtype=new_dtype)

for name in event_list.dtype.names:
    new_event_list[name] = event_list[name]

print("new FITS binary file will be created.")

for k, v in new_cols.items():
    new_event_list[k] = v
    
#####
#
#phasum にgain補正をかけた場合（参照）
#phasum = event_list['PHAsum'].copy()  # 元データをコピー
#PhSumCtiCor = []
##
#for i, (x, ph) in enumerate(zip(camexx, phasum)):
#    slope = cti_dict.get(int(x), 1.0)
#    offset = cti_off_dict.get(int(x),1.0)
#    denom = 1 + dety[i]*slope/offset if offset !=0 else 1
#    PhSumCtiCor.append(ph/ denom if denom != 0 else ph)
##
#
#table_dict = {name: event_list[name] for name in event_list.names}
#table_dict['PhSumCtiCor'] = np.array(PhSumCtiCor, dtype=np.float32)
#
#org_cols = event_list.columns
#new_col = ap.Column(name='PhSumCtiCor', array=PhSumCtiCor, format='E')
#new_cols = ap.ColDefs([
#        ap.Column(name=colname, array=table_dict[colname], format=cols[colname].format)
#        for colname in event_list.names
#    ])
#new_cols = ap.ColDefs(list(org_cols)+[new_col])

##HEADER情報の継承
new_hdu = ap.BinTableHDU(new_event_list,name='EVENT_LIST')
new_hdu.header['CCDCOL']= CCD_col
new_hdu.header['CCDROW']= CCD_row
new_hdu.header['CAMEXNUM'] = 1
new_hdu.header['SP_READ']= 1
new_hdu.header['READCH']= read_channel
new_hdu.header['EVENT_TH']= event_th
new_hdu.header['SPLIT_TH']= split_th
new_hdu.header['S_Mn']=single_Mn
hdulist = ap.HDUList([ap.PrimaryHDU(), new_hdu])
new_hdu.writeto("eventlist_cticor3x3.fits", overwrite=True)

#######確認のために以下でgain mapとスペクトルを作成。ROOTでfor文を回すので時間がかかる
#h1fs = TH1F("TH1F_single","TH1F_single", 10000, 0, 10000)
#h1fm = TH1F("TH1F_multi","TH1F_multi", 10000, 0, 10000)
#h1fs1 = TH1F("TH1F_single1","TH1F_single1", 10000, 0, 10000)
#h1fm1 = TH1F("TH1F_multi1","TH1F_multi1", 10000, 0, 10000)
##h1fs2 = TH1F("TH1F_single","TH1F_single", 10000, 0, 10000)
##h1fs3 = TH1F("TH1F_single","TH1F_single", 10000, 0, 10000)
##h1fm3 = TH1F("TH1F_multi","TH1F_multi", 10000, 0, 10000)
##h1fm2 = TH1F("TH1F_multi","TH1F_multi", 10000, 0, 10000)
#h2d = TH2F("TH2d_gain","TH2d_gain", read_channel, 0, read_channel, 600, 0, 6000)
#h2d_cti = TH2F("TH2d_cti","TH2d_cti", transfer, 0, transfer, 60, 0, 6000)
##print (type(image))
##print (image.shape[0])
##print (image[1:256, 181])
#for j in range(event_list.shape[0]):
#    if j % 100000 == 0:
#        print(f"{j}th event processing.......")
#    if( event_list['vortex'][j]==0 or event_list['vortex'][j]==1 or event_list['vortex'][j]==16):
#        h2d.Fill(event_list['CAMEX_X'][j],table_dict['PhSumCtiCor'][j])
##    if(event_list['CCDX'][j] < 101 or  event_list['CCDX'][j]>126): continue
#    if( event_list['vortex'][j]==0 ):
#        h1fs1.Fill(event_list['PHAsum'][j])
#        h1fs.Fill(table_dict['PhSumCtiCor'][j])
##        print(event_list['vortex'][5],event_list['PHAsum'][j],table_dict['PhSumCor'][j])
#    #elif(image[j,5]==1 or image[j,5]==16 or image[j,5]==4 or image[j,5]==64):
#    #elif(image[j,5]==1 or image[j,5]==4 or image[j,5]==64 ):
#    else:
#        h1fm1.Fill(event_list['PHAsum'][j])
#        h1fm.Fill(table_dict['PhSumCtiCor'][j])
##
##for j in range(event_list.shape[0]):
##    if( event_list['PHAsum'][j]==0 or image[j,5]==1 or image[j,5]==16):
##        h2d.Fill(image[j,2],image[j,4])
##    if(image[j,2] < 97 or  image[j,2]>128): continue
##    if( image[j,5]==0 ):
##        h1fs.Fill(image[j,4])
##    #elif(image[j,5]==1 or image[j,5]==16 or image[j,5]==4 or image[j,5]==64):
##    elif(image[j,5]==1 or image[j,5]==4 or image[j,5]==64 ):
##        h1fm.Fill(image[j,4])
#        
#model = TF1('gauss', '[0]/([2]*sqrt(2*pi))*exp(-(x-[1])*(x-[1])*0.5/[2]/[2])+[3]', 0, 10000)
#model.SetNpx(5000)
#model.SetLineColor(2)
#model.SetLineWidth(2)
#
#area=10*h1fs.GetMaximum()
#mean=h1fs.GetMean()
#std=0.5*h1fs.GetStdDev()
#xs=mean-250
#xe=mean+100
#area=500
#mean=single_Mn
#std=40
#xs=mean-500
#xe=mean+500
#model.SetParameters(area, mean, std,0)
#center1=model.GetParameter(1)
#sigma1=model.GetParameter(2)
#gStyle.SetOptFit(1111)
#
#c1  = TCanvas("can", "histograms   ", 1000, 800)
#c1.Divide(1,2)
#c1.SetLogy(1)
#c1.SetGrid(1)
#c1.cd(1)
#gPad.SetLogy(1)
#xmin=0
#xmax=mean+4000
#h1fs1.GetXaxis().SetRangeUser(xmin,xmax)
#h1fs1.Fit(model,'R', "",3000,5000)
##h1fs.SetLineWidth(1)
#h1fs1.Draw()
#
#c1.cd(2)
#gPad.SetLogy(1)
#h1fm1.GetXaxis().SetRangeUser(xmin,xmax)
#h1fm1.Fit(model,'R', "",3000,5000)
##h1fm.SetLineWidth(1)
#h1fm1.Draw()
#c1.SaveAs("sigle-Multi_spectrum_cticor.pdf")
#
#c2  = TCanvas("gain", "gain", 500, 1000)
#c2.cd()
#h2d.Draw("colz")
#c2.SaveAs("gain_variation_cticor.pdf")
#



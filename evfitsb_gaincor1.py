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

#input binary table event list , binary table gain list
#additional gain correction for adjacent 8 pixels

# cp /Users/jhiraga/work/pnCCD/CXC_sample/evfitsb_calcgain1.py 2025.07.08
# phasum --> phaedit 2026.07.09

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

cti_on = True #False 統計が不十分でCTIの計算が適切にできない場合、CTI補正していなPHA情報を使う


lsArgv = sys.argv
if len(lsArgv) <= 2:
    print('usage:')
    print('evfitsb_gaincor1.py [fits_bintable_eventfile] [fits_bintable_gainlist]')
    sys.exit()
    
strFileName1 = lsArgv[1]
arr = ap.open(strFileName1)
header0 = arr[0].header
image0 = arr[0].data
header = arr[1].header
event_list = arr[1].data
cols = arr[1].columns

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
split_then = split_th*5900/single_Mn #[eV]

strFileName2 = lsArgv[2]
hdul = ap.open(strFileName2)
gain_list = hdul[1].data

print(f"{strFileName2} is read.")


if cti_on :
    PhEdited='PHAsum_cti' #<--'PhSumCtiCor' PHA_sumに対する補正値コラム名変更2025.07.09
    ph_list = [f'ph{i}_cti' for i in range(9)]
    
else:
    PhEdited='PHAsum'
    ph_list = [f'ph{i}' for i in range(9)]
    
ph0, ph1, ph2,ph3,ph4,ph5,ph6,ph7,ph8 = ph_list
    
gain_dict = {int(row[0]): row[1] for row in gain_list}


    

camexx = event_list['CAMEX_X']
camexid = event_list['CAMEX_ID']

#PHAsum,ph0-ph9 のそれぞれに対応するコラムのgain補正を施す。
ph_keys = [PhEdited, 'ph0_cti', 'ph1_cti', 'ph2_cti', 'ph3_cti', 'ph4_cti', 'ph5_cti', 'ph6_cti', 'ph7_cti', 'ph8_cti']
ph_keys = [PhEdited, ph0, ph1, ph2,ph3,ph4,ph5,ph6,ph7,ph8]
#ph_offsets = [0, 0,-1,0,1,-1,1,-1,0,1] #vortexの定義に沿ってなかった。。
ph_offsets = [0, 0, 0, 1,1, 1,0,-1,-1,-1]
new_cols = {}

for ph_key, dx in zip(ph_keys, ph_offsets):
    neighbor_x = camexx + dx
    gain_cor = single_Mn/np.array([gain_dict.get(int(x), 1.0) for x in neighbor_x])
    ph_raw = event_list[ph_key]
    ph_cor = ph_raw * gain_cor
    new_cols[ph_key + '_cor'] = ph_cor
    
    #column gain = 5900[eV]/peak[ch] ではなく、gain_cor = total_Mn[ch]/column_Mn[ch] に変更　2025.07.09

new_dtype = event_list.dtype.descr + [(k,'f4') for k in new_cols.keys()] + [('PhSumCorNew', 'f4'), ('vortex_r', 'u1')]
new_event_list = np.empty(len(event_list),dtype=new_dtype)

for name in event_list.dtype.names:
    new_event_list[name] = event_list[name]
    
for k, v in new_cols.items():
    new_event_list[k] = v

#phasum にgain補正をかけた場合（参照）
#phasum = event_list[PhEdited].copy()  # 元データをコピー
#PhSumCor = []

#for i, (x, ph) in enumerate(zip(camexx, phasum)):
#    gain = 5900/gain_dict.get(int(x), 1.0)
#    PhSumCor.append(ph*gain)
#    
#    if i % 100000 == 0:
#        print(f"{i}th event calc.......{ph} and {gain}")
##
#table_dict = {name: event_list[name] for name in event_list.names}
#table_dict['PhSumCor'] = np.array(PhSumCor, dtype=np.float32)
#

##3x3pixelsの補正されたADUを用いて、split閾値をgain（代表値）補正した値を用いてパターン識別を再構築
ph0=new_event_list['ph0_cti_cor' if cti_on else 'ph0_cor']
phs = [new_event_list[f'ph{i}_cti_cor' if cti_on else f'ph{i}_cor'] for i in range(1, 9)]
#phs = [new_event_list[f'ph{i}_cti_cor'] for i in range(1,9)] #ph1-ph8
phs_over_thr = [np.where(ph > split_th, ph, 0) for ph in phs]
phasum_cor_new = ph0 + sum(phs_over_thr)

flags = [np.where(ph > split_then,1,0).astype(np.uint8) for ph in phs]
binary_pow = [2**i for i in range(8)]
vortex_r = sum(f*w for f,w in zip(flags[::-1], binary_pow))

new_event_list['PhSumCorNew'] = phasum_cor_new
new_event_list['vortex_r'] = vortex_r

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
hdulist.writeto("eventlist_CtiGainCorMulti.fits", overwrite=True)

######いつもの確認ROOT でforループ　gain mapの補正前と補正後、補正後のスペクトルを表示
h1fs = TH1F("TH1F_single","TH1F_single", 10000, 0, 10000)
h1fm = TH1F("TH1F_multi","TH1F_multi", 10000, 0, 10000)
h1fs1 = TH1F("TH1F_single1","TH1F_single1", 10000, 0, 10000)
h1fm1 = TH1F("TH1F_multi1","TH1F_multi1", 10000, 0, 10000)
h2d = TH2F("TH2d_gain","TH2d_gain", read_channel, 0, read_channel, 600, 1000, 7000)
h2d_cor = TH2F("TH2d_cor","TH2d_cor", read_channel, 0, read_channel, 600, 1000, 7000)
for j in range(event_list.shape[0]):
    if j % 100000 == 0:
        print(f"{j}th event processing.......")
#    if(event_list['FrameID'][j]>100000 or event_list['FrameID'][j]<90000): continue
#    if( new_event_list['vortex_r'][j]==0 or new_event_list['vortex_r'][j]==1 or new_event_list['vortex_r'][j]==16):
#        
#    if(event_list['DETX'][j] < 105 or  event_list['DETX'][j]>124): continue
    if( new_event_list['vortex_r'][j]==0):
        h2d.Fill(event_list['CAMEX_X'][j],event_list['PHAsum'][j])
        h2d_cor.Fill(event_list['CAMEX_X'][j],new_event_list['PhSumCorNew'][j])
#        if(event_list[PhEdited][j]<1000): continue
        h1fs.Fill(new_event_list['PhSumCorNew'][j])
        h1fs1.Fill(new_event_list['PHAsum_cti'][j])
#        print(event_list['vortex'][5],event_list['PHAsum'][j],table_dict['PhSumCor'][j])
    #elif(image[j,5]==1 or image[j,5]==16 or image[j,5]==4 or image[j,5]==64):
    #elif(image[j,5]==1 or image[j,5]==4 or image[j,5]==64 ):
    else:
#        if(event_list[PhEdited][j]<1000): continue
        h1fm.Fill(new_event_list['PhSumCorNew'][j])
        h1fm1.Fill(new_event_list['PHAsum_cti'][j])
#PHAsum_cor--->PHAsum_cti column name exchanged on 20250709
   
        
model = TF1('gauss', '[0]/([2]*sqrt(2*pi))*exp(-(x-[1])*(x-[1])*0.5/[2]/[2])+[3]', 0, 10000)
model.SetNpx(5000)
model.SetLineColor(2)
model.SetLineWidth(2)

area=10*h1fs.GetMaximum()
mean=h1fs.GetMean()
mean=4530
std=0.5*h1fs.GetStdDev()
xs=mean-350
xe=mean+300
xss=mean-500
xee=mean+1000
area=500
#mean=5900
std=40
model.SetParameters(area, mean, std,0)
center1=model.GetParameter(1)
sigma1=model.GetParameter(2)
gStyle.SetOptFit(1111)
xmin=500
xmax=mean+5000
c1  = TCanvas("can", "histograms   ", 1000, 800)
c1.Divide(1,2)
c1.SetLogy(1)
c1.SetGrid(1)
c1.cd(1)
gPad.SetLogy(1)
h1fs.GetXaxis().SetRangeUser(xmin,xmax)
h1fs.Fit(model,'R', "",xs,xe)
#h1fs.SetLineWidth(1)
h1fs.Draw()

c1.cd(2)
gPad.SetLogy(1)
h1fm.GetXaxis().SetRangeUser(xmin,xmax)
h1fm.Fit(model,'R', "",xs,xe)
#h1fm.SetLineWidth(1)
h1fm.Draw()
c1.SaveAs("single-Multi_spectrum_gaincor1.pdf")

c2  = TCanvas("gain", "gain", 1000, 700)
h2d.GetYaxis().SetRangeUser(xss,xee)
h2d_cor.GetYaxis().SetRangeUser(xss,xee)
c2.Divide(1,2)
c2.cd(1)
h2d.Draw("colz")
c2.cd(2)
h2d_cor.Draw("colz")
c2.SaveAs("gain_variation_gaincor1.pdf")



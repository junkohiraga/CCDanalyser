#! /usr/bin/env python -i
# coding:utf-8
# original version by J.S.Hiraga
# 2022.06.01
# matplot lib does not work correctly
# ROOT module re-installed
# edited for root hist fitting 2022.06.14
# update 2025.06.19 for CXC and data type :FRAME_ID I-->J
# line 70 astropy numpy arrayの名前が間違っていたので修正
# split readout に対応するため、CAMEX_ID, CAMEX_X, CAMEX_Yを導入する。optionはめんどくさい。
# corrected phaのためのコラムをあらかじめ用意していたが、以降のプロセスで追加コラムが多数発生するのでやめた。2025/06/26
# CXC or Radiated-CCD など、split readout or not に対応できるように、CAMEXが１の場合とそれ以外の場合で分けた 2025/06/30
# fits headerに情報を追記してheaderから読み込んでもらうのは未実装：2025/6/30
# eventfile.fits のheader情報のうち、rframe fitsのファイルパスが全フレーム記述され重すぎるので、コピーすることをやめました。2025/7/2
# 2025/07/08

#################CCD素子の仕様に応じて適切に書き換える###############
CCD_col = 264
CCD_row = 264
CAMEX_num = 4
split_read = 1
cti_on = 1
read_channel = int(CCD_col*(split_read+1))
#################EDIT appropriate numbers for your CCD property###############


#import PIL as pil
#from my_functions_20190830 import *
from xaizalibs.CMOSanalyzerlib import *
import numpy as np
import matplotlib.pyplot as plt
import re, sys,os
from ROOT import TH1F, TCanvas, TF1,gROOT, gStyle,gPad,TH2F
import astropy.io.fits as ap


lsArgv = sys.argv
if len(lsArgv) <= 1:
    print('usage:')
    print('fitsim2ev.py [fits_filename]')
    quit()

strFileName = lsArgv[1]

if not re.search(r'\.fits$',strFileName):
    print("Error: It is not FITS file.")
    sys.exit()
    
match = re.search(r'(.+)\.fits',strFileName)
basename=match.group(1)
outfits=basename + "_bt.fits"


arr = ap.open(strFileName)
header = arr[0].header.copy()
image = arr[0].data
image = image.astype(np.float32)
rows, cols = image.shape
event_th=int(float(header.get('EVENT_TH')))
split_th=int(float(header.get('SPLIT_TH')))
#THはfloatで格納したので。

h1fs = TH1F("TH1F_single","TH1F_single", 10000, 0, 20000)
h1fm = TH1F("TH1F_multi","TH1F_multi", 10000, 0, 20000)
#h2d = TH2F("TH2d_gain","TH2d_gain", 528, 0, 528, 700, 4200, 4900)
h2d = TH2F("TH2d_gain","TH2d_gain", read_channel, 0, read_channel, 300, 3000, 6000)


#CAMEX_1 = 1 < x  <= CCD_col/2 and 1 < y < CCD_row/2, CAMEX_X = x, CAMEX_Y = y
#CAMEX_2 = CCD_col/2 < x <= CCD_col and 1 < y < CCD_row/2, CAMEX_X = x, CAMEX_Y = y
#CAMEX_3 = CCD_col/2 < x <= CCD_col and CCD_row/2 < y < CCD_row, CAMEX_X = CCD_col - x, CAMEX_Y = CCD_row - y
#CAMEX_4 = 1 < x < CCD_col/2 and CCD_row/2 < y < CCD_roaw, CAMEX_X = CCD_col - x, CAMEX_Y = CCD_roaw - y

# x, y座標の配列（多分　0始まり）
x_coords = image[:,2]
y_coords = image[:,1]


## CAMEX座標の初期化
camex_x = np.zeros_like(image[:,2])
camex_y = np.zeros_like(image[:,1])
camex_id = np.full(rows, 0.0, dtype=np.int16)


if (CAMEX_num ==1):
    camex_x[:] = x_coords[:]
    camex_y[:] = y_coords[:]
    camex_id[:] = 0

else:

    ## CAMEX1: 0 < x <= CCD_col/2 and 1 < y < CCD_row/2
    camex1 = (x_coords > 0) & (x_coords <= CCD_col / 2) & (y_coords > 0) & (y_coords <= CCD_row / 2)
    camex_x[camex1] = x_coords[camex1]
    camex_y[camex1] = y_coords[camex1]
    camex_id[camex1] = 0

    ## CAMEX2: CCD_col/2 < x <= CCD_col and 1 < y < CCD_row/2
    camex2 = (x_coords > CCD_col / 2) & (x_coords <= CCD_col) & (y_coords > 0) & (y_coords <= CCD_row / 2)
    camex_x[camex2] = x_coords[camex2]
    camex_y[camex2] = y_coords[camex2]
    camex_id[camex2] = 1
    #
    ## 条件3: CCD_col/2 < x <= CCD_col and CCD_row/2 < y < CCD_row
    camex3 = (x_coords > CCD_col / 2) & (x_coords <= CCD_col) & (y_coords > CCD_row / 2) & (y_coords <= CCD_row)
    camex_x[camex3] = CCD_col - x_coords[camex3] + CCD_col
    camex_y[camex3] = CCD_row - y_coords[camex3]
    camex_id[camex3] = 2
    #
    ## 条件4: 1 < x < CCD_col/2 and CCD_row/2 < y < CCD_row
    camex4 = (x_coords > 0) & (x_coords <= CCD_col / 2) & (y_coords > CCD_row / 2) & (y_coords <= CCD_row)
    camex_x[camex4] = CCD_col - x_coords[camex4] + CCD_col
    camex_y[camex4] = CCD_row - y_coords[camex4]
    camex_id[camex4] = 3
    #

phsumcor_value = 0.0  # すべての行にこの値を入れる
phsumcor_data = np.full(rows, phsumcor_value, dtype=np.float32)


column_names = ['FrameID','CCDY','CCDX','MAXleak','PHAsum','vortex','ph0','ph1','ph2','ph3','ph4','ph5','ph6','ph7','ph8']
format = ['J','I','I','I','E','I','E','E','E','E','E','E','E','E','E']
columns = []

for i, name in enumerate(column_names):
    col_data = image[:, i]
    col = ap.Column(name=name, format=format[i], array=col_data)
    columns.append(col)

coldefs = ap.ColDefs(columns)
table_hdu = ap.BinTableHDU.from_columns(coldefs, name='EVENT LIST')
#table_hdu.header.extend(header, update=True)  # update=True で重複キーは上書きされる

# PrimaryHDUは空でOK（ただしヘッダーつけたければこちらにも追加できる）
primary_hdu = ap.PrimaryHDU()

old_cols = table_hdu.columns

new_cols = old_cols + ap.ColDefs([
    ap.Column(name='CAMEX_X', format='I', array=camex_x),
    ap.Column(name='CAMEX_Y', format='I', array=camex_y),
    ap.Column(name='CAMEX_ID', format='I', array=camex_id)])

new_table_hdu = ap.BinTableHDU.from_columns(new_cols, name='EVENT_LIST')
#new_table_hdu.header.extend(header, update=True)
new_table_hdu.header['CCDCOL']= CCD_col
new_table_hdu.header['CCDROW']= CCD_row
new_table_hdu.header['CAMEXNUM'] = 1
new_table_hdu.header['SP_READ']= 1
new_table_hdu.header['READCH']= read_channel
new_table_hdu.header['EVENT_TH']= event_th
new_table_hdu.header['SPLIT_TH']= split_th


#################確認のためにgain map とスペクトルを書き出す。
##補正前のシングルスペクトルMn-Kaピークを抽出してイベントファイルヘッダーに書き込む
for j in range(image.shape[0]):
    h2d.Fill(camex_x[j],image[j,4])
#    h2d.Fill(image[j,2],image[j,4])
    if( image[j,5]==0):
        h1fs.Fill(image[j,4])
#        h2d.Fill(image[j,2],image[j,4])
    else:
        h1fm.Fill(image[j,4])

model = TF1('gauss', '[0]/([2]*sqrt(2*pi))*exp(-(x-[1])*(x-[1])*0.5/[2]/[2])+[3]', 0, 20000)
model.SetNpx(5000)
model.SetLineColor(2)
model.SetLineWidth(2)

area=10*h1fs.GetMaximum()
mean=h1fs.GetMean()
std=0.5*h1fs.GetStdDev()
area=50
mean=4500
std=40
xs=mean-500
xe=mean+500
model.SetParameters(area, mean, std,0)
center1=model.GetParameter(1)
sigma1=model.GetParameter(2)
gStyle.SetOptFit(1111)

c1  = TCanvas("can", "histograms   ", 1000, 800)
c1.Divide(1,2)
c1.SetLogy(1)
c1.SetGrid(1)
c1.cd(1)
gPad.SetLogy(1)
xmin=0
xmax=mean+4500
h1fs.GetXaxis().SetRangeUser(xmin,xmax)
h1fs.Fit(model,'R', "",xs,xe)
#h1fs.SetLineWidth(1)
single_Mn=model.GetParameter(1)
h1fs.Draw()

c1.cd(2)
gPad.SetLogy(1)
h1fm.GetXaxis().SetRangeUser(xmin,xmax)
h1fm.Fit(model,'R', "",xs,xe)
#h1fm.SetLineWidth(1)
h1fm.Draw()
c1.SaveAs("sigle-Multi_spectrum.pdf")

c2  = TCanvas("gain", "gain", 1000, 500)
c2.cd()
h2d.Draw("colz")
c2.SaveAs("gain_variation.pdf")


new_table_hdu.header['S_Mn']=single_Mn
hdulist = ap.HDUList([ap.PrimaryHDU(), new_table_hdu])
hdulist.writeto(outfits, overwrite=True)

print(f"File format was transfered from {strFileName} to {outfits}")


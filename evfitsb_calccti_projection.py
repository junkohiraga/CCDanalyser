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
# edit for flip readout(CXC) refering to the additional column of event file of CAMEX_X, CAMEX_Y, CAMEX_ID
# CCD parameters are read via fits header information 2025/07/02
# 2025.07.08
# 2025.07.09 CTI factors of slope and offset should be saved (before slope only)

#import PIL as pil
#from my_functions_20190830 import *
#from xaizalibs.CMOSanalyzerlib import *
import numpy as np
import matplotlib.pyplot as plt
import re, sys,os
from ROOT import TH1F, TCanvas, TF1,gROOT, gStyle,gPad,TH2F,TGraph
import astropy.io.fits as ap
### added
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress, mode
#graph = TGraph()

# lnear function
def linear(x, slope,offset):
    return slope * x + offset

lsArgv = sys.argv
if len(lsArgv) <= 1:
    print('usage:')
    print('evfits_calccti.py [fits_filename(binary table)]')
    sys.exit()
    
strFileName = lsArgv[1]
arr = ap.open(strFileName)
header0 = arr[0].header
image0 = arr[0].data
header = arr[1].header
event_list = arr[1].data
org_column = arr[1].columns
orig_hdu = arr[1]

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
Mn_min = 4000
Mn_max = 8000


print(f"{strFileName} is read.")

####Radiation damaged @2023 CCD's pixel format
#CCD_nrows = 256
#CCD_ncolumns = 128
#CAMEX_num = 4
#split_read = 1
#read_channel = CCD_ncolumns

###FF mode of CXC pixel format
#CCD_col = 264
#CCD_row = 264
#CAMEX_num = 4
#split_read = 1


ccdx = event_list['CCDX']
ccdy = event_list['CCDY']
camexx = event_list['CAMEX_X']
camexy = event_list['CAMEX_Y']
phasum = event_list['PHAsum']
vortex = event_list['vortex']

PhSumCtiCor = phasum.copy() #initialization

n_event = len(ccdx)

uniqueX= np.unique(camexx)

result_X = []
result_slope = []
result_offset = []

result_slopeRoot = []
result_offsetRoot = []

linear_model = TF1("fit_func", "[0]*x + [1]", 0, 512)

bin_width = 5
bins = np.arange(0, 256+bin_width, bin_width)


# ==== ガウス関数 ====
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2 / (2*sigma**2))


for x_val in uniqueX:
    idx = np.where(camexx == x_val)[0]
    
    camexy_sub = camexy[idx]
    phasum_sub = phasum[idx]
    vortex_sub = vortex[idx]
    
    mask = (phasum_sub > Mn_min) & (phasum_sub < Mn_max) & (vortex_sub==0)
#    mask = (phasum_sub > 4000) & (phasum_sub < 4800) &(vortex ==0)
    if np.sum(mask)<2: continue
    print(np.sum(mask))
    camexy_fit = camexy_sub[mask]
    phasum_fit = phasum_sub[mask]
    
    camexy_fit_f = camexy_fit.astype(np.float32)
    phasum_fit_f = phasum_fit.astype(np.float32)
    
    graph=TGraph(len(camexy_fit), camexy_fit_f, phasum_fit_f)
    
    
    x_centers = []
    y_means = []
    y_mode = []
    slope, offset, r_value =0.,0.,0.,
    for i in range(len(bins)-1):
        mask = (camexy_fit >= bins[i]) & (camexy_fit < bins[i+1])
        y_in_bin = phasum_fit[mask]
        if len(y_in_bin) < 10:  # データ少なすぎならスキップ
            continue
    
    # ヒストグラム
        counts, edges = np.histogram(y_in_bin, bins=30)
        centers = 0.5*(edges[1:]+edges[:-1])
#        plt.hist(y_in_bin, bins=30)
#        plt.show()
        
        
    # 初期値: 最大値, 平均, 標準偏差
        p0 = [counts.max(), np.mean(y_in_bin), np.std(y_in_bin)]
    
        try:
            popt, _ = curve_fit(gauss, centers, counts, p0=p0)
            A, mu, sigma = popt
            x_centers.append((bins[i]+bins[i+1])/2)  # ビンの中心
            y_means.append(mu)  # ガウスの平均
            y_mode.append(edges[np.argmax(counts)])
            print(f"{i}, mode:{y_mode[-1]:.2f}, mean:{y_means[-1]:.2f}, {x_centers[-1]}")
            
        except RuntimeError:
            pass  # フィット失敗時は無視
#        print(f"{edges[np.argmax(counts)]}, {mu}")
    # ==== 結果プロット ====
#    plt.figure(figsize=(8,5))
#    plt.scatter(camexy_fit, phasum_fit, s=10, alpha=0.3, label="original data")
#    plt.plot(x_centers, y_means, "ro-", label="Gaussian fit mean per bin")
#    plt.plot(x_centers, y_mode, "yo-")
#    plt.xlabel("X")
#    plt.ylabel("Y")
#    plt.legend()
#    plt.show()
    
    try:
#        popt, _ = curve_fit(linear, camexy_fit, phasum_fit,p0=[-4.0,6250])
#        slope,offset = popt
#        slope,offset = [-4.0, 6250]
#        result = linregress(camexy_fit, phasum_fit)
#        print(y_means)
#       result = linregress(x_centers, y_means)
        print(f"{len(x_centers)} is not equal to {len(y_means)} or {len(y_mode)}")
        result = linregress(x_centers, y_mode)
        slope, offset, r_value = result.slope, result.intercept, result.rvalue
        x_centers = np.array(x_centers)
        y_fit = slope * x_centers + offset
#        print(f"{x_val}  fitting:{slope:.2f}, {offset:.2f}, {r_value:.2f}")
        result_X.append(x_val)
        result_slope.append(slope)
        result_offset.append(offset)
        
#        graph.Fit(linear_model,"Q")
#        slope = linear_model.GetParameter(0)
#        offset = linear_model.GetParameter(1)
#        result_slopeRoot.append(slope)
#        result_offsetRoot.append(offset)
        
        
    except RuntimeError:
        slope, offset = 0.0, 1.0 # no correction if fitting failure
    
    print(f"{x_val} is processing .......")

#    print(f"{slope:.2f}, {offset:.2f}, {r_value:.2f}")
#    print(", ".join(f"{val:.2f}" for val in y_mode))
#    print(",".join(f"{val:.2f}" for val in y_means))
    plt.figure(figsize = (6,4))
    plt.scatter(camexy_fit, phasum_fit, s=0.2, label='data')
    plt.plot(x_centers, y_means, "ro-", label="Gaussian fit mean per bin")
#    plt.plot(x_centers, y_mode, "yo-")
#    plt.plot(x_centers, y_fit, "b-")
    camexy_range = np.linspace(0, transfer, transfer)
    #plt.plot(camexy_range, linear(camexy_range, slope, offset), 'r-', label='Fit')
    plt.plot(x_centers, linear(x_centers, slope, offset), 'r-', label='Fit')
#    plt.plot(camexy_range, linear(camexy_range, -5.0, 6250), 'r-', label='Fit')
    plt.xlabel('camexY:transfer')
    plt.ylabel('PHAsum')
    plt.title(f'X={x_val}, slope={slope:.4f}')
    plt.legend()
    
    save_dir = './cti'
    os.makedirs(save_dir,exist_ok=True)
    
    fname = f"fit_X{x_val:03.0f}.pdf"
    filepath =os.path.join(save_dir,fname)
    plt.savefig(filepath, format='pdf')
#    plt.show()
    plt.close()
    
#X=527 の表示に時間がかかる謎の挙動のためにコメントアウト
#    c1 = TCanvas("", "", 800, 600)
##    c1.SaveAs("multi_page.pdf[")  # PDF 開始
#    title = f'X={x_val}, slope={slope:.4f}'
#    graph.SetTitle(title)
#    graph.Draw("AP")
#    fname = f"fit_X{x_val:03.0f}_TGraph.pdf"
#    filepath =os.path.join(save_dir,fname)
#    c1.SaveAs(filepath)

col1 = ap.Column(name='readCH', format='J', array=np.array(result_X))
col2 = ap.Column(name='CTIslope', format='E', array=np.array(result_slope))
col3 = ap.Column(name='CTIoff', format='E', array=np.array(result_offset))

hdu = ap.BinTableHDU.from_columns([col1, col2, col3],name='COL_CTI')
hdu.writeto("cti_list.fits", overwrite=True)


df = pd.DataFrame({
#    "readCH": range(len(gain_list)),
    "readCH": [int(v) for v in result_X],
    "CTI": [float(v) for v in result_slope]
})

df.to_csv("cti_list.csv", index=True)

#
#h1fs = TH1F("TH1F_single","TH1F_single", 10000, 0, 20000)
#h1fm = TH1F("TH1F_multi","TH1F_multi", 10000, 0, 20000)
#h2d = TH2F("TH2d_gain","TH2d_gain", read_channel, 0, read_channel, 300, 3000, 6000)
#h2d_cti = TH2F("TH2d_cti","TH2d_cti", transfer, 0, transfer, 60, 0, 6000)
#
#for j in range(event_list.shape[0]):
#    if j % 100000 == 0:
#        print(f"{j}th event processing.......")
#    if( event_list['vortex'][j]==0 or event_list['vortex'][j]==1 or event_list['vortex'][j]==16):
#        h2d.Fill(event_list['CAMEX_X'][j],event_list['PHAsum'][j])
##        h2d_cor.Fill(event_list['DETX'][j],table_dict['PhSumCor'][j])
##    if(event_list['CCDX'][j] < 101 or  event_list['CCDX'][j]>126): continue
#    if( event_list['vortex'][j]==0 ):
#        h1fs.Fill(event_list['PHAsum'][j])
##        h1fs1.Fill(table_dict['PhSumCor'][j])
##        print(event_list['vortex'][5],event_list['PHAsum'][j],table_dict['PhSumCor'][j])
#    #elif(image[j,5]==1 or image[j,5]==16 or image[j,5]==4 or image[j,5]==64):
#    #elif(image[j,5]==1 or image[j,5]==4 or image[j,5]==64 ):
#    else:
#        h1fm.Fill(event_list['PHAsum'][j])
# #       h1fm1.Fill(table_dict['PhSumCor'][j])
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
#area=50
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
#h1fs.GetXaxis().SetRangeUser(xmin,xmax)
#h1fs.Fit(model,'R', "",xs,xe)
##h1fs.SetLineWidth(1)
#h1fs.Draw()
#
#c1.cd(2)
#gPad.SetLogy(1)
#h1fm.GetXaxis().SetRangeUser(xmin,xmax)
#h1fm.Fit(model,'R', "",xs,xe)
##h1fm.SetLineWidth(1)
#h1fm.Draw()
#c1.SaveAs("sigle-Multi_spectrum_calccti.pdf")
#
#c2  = TCanvas("gain", "gain", 1000, 500)
#c2.cd()
#h2d.Draw("colz")
#c2.SaveAs("gain_variation_calccti.pdf")
#

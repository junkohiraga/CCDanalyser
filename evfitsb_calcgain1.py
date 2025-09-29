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
# edit for flip readout(CXC) refering to the additional column of event file of CAMEX_X, CAMEX_Y, CAMEX_ID 20250624
# cp /Users/jhiraga/work/pnCCD/CXC_sample/evfitsb_calcgain1.py 2025.07.08


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



lsArgv = sys.argv
if len(lsArgv) <= 1:
    print('usage:')
    print('evfitsb_calcgain1.py [fits_filename(binary table)]')
    sys.exit()
 
strFileName = lsArgv[1]
arr = ap.open(strFileName)
header0 = arr[0].header
image0 = arr[0].data
header = arr[1].header
event_list = arr[1].data
org_column = arr[1].columns
orig_hdu = arr[1]


#########各コラムの統計が不十分でCTIが求められない場合は、cti_on=False とする。CTI補正PHAsum_cti+ph?_ctiではなくCTI補正前PHAsum+ph?を用いる
#####
cti_on = True
#####
#########
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

print(f"{strFileName} is read.")

ccdx = event_list['CCDX']
ccdy = event_list['CCDY']
camexx = event_list['CAMEX_X']
camexy = event_list['CAMEX_Y']
phasum = event_list['PHAsum']


if cti_on :
        PhEdited='PHAsum_cti' #<--'PhSumCtiCor' PHA_sumに対する補正値コラム名変更2025.07.09
else:
    PhEdited='PHAsum'

one_column_event = [
    np.array(event_list[PhEdited][(event_list['CAMEX_X'] == col) & (event_list[PhEdited] < 4800) & (event_list[PhEdited] > 4200) ])
    for col in range(read_channel)
]
# vortexで判断していない、つまり全イベントを使っていた-->vortex=0のみ使用？2025.07.09
#    & (event_list['FrameID']>=90000) & (event_list['FrameID']<100000)
# カラムごとのイベントスペクトルのKα輝線からgain arrayを作成
def gauss_fit(x, amp, mu, sigma):
    return amp * np.exp(-(x-mu)**2/(2*(sigma**2)))

def calc_gain(events, index, target_mu=5900, bins=50, range=(4200,4800)):#汎用性は高くない。Mn-Ka=5900eV, Mn-Kaピークが4500ch付近以外の時は編集が必要
    hist, edges = np.histogram(events, bins=bins)
    centers = edges[:-1] + np.diff(edges) / 2
    mean = np.mean(events)
    init_vals = [np.max(hist), mean, 50]

    
    try:
        evnum=len(events)
#        print(f"index={index:03.0f} with event num = {evnum:05d}")
#        popt, _ = curve_fit(gauss_fit, centers, hist, p0=init_vals)
##        gain = target_mu / popt[1]
#        gain = popt[1]
#        std = popt[2]
#        print(target_mu, popt[1],popt[2])
        if(evnum<50):
            print(f"index={index:03.0f} low-counting column thus excloded")
            gain = mean
            std = 50
            popt = [1, 50, 1]
        else:
            popt, _ = curve_fit(gauss_fit, centers, hist, p0=init_vals)
#        gain = target_mu / popt[1]
            gain = popt[1]
            std = popt[2]
#            print(target_mu, popt[1],popt[2])

    except Exception:
        gain = 1.0  # フィッティング失敗時は１
        std = -1.0
        popt = [0, 0, 1]
        print(f"ex index={index:03.0f} with event numbrt={events[0]}")
        
    plt.figure(figsize = (6,4))
    plt.hist(events, bins=bins, range=(4200,4800),  label='data')
    x_fit = np.linspace(4200, 4800, 6000)
    y_fit = gauss_fit(x_fit, *popt)
    plt.plot(x_fit,y_fit,color='red')
    plt.grid(True)
    plt.xlabel('PHAsum')
    plt.ylabel('count/bin')
    plt.title(f'read_channel={index}, mu={gain:.4f}')
    plt.legend()
    
    save_dir = './gain'
    os.makedirs(save_dir,exist_ok=True)
    
    fname = f"fit_X{index:03.0f}.pdf"
    filepath =os.path.join(save_dir,fname)
    plt.savefig(filepath, format='pdf')
    plt.close()
    
#    c1 = TCanvas("", "", 800, 600)
##    c1.SaveAs("multi_page.pdf[")  # PDF 開始
#    title = f'X={x_val}, slope={slope:.4f}'
#    graph.SetTitle(title)
#    graph.Draw("AP")
#    fname = f"fit_X{x_val:03.0f}_TGraph.pdf"
#    filepath =os.path.join(save_dir,fname)
#    c1.SaveAs(filepath)

    return [gain,mean,std]

gain_list = [calc_gain(events,index) for index, events in enumerate(one_column_event)]

readCH = np.arange(len(gain_list))
gains = np.array([float(v[0]) for v in gain_list], dtype=np.float32)

col1 = ap.Column(name='readCH', format='J', array=readCH)  # J = 32bit int
col2 = ap.Column(name='GAIN', format='E', array=gains)     # E = 32bit float

hdu = ap.BinTableHDU.from_columns([col1, col2],name='COL_GAIN')
hdu.writeto("ctigain_list.fits", overwrite=True)


df = pd.DataFrame({
    "readCH": range(len(gain_list)),
    "GAIN": [float(v[0]) for v in gain_list],
    "Mean": [float(v[1]) for v in gain_list]
})

df.to_csv("ctigain_list.csv", index=True)


######確認のためにgain mapをスペクトルを作成する場合はコメントアウトを外す（ROOTでforを回すので時間がかかる。）
#h1fs = TH1F("TH1F_single","TH1F_single", 10000, 0, 10000)
#h1fm = TH1F("TH1F_multi","TH1F_multi", 10000, 0, 10000)
#h1fs1 = TH1F("TH1F_single1","TH1F_single", 10000, 0, 10000)
#h1fm1 = TH1F("TH1F_multi1","TH1F_multi", 10000, 0, 10000)
#h2d = TH2F("TH2d_gain","TH2d_gain", read_channel, 0, read_channel, 600, 0, 6000)
#h2d_cti = TH2F("TH2d_cti","TH2d_cti", 256, 0, 256, 60, 0, 6000)
#for j in range(event_list.shape[0]):
#    if j % 100000 == 0:
#        print(f"{j}th event processing.......")
#    if( event_list['vortex'][j]==0 or event_list['vortex'][j]==1 or event_list['vortex'][j]==16):
#        #h2d.Fill(event_list['DETX'][j],event_list['PHAsum'][j])
#        h2d.Fill(event_list['CAMEX_X'][j],event_list[PhEdited][j])
##    if(event_list['CAMEX_X'][j] < 105 or  event_list['CAMEX_X'][j]>124): continue
#    if( event_list['vortex'][j]==0 ):
#        h1fs1.Fill(event_list['PHAsum'][j])
#        h1fs.Fill(event_list[PhEdited][j])
##        print(event_list['vortex'][5],event_list['PHAsum'][j],table_dict['PhSumCor'][j])
#    #elif(image[j,5]==1 or image[j,5]==16 or image[j,5]==4 or image[j,5]==64):
#    #elif(image[j,5]==1 or image[j,5]==4 or image[j,5]==64 ):
#    else:
#        h1fm1.Fill(event_list['PHAsum'][j])
#        h1fm.Fill(event_list[PhEdited][j])
##
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
#mean=4530
#std=40
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
#h1fs.Fit(model,'R', "",3000,5000)
##h1fs.SetLineWidth(1)
#h1fs.Draw()
#
#c1.cd(2)
#gPad.SetLogy(1)
#h1fm.Fit(model,'R', "",3000,5000)
##h1fm.SetLineWidth(1)
#h1fm.Draw()
#c1.SaveAs("sigle-Multi_spectrum_calcgain.pdf")
#
#c2  = TCanvas("gain", "gain", 500, 1000)
#c2.cd()
#h2d.Draw("colz")
#c2.SaveAs("gain_variation_calcgain.pdf")
#

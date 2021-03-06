

import numpy as np
import matplotlib.pyplot as plt
import MCD19
from scipy.stats.stats import pearsonr
import EMFit
from os import path


dir='/Users/chili/Downloads/AvgMCD/SepWs/RecordsAll/'
outdir='/Users/chili/Downloads/AvgMCD/SepWs/RecordsAll/FigsSec/'


seasons=['winter','spring','summer','fall']
samplewd=5
sDom=True
goodr=0.9

SO2file='/Users/chili/Downloads/MAIAC/SO2-2014-US.csv'
csvdata = MCD19.CSVload(SO2file)
NO2Lats = csvdata[1:, 1].astype(float)
NO2Lons = csvdata[1:, 2].astype(float)
Citys = csvdata[1:,5]

NO2file='/Users/chili/Downloads/MAIAC/NO2-Lu-2015.csv'
csvdata = MCD19.CSVload(NO2file)
NO2Lats = np.append(NO2Lats,csvdata[1:, 2].astype(float))
NO2Lons = np.append(NO2Lons,csvdata[1:, 3].astype(float))
Citys = np.append(Citys,csvdata[1:, 0])

#the source location should be at most 300 km away otherwise too poorly chosen
# maxxsc=60.

Fitpardicts=dict()
Fitpardicts['EMG']=np.array(['a','x0','xsc','sigma','b'])

if sDom==True:
    Fitpardicts['EMA'] = np.array(['c', 'xa', 'xc', 'xscc', 'sigmac', 'b'])
else:
    Fitpardicts['EMA']=np.array(['a','c', 'xa', 'xc', 'xscc', 'sigmaa', 'sigmac', 'b'])



print ('season, City, Fit, Lat, Lon, Dir, taua, tauc')

for season in seasons:

    for iCity in np.arange(len(Citys)):

        City=Citys[iCity]
        Lat=NO2Lats[iCity]
        Lon=NO2Lons[iCity]

        if sDom==True:
            file = dir + City + '.' + season + '.rcd'
        else:
            file = dir + City + '.' + season + '.alt.rcd'


        if path.exists(file)==False:
            continue

        fitRpt = EMFit.ExamFit(file, Fitpardicts, GoodR=goodr)

        if fitRpt==None:
            continue

        for (Dir,Fitdata) in fitRpt.items():

            outfile = outdir + City + '.' + season + '.' + Dir + '.png'

            startplot=False
            for key,value in Fitpardicts.items():

                if value[0]+'_'+key in Fitdata:

                    if startplot==False:
                        AvgW = Fitdata['Y']

                        xW = Fitdata['xW']
                        inds = np.argwhere(AvgW > 0.)

                        effxW = xW[inds]


                        relAODerr=(0.05+0.1*np.mean(AvgW[inds]))/np.mean(AvgW[inds])


                        fig, axs = plt.subplots()
                        plt.xlabel('X (km)')
                        plt.ylabel('AOD')
                        axs.plot(xW[inds] * samplewd, AvgW[inds], 'ro', mfc='none')
                        plt.text(0.03, 0.95, City + ', Lat=' + '{:10.2f}'.format(Lat).strip() + ', Lon=' + \
                                 '{:10.2f}'.format(Lon).strip(), transform=axs.transAxes)
                        plt.text(0.03, 0.8, 'Wind speed=' + '{:10.1f}'.format(Fitdata['ws']).strip() + ' km/h', \
                                 transform=axs.transAxes)
                        startplot=True



                    if key == 'EMG':

                        b=Fitdata['b_' + key]

                        xsc=Fitdata['xsc_' + key]
                        x0=Fitdata['x0_'+key]
                        sigma=Fitdata['sigma_'+key]


                        FitAvgW = EMFit.EMG(xW[inds], Fitdata['a_' + key], x0, \
                                            xsc, sigma, b)

                        if np.mean(FitAvgW[len(inds)-10:len(inds)])-b>=0.85*(np.max(FitAvgW)-b):
                            continue

                        corvalue = pearsonr(AvgW[inds].flatten(),FitAvgW.flatten())

                        #The average of last 10 numbers should be smaller than maxAvgW (both minus b)



                        # if ((xsc+x0+sigma)>effxW[-1]) | \
                        #         (x0<sigma):
                        #     startplot=False

                        tau = Fitdata['x0_' + key] * samplewd / Fitdata['ws']
                        dtau = tau * np.sqrt((Fitdata['x0std_' + key] / Fitdata['x0_' + key]) ** 2 + \
                                             (Fitdata['wsstd'] / Fitdata['ws']) ** 2)

                        print (season, ',', City, ',EMG,', Lat, ',', Lon, ',', Dir, ',', \
                               tau, '+/-', dtau, ',', 0.)


                        axs.plot(xW[inds] * samplewd, FitAvgW, color='black')
                        plt.text(0.03, 0.9,
                                 key + ' x0=' + '{:10.1f}'.format(Fitdata['x0_' + key] * samplewd).strip() \
                                 + 'km, xsc=' + '{:10.1f}'.format(Fitdata['xsc_' + key] * samplewd).strip()
                                 # r'$\pm $' + '{:10.1f}'.format(Fitdata['x0std_EMG'] * samplewd).strip()\
                                 + ' km, correlation=' + '{:10.2f}'.format(corvalue[0]).strip(),
                                 transform=axs.transAxes)



                    if key == 'EMA':
                        c = Fitdata['c_' + key]
                        xa = Fitdata['xa_' + key]

                        xc = Fitdata['xc_' + key]

                        if np.absolute(xc-xa)/xa<0.0000001:
                            continue
                        xsca = Fitdata['xscc_' + key]
                        sigmac = Fitdata['sigmac_' + key]
                        xscc = xsca

                        b = Fitdata['b_' + key]
                        if sDom == True:
                            a = 0.
                            sigmaa = sigmac
                        else:
                            a = Fitdata['a_' + key]
                            sigmaa = Fitdata['sigmaa_' + key]

                        FitAvgW = EMFit.EMA(xW[inds], a, c, xa, xc, xsca, xscc, sigmaa, sigmac, b)
                        if np.mean(FitAvgW[len(inds)-10:len(inds)])-b>=0.85*(np.max(FitAvgW)-b):
                            continue
                        corvalue = pearsonr(AvgW[inds].flatten(),FitAvgW.flatten())


                        if City == "Barry":
                            print (season,Dir, np.mean(FitAvgW[-11:-1]) - b, np.max(FitAvgW) - b)

                        # if (xscc+xc+sigmac>xW[-1]) | (xsca+xa+sigmaa>effxW[-1]) | (xc<sigmac) | (xa<sigmaa):
                        #     startplot = False

                        tau = Fitdata['xa_' + key] * samplewd / Fitdata['ws']
                        dtau = tau * np.sqrt((Fitdata['xastd_' + key] / Fitdata['xa_' + key]) ** 2 + \
                                             (Fitdata['wsstd'] / Fitdata['ws']) ** 2+\
                                             relAODerr**2+2*0.01)

                        print (season, ',', City, ',EMA,', Lat, ',', Lon, ',', Dir, ',', \
                               xa * samplewd / Fitdata['ws'], '+/-', dtau, ',', \
                               xc * samplewd / Fitdata['ws'])


                        axs.plot(xW[inds] * samplewd, FitAvgW, color='blue')
                        plt.text(0.03, 0.85,
                                 key + ' xa=' + '{:10.1f}'.format(Fitdata['xa_' + key] * samplewd).strip() \
                                 + 'km, xc=' + '{:10.1f}'.format(Fitdata['xc_' + key] * samplewd).strip() \
                                 + 'km, xsc=' + '{:10.1f}'.format(Fitdata['xscc_' + key] * samplewd).strip() \
                                 # r'$\pm $' + '{:10.1f}'.format(Fitdata['x0std_EMG'] * samplewd).strip()\
                                 + ' km, correlation=' + '{:10.2f}'.format(corvalue[0]).strip(),
                                 transform=axs.transAxes, \
                                 color='blue')



            if startplot==True:
                plt.savefig(outfile, dpi=600)
            plt.close('all')





































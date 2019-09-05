

import numpy as np
import matplotlib.pyplot as plt
import MCD19
from scipy.stats.stats import pearsonr
import EMFit



EMGdir='/Users/chili/Downloads/AvgBEHR/Citys/'
EMFdir='/Users/chili/Downloads/AvgBEHR/Citys/EMF/'
outEMFdir='/Users/chili/Downloads/AvgBEHR/Citys/EMFFigs/'
outEMGdir='/Users/chili/Downloads/AvgBEHR/Citys/EMGFigs/'

seasons=['winter','spring','summer','fall']
samplewd=5

NO2file='/Users/chili/Downloads/MAIAC/NO2-Lu-2015.csv'
csvdata = MCD19.CSVload(NO2file)
NO2Lats = csvdata[1:, 2].astype(float)
NO2Lons = csvdata[1:, 3].astype(float)
Citys = csvdata[1:,0]

# NO2file='/Users/chili/Downloads/MAIAC/NO2-China-2016.csv'
# csvdata = MCD19.CSVload(NO2file)
# NO2Lats = csvdata[1:, 1].astype(float)
# NO2Lons = csvdata[1:, 2].astype(float)
# Citys = csvdata[1:,0]

print 'City, Lat, Lon, season, Dir, WS, tau, tau_sigma, r'

for season in seasons:

    for iCity in np.arange(len(Citys)):

        City=Citys[iCity]
        Lat=NO2Lats[iCity]
        Lon=NO2Lons[iCity]

        # file=EMGdir+City+'.'+season+'.rcd'
        #
        # fitRpt = EMFit.ExamFit(file, DoEMG=True, DoEMF=False, GoodR=0.9)
        #
        #
        # for (Dir,Fitdata) in fitRpt.items():
        #
        #     if Fitdata['x0_EMG'] * samplewd/Fitdata['ws']>1000.:
        #         continue
        #
        #     if np.isnan(Fitdata['x0std_EMG']):
        #         Fitdata['x0std_EMG']=0.3*Fitdata['x0_EMG']
        #
        #     outfile=outEMGdir+City+'.'+season+'.'+Dir+'.png'
        #
        #     AvgW=Fitdata['AvgW']
        #     xW=Fitdata['xW']
        #
        #     FitAvgWEMG = EMFit.EMG(xW, Fitdata['a_EMG'], Fitdata['x0_EMG'], Fitdata['xsc_EMG'], \
        #                            Fitdata['sigma_EMG'], Fitdata['b_EMG'])
        #     corvalueEMG = pearsonr(AvgW,FitAvgWEMG)
        #
        #     if corvalueEMG[0]<0.85:
        #         continue
        #
        #     print City, Lat,',',Lon,',',season,',', Dir,',', Fitdata['ws'],',', \
        #         Fitdata['x0_EMG'] * samplewd/Fitdata['ws'],',', Fitdata['x0std_EMG'] * samplewd/Fitdata['ws'],','\
        #         ,corvalueEMG[0]
        #
        #     fig, axs = plt.subplots()
        #     plt.xlabel('X (km)')
        #     plt.ylabel('AOD')
        #
        #     axs.plot(xW * samplewd, AvgW, 'ro')
        #
        #     axs.plot(xW * samplewd, FitAvgWEMG, color='black')
        #     plt.text(0.03, 0.9, 'x0: ' + '{:10.1f}'.format(Fitdata['x0_EMG'] * samplewd).strip() + \
        #              r'$\pm $'+'{:10.1f}'.format(Fitdata['x0std_EMG'] * samplewd).strip()+' km',\
        #              transform=axs.transAxes)
        #     plt.text(0.03, 0.85, 'Correlation (p): ' + '{:10.2f}'.format(corvalueEMG[0]).strip() \
        #              + ' (' + '{:10.3f}'.format(corvalueEMG[1]).strip() + ')', transform=axs.transAxes)
        #
        #     plt.text(0.03, 0.95, City+' '+season+', '+Dir, transform=axs.transAxes)
        #     plt.text(0.03, 0.7, 'wind speed: ' + '{:10.1f}'.format(Fitdata['ws']).strip() + ' km/h, '\
        #              +r'$\tau $= ' + '{:10.1f}'.format(Fitdata['x0_EMG'] * samplewd/Fitdata['ws']).strip() + \
        #              r'$\pm $'+'{:10.1f}'.format(Fitdata['x0std_EMG'] * samplewd/Fitdata['ws']).strip()+ ' hr',\
        #              transform=axs.transAxes)
        #     plt.savefig(outfile, dpi=600)
        #     plt.close()

        file = EMFdir + City + '.' + season + '.rcd'

        fitRpt = EMFit.ExamFit(file, DoEMG=False, DoEMF=True, GoodR=0.9)
        print fitRpt
        for (Dir, Fitdata) in fitRpt.items():

            if np.isnan(Fitdata['x0std_EMF'])==True:
                Fitdata['x0std_EMF'] = 0.05 * Fitdata['x0_EMF']

            outfile = outEMFdir + City + '.' + season + '.' + Dir + '.png'

            AvgW = Fitdata['AvgW']
            xW = Fitdata['xW']
            AvgC = Fitdata['AvgC']
            xC = Fitdata['xC']
            print Fitdata
            [nhf, CMatrix, startind, endind] = EMFit.PrepareEMF(xW, xC, AvgC, Fitdata['xmax'])


            FitAvgWEMF = EMFit.EMF(xW,nhf,CMatrix,startind,endind,Fitdata['a_EMF'],Fitdata['b_EMF'],Fitdata['x0_EMF'])

            ARatio=np.mean(AvgW[startind:endind + 1])/np.mean(FitAvgWEMF)

            Fitdata['a_EMF'] = Fitdata['a_EMF'] * ARatio
            Fitdata['b_EMF'] = Fitdata['b_EMF'] * ARatio

            FitAvgWEMF = EMFit.EMF(xW, nhf, CMatrix, startind, endind, Fitdata['a_EMF'], Fitdata['b_EMF'],
                                   Fitdata['x0_EMF'])
            corvalueEMF = pearsonr(AvgW[startind:endind + 1], FitAvgWEMF)
            print corvalueEMF[0]
            if corvalueEMF[0] < 0.85:
                continue

            print City, Lat, ',', Lon, ',', season, ',', Dir, ',', Fitdata['ws'], ',', \
                Fitdata['x0_EMF'] * samplewd/Fitdata['ws'],',', Fitdata['x0std_EMF'] * samplewd/Fitdata['ws'],','\
                ,corvalueEMG[0]

            fig, axs = plt.subplots()
            plt.xlabel('X (km)')
            plt.ylabel('AOD')

            axs.plot(xW * samplewd, AvgW, 'ro')
            axs.plot(xC * samplewd, AvgC, 'bo')

            axs.plot(xW[startind:endind+1] * samplewd, FitAvgWEMF, color='black')
            plt.text(0.03, 0.9, 'x0: ' + '{:10.1f}'.format(Fitdata['x0_EMF'] * samplewd).strip() + \
                     r'$\pm $' + '{:10.1f}'.format(Fitdata['x0std_EMF'] * samplewd).strip() + ' km', \
                     transform=axs.transAxes)
            plt.text(0.03, 0.85, 'Correlation (p): ' + '{:10.2f}'.format(corvalueEMF[0]).strip() \
                     + ' (' + '{:10.3f}'.format(corvalueEMF[1]).strip() + ')', transform=axs.transAxes)

            plt.text(0.03, 0.95, City + ' ' + season + ', ' + Dir, transform=axs.transAxes)
            plt.text(0.03, 0.7, 'wind speed: ' + '{:10.1f}'.format(Fitdata['ws']).strip() + ' km/h, ' \
                     + r'$\tau $= ' + '{:10.1f}'.format(Fitdata['x0_EMF'] * samplewd / Fitdata['ws']).strip() + \
                     r'$\pm $' + '{:10.1f}'.format(Fitdata['x0std_EMF'] * samplewd / Fitdata['ws']).strip() + ' hr', \
                     transform=axs.transAxes)
            plt.savefig(outfile, dpi=600)
            plt.close()

































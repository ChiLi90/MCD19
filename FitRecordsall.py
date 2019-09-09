
import numpy as np
from netCDF4 import Dataset
import MCD19
import math
from scipy.stats.stats import pearsonr
import EMFit
import argparse
from os import path
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

parser=argparse.ArgumentParser()
parser.add_argument("--season",type=str)
parser.add_argument("--loc",type=str)
parser.add_argument("--sDom",type=str)

args=parser.parse_args()
season=args.season
Loc=args.loc
sDom=args.sDom

Rearth = 6373.0
Dirs = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
Vectors = np.array([[1.414, 0], [1, 1], [0, 1.414], [-1, 1], [-1.414, 0], [-1, 1], [0, -1.414], [1, -1]])
ndirs=len(Dirs)

indir = '/global/scratch/chili/AvgMCD/SepWs/'
minobs = 1
outdir = "/global/scratch/chili/AvgMCD/RecordsAll/"
SO2file='/global/scratch/chili/AvgMCD/SO2-2014-US.csv'
NO2file='/global/scratch/chili/AvgMCD/NO2-Lu-2015.csv'
Aerfile = indir + season + '.2001-2013.nc'

varname='AOD'
sfactor=0.001

usewsbins=[0,1]

xmin=-300.
xmax=600.   #km
xEMG=np.array([xmin,xmax])
ymax=150.
minacross=ymax
samplewd=5.
nsample=50
xmax=np.round(xmax/samplewd).astype(int)
xmin=np.round(xmin/samplewd).astype(int)
xEMG=np.round(xEMG/samplewd).astype(int)
ymax=np.round(ymax/samplewd)
minacross=np.round(minacross/samplewd).astype(int)


EMGFit=True
EMAFit=True
EMGPars=np.array(['a','x0','xsc','sigma','b'])
nEMG=len(EMGPars)

if sDom=='True':
    EMAPars = np.array(['c', 'xa', 'xc', 'xscc', 'sigmac', 'b'])
else:
    EMAPars = np.array(['a','c', 'xa', 'xc', 'xscc', 'sigmaa','sigmac', 'b'])

nEMA=len(EMAPars)



if Loc=='NO2':
    Cityfile = NO2file
    csvdata = MCD19.CSVload(Cityfile)
    CityLats = csvdata[1:, 2].astype(float)
    CityLons = csvdata[1:, 3].astype(float)
    Citys = csvdata[1:, 0]
else:
    Cityfile = SO2file
    csvdata = MCD19.CSVload(Cityfile)
    CityLats = csvdata[1:, 1].astype(float)
    CityLons = csvdata[1:, 2].astype(float)
    Citys = csvdata[1:, 5]

nrows=xmax-xmin + 1 + nEMG + nEMA + 3
Column1=np.chararray(nrows,itemsize=20,unicode=True)
strxfull=(np.arange(xmax-xmin+1)+xmin).astype('str')
Column1[0:xmax-xmin+1]=strxfull

Column1[xmax-xmin+1:xmax-xmin+nEMG+1]=list(map(''.join, zip(EMGPars,np.stack(['_EMG']*nEMG))))
Column1[xmax-xmin+nEMG+1]='R_EMG'  #store chisqure for the second column
Column1[xmax-xmin+nEMG+2:xmax-xmin+nEMG+nEMA+2]=list(map(''.join, zip(EMAPars,np.stack(['_EMA']*nEMA))))
Column1[xmax-xmin+nEMG+nEMA+2]='R_EMA'
Column1[xmax-xmin+nEMG+nEMA+3]='Ws(km/h)'

ds = Dataset(Aerfile, 'r')
hvLat = ds['latitude'][:]
hvLon = ds['longitude'][:]

for ist in np.arange(len(Citys)):   #len(Citys)

    City = Citys[ist]
    RtCenter = np.array([CityLons[ist], CityLats[ist]]).astype(float)
    if sDom=='True':
        outfile=outdir+City+'.'+season+'.rcd'
    else:
        outfile=outdir+City+'.'+season+'.alt.rcd'
    if path.exists(outfile):
        continue
    # calculate distance
    dy = (hvLat - RtCenter[1]) * math.pi / 180. * Rearth
    dx = np.cos(RtCenter[1] * math.pi / 180.) * (hvLon - RtCenter[0]) * math.pi / 180. * Rearth

    # sample by distance, minimum is 1 km (original resolution)
    dy = dy / samplewd
    dx = dx / samplewd

    rgind = ((np.abs(dy) <= ymax) & (dx >= xmin) & (dx <= xmax)).nonzero()
    mindx = np.min(dx) - 1
    dx = dx - mindx

    outdata = np.zeros([nrows, 2 * ndirs])

    for iwdbin in np.arange(ndirs):
        bincount = 0
        for iwsbin in usewsbins:
            AOD = ds[varname][:, :, iwsbin, iwdbin] * sfactor
            Sample = ds['Sample'][:, :, iwsbin, iwdbin]
            u10 = ds['u10'][:, :, iwsbin, iwdbin] * 0.001
            v10 = ds['v10'][:, :, iwsbin, iwdbin] * 0.001

            AOD[(Sample <= 0.) | (AOD<=0.) | (u10 <= -900.) | (v10 <= -900.)] = 0.
            u10[(Sample <= 0.) | (AOD<=0.) | (u10 <= -900.) | (v10 <= -900.)] = 0.
            v10[(Sample <= 0.) | (AOD<=0.) | (u10 <= -900.) | (v10 <= -900.)] = 0.
            Sample[(Sample <= 0.) | (AOD<=0.) | (u10 <= -900.) | (v10 <= -900.)] = 0

            if bincount == 0:

                WindyAOD = AOD * Sample
                Windyu10 = u10 * Sample
                Windyv10 = v10 * Sample
                WindyNo = Sample

            else:
                WindyAOD = WindyAOD + AOD * Sample
                Windyu10 = Windyu10 + u10 * Sample
                Windyv10 = Windyv10 + v10 * Sample
                WindyNo = WindyNo + Sample

        WindyAOD = WindyAOD / WindyNo
        Windyu10 = Windyu10 / WindyNo
        Windyv10 = Windyv10 / WindyNo

        WindyAOD[WindyNo <= 0] = np.nan
        Windyu10[WindyNo <= 0] = np.nan
        Windyv10[WindyNo <= 0] = np.nan

        # rotate windy data for consistent handling of 8 wind directions
        [rtWAOD, u, v] = MCD19.RotateAOD(MCD19.CutEdge(WindyAOD), hvLon, hvLat, RtCenter[0], RtCenter[1], \
                                         uv=Vectors[iwdbin, :])

        [rtw10, u, v] = MCD19.RotateAOD(MCD19.CutEdge(np.sqrt(Windyu10 ** 2 + Windyv10 ** 2)), hvLon, hvLat, \
                                        RtCenter[0], RtCenter[1], uv=Vectors[iwdbin, :])

        rtWAOD[rtWAOD <= 0] = np.nan
        rtw10[rtw10 <= 0] = np.nan

        avgw10 = np.nanmean(rtw10[rgind])
        stdw10 = np.nanstd(rtw10[rgind])
        outdata[xmax-xmin+nEMG+nEMA+3, 2 * iwdbin] = avgw10 * 3.6
        outdata[xmax-xmin+nEMG+nEMA+3, 2 * iwdbin + 1] = stdw10 * 3.6

        minx0 = avgw10 * 3.6 / samplewd  # 1 hr
        maxx0 = minx0 * 24. * 10.  # 10 days

        DoEMGFit = EMGFit
        DoEMAFit = EMAFit

        [AvgW, WNo, xW] = MCD19.SortAvg(rtWAOD[rgind].flatten(), np.round(dx[rgind].flatten()).astype(int),
                                        mincount=minacross)
        xW = np.round((xW + mindx)).astype(int)

        outinds=(xW-xmin).astype(int)
        outdata[outinds,2*iwdbin]=AvgW
        outdata[outinds, 2 * iwdbin+1] = WNo

        EMGinds = np.nonzero((xW <= xEMG[1]) & (xW >= xEMG[0]))
        if len(xW[EMGinds]) < 2. / 3 * (xmax - xmin + 1):
            continue

        # 1. Do exponential modified gaussian fitting
        if DoEMGFit == True:


            [outEMG, successG] = EMFit.EMGFit(xW[EMGinds], AvgW[EMGinds], samplewd, minx0,20,
                                              solver='trust-constr')

            if successG == False:
                DoEMGFit = False
            else:
                DoEMGFit = True
                yEMG=EMFit.EMG(xW[EMGinds], outEMG['a'], outEMG['x0'], \
                                                 outEMG['xsc'], outEMG['sigma'], outEMG['b'])
                corvalueEMG = pearsonr(AvgW[EMGinds],yEMG)

                for ipar in np.arange(nEMG):
                    outdata[ipar + xmax-xmin+1, 2 * iwdbin] = outEMG[EMGPars[ipar]].value
                    outdata[ipar + xmax-xmin+1, 2 * iwdbin + 1] = outEMG[EMGPars[ipar]].brute_step
                outdata[xmax-xmin+1 + nEMG, 2 * iwdbin] = corvalueEMG[0]
                outdata[xmax - xmin + 1 + nEMG, 2 * iwdbin+1]=\
                    np.sqrt(np.sum((yEMG-AvgW[EMGinds])**2))/np.nanmean(AvgW[EMGinds])

        if DoEMAFit == True:

            if sDom=='True':
                [outEMA, successA] = EMFit.EMAFit(xW[EMGinds], AvgW[EMGinds], samplewd, minx0, nsample, \
                                                  sameSource=True, sDom=True)
            else:
                [outEMA, successA] = EMFit.EMAFit(xW[EMGinds], AvgW[EMGinds], samplewd, minx0, nsample, \
                                                  sameSource=True)


            if successA == False:
                DoEMAFit = False
            else:
                DoEMAFit = True
                yEMA= EMFit.EMA(xW[EMGinds], outEMA['a'], outEMA['c'], \
                                outEMA['xa'], outEMA['xc'],\
                                outEMA['xsca'], outEMA['xscc'],\
                                outEMA['sigmaa'], outEMA['sigmac'],\
                                outEMA['b'])
                corvalueEMA = pearsonr(AvgW[EMGinds],yEMA)

                for ipar in np.arange(nEMA):
                    outdata[ipar + xmax-xmin+2+nEMG, 2 * iwdbin] = outEMA[EMAPars[ipar]].value
                    outdata[ipar + xmax-xmin+2+nEMG, 2 * iwdbin + 1] = outEMA[EMAPars[ipar]].brute_step
                outdata[xmax-xmin+2+nEMG+nEMA, 2 * iwdbin] = corvalueEMA[0]

                outdata[xmax-xmin+2+nEMG+nEMA, 2 * iwdbin+1] =\
                    np.sqrt(np.sum((yEMA-AvgW[EMGinds])**2))/np.nanmean(AvgW[EMGinds])


        if (DoEMGFit == False) & (DoEMAFit == False):
            continue

    #if sDom=='True':
    #    outfile = outdir+City + '.' + season + '.rcd'
    #else:
    #    outfile = outdir + City + '.' + season + '.alt.rcd'
    outF = open(outfile, "w")
    # first line (header)
    header=City+' Lat='+'{:10.2f}'.format(RtCenter[1]).strip()+', Lon='+'{:10.2f}'.format(RtCenter[0]).strip()
    outF.write(header)
    outF.write("\n")
    outF.write('{:>10}'.format('Direction'))
    outF.write(''.join(list(map('{:>20}'.format,Dirs))))
    outF.write("\n")

    for iline in np.arange(len(Column1)):
        outF.write('{:>10}'.format(Column1[iline]))

        outF.write(''.join(list(map('{:10.2e}'.format, outdata[iline,:]))))
        outF.write('\n')
    outF.close()

ds.close()


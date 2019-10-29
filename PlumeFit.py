
import numpy as np
from netCDF4 import Dataset
import MCD19
from scipy.stats.stats import pearsonr
import EMFit
import argparse
import glob
import os

def sortSum2D(dirAOD,dirAODsq,dirw10,dirw10sq,dirNo,dx,dy,xmin,xmax,y,samplewd,minsample):

    #data has the same dimension as dx dy
    sdx=np.round(dx/samplewd).astype(int)
    sdy = np.round(dy / samplewd).astype(int)

    sxmin=np.round(xmin/samplewd).astype(int)
    sxmax=np.round(xmax/samplewd).astype(int)
    sy=np.round(y/samplewd).astype(int)

    siteAOD=np.zeros([sxmin+sxmax+1,2*sy+1])
    siteAODsq = np.zeros([sxmin + sxmax + 1, 2 * sy + 1])
    sitew10 = np.zeros([sxmin + sxmax + 1, 2 * sy + 1])
    sitew10sq = np.zeros([sxmin + sxmax + 1, 2 * sy + 1])
    siteNo = np.zeros([sxmin + sxmax + 1, 2 * sy + 1])

    for ix in np.arange(sxmin+sxmax+1):
        for iy in np.arange(2*sy+1):

            inds=((sdx==ix-sxmin) & (sdy==iy-sy)).nonzero()
            if np.array(inds).size>0:
                siteAOD[ix,iy]=np.sum(dirAOD[inds])
                siteAODsq[ix, iy] = np.sum(dirAODsq[inds])
                sitew10[ix, iy] = np.sum(dirw10[inds])
                sitew10sq[ix, iy] = np.sum(dirw10sq[inds])
                siteNo[ix,iy]=np.sum(dirNo[inds])
    
    nvinds=(siteNo<minsample).nonzero()
    if np.array(nvinds).size>0:
        siteNo[nvinds] = 0
        siteAOD[nvinds] = 0.0
        siteAODsq[nvinds] = 0.0
        sitew10[nvinds] = 0.0
        sitew10sq[nvinds] = 0.0



    return [siteAOD, siteAODsq, sitew10, sitew10sq, siteNo]

def FitOut(xarray,siteAOD, siteAODsq, sitew10, sitew10sq, siteNo, outfile,minsample,complete,x0,y0,samplewd,FitType):


    nx,ny=siteNo.shape

    #check completeness
    nvinds=(siteNo<minsample).nonzero()
    if np.array(nvinds).size/2.>(1-complete)*siteNo.size:
        return False

    siteNo[nvinds]=0
    siteAOD[nvinds]=0
    siteAODsq[nvinds] = 0
    sitew10[nvinds] = 0
    sitew10sq[nvinds] = 0

    acrossNo=np.sum(siteNo,axis=1)
    acrossAOD = np.sum(siteAOD, axis=1)/acrossNo
    acrossAODsq = np.sum(siteAODsq, axis=1)
    acrossAODstd = np.sqrt((acrossAODsq - acrossNo * (acrossAOD ** 2)) / (acrossNo - 1))

    #standard deviation of w10 for all grids:
    avgw10=np.sum(sitew10)/np.sum(siteNo)
    stdw10=np.sqrt((np.sum(sitew10sq) - np.sum(siteNo) * (avgw10 ** 2)) / (np.sum(siteNo) - 1))

    vinds=(acrossNo>0).nonzero()
    minx0=avgw10 * 3.6 / samplewd

    if FitType=='EMG':
        pars = ['a', 'x0', 'xsc', 'sigma', 'b', 'w10', 'r']

        [outEMG, successG] = EMFit.EMGFit(xarray[vinds], acrossAOD[vinds], samplewd, minx0, 20,
                                          solver='trust-constr')

        if successG == False:
            DoEMGFit = False

        else:
            DoEMGFit = True
            yEMG = EMFit.EMG(xarray[vinds], outEMG['a'], outEMG['x0'], \
                             outEMG['xsc'], outEMG['sigma'], outEMG['b'])
            corvalueEMG = pearsonr(acrossAOD[vinds], yEMG)

            outdata = np.zeros([2, len(pars)])
            for ipar in np.arange(len(pars) - 2):
                outdata[0, ipar] = outEMG[pars[ipar]].value
                outdata[1, ipar] = outEMG[pars[ipar]].brute_step

            outdata[0, len(pars) - 2] = avgw10 * 3.6
            outdata[1, len(pars) - 2] = stdw10 * 3.6
            outdata[:, len(pars) - 1] = corvalueEMG
            # write outfile

            outF = open(outfile, "w")

            header = 'Lat=' + '{:10.2f}'.format(y0).strip() + ', Lon=' + '{:10.2f}'.format(x0).strip() + \
                     ', Sample=' + '{:10.2f}'.format(samplewd).strip()
            outF.write(header)
            outF.write('\n')
            outF.write('{:>11}'.format('x/y,'))
            outF.write(','.join(list(map('{:10.0f}'.format, xarray))))
            outF.write('\n')
            # x,y 2-D data
            for iline in np.arange(ny):
                outF.write('{:10.0f}'.format(-1 * ny / 2 + iline) + ',')
                outF.write(','.join(list(map('{:10.3f}'.format, siteAOD[:, iline] / siteNo[:, iline]))))
                outF.write('\n')

            # x, 1-D data
            outF.write('{:>11}'.format('avgAOD,'))
            outF.write(','.join(list(map('{:10.3f}'.format, acrossAOD))))
            outF.write('\n')

            outF.write('{:>11}'.format('Sample,'))
            outF.write(','.join(list(map('{:10.0f}'.format, acrossNo))))
            outF.write('\n')

            outF.write('{:>11}'.format('std,'))
            outF.write(','.join(list(map('{:10.3f}'.format, acrossAODstd))))
            outF.write('\n')

            outF.write('{:>11}'.format('Pars,'))
            outF.write(','.join(list(map('{:>10}'.format, pars))))
            outF.write('\n')

            outF.write('{:>11}'.format('avg,'))
            outF.write(','.join(list(map('{:10.3e}'.format, outdata[0, :]))))
            outF.write('\n')

            outF.write('{:>11}'.format('std,'))
            outF.write(','.join(list(map('{:10.3e}'.format, outdata[1, :]))))
            outF.write('\n')

        return DoEMGFit

    if FitType=='EMA':
        pars = ['a', 'c', 'xa', 'xc', 'xsca', 'xscc', 'sigmaa', 'sigmac', 'b', 'w10', 'r']

        [outEMA, successA] = EMFit.EMAFit(xarray[vinds], acrossAOD[vinds], samplewd, minx0, 50, \
                                          sameSource=True,solver='trust-constr')

        if successA == False:
            DoEMAFit = False

        else:
            DoEMAFit = True
            yEMA = EMFit.EMA(xarray[vinds], outEMA['a'], outEMA['c'], \
                             outEMA['xa'], outEMA['xc'], \
                             outEMA['xsca'], outEMA['xscc'], \
                             outEMA['sigmaa'], outEMA['sigmac'], \
                             outEMA['b'])
            corvalueEMA = pearsonr(acrossAOD[vinds], yEMA)


            outdata = np.zeros([2, len(pars)])
            for ipar in np.arange(len(pars) - 2):
                outdata[0, ipar] = outEMA[pars[ipar]].value
                outdata[1, ipar] = outEMA[pars[ipar]].brute_step

            outdata[0, len(pars) - 2] = avgw10 * 3.6
            outdata[1, len(pars) - 2] = stdw10 * 3.6
            outdata[:, len(pars) - 1] = corvalueEMA
            # write outfile

            outF = open(outfile, "w")

            header = 'Lat=' + '{:10.2f}'.format(y0).strip() + ', Lon=' + '{:10.2f}'.format(x0).strip() + \
                     ', Sample=' + '{:10.2f}'.format(samplewd).strip()
            outF.write(header)
            outF.write('\n')
            outF.write('{:>11}'.format('x/y,'))
            outF.write(','.join(list(map('{:10.0f}'.format, xarray))))
            outF.write('\n')
            # x,y 2-D data
            for iline in np.arange(ny):
                outF.write('{:10.0f}'.format(-1 * ny / 2 + iline) + ',')
                outF.write(','.join(list(map('{:10.3f}'.format, siteAOD[:, iline] / siteNo[:, iline]))))
                outF.write('\n')

            # x, 1-D data
            outF.write('{:>11}'.format('avgAOD,'))
            outF.write(','.join(list(map('{:10.3f}'.format, acrossAOD))))
            outF.write('\n')

            outF.write('{:>11}'.format('Sample,'))
            outF.write(','.join(list(map('{:10.0f}'.format, acrossNo))))
            outF.write('\n')

            outF.write('{:>11}'.format('std,'))
            outF.write(','.join(list(map('{:10.3f}'.format, acrossAODstd))))
            outF.write('\n')

            outF.write('{:>11}'.format('Pars,'))
            outF.write(','.join(list(map('{:>10}'.format, pars))))
            outF.write('\n')

            outF.write('{:>11}'.format('avg,'))
            outF.write(','.join(list(map('{:10.3e}'.format, outdata[0, :]))))
            outF.write('\n')

            outF.write('{:>11}'.format('std,'))
            outF.write(','.join(list(map('{:10.3e}'.format, outdata[1, :]))))
            outF.write('\n')

        return DoEMAFit







parser = argparse.ArgumentParser()
parser.add_argument("--season")
parser.add_argument("--y",type=float)
parser.add_argument("--Combine")
parser.add_argument("--incCalm")
parser.add_argument("--xmin",type=float)
parser.add_argument("--xmax",type=float)
parser.add_argument("--start",type=int)
parser.add_argument("--end",type=int)
parser.add_argument("--FitType")

args = parser.parse_args()
season=args.season
y=args.y
xmin=np.absolute(args.xmin)
xmax=np.absolute(args.xmax)
startyr=args.start
endyr=args.end
FitType=args.FitType

samplewd=5. #native resolution, 5km
xno=np.round(xmax/samplewd).astype(int)+np.round(xmin/samplewd).astype(int)+1
yno=2*np.round(y/samplewd).astype(int)+1

xarray=np.arange(xno)-np.round(xmin/samplewd).astype(int)


#whether combine the 8 directions together
if args.Combine=='True':
    CombineDirs=True
    outdir='/global/scratch/chili/AvgMCD/SepWs/Sqr/100Fit/CN/Records/'
else:
    CombineDirs=False
    outdir='/global/scratch/chili/AvgMCD/SepWs/Sqr/100Fit/CN/RecordsDir/'

if args.incCalm=='True':
    wsinds=[0,1]
else:
    wsinds=[1]

Rearth = 6373.0
complete=0.667

#every pixle contain at least 1 observations each year for this season (180 days)
minsample=endyr-startyr+1

Datadir='/global/scratch/chili/AvgMCD/SepWs/Sqr/CN/Combined/'
sfile='/global/scratch/chili/AvgMCD/SepWs/Plumes/a15b20/Sources.txt'
outdir=outdir+args.incCalm+'_'+'{:10.0f}'.format(args.xmin).strip()+'/'
NO2file='/global/scratch/chili/AvgMCD/NO2-Lu-2015.csv'
SO2file='/global/scratch/chili/AvgMCD/SO2-2014-US.csv'
CNfile='/global/scratch/chili/AvgMCD/NO2-China-2016.csv'
if not os.path.exists(outdir):
    os.makedirs(outdir)

Dirs = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
Vectors = np.array([[1.414, 0], [1, 1], [0, 1.414], [-1, 1], [-1.414, 0], [-1, 1], [0, -1.414], [1, -1]])

#read the data for 8 directions
fcount=0
for yr in startyr+np.arange(endyr-startyr+1):

    stryr='{:10.0f}'.format(yr).strip()
    if season=='annual':
        files=glob.glob(Datadir+'*'+stryr+'-'+stryr+'.nc')
    else:
        files=glob.glob(Datadir+season+'.'+stryr+'-'+stryr+'.nc')

    for file in files:
        ds=Dataset(file,'r')
        if fcount==0:

            fsample=ds['Sample'][:]
            fsample[fsample<=0]=0

            Lat=ds['latitude'][:]
            Lon=ds['longitude'][:]
            AOD=ds['AOD'][:]*0.001*fsample
            AODsq=ds['AODsq'][:]*0.0001*fsample
            w10=np.sqrt(ds['u10'][:]**2+ds['v10'][:]**2)*0.001*fsample
            w10sq = ds['w10sq'][:] * 0.0001 * fsample
            sample=fsample
            nx,ny,nws,nwd=AOD.shape
        else:

            fsample = ds['Sample'][:]
            fsample[fsample <= 0] = 0
            AOD = AOD+ds['AOD'][:] * 0.001 * fsample
            AODsq = AODsq+ds['AODsq'][:] * 0.0001 * fsample
            w10 = w10+np.sqrt(ds['u10'][:] ** 2 + ds['v10'][:] ** 2) * 0.001 * fsample
            w10sq = w10sq+ds['w10sq'][:] * 0.0001 * fsample
            sample = sample+fsample

        ds.close()
        fcount=fcount+1

#accumulate with wind speed bins first
AOD=np.sum(AOD[:,:,wsinds,:],axis=2)
AODsq=np.sum(AODsq[:,:,wsinds,:],axis=2)
w10=np.sum(w10[:,:,wsinds,:],axis=2)
w10sq=np.sum(w10sq[:,:,wsinds,:],axis=2)
sample=np.sum(sample[:,:,wsinds,:],axis=2)

#read the source file
# csvdata = MCD19.CSVload(sfile)
# CityLats = csvdata[1:, 1].astype(float)
# CityLons = csvdata[1:, 0].astype(float)
# Citys=list(map('{:10.0f}'.format,np.arange(len(CityLons))))
#Cityfile = SO2file
#csvdata = MCD19.CSVload(Cityfile)
#CityLats = csvdata[1:, 1].astype(float)
#CityLons = csvdata[1:, 2].astype(float)
#Citys = csvdata[1:, 5]
#Cityfile = NO2file
#csvdata = MCD19.CSVload(Cityfile)
#CityLats = csvdata[1:, 2].astype(float)
#CityLons = csvdata[1:, 3].astype(float)
#Citys = csvdata[1:, 0]
Cityfile = CNfile
csvdata=MCD19.CSVload(Cityfile)
CityLats=csvdata[1:,1].astype(float)
CityLons=csvdata[1:,2].astype(float)
Citys=csvdata[1:,0]


for iloc in np.arange(len(CityLats)):

    x0=CityLons[iloc]
    y0=CityLats[iloc]
    #strloc = '{:10.0f}'.format(iloc).strip()

    #record the sum, sum-square, sample and w10 squre of the "rectangle"
    if CombineDirs==True:
        outfile = outdir + Citys[iloc].strip() +'.' +season+'.txt'
        if os.path.exists(outfile):
            continue

        siteAOD=np.zeros([xno,yno])
        siteAODsq=np.zeros([xno,yno])
        sitew10=np.zeros([xno,yno])
        sitew10sq=np.zeros([xno, yno])
        siteNo=np.zeros([xno,yno])

    for idir in np.arange(nwd):

        if CombineDirs==False:

            outfile=outdir+Citys[iloc].strip()+'.'+Dirs[idir]+'.'+season+'.txt'
            if os.path.exists(outfile):
                continue

        dirNo = sample[:, :, idir]
        dirAOD = AOD[:, :, idir]
        dirAODsq = AODsq[:, :, idir]
        dirw10=w10[:, :, idir]
        dirw10sq = w10sq[:, :, idir]

        # Do rotation
        if idir > 0:
            [rtLon, rtLat] = MCD19.RotateAOD(dirAOD, Lon, Lat, x0, y0, uv=Vectors[idir, :], ToOrigin=False)
        else:
            [rtLon, rtLat] = [Lon, Lat]

        dy = (rtLat - y0) * np.pi / 180. * Rearth
        dx = np.cos(y0 * np.pi / 180.) * (rtLon - x0) * np.pi / 180. * Rearth

        if CombineDirs==False:

            [siteAOD,siteAODsq,sitew10,sitew10sq,siteNo] = \
                sortSum2D(dirAOD,dirAODsq,dirw10,dirw10sq,dirNo, dx, dy, xmin, xmax, y, samplewd,minsample)
            success = FitOut(xarray,siteAOD, siteAODsq, sitew10, sitew10sq, siteNo, outfile,minsample,complete,x0,y0,\
                             samplewd,FitType)
            if success == False:
                print('unsuccessful fitting for file: ' + outfile)
        else:

            [isiteAOD,isiteAODsq,isitew10,isitew10sq,isiteNo] = \
                sortSum2D(dirAOD,dirAODsq,dirw10,dirw10sq,dirNo, dx, dy, xmin, xmax, y, samplewd,minsample)
            siteAOD=siteAOD+isiteAOD
            siteAODsq = siteAODsq + isiteAODsq
            sitew10 = sitew10 + isitew10
            sitew10sq = sitew10sq + isitew10sq
            siteNo = siteNo + isiteNo

    if CombineDirs==True:
        success = FitOut(xarray,siteAOD, siteAODsq, sitew10, sitew10sq, siteNo, outfile,minsample,complete,x0,y0,\
                         samplewd,FitType)

        if success==False:
            print('unsuccessful fitting for file: '+outfile)












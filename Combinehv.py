


#Combine several tiles of MAIAC averaged files for easier handling (e.g. fitting)
#rebin the size of data to (5x5)
import argparse
import numpy as np
from netCDF4 import Dataset
import MCD19
import glob

indir='/global/scratch/chili/AvgMCD/SepWs/Sqr/CN/'
outdir='/global/scratch/chili/AvgMCD/SepWs/Sqr/CN/Combined/'
strhvs = ['h26v05','h26v06','h27v05','h27v06','h28v05','h28v06']
hs=np.arange(3)+26
vs=np.arange(2)+5
#strhvs=['h08v04','h09v04']   #,'h08v05','h09v04','h09v05','h10v04','h10v05','h11v04','h11v05','h12v04','h12v05'
seasons = ['winter','spring','summer','fall']
varnames=['AOD','AODsq','u10','v10','w10sq','tp','tcc']
units=['unitless','unitless','m/s','m/s','m2/s2','um','unitless']
varNo='Sample'

parser=argparse.ArgumentParser()
parser.add_argument("--start",type=int)
parser.add_argument("--end",type=int)

args=parser.parse_args()
startyr=args.start
endyr=args.end
Dorebin=False

newsp=[1,1]

strse='{:10.0f}'.format(startyr).strip()+'-'+'{:10.0f}'.format(endyr).strip()


for season in seasons:

    outfile=outdir+season+'.'+strse+'.nc'

    dso = Dataset(outfile, mode='w', format='NETCDF4')


    for ivar in np.arange(len(varnames)):

        varname=varnames[ivar]

        #for strhv in strhvs:


        #vary in h, add on the axis=1
        for ih in np.arange(len(hs)):
            h=hs[ih]
            #vary in v, add on the axis=0
            for iv in np.arange(len(vs)):

                v=vs[iv]
                strhv = 'h' + '{:10.0f}'.format(h + 100).strip()[1:3] + 'v' + '{:10.0f}'.format(v + 100).strip()[1:3]
                files = glob.glob(indir + strhv + '.' + season + '.*.nc')
                starts = np.zeros(len(files), dtype=int)
                ends = np.zeros(len(files), dtype=int)
                for ifile in np.arange(len(files)):
                    file = files[ifile]
                    strses = file.split('/')[-1].split('.')[2].split('-')
                    starts[ifile] = strses[0]
                    ends[ifile] = strses[1]

                # do the years first for each hv:
                for year in np.arange(endyr-startyr+1) + startyr:  # endyr-startyr+

                    fileind = -2

                    for ifile in np.arange(len(files)):
                        if (starts[ifile] <= year) & (ends[ifile] >= year):
                            fileind = ifile
                            break

                    Aerfile = files[fileind]

                    ds = Dataset(Aerfile, 'r')
                    stryear = '{:10.0f}'.format(year).strip()

                    yrdata = ds[varname + '_' + stryear][:]
                    yrdata = yrdata.astype(float)
                    yrSample = ds[varNo + '_' + stryear][:]

                    yrdata[yrSample <= 0] = 0
                    yrSample[yrSample <= 0] = 0

                    if year == startyr:
                        mydata = yrdata * yrSample
                        mySample = yrSample
                        Lat = ds['latitude'][:]
                        Lon = ds['longitude'][:]
                        mws = ds['MaxWindSpeed'][:]

                    else:
                        mydata = mydata + yrdata * yrSample
                        mySample = mySample + yrSample

                    ds.close()

                [nx, ny, na, nb] = mydata.shape
                if Dorebin == False:
                    newx = nx
                    newy = ny
                else:
                    newx = nx / newsp[0]
                    newy = ny / newsp[1]

                mydata[mySample <= 0] = 0
                mySample[mySample <= 0] = 0

                if Dorebin == False:
                    newdata = mydata
                    newsample = mySample
                else:
                    newdata = np.zeros([newx, newy, na, nb], dtype=int)
                    newsample = np.zeros([newx, newy, na, nb], dtype=int)

                    for ia in np.arange(na):
                        for ib in np.arange(nb):
                            newdata[:, :, ia, ib] = MCD19.rebin(mydata[:, :, ia, ib], [newx, newy], Add=True)
                            newsample[:, :, ia, ib] = MCD19.rebin(mySample[:, :, ia, ib], [newx, newy], Add=True)

                newdata[np.isnan(newsample) == True] = 0
                newsample[np.isnan(newsample) == True] = 0

                newdata = np.round(newdata * 1. / newsample)
                newdata[newsample <= 0] = -999000
                newsample[newsample <= 0] = -999

                if iv == 0:
                    print(newdata.shape)
                    vcomdata = newdata
                    vcomSample = newsample
                    if Dorebin == False:
                        vcomLat = Lat
                        vcomLon = Lon
                    else:
                        vcomLat = MCD19.rebin(Lat, [newx, newy])
                        vcomLon = MCD19.rebin(Lon, [newx, newy])
                else:
                    print(v, vs, newdata.shape)
                    vcomdata = np.append(vcomdata, newdata, axis=0)
                    vcomSample = np.append(vcomSample, newsample, axis=0)
                    if Dorebin == False:
                        vcomLat = np.append(vcomLat, Lat, axis=0)
                        vcomLon = np.append(vcomLon, Lon, axis=0)
                    else:
                        vcomLat = np.append(vcomLat, MCD19.rebin(Lat, [newx, newy]), axis=0)
                        vcomLon = np.append(vcomLon, MCD19.rebin(Lon, [newx, newy]), axis=0)






            if ih==0:

                comdata=vcomdata
                comSample=vcomSample
                comLat = vcomLat
                comLon = vcomLon

            else:
                comdata = np.append(comdata, vcomdata, axis=1)
                comSample = np.append(comSample, vcomSample, axis=1)
                comLat = np.append(comLat, vcomLat, axis=1)
                comLon = np.append(comLon, vcomLon, axis=1)


        if varname==varnames[0]:
            nx, ny, nws, nwd = comdata.shape
            dso.createDimension('x', nx)
            dso.createDimension('y', ny)
            dso.createDimension('wsbin', nws)
            dso.createDimension('wdbin', nwd)

            outdata = dso.createVariable('latitude', np.float32, ('x', 'y'))
            outdata.units = 'degree'
            outdata[:] = comLat
            outdata = dso.createVariable('longitude', np.float32, ('x', 'y'))
            outdata.units = 'degree'
            outdata[:] = comLon
            outdata = dso.createVariable('MaxWindSpeed', np.float32, 'wsbin')
            outdata.units = 'm/s'
            outdata[:] = mws
            outdata = dso.createVariable('Sample', np.int, ('x', 'y', 'wsbin', 'wdbin'))
            outdata.units = 'unitless'
            outdata[:] = comSample.astype(int)

        outdata = dso.createVariable(varname, np.int, ('x', 'y', 'wsbin', 'wdbin'))
        outdata.units = units[ivar]
        outdata[:] = np.round(comdata).astype(int)

    dso.close()






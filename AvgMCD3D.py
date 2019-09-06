#This program calculate and output multi-year average of MCD19 1-km AOD

import MCD19
import numpy as np
from netCDF4 import Dataset
import os
import glob
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--h",nargs='+',type=int)
parser.add_argument("--v",nargs='+',type=int)
parser.add_argument('--start',type=int)
parser.add_argument('--end',type=int)

args=parser.parse_args()
hs=args.h
vs=args.v
start=args.start
end=args.end

Aerdir = '/global/scratch/chili/MCD19A2.006/'

outdir = '/global/scratch/chili/AvgMCD/SepWs/MWS/'
if not os.path.exists(outdir):
        os.makedirs(outdir)

complete=0.3
wsbin=[1.5,4.,8.] #m/s
seasons=['winter','spring','summer','fall']  #
#georange=[-116., 35., -114., 37.]

#Center=[46.69,24.70]
#Center=[-75.,43.]
#georange=[Center[0]-4.5,Center[1]-4.5, Center[0]+4.5, Center[1]+4.5]

MetSum=True
foundflag=True
samplewd=5  #do a 5x5 window average to increase observation number and reduce zero counts.

strse='{:10.0f}'.format(start).strip()+'-'+'{:10.0f}'.format(end).strip()

for h in hs:

    for v in vs:
        strhv = 'h' + '{:10.0f}'.format(h + 100).strip()[1:3] + 'v' + '{:10.0f}'.format(v + 100).strip()[1:3]

        for season in seasons:
            accflag=False
            for year in np.arange(end - start +1) + start:   #


                stryear = '{:10.0f}'.format(year).strip()
                startdate = year * 1000 + 1
                enddate = year * 1000 + 366


                if MetSum == True:
                    [accumAOD, accumNo, accumu10, accumv10,accumtp,accumtcc, hvLat, hvLon, totalNo] = \
                        MCD19.AccumAOD(Aerdir, strhv, startdate, enddate, wsbins=wsbin, season=season, wdbins=True, \
                                       MetSum=MetSum, wsgrid=True, samplewd=samplewd)  # ,wsgrid=True , limit=georange
                else:
                    [accumAOD, accumNo, hvLat, hvLon, totalNo] = \
                        MCD19.AccumAOD(Aerdir, strhv, startdate, enddate, wsbins=wsbin, season=season, wdbins=True, \
                                       wsgrid=True,samplewd=samplewd)  # ,wsgrid=True , limit=georange

                if accumAOD is None:
                    print('No data for year '+stryear+' '+season)
                    continue

                if accflag==False:
                    outfile = outdir + strhv + "." + season + "."+strse+".nc"
                    accflag=True
                    dso = Dataset(outfile, mode='w', format='NETCDF4')
                    nx, ny, nws, nwd = accumAOD.shape
                    dso.createDimension('x', nx)
                    dso.createDimension('y', ny)
                    dso.createDimension('wsbin', nws)
                    dso.createDimension('wdbin', nwd)

                    outdata = dso.createVariable('latitude', np.float32, ('x', 'y'))
                    outdata.units = 'degree'
                    outdata[:] = hvLat
                    outdata = dso.createVariable('longitude', np.float32, ('x', 'y'))
                    outdata.units = 'degree'
                    outdata[:] = hvLon
                    outdata = dso.createVariable('MaxWindSpeed', np.float32, 'wsbin')
                    outdata.units = 'm/s'
                    outdata[:] = np.append(wsbin, 10000.)


                outdata = dso.createVariable('AOD_' + stryear, np.int, ('x', 'y', 'wsbin', 'wdbin'))
                outdata.units = 'unitless'
                outdata[:] = np.round(accumAOD * 1000.).astype(int)

                if MetSum == True:
                    outdata = dso.createVariable('u10_' + stryear, np.int, ('x', 'y', 'wsbin', 'wdbin'))
                    outdata.units = 'm/s'
                    outdata[:] = np.round(accumu10 * 1000.).astype(int)

                    outdata = dso.createVariable('v10_' + stryear, np.int, ('x', 'y', 'wsbin', 'wdbin'))
                    outdata.units = 'm/s'
                    outdata[:] = np.round(accumv10 * 1000.).astype(int)

                    outdata = dso.createVariable('tcc_' + stryear, np.int, ('x', 'y', 'wsbin', 'wdbin'))
                    outdata.units = 'unitless'
                    outdata[:] = np.round(accumtcc * 1000.).astype(int)

                    outdata = dso.createVariable('tp_' + stryear, np.int, ('x', 'y', 'wsbin', 'wdbin'))
                    outdata.units = 'um'
                    outdata[:] = np.round(accumtp * 1.e6).astype(int)

                outdata = dso.createVariable('Sample_' + stryear, np.int, ('x', 'y', 'wsbin', 'wdbin'))
                outdata.units = 'unitless'
                outdata[:] = accumNo			
            if accflag==False:
                continue
            dso.close()








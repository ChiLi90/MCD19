
import argparse
import numpy as np
from netCDF4 import Dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int)
parser.add_argument("--ny", type=int)
parser.add_argument('--season')
parser.add_argument('--strab')

args = parser.parse_args()

season=args.season  #0 1
nchx=args.nx
nchy=args.ny
strab=args.strab

strse='2010-2015'
wdir='/global/scratch/chili/AvgMCD/SepWs/Plumes/'+strse+'/'+strab+'/'
outfile=wdir+season+'.'+strse+'.SNR.nc'

filecount=0
for ix in np.arange(nchx):
    for iy in np.arange(nchy):
        strchk = 'x' + '{:10.0f}'.format(ix).strip() + 'y' + '{:10.0f}'.format(iy).strip()
        ncfile=wdir+season+'.'+strse+'.'+strchk+'.SNR.nc'
        if os.path.exists(ncfile)==False:
            continue
        ds = Dataset(ncfile,'r')

        if filecount==0:
            varnames=list(ds.variables.keys())

            nvar=len(varnames)


        for ivar in np.arange(nvar):

            xydata=ds[varnames[ivar]][:]

            if (filecount==0) & (ivar==0):
                nx,ny=xydata.shape
                outdata=np.zeros([nvar,nx*nchx,ny*nchy])-999

            outdata[ivar,ix*nx:(ix+1)*nx,iy*ny:(iy+1)*ny]=xydata

        ds.close()

        filecount=filecount+1

dso = Dataset(outfile, mode='w', format='NETCDF4')
dso.createDimension('x', nx*nchx)
dso.createDimension('y', ny*nchy)

for ivar in np.arange(nvar):
    varname = varnames[ivar]
    if (varname == 'Lat') | (varname == 'Lon'):
        varout = dso.createVariable(varname, np.float32, ('x', 'y'))
        varout.units = 'degree'
        varout[:] = outdata[ivar, :, :]
    else:
        varout = dso.createVariable(varname, np.int, ('x', 'y'))
        varout.units = 'unitless'
        varout[:] = np.round(outdata[ivar, :, :]).astype(int)

dso.close()



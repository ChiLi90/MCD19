import numpy as np
import argparse
import glob
from netCDF4 import Dataset
import MCD19
import os


parser = argparse.ArgumentParser()
parser.add_argument("--a", type=float)
parser.add_argument("--b", type=float)
parser.add_argument("--chunckx",type=int)
parser.add_argument("--chuncky",type=int)
#e.g. a=15, b=20  a roughly represents the "width" of the source (i.e. sigma), b is the distance along the wind to integrate for both upwind and downwind
# parser.add_argument("--v", type=int)
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--season')

args = parser.parse_args()
# hs = args.h
# vs = args.v
chunckx=args.chunckx   #0 1
chuncky=args.chuncky   #0 1 2 3 4
chunckInterval=120
strchk ='x'+ '{:10.0f}'.format(chunckx).strip()+'y'+ '{:10.0f}'.format(chuncky).strip()


print("Now we are doing: ")
print(args.season,strchk,args.a,args.b)

startyr = args.start
endyr = args.end
a = args.a    #
b = args.b

indir='/global/scratch/chili/AvgMCD/SepWs/Sqr/Combined/'
outdir='/global/scratch/chili/AvgMCD/SepWs/Plumes/a'+'{:10.0f}'.format(a).strip()+'b'+'{:10.0f}'.format(b).strip()+'/'

if not os.path.exists(outdir):
    os.makedirs(outdir)


strse = '{:10.0f}'.format(startyr).strip() + '-' + '{:10.0f}'.format(endyr).strip()
seasons = [args.season]
Dirs = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
Vectors = np.array([[1.414, 0], [1, 1], [0, 1.414], [-1, 1], [-1.414, 0], [-1, 1], [0, -1.414], [1, -1]])
ndirs = len(Dirs)
wsinds = [1]

# to be modified and tested

b1 = a
b2 = b+b1
complete = 0.667  # at least 2/3 of grids with available data for SNR calculation

Rearth = 6373.0

for season in seasons:

    files = glob.glob(indir + season + '.*.nc')

    starts = np.zeros(len(files), dtype=int)
    ends = np.zeros(len(files), dtype=int)
    for ifile in np.arange(len(files)):
        file = files[ifile]
        strses = file.split('/')[-1].split('.')[1].split('-')
        starts[ifile] = strses[0]
        ends[ifile] = strses[1]

    # do the years first for each hv:
    findfile = False
    for year in np.arange(endyr - startyr + 1) + startyr:  # endyr-startyr+

        fileind = -2

        for ifile in np.arange(len(files)):
            if (starts[ifile] <= year) & (ends[ifile] >= year):
                fileind = ifile
                break

        if fileind == -2:
            continue
        else:
            findfile = True

        Aerfile = files[fileind]

        # accumulate AOD and AODsq
        ds = Dataset(Aerfile, 'r')
        stryear = '{:10.0f}'.format(year).strip()

        AOD = ds['AOD'][:]
        AODsq = ds['AODsq'][:]
        Sample = ds['Sample'][:]

        AOD[Sample <= 0] = 0
        AODsq[Sample <= 0] = 0
        Sample[Sample <= 0] = 0

        if year == startyr:
            accumAOD = AOD * Sample
            accumAODsq = AODsq * Sample
            accumNo = Sample
            Lat = ds['latitude'][:]
            Lon = ds['longitude'][:]

        else:
            accumAOD = accumAOD + AOD * Sample
            accumAODsq = accumAODsq + AODsq * Sample
            accumNo = accumNo + Sample

        ds.close()

    if findfile == False:
        continue

    [nx, ny, nws, nwd] = accumAOD.shape
    outfile = outdir  + season + '.' + strse +'.'+strchk+ '.SNR.nc'
    SNR=np.zeros([nx,ny])
    
    AODuw=np.zeros([nx,ny])
    AODdw=np.zeros([nx,ny])
    AODuwsq=np.zeros([nx,ny])
    AODdwsq=np.zeros([nx,ny])
    uwSample=np.zeros([nx,ny],dtype=int)
    dwSample=np.zeros([nx,ny],dtype=int)

    dirAODs = np.sum(accumAOD[:,:,wsinds,:],axis=2)*0.001
    dirAODsqs = np.sum(accumAODsq[:,:,wsinds,:],axis=2)*0.0001
    dirNos = np.sum(accumNo[:,:,wsinds,:],axis=2)

    outAOD=np.sum(dirAODs,axis=2)
    outNo=np.sum(dirNos,axis=2)




    for ix in np.arange(chunckInterval)+chunckInterval*chunckx:     #
        for iy in np.arange(chunckInterval)+chunckInterval*chuncky:   #

            RtCenter = [Lon[ix, iy], Lat[ix, iy]]

            diraccum = False

            for idir in np.arange(ndirs):
                dirAOD=dirAODs[:,:,idir]
                dirAODsq=dirAODsqs[:,:,idir]
                dirNo=dirNos[:,:,idir]
                # Do rotation
                if idir > 0:
                    [rtLon, rtLat] = MCD19.RotateAOD(dirAOD, Lon, Lat, \
                                                     RtCenter[0], RtCenter[1], uv=Vectors[idir, :], ToOrigin=False)
                else:
                    [rtLon, rtLat] = [Lon, Lat]

                dy = (rtLat - RtCenter[1]) * np.pi / 180. * Rearth
                dx = np.cos(RtCenter[1] * np.pi / 180.) * (rtLon - RtCenter[0]) * np.pi / 180. * Rearth

                if (np.min(dx) > 5.) | (np.min(dy) > 5.):
                    continue

                # upwind
                uwind = ((np.abs(dy) <= a) & (dx >= -1 * b2) & (dx <= -1 * b1)).nonzero()
                dwind = ((np.abs(dy) <= a) & (dx >= b1) & (dx <= b2)).nonzero()

                uwAOD = dirAOD[uwind]
                uwAODsq = dirAODsq[uwind]
                uwNo = dirNo[uwind]

                dwAOD = dirAOD[dwind]
                dwAODsq = dirAODsq[dwind]
                dwNo = dirNo[dwind]

                # completeness check
                if (np.array((dwNo > 0).nonzero()).flatten().size * 1. < complete * len(dwNo)) | (
                        np.array((uwNo > 0).nonzero()).flatten().size * 1. < complete * len(uwNo)):
                    continue

                # the upwind and downwind sampling pixels should be close (<33% diff)
                if (len(uwNo) <= 0) | (len(dwNo) <= 0) | \
                        (np.absolute(len(dwNo) - len(uwNo)) > (1. - complete) * np.max([len(dwNo), len(uwNo)])):
                    continue

                if diraccum == False:
                    sampleuw = np.sum(uwNo[uwNo > 0])
                    sampledw = np.sum(dwNo[dwNo > 0])

                    omegauw = np.sum(uwAOD[uwNo > 0])
                    omegadw = np.sum(dwAOD[dwNo > 0])

                    sigmauw = np.sum(uwAODsq[uwNo > 0])
                    sigmadw = np.sum(dwAODsq[dwNo > 0])
                    diraccum = True
                else:

                    sampleuw = sampleuw + np.sum(uwNo[uwNo > 0])
                    sampledw = sampledw + np.sum(dwNo[dwNo > 0])

                    omegauw = omegauw + np.sum(uwAOD[uwNo > 0])
                    omegadw = omegadw + np.sum(dwAOD[dwNo > 0])

                    sigmauw = sigmauw + np.sum(uwAODsq[uwNo > 0])
                    sigmadw = sigmadw + np.sum(dwAODsq[dwNo > 0])

            if diraccum == False:
                continue

            omegauw = omegauw / sampleuw
            omegadw = omegadw / sampledw
            if omegauw >= omegadw:
                continue

            AODdw[ix,iy]=omegadw
            AODuw[ix,iy]=omegauw
            dwSample[ix,iy]=sampledw
            uwSample[ix,iy]=sampleuw

            AODdwsq[ix,iy]=sigmadw/sampledw
            AODuwsq[ix,iy]=sigmauw/sampleuw


            sigmauw = np.sqrt((sigmauw - sampleuw * (omegauw ** 2)) / (sampleuw - 1))
            sigmadw = np.sqrt((sigmadw - sampledw * (omegadw ** 2)) / (sampledw - 1))

            # Calculate SNR in Mclinden et al 2016

            SNR[ix,iy] = (omegadw - omegauw) / (sigmauw / np.sqrt(sampleuw) + sigmadw / np.sqrt(sampledw))
            #print(RtCenter, omegauw, omegadw, sigmauw, sigmadw, SNR[ix, iy])
    #
    #         if SNR[ix,iy]>2.:
    #             print(RtCenter,omegauw,omegadw,sigmauw,sigmadw,SNR[ix,iy])

            # Welch's t-test
            # For this location, all the pixels downwind between b1 and b2 are signicantly larger than pixels upwind
            # tvalue=(omegadw-omegauw)/np.sqrt((sigmauw**2)/sampleuw+(sigmadw**2)/sampledw)
            # degf=np.round(((sigmauw**2)/sampleuw+(sigmadw**2)/sampledw)**2/ \
            #      (sigmauw**4/((sampleuw**2)*(sampleuw-1))+sigmadw**4/((sampledw**2)*(sampledw-1))))

            # p-value (one tail)
            # p = 1. - stats.t.cdf(tvalue, df=degf)

    outAOD[outNo>0]=outAOD[outNo>0]/outNo[outNo>0]
    outAOD[outNo<=0]=np.nan

    dso = Dataset(outfile, mode='w', format='NETCDF4')
    dso.createDimension('x', nx)
    dso.createDimension('y', ny)

    outdata = dso.createVariable('Lat', np.float32, ('x', 'y'))
    outdata.units = 'degree'
    outdata[:] = Lat

    outdata = dso.createVariable('Lon', np.float32, ('x', 'y'))
    outdata.units = 'degree'
    outdata[:] = Lon

    outdata = dso.createVariable('AOD', np.int, ('x', 'y'))
    outdata.units = 'unitless'
    outdata[:] = np.round(outAOD*1000).astype(int)

    outdata = dso.createVariable('Sample', np.int, ('x', 'y'))
    outdata.units = 'unitless'
    outdata[:] = outNo.astype(int)
    
    outdata = dso.createVariable('dwSample', np.int, ('x', 'y'))
    outdata.units = 'unitless'
    outdata[:] = dwSample.astype(int)
    
    outdata = dso.createVariable('uwSample', np.int, ('x', 'y'))
    outdata.units = 'unitless'
    outdata[:] = uwSample.astype(int)

    outdata = dso.createVariable('SNR', np.int, ('x', 'y'))
    outdata.units = 'unitless'
    outdata[:] = np.round(SNR*1000).astype(int)
    
    outdata=dso.createVariable('dwAOD', np.int, ('x', 'y'))
    outdata.units='unitless'
    outdata[:]=np.round(AODdw*1000).astype(int)
    
    outdata=dso.createVariable('uwAOD', np.int, ('x', 'y'))
    outdata.units='unitless'
    outdata[:]=np.round(AODuw*1000).astype(int)
    
    outdata=dso.createVariable('dwAODsq', np.int, ('x', 'y'))
    outdata.units='unitless'
    outdata[:]=np.round(AODdwsq*10000).astype(int)
    
    outdata=dso.createVariable('uwAODsq', np.int, ('x', 'y'))
    outdata.units='unitless'
    outdata[:]=np.round(AODuwsq*10000).astype(int)


    dso.close()



    # ax = plt.axes([0.05, 0.1, 0.95, 1])
    # ax.set_aspect('equal')
    # m = Basemap(projection='cyl', resolution='i', \
    #             llcrnrlon=np.min(Lon), llcrnrlat=np.min(Lat), urcrnrlon=np.max(Lon), urcrnrlat=np.max(Lat), \
    #             ax=ax)
    #
    # x, y = m(Lon, Lat)
    #
    # m.drawcoastlines(linewidth=0.3)
    # m.drawcountries(linewidth=0.3)
    # m.drawstates(linewidth=0.3)
    #
    # AODout = outAOD / outNo
    # AODout[AODout > 0.3] = 0.3
    # AODout[outNo<=0]=np.nan
    # clevs = np.arange(21) * 0.3 / 20.
    # title = 'AOD'
    #
    # cs = m.contourf(x, y, AODout, clevs, cmap=nclcmaps.cmap('WhBlGrYeRe'))
    #
    # x0, y0 = m(plumelons, plumelats)
    # m.scatter(x0, y0, s=3, facecolors='none', edgecolors='black', linewidths=0.1)
    #
    # cbar = m.colorbar(cs, location='bottom', pad="5%")
    # cbar.set_label(title)
    #
    # plt.savefig(outfile, dpi=600)
    # plt.close('all')

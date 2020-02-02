from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
import glob
from datetime import datetime
from datetime import timedelta
import pyproj
import csv
import scipy.interpolate as impdata
import SpatialFunctions

# @jit("int64[:](float64[:],float32[:])",nopython=True,nogil=True)
# def jit_findnearest (x,y):
#
#     result=[]
#     for aa in x:
#         result.append(np.argmin(np.absolute(y - aa)))
#
#     return np.array(result)

# @cuda.jit("void(float32[:],float32[:],int16[:])",nopython=True,nogil=True)
# def cuda_findnearest (x,y,result):
#     for l in range(0, len(x)):
#         xthis=x[l]
#         for s in range(0,len(y)):
#             if (s==0):
#                 minabs=abs(y[s]-xthis)
#                 result[l]=s
#             else:
#                 if (minabs>abs(y[s]-xthis)):
#                     minabs=abs(y[s]-xthis)
#                     result[l]=s


def rebin(arr, new_shape,**kwargs):

    new_shape=np.array(new_shape).astype(int)

    DoAdd=False
    if 'Add' in kwargs:
        if kwargs['Add']==True:
            DoAdd=True

    if new_shape[0]<arr.shape[0]:
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        if DoAdd:
            return np.nansum(np.nansum(arr.reshape(shape), axis=-1), axis=1)
        else:
            return np.nanmean(np.nanmean(arr.reshape(shape), axis=-1), axis=1)
    else:
        shape = (arr.shape[0], new_shape[0] // arr.shape[0],
                 arr.shape[1], new_shape[1] // arr.shape[1])

        stackarr = np.stack([np.stack([arr] * shape[1], axis=1)]*shape[3],axis=3)
        return stackarr.reshape(new_shape)


def AccumAOD(Aerdir, strhv, start, end,**kwargs):

    #maximum possible number of observations
    totalNo = 0
    if "samplewd" in kwargs:
        samplewd=kwargs["samplewd"]

    for date in np.arange(end - start + 1) + start:

        strdate='{:d}'.format(date).strip()
        year=np.int(strdate[0:4])
        refday = datetime(year, 1, 1) + timedelta(days=(np.int(strdate[4:7]) - 1))

        #select season
        if ('season' in kwargs):
            monthind=refday.month
            season=kwargs['season']

            if (season == 'spring') & ((monthind>5) | (monthind <3)):
                continue

            if (season == 'winter') & ((monthind<12) & (monthind >2)):
                continue

            if (season == 'summer') & ((monthind>8) | (monthind <6)):
                continue

            if (season == 'fall') & ((monthind>11) | (monthind <9)):
                continue

        stryymm = '{:10.0f}'.format(year).strip() \
                  + '.'+'{:10.0f}'.format(refday.month + 100).strip()[1:3] \
                  + '.'+'{:10.0f}'.format(refday.day + 100).strip()[1:3]

        #for this date, at most only one file is found
        Aerfile = glob.glob(Aerdir + stryymm+'/' + 'MCD19A2.A' + strdate + '.' + strhv + '.006' + '*.hdf')

        if len(Aerfile)==0:
            print("No file found for date: " + strdate)
            continue

        if totalNo>0:
            [AOD, obtime] = ReadAOD(Aerfile[0])
        else:
            [AOD, obtime, orLat, orLon] = ReadAOD(Aerfile[0],CalLatLon=True)

        #if small lat-lon region is defined
        if ('limit' in kwargs):
            if (totalNo==0):
                georange = kwargs['limit']

                vinds = np.argwhere((orLat >= georange[1]) & (orLon >= georange[0]) & \
                                    (orLat <= georange[3]) & (orLon <= georange[2]))
                minx = np.min(vinds[:, 0])
                maxx = np.max(vinds[:, 0])
                miny = np.min(vinds[:, 1])
                maxy = np.max(vinds[:, 1])


                maxx = (minx + 10 * (round((maxx - minx + 1) / 10.))).astype(int)
                maxy = (miny + 10 * (round((maxy - miny + 1) / 10.))).astype(int)


                Lat = orLat[minx:maxx, miny:maxy]
                Lon = orLon[minx:maxx, miny:maxy]
        else:
            Lat=orLat
            Lon=orLon
            georange=[np.min(Lon),np.min(Lat),np.max(Lon),np.max(Lat)]



        #for each orbit
        for iob in np.arange(len(obtime)):

            AODob = AOD[iob, :, :]
            wsgridflag=False


            if ('limit' in kwargs):
                AODob = AODob[minx:maxx,miny:maxy]

            AODob[np.isnan(AODob)==True]=-999.
            if np.all(AODob<=0.):
                print('orbit ' + '{:10.0f}'.format(
                    iob + 1).strip() + ' of date ' + strdate + ' contains no meaningful AOD')
                continue

            if ('Completeness' in kwargs):
                vinds = np.nonzero((Lat >= georange[1]) & (Lon >= georange[0]) & \
                                    (Lat <= georange[3]) & (Lon <= georange[2]))

                nvinds=len(np.array(vinds)[0,:])

                mvinds = np.nonzero(AODob[vinds]>0.)
                nmvinds = len(np.array(mvinds)[0,:])
                if nmvinds*1./nvinds<kwargs['Completeness']:
                    print('orbit ' + '{:10.0f}'.format(
                        iob + 1).strip() + ' of date ' + strdate + ' is incomplete than '+'{:10.2f}'.format(\
                        kwargs['Completeness']).strip())
                    continue

            if totalNo == 0:

                if ('wsbins' in kwargs):
                    wsbins=kwargs['wsbins']   #maximum wind speed of each bin
                    nwsbin=len(wsbins)+1
                else:
                    nwsbin=1

                if ('wdbins' in kwargs):
                    if kwargs['wdbins']==True:
                        nwdbin=8
                    else:
                        nwdbin=1
                else:
                    nwdbin = 1



                nx,ny=AODob.shape
                accumAOD = np.zeros([nx,ny,nwsbin,nwdbin])
                accumsqAOD = np.zeros([nx, ny, nwsbin, nwdbin])
                accumNo = np.zeros([nx,ny,nwsbin,nwdbin], dtype=int)

                if ('MetSum' in kwargs):
                    if (kwargs['MetSum']==True):
                        accumu10 = np.zeros([nx, ny, nwsbin, nwdbin])
                        accumv10 = np.zeros([nx, ny, nwsbin, nwdbin])
                        accumsqw10 = np.zeros([nx, ny, nwsbin, nwdbin])
                        accumtp = np.zeros([nx, ny, nwsbin, nwdbin])
                        accumtcc = np.zeros([nx, ny, nwsbin, nwdbin])

            if ('wdbins' in kwargs):

                [u10,v10] = GetWindSpeed(Lon, Lat, obtime[iob])

                if ('MetSum' in kwargs):
                    if (kwargs['MetSum'] == True):
                        tp = GetERA(Lon, Lat, obtime[iob], 'tp')
                        tcc = GetERA(Lon, Lat, obtime[iob], 'tcc')
                if ('wsgrid' in kwargs):
                    if kwargs['wsgrid']==True:
                        wsgridflag = True
                else:
                    u10m = np.nanmean(u10)
                    v10m = np.nanmean(v10)
                    tanwd = v10m / u10m

                    if (tanwd >= -0.414) & (tanwd < -0.414):
                        if (u10m > 0):
                            iwdbin = 0
                        else:
                            iwdbin = 4

                    elif (tanwd >= 0.414) & (tanwd < 2.414):
                        if (u10m > 0):
                            iwdbin = 1
                        else:
                            iwdbin = 5

                    elif (tanwd >= 2.414) | (tanwd < -2.414):
                        if (v10m > 0):
                            iwdbin = 2
                        else:
                            iwdbin = 6

                    else:
                        if (u10m > 0):
                            iwdbin = 7
                        else:
                            iwdbin = 3



            else:
                iwdbin = 0

            if ('wsbins' in kwargs):

                if ('wdbins' in kwargs) == False:
                    [u10,v10] = GetWindSpeed(Lon, Lat, obtime[iob])

                    if ('MetSum' in kwargs):
                        if (kwargs['MetSum'] == True):
                            tp = GetERA(Lon, Lat, obtime[iob], 'tp')
                            tcc = GetERA(Lon, Lat, obtime[iob], 'tcc')

                if ('wsgrid' in kwargs):
                    if kwargs['wsgrid']==True:
                        wsgridflag=True

                        #collect pixels that correspond to each wind direction and each wind strength bin
                        tanwd = v10 / u10
                        wdob=np.zeros([nx,ny],dtype=int)

                        wdob[(tanwd >= -0.414) & (tanwd < 0.414) & (u10 >= 0.)]=0
                        wdob[(tanwd >= -0.414) & (tanwd < 0.414) & (u10 < 0.)] = 4
                        wdob[(tanwd >= 0.414) & (tanwd < 2.414) & (u10 >= 0.)] = 1
                        wdob[(tanwd >= 0.414) & (tanwd < 2.414) & (u10 < 0.)] = 5
                        wdob[((tanwd >= 2.414) | (tanwd < -2.414)) & (v10 >= 0.)] = 2
                        wdob[((tanwd >= 2.414) | (tanwd < -2.414)) & (v10 < 0.)] = 6
                        wdob[(tanwd >= -2.414) & (tanwd < -0.414) & (u10 >= 0.)] = 7
                        wdob[(tanwd >= -2.414) & (tanwd < -0.414) & (u10 < 0.)] = 3


                        w10=np.sqrt(u10**2+v10**2)
                        binAOD=np.zeros([nx,ny,nwsbin,nwdbin])
                        binsqAOD=np.zeros([nx,ny,nwsbin,nwdbin])
                        binNo=np.zeros([nx,ny,nwsbin,nwdbin],dtype=int)

                        if ('MetSum' in kwargs):
                            if (kwargs['MetSum'] == True):
                                binu10 = np.zeros([nx, ny, nwsbin,nwdbin])
                                binv10 = np.zeros([nx, ny, nwsbin,nwdbin])
                                binsqw10 = np.zeros([nx, ny, nwsbin,nwdbin])
                                bintp = np.zeros([nx, ny, nwsbin,nwdbin])
                                bintcc = np.zeros([nx, ny, nwsbin,nwdbin])

                        for ibin in np.arange(nwsbin):

                            if ibin==nwsbin-1:
                                maxws=1.e5
                            else:
                                maxws=wsbins[ibin]

                            if ibin==0:
                                minws=0.
                            else:
                                minws=wsbins[ibin-1]

                            for idbin in np.arange(nwdbin):

                                ibinAOD = np.zeros([nx, ny])
                                ibinsqAOD=np.zeros([nx, ny])
                                ibinNo = np.zeros([nx, ny], dtype=int)

                                if ('MetSum' in kwargs):
                                    if (kwargs['MetSum'] == True):
                                        ibinu10 = np.zeros([nx, ny])
                                        ibinv10 = np.zeros([nx, ny])
                                        ibinsqw10=np.zeros([nx, ny])
                                        ibintp = np.zeros([nx, ny])
                                        ibintcc = np.zeros([nx, ny])

                                wsliminds = ((AODob > 0) & (w10 >= minws) & (w10 < maxws) & (wdob==idbin)).nonzero()

                                ibinAOD[wsliminds] = AODob[wsliminds]
                                ibinsqAOD[wsliminds] = AODob[wsliminds]**2
                                ibinNo[wsliminds] = 1

                                binAOD[:, :, ibin,idbin] = ibinAOD
                                binsqAOD[:, :, ibin, idbin] = ibinsqAOD
                                binNo[:, :, ibin,idbin] = ibinNo

                                if ('MetSum' in kwargs):
                                    if (kwargs['MetSum'] == True):
                                        ibinu10[wsliminds] = u10[wsliminds]
                                        ibinv10[wsliminds] = v10[wsliminds]
                                        ibinsqw10[wsliminds] = u10[wsliminds]**2+v10[wsliminds]**2
                                        ibintp[wsliminds] = tp[wsliminds]
                                        ibintcc[wsliminds] = tcc[wsliminds]

                                        binu10[:, :, ibin,idbin] = ibinu10
                                        binv10[:, :, ibin,idbin] = ibinv10
                                        binsqw10[:, :, ibin,idbin] = ibinsqw10
                                        bintcc[:, :, ibin, idbin] = ibintcc
                                        bintp[:, :, ibin, idbin] = ibintp


                        accumAOD = accumAOD + binAOD
                        accumsqAOD=accumsqAOD+binsqAOD
                        accumNo = accumNo + binNo
                        if ('MetSum' in kwargs):
                            if (kwargs['MetSum'] == True):
                                accumu10 = accumu10 + binu10
                                accumv10 = accumv10 + binv10
                                accumsqw10=accumsqw10+binsqw10
                                accumtp = accumtp + bintp
                                accumtcc = accumtcc + bintcc

                else:

                    u10m = np.nanmean(u10)
                    v10m = np.nanmean(v10)
                    w10 = np.sqrt(u10m ** 2 + v10m ** 2)

                    if w10 > wsbins[nwsbin - 2]:
                        iwsbin = nwsbin - 1
                    else:

                        wsdiff = wsbins - w10
                        wsp1diff = np.append(-1., wsdiff[0:nwsbin - 2])
                        iwsbin = (np.argwhere((wsdiff >= 0) & (wsp1diff < 0))).item(0)
            else:
                iwsbin = 0


            if wsgridflag==False:
                binAOD = np.zeros([nx, ny])
                binsqAOD=np.zeros([nx, ny])
                binNo = np.zeros([nx, ny],dtype=int)

                bininds = (AODob > 0).nonzero()
                binAOD[bininds] = AODob[bininds]
                binsqAOD[bininds]=AODob[bininds]**2
                binNo[bininds] = 1
                accumAOD[:, :, iwsbin, iwdbin] = accumAOD[:, :, iwsbin, iwdbin] + binAOD
                accumsqAOD[:, :, iwsbin, iwdbin] = accumsqAOD[:, :, iwsbin, iwdbin] + binsqAOD
                accumNo[:, :, iwsbin, iwdbin] = accumNo[:, :, iwsbin, iwdbin] + binNo

                if ('MetSum' in kwargs):
                    if (kwargs['MetSum'] == True):
                        binu10 = np.zeros([nx, ny])
                        binv10 = np.zeros([nx, ny])
                        binsqw10=np.zeros([nx, ny])
                        bintp = np.zeros([nx, ny])
                        bintcc = np.zeros([nx, ny])

                        binu10[bininds] = u10[bininds]
                        binv10[bininds] = v10[bininds]
                        binsqw10[bininds] = u10[bininds]**2+v10[bininds]**2
                        bintp[bininds] = tp[bininds]
                        bintcc[bininds] = tcc[bininds]

                        accumu10[:, :, iwsbin, iwdbin] = accumu10[:, :, iwsbin, iwdbin] + binu10
                        accumv10[:, :, iwsbin, iwdbin] = accumv10[:, :, iwsbin, iwdbin] + binv10
                        accumsqw10[:, :, iwsbin, iwdbin]=accumsqw10[:, :, iwsbin, iwdbin]+binsqw10
                        accumtp[:, :, iwsbin, iwdbin] = accumtp[:, :, iwsbin, iwdbin] + bintp
                        accumtcc[:, :, iwsbin, iwdbin] = accumtcc[:, :, iwsbin, iwdbin] + bintcc

            totalNo = totalNo + 1

    if totalNo<1:
        if ('MetSum' in kwargs):
            if (kwargs['MetSum'] == True):
                return [None,None,None,None,None,None,None,None,None,None,None]

        return [None,None,None,None,None]

    if "samplewd" in kwargs:

        newx = np.int(nx / samplewd)
        newy = np.int(ny / samplewd)
        Lat = rebin(Lat,[newx,newy])
        Lon = rebin(Lon, [newx, newy])
        naccumAOD=np.zeros([newx,newy,nwsbin,nwdbin])
        naccumsqAOD=np.zeros([newx,newy,nwsbin,nwdbin])
        naccumNo=np.zeros([newx,newy,nwsbin,nwdbin])
        for ix in np.arange(nwdbin):
            for iy in np.arange(nwsbin):
                naccumAOD[:,:,iy,ix] = rebin(accumAOD[:,:,iy,ix],[newx,newy], Add = True)
                naccumsqAOD[:, :, iy, ix] = rebin(accumsqAOD[:, :, iy, ix], [newx, newy], Add=True)
                naccumNo[:, :, iy, ix] = rebin(accumNo[:, :, iy, ix], [newx, newy], Add=True)

        accumAOD=naccumAOD
        accumsqAOD=naccumsqAOD
        accumNo=naccumNo

    accumAOD[accumNo > 0] = accumAOD[accumNo > 0] / accumNo[accumNo > 0]
    accumsqAOD[accumNo > 0] = accumsqAOD[accumNo > 0] / accumNo[accumNo > 0]
    accumAOD[accumAOD<=0]=-999.
    accumNo[accumNo<=0]=-999

    if ('MetSum' in kwargs):
        if (kwargs['MetSum'] == True):
            if "samplewd" in kwargs:
                naccumu10 = np.zeros([newx, newy, nwsbin, nwdbin])
                naccumv10 = np.zeros([newx, newy, nwsbin, nwdbin])
                naccumsqw10=np.zeros([newx, newy, nwsbin, nwdbin])
                naccumtp = np.zeros([newx, newy, nwsbin, nwdbin])
                naccumtcc = np.zeros([newx, newy, nwsbin, nwdbin])
                for ix in np.arange(nwdbin):
                    for iy in np.arange(nwsbin):
                        naccumu10[:, :, iy, ix] = rebin(accumu10[:, :, iy, ix], [newx, newy], Add=True)
                        naccumv10[:, :, iy, ix] = rebin(accumv10[:, :, iy, ix], [newx, newy], Add=True)
                        naccumsqw10[:, :, iy, ix] = rebin(accumsqw10[:, :, iy, ix], [newx, newy], Add=True)
                        naccumtp[:, :, iy, ix] = rebin(accumtp[:, :, iy, ix], [newx, newy], Add=True)
                        naccumtcc[:, :, iy, ix] = rebin(accumtcc[:, :, iy, ix], [newx, newy], Add=True)

                accumu10=naccumu10
                accumv10=naccumv10
                accumsqw10=naccumsqw10
                accumtp=naccumtp
                accumtcc=naccumtcc

            accumu10[accumNo > 0] = accumu10[accumNo > 0] / accumNo[accumNo > 0]
            accumv10[accumNo > 0] = accumv10[accumNo > 0] / accumNo[accumNo > 0]
            accumsqw10[accumNo > 0] = accumsqw10[accumNo > 0] / accumNo[accumNo > 0]
            accumtp[accumNo > 0] = accumtp[accumNo > 0] / accumNo[accumNo > 0]
            accumtcc[accumNo > 0] = accumtcc[accumNo > 0] / accumNo[accumNo > 0]

            accumu10[accumNo <= 0] = -999.
            accumv10[accumNo <= 0] = -999.
            accumsqw10[accumNo <= 0]=-999.
            accumtp[accumNo <= 0] = -999.
            accumtcc[accumNo <= 0] = -999.

    if ('MetSum' in kwargs):
        if (kwargs['MetSum'] == True):
            return [accumAOD,accumsqAOD, accumNo, accumu10, accumv10, accumsqw10, accumtp,accumtcc,Lat, Lon, totalNo]

    return [accumAOD,accumsqAOD, accumNo, Lat, Lon, totalNo]

def GetWindSpeed(Lons,Lats,obtime,**kwargs):

    ECdir = '/global/scratch/chili/ERA5UV/'
    # extract year month day UTC from obtime

    datetype=kwargs['datetype']
    if datetype=='NOD':
        year = np.int(obtime[0:4])
        dayno = np.int(obtime[4:7])
        hour = np.int(obtime[7:9]) + np.float(obtime[9:11]) / 60.

        refday = datetime(year, 1, 1) + timedelta(days=(dayno - 1))

        stryymm = '{:10.0f}'.format(year).strip() + '{:10.0f}'.format(refday.month + 100).strip()[1:3]
    if datetype=='YYMM':
        year = np.int(obtime[0:4])
        month=np.int(obtime[4:6])
        day=np.int(obtime[6:8])
        refday = datetime(year, month, day)
        stryymm = obtime[0:6]
        hour = np.int(obtime[8:10]) + np.float(obtime[10:12]) / 60.



    ECUfile = ECdir + stryymm + '.U.nc'
    ECVfile = ECdir + stryymm + '.V.nc'
    dsu = Dataset(ECUfile, 'r')
    dsv = Dataset(ECVfile, 'r')

    lat = dsu["latitude"][:].flatten()
    lon = dsu["longitude"][:].flatten()
    if (np.max(lon) > 180.):
        lontemp = lon
        lontemp[lon > 180] = lontemp[lon > 180] - 360.
        lon = lontemp

    noday = (len(dsu["time"][:]) - 1) / 8
    thour = (refday.day - 1) * 24. + hour
    thours = np.arange(noday * 24)

    tind = np.argmin(np.absolute(thours - thour))
    usf = dsu["u"]
    vsf = dsv["v"]

    usft = usf[tind, :, :, :]
    vsft = vsf[tind, :, :, :]

    if Lons.size==1:

        xind = np.argmin(np.absolute(lon - Lons[0]))
        yind = np.argmin(np.absolute(lat - Lats[0]))

        if 'bufferzone' in kwargs:
            xleft,yleft,xright,yright=kwargs['bufferzone']

            ys=np.max([yind-yleft,0])
            ye=np.min([yind+yright+1,len(lat)])

            xs = np.max([xind - xleft, 0])
            xe = np.min([xind + xright+1, len(lon)])

            udata = np.mean(usft[:, ys:ye, xs:xe], axis=0)
            vdata = np.mean(vsft[:, ys:ye, xs:xe], axis=0)
        else:
            udata = np.mean(usft[:, yind, xind], axis=0)
            vdata = np.mean(vsft[:, yind, xind], axis=0)

        dsu.close()
        dsv.close()

        if 'bufferzone' in kwargs:

            outlat=np.stack([lat[ys:ye]]*(xe-xs),axis=1)
            outlon=np.stack([lon[xs:xe]]*(ye-ys),axis=0)

            return [outlat,outlon,udata,vdata]

        return [udata,vdata]



    else:
        nx, ny = Lons.shape

        rLons = rebin(Lons, [nx / 10, ny / 10])
        rLats = rebin(Lats, [nx / 10, ny / 10])
        # parallizible?
        xind = np.array(list(map(lambda x0: np.argmin(np.absolute(lon - x0)), rLons.flatten())))
        yind = np.array(list(map(lambda y0: np.argmin(np.absolute(lat - y0)), rLats.flatten())))

        udata = np.mean(usft[:, yind, xind], axis=0)
        vdata = np.mean(vsft[:, yind, xind], axis=0)
        dsu.close()
        dsv.close()

        # rwsdata=np.sqrt(udata**2+vdata**2).reshape([nx/10,ny/10])
        return [rebin(udata.reshape(np.array([nx / 10, ny / 10]).astype(int)), [nx, ny]), \
                rebin(vdata.reshape(np.array([nx / 10, ny / 10]).astype(int)), [nx, ny])]


def GetERA(Lons,Lats,obtime,parameter):


    nx,ny=Lons.shape
    if (nx % 10==0) & (ny % 10==0):
        rLons = rebin(Lons, [nx / 10, ny / 10])
        rLats = rebin(Lats, [nx / 10, ny / 10])
    else:
        rLons=Lons
        rLats=Lats

    ECdir = '/global/scratch/chili/ERA5/'
    # extract year month day UTC from obtime
    year = np.int(obtime[0:4])
    dayno = np.int(obtime[4:7])
    hour = np.int(obtime[7:9]) + np.float(obtime[9:11]) / 60.

    refday = datetime(year, 1, 1) + timedelta(days=(dayno - 1))

    stryymm = '{:10.0f}'.format(year).strip() + '{:10.0f}'.format(refday.month + 100).strip()[1:3]

    ECfile = ECdir + stryymm + '.nc'
    ds = Dataset(ECfile, 'r')

    lat = ds["latitude"][:].flatten()
    lon = ds["longitude"][:].flatten()
    if (np.max(lon) > 180.):
        lontemp = lon
        lontemp[lon > 180] = lontemp[lon > 180] - 360.
        lon = lontemp

    #parallizible?
    xind = np.array(list(map(lambda x0: np.argmin(np.absolute(lon - x0)), rLons.flatten())))
    yind = np.array(list(map(lambda y0: np.argmin(np.absolute(lat - y0)), rLats.flatten())))

    noday = (len(ds.variables["time"][:]) - 1) / 8
    thour = (refday.day - 1) * 24. + hour
    thours = np.arange(noday * 24)

    tind = np.argmin(np.absolute(thours - thour))

    varinfo = ds.variables[parameter]
    vardata = varinfo[tind,:,:]
    varout = vardata[yind, xind]
    ds.close()


    if (nx % 10==0) & (ny % 10==0):
        outdata = varout.reshape(np.array([nx / 10, ny / 10]).astype(int))
        return rebin(outdata,[nx,ny])
    else:
        outdata = varout.reshape([nx, ny])
        return outdata

# x0, x coordinate (longitude) of the source
# y0, y coordinate (latitude) of the source
# xlocs, longitudes of each AOD grid cell
# ylocs, latitudes of each AOD grid cell
def RotateAOD(AOD,xlocs,ylocs,x0,y0,**kwargs):

    if 'uv' in kwargs:
        udata=[kwargs['uv'][0]]
        vdata=[kwargs['uv'][1]]
    else:
        print ("no wind information found!")
        return [0,0]


    if 'Uthres' in kwargs:
        wddata=np.sqrt(udata[0]**2+vdata[0]**2)
        if wddata<kwargs['Uthres']:
            return [None,None,None]

    # locind = np.argmin(np.absolute((xlocs - x0)**2+(ylocs-y0)**2))
    # x0 = xlocs.flatten()[locind]
    # y0 = ylocs.flatten()[locind]

    [xr, yr] = Rotate2East(udata[0],vdata[0], xlocs-x0, ylocs-y0)

    if 'ToOrigin' in kwargs:
        if kwargs['ToOrigin']==False:
            return [xr + x0, yr + y0]

    ocords = np.stack(((xlocs - x0).flatten(), (ylocs - y0).flatten()), axis=1)
    rcords = np.stack((xr.flatten(), yr.flatten()), axis=1)

    rtAOD = impdata.griddata(rcords, AOD.flatten(), ocords, method='linear').reshape(AOD.shape)
    # AOD[AOD<=0]=np.nan
    # rtAOD[rtAOD<=0]=np.nan

    return [rtAOD, udata[0], vdata[0]]




# rotate the fields of data (e.g. AOD according to x0,y0)
# x0, x coordinate (longitude) of the source
# y0, y coordinate (latitude) of the source
# U0, x wind component of the source
# V0, y wind component of the source
# data, 2-D data field, ~200 km wide (200 pixel x 200 pixel for 1 km)
# The source is by default the original (0,0) point
def Rotate2East(U0, V0, xs, ys):
    # calculate sin and cos (math) wind direction
    sinwd = -1 * V0 / np.sqrt(U0 ** 2 + V0 ** 2)
    coswd = U0 / np.sqrt(U0 ** 2 + V0 ** 2)

    # rotated coordinate, rotate that the wind points to the +x direction
    xr = xs * coswd - ys * sinwd
    yr = xs * sinwd + ys * coswd

    return [xr,yr]

def ReadTROPOMI(file,dataname,datauncname,minqa,maxrelerr):
    #dataname="carbonmonoxide_total_column"
    #datauncname="carbonmonoxide_total_column_precision"
    #minqa=0.5
    #maxrealerr=0.3
    try:
        fds = Dataset(file, 'r')
        ds = fds.groups['PRODUCT']
        lat = ds["latitude"][:]
        lon = ds["longitude"][:]
        CO = ds[dataname][:]
        COunc = ds[datauncname][:]
        QA = ds["qa_value"][:]

        hours = ds["delta_time"][:] / 1000. / 3600.

        hours = np.stack([hours] * (CO.shape)[2], axis=2)

        CO[(QA < minqa) | (COunc / CO > maxrelerr)] = np.nan

        fds.close()

        return [np.squeeze(CO), np.squeeze(COunc), np.squeeze(hours), np.squeeze(lat), np.squeeze(lon)]
    except:
        print('File not openable: ' + file)
        return [0, 0, 0, 0, 0]


def ReadAOD(Aerfile,**kwargs):

    try:
        ds = SD(Aerfile, SDC.READ)
        Aobj = ds.select('Optical_Depth_047')
        AOD = Aobj.get() * 0.001

        #AOD uncertainty from the file is useless
        AODunc = ds.select('AOD_Uncertainty').get() * 0.0001

        obtime = ds.attributes()['Orbit_time_stamp'].split()

        QA = ds.select('AOD_QA').get()

        AOD[(QA != 1) & (QA != 8193) & (QA != 16385) & (QA != 12289) & (QA != 20481) & (QA != 4097) & \
            (QA != 9) & (QA != 8201) & (QA != 16393) & (QA != 12297) & (QA != 20489) & (QA != 4105) ] = np.nan
            #(QA != 17) & (QA != 8209) & (QA != 16401) & (QA != 12305) & (QA != 20497) & (QA != 4113) & \
            #(QA != 25) & (QA != 8217) & (QA != 16409) & (QA != 12313) & (QA != 20505) & (QA != 4121)] = np.nan

        AODunc[(np.isnan(AODunc)) | (AODunc<=0.)]=AOD[np.isnan(AODunc)| (AODunc<=0.)]*0.1+0.05
        AOD[((AODunc / AOD) > 0.3)] = np.nan    #
        AODunc[np.isnan(AOD)] = np.nan

        if 'SMokeHeight' in kwargs:
            if kwargs['SMokeHeight']==True:
                SmokeHeight = ds.select('Injection_Height').get()
                SmokeHeight[np.isnan(AOD)]=np.nan

        if ('CalLatLon' in kwargs):
            if kwargs['CalLatLon'] == True:

                [ny, nx] = (AOD[0, :, :].shape)
                lookup1 = 'UpperLeftPointMtrs'
                lookup2 = 'LowerRightMtrs'
                StrMeta = ds.attributes()['StructMetadata.0']
                for line in StrMeta.splitlines():
                    if lookup1 in line:
                        xy = (line.split('('))[1].split(',')
                        xul = np.float(xy[0])
                        yul = np.float((xy[1].split(')'))[0])
                    if lookup2 in line:
                        xy = (line.split('('))[1].split(',')
                        xlr = np.float(xy[0])
                        ylr = np.float((xy[1].split(')'))[0])
                        break
                XDim = xul + (0.5 + np.arange(nx)) * (xlr - xul) / nx
                YDim = yul + (0.5 + np.arange(ny)) * (ylr - yul) / ny
                # YDim=np.flip(YDim)
                xv, yv = np.meshgrid(XDim, YDim)

                # In basemap, the sinusoidal projection is global, so we won't use it.
                # Instead we'll convert the grid back to lat/lons.
                sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")

                wgs84 = pyproj.Proj("+proj=latlong +R=6371007.181")
                lon, lat = pyproj.transform(sinu, wgs84, xv, yv)
        ds.end()

        #return data
        if ('CalLatLon' in kwargs):
            if kwargs['CalLatLon'] == True:

                if 'SMokeHeight' in kwargs:
                    if kwargs['SMokeHeight'] == True:
                        return [AOD,AODunc,SmokeHeight,obtime,lat,lon]

                return [AOD, AODunc, obtime, lat, lon]

        if 'SMokeHeight' in kwargs:
            if kwargs['SMokeHeight'] == True:
                return [AOD, AODunc, SmokeHeight, obtime]

        return [AOD, AODunc, obtime]
    except:

        # return data
        print('File not openable: ' + file)
        if ('CalLatLon' in kwargs):
            if kwargs['CalLatLon'] == True:
                if 'SMokeHeight' in kwargs:
                    if kwargs['SMokeHeight'] == True:
                        return [[0], [0], [0], [0], [0],[0]]
                return [[0], [0], [0], [0], [0]]

        if 'SMokeHeight' in kwargs:
            if kwargs['SMokeHeight'] == True:
                return [[0], [0], [0],[0]]
        return [[0], [0], [0]]

def ReadTile(Tilefile):
    ds = Dataset(Tilefile, 'r')
    Lat = ds['lat'][:]
    Lon = ds['lon'][:]

    if (np.max(Lon) > 180.):
        lontemp = Lon
        lontemp[Lon > 180] = lontemp[Lon > 180] - 360.
        Lon = lontemp

    ds.close()

    return [Lat, Lon]

def CutEdge(AOD):

    [ny, nx] = AOD.shape
    AOD[np.isnan(AOD)] = -999.
    AOD[AOD <= 0.] = -999.
    AOD[ny - 1, :] = -999.
    AOD[0, :] = -999.
    AOD[:, nx - 1] = -999.
    AOD[:, 0] = -999.
    return AOD



def FindPeaks(indata,wdsize,stdtime):

    #reform the AOD matrix so that we could calculate the standard deviation of windows
    orishape=indata.shape
    shape = (orishape[0]/wdsize[0], wdsize[0],
             orishape[1]/wdsize[1], wdsize[1])

    nx=orishape[0] / wdsize[0]*wdsize[0]
    ny=orishape[1] / wdsize[1]*wdsize[1]

    subdata=indata[0:nx,0:ny]

    #high-dimension data
    hddata = indata[0:nx,0:ny].reshape(shape)
    avgdata = rebin(np.nanmean(np.nanmean(hddata, axis=-1), axis=1),[nx,ny])

    #Calculate the difference for each element, then calculate the std
    hddiff = (subdata-avgdata).reshape(shape)
    hdstd = rebin(np.sqrt(np.nanmean(np.nanmean((hddiff)**2, axis=-1), axis=1)), [nx, ny])

    return (subdata>np.nanpercentile(subdata,99.))&((subdata-avgdata)>stdtime*hdstd)

def SortAvg(data,loc,mincount):

    #sort data first
    valind=(np.isnan(data)==False).nonzero()
    if np.array(valind).size <= 0:
        return [[0],[0],[0]]

    data=data[valind]
    loc=loc[valind]

    sortinds=np.argsort(loc)
    data=data[sortinds]
    loc=loc[sortinds]

    loc=np.round(loc).astype(int)
    locsum=np.bincount(loc, weights=data, minlength=0)
    loccount=np.bincount(loc, minlength=0)

    locsum=locsum[loccount>0]
    loccount = loccount[loccount > 0]
    uloc = np.unique(loc)


    inds=(loccount>mincount).nonzero()
    if np.array(inds).size <= 0:
        return [[0],[0],[0]]

    return [(locsum/loccount)[inds],loccount[inds],uloc[inds]]

def readAvgData(ds,varname,dir,ibin,minobs, **kwargs):  #kwargs could be 'years' or 'sf'

    if 'years' in kwargs:
        inyears=kwargs['years']
        startyear=inyears[0]
    else:
        inyears=[0]

    if 'sf' in kwargs:
        sf=kwargs['sf']
    else:
        sf=1.

    #'dir<0' meant that we combine all directions
    if dir<0:
        dirs=np.arange(kwargs['ndirs'])
    else:
        dirs=[dir]

    for iwdbin in np.arange(len(dirs)):

        idir=dirs[iwdbin]

        for iyear in np.arange(len(inyears)):

            if 'years' in kwargs:
                stryear = '{:10.0f}'.format(startyear + iyear).strip()
                idata = ds[varname + '_' + stryear][:, :, ibin, idir] * sf
                obsNo = ds['Sample_' + stryear][:, :, ibin, idir]
            else:
                idata = ds[varname][:, :, ibin, idir] * sf
                obsNo = ds['Sample'][:, :, ibin, idir]


            if (iwdbin == 0) & (iyear == 0):
                outdata = np.zeros(idata.shape)
                outNo = np.zeros(idata.shape)

            outdata[obsNo > 0] = outdata[obsNo > 0] + idata[obsNo > 0] * obsNo[obsNo > 0]
            outNo[obsNo > 0] = outNo[obsNo > 0] + obsNo[obsNo > 0]

    outdata = outdata / outNo
    outdata[outNo <= minobs] = np.nan

    if 'LatLon' in kwargs:
        if kwargs['LatLon']==True:
            lat=ds['latitude'][:]
            lon=ds['longitude'][:]
            return [outdata,outNo,lat,lon]

    return [outdata,outNo]

def CSVload(file):

    csvdata=[]
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csvdata.append([[x.strip() for x in row] for row in csv_reader])
    return np.array(csvdata)[0,:,:]

def gethv(Datadir,x0,y0,**kwargs):


    files=glob.glob(Datadir+'2008.06.18/*.hdf')
    strhv=[]
    for Aerfile in files:
        ds = SD(Aerfile, SDC.READ)
        lookup1 = 'UpperLeftPointMtrs'
        lookup2 = 'LowerRightMtrs'
        StrMeta = ds.attributes()['StructMetadata.0']
        for line in StrMeta.splitlines():
            if lookup1 in line:
                xy = (line.split('('))[1].split(',')
                xul = np.float(xy[0])
                yul = np.float((xy[1].split(')'))[0])
            if lookup2 in line:
                xy = (line.split('('))[1].split(',')
                xlr = np.float(xy[0])
                ylr = np.float((xy[1].split(')'))[0])
                break

        Aobj = ds.select('Optical_Depth_047')
        AOD = Aobj.get()
        [nx, ny] = AOD[0, :, :].shape
        XDim = xul + (0.5 + np.arange(nx)) * (xlr - xul) / nx
        YDim = yul + (0.5 + np.arange(ny)) * (ylr - yul) / ny
        # YDim=np.flip(YDim)
        xv, yv = np.meshgrid(XDim, YDim)
        # In basemap, the sinusoidal projection is global, so we won't use it.
        # Instead we'll convert the grid back to lat/lons.
        sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
        wgs84 = pyproj.Proj("+proj=latlong +R=6371007.181")
        lon, lat = pyproj.transform(sinu, wgs84, xv, yv)
        ds.end()

        if 'mindist' in kwargs:
            mindist = kwargs['mindist']
            distance = (lat - y0) ** 2 + (lon - x0) ** 2
            if np.min(distance) < mindist:
                strhv = np.append(strhv, ((Aerfile.split('/'))[-1].split('.'))[-4])

        if 'dx' in kwargs:
            dx = kwargs['dx']
            dy = kwargs['dy']
            if (np.min(np.absolute(lat-y0))<dy) & (np.min(np.absolute(lon-x0))<dx):
                strhv = np.append(strhv, ((Aerfile.split('/'))[-1].split('.'))[-4])


    return strhv


def getAODwind(Datadir,x0,y0,date,**kwargs):


    res=kwargs['resolution']
    dx = kwargs['dx']
    dy = kwargs['dy']
    nx=np.int(np.round(2*dx/res))
    ny = np.int(np.round(2*dy/res))

    # "date" is in the format of YYYY.MM.DD
    if 'strhv' in kwargs:
        strhvs=kwargs['strhv']
    else:
        if 'mindist' in kwargs:
            mindist = kwargs['mindist']
            strhvs = gethv(Datadir, x0, y0, mindist=mindist)
        else:
            strhvs = gethv(Datadir, x0, y0, dx=dx, dy=dy)
    obno = 0

    if 'TA' in kwargs:
        TA=kwargs['TA']

    if 'SmokeHeight' in kwargs:
        SmokeHeight=kwargs['SmokeHeight']
    else:
        SmokeHeight=False


    outAOD=np.zeros([ny,nx])-999
    outAODunc = np.zeros([ny, nx])-999
    outSample = np.zeros([ny, nx],dtype=int)
    outhours = np.zeros([ny, nx])-999
    outu = np.zeros([ny, nx])-999
    outv = np.zeros([ny, nx])-999
    if SmokeHeight == True:
        outSmokeH=np.zeros([ny,nx])-999

    for strhv in strhvs:

        files = glob.glob(Datadir + date + '/' + 'MCD19*.' + strhv + '.*hdf')

        if (len(files) < 1) | (len(files) > 1):
            print("wrong number of files for date " + date+' and '+strhv + ': ', len(files))
            continue

        if SmokeHeight==False:
            [AOD, AODunc,obtime, hvLat, hvLon] = ReadAOD(files[0], CalLatLon=True)
        else:
            [AOD, AODunc, SmokeH, obtime, hvLat, hvLon] = ReadAOD(files[0], CalLatLon=True,SMokeHeight=SmokeHeight)

        hvnx,hvny=hvLat.shape

        if 'TA' in kwargs:
            TAinds = np.array((np.char.find(obtime, TA) >= 0).nonzero()).flatten()

            if np.array(TAinds).flatten().size <= 0:
                continue
            AOD = AOD[TAinds, :, :]
            AODunc = AODunc[TAinds, :, :]
            if SmokeHeight == True:
                SmokeH = SmokeH[TAinds, :, :]
            obtime=np.array(obtime)[TAinds]
            print(obtime)

        if (np.all(AOD <= 0.)) | (np.all(np.isnan(AOD))):
            continue

        # We only select (at most) one TERRA and one AQUA overpass for each day and each hv
        recordno=0
        for iob in np.arange(len(obtime)):
            ovpsinds=(AOD[iob, :, :]>0.).nonzero()
            if np.array(ovpsinds).size>recordno:
                recordno=np.array(ovpsinds).size
                obt = obtime[iob]
                AODob = AOD[iob, :, :]
                AODuncob = AODunc[iob, :, :]
                if SmokeHeight == True:
                    SmokeHob = SmokeH[iob, :, :]

        if recordno==0:
            continue

        if (np.all(AODob <= 0.)) | (np.all(np.isnan(AODob))):
            continue

        [metlat, metlon, u, v] = \
            GetWindSpeed(np.array([[x0]]), np.array([[y0]]), obt, bufferzone=kwargs['bufferzone'], datetype='NOD')

        AODsample = AODob - AODob
        AODsample[AODob > 0.] = 1

        obhr = np.zeros([hvnx, hvny])
        obhr[AODob > 0] = np.int(obt[7:9]) + np.float(obt[9:11]) / 60.

        if SmokeHeight == False:
            AODdata = np.stack((AODob, AODuncob ** 2, obhr), axis=2)
        else:
            AODdata = np.stack((AODob, AODuncob ** 2, obhr, SmokeHob), axis=2)

        [rgAOD, rgsample] = SpatialFunctions.Regrid3Ddata(AODdata, AODsample, hvLat, hvLon, x0, y0, dx, dy, res)

        inds = (rgsample > 0).nonzero()
        if np.array(inds).size <= 0:
            continue

        outAOD[inds] = (rgAOD[:, :, 0])[inds]
        outAODunc[inds] = np.sqrt((rgAOD[:, :, 1])[inds] / rgsample[inds])

        outSample[inds] = rgsample[inds]
        outhours[inds] = (rgAOD[:, :, 2])[inds]
        if SmokeHeight == True:
            outSmokeH[inds] = (rgAOD[:, :, 3])[inds]

        uvdata = np.stack((u, v), axis=2)
        [rguv, uvsample] = SpatialFunctions.Regrid3Ddata(uvdata, (u - u + 1).astype(int), \
                                                         metlat, metlon, x0, y0, dx, dy, res)

        rgu = rguv[:, :, 0]
        rgv = rguv[:, :, 1]

        rgu[uvsample <= 0] = np.nan
        rgv[uvsample <= 0] = np.nan
        outu[inds] = rgu[inds]
        outv[inds] = rgv[inds]

        obno = obno + 1

    outu[outu<-990]=np.nan
    outv[outv<-990]=np.nan
    outAOD[outAOD<-990]=np.nan
    outhours[outhours<-990]=np.nan
    outAODunc[outAODunc<-990]=np.nan
    if SmokeHeight == True:
        outSmokeH[outSmokeH<=-990.]=np.nan

    if obno==0:
        if SmokeHeight==False:
            return [strhvs, 0, 0, 0, 0, 0, 0]
        else:
            return [strhvs, 0, 0, 0, 0, 0, 0,0]
    if SmokeHeight==False:
        return [strhvs, outAOD, outAODunc, outSample, outhours, outu, outv]
    else:
        return [strhvs, outAOD, outAODunc, outSmokeH, outSample, outhours, outu, outv]



def getTROPOMIwind(Datadir,filekey,dataname,datauncname,x0,y0,strdate,**kwargs):
    # dataname="carbonmonoxide_total_column"
    # datauncname="carbonmonoxide_total_column_precision"
    # minqa=0.5
    # maxrealerr=0.3
    # filekey=''S5P_OFFL_L2__CO_____''

    dx = kwargs['dx']
    dy = kwargs['dy']
    res = kwargs['resolution']

    minqa=0.5
    maxrelerr=1.
    if 'minqa' in kwargs:
        minqa=kwargs['minqa']

    if 'maxrelerr' in kwargs:
        maxrelerr=kwargs['maxrelerr']

    nx = np.int(np.round(2 * dx / res + 1))
    ny = np.int(np.round(2 * dy / res + 1))


    obno = 0

    outCO = np.zeros([ny, nx])
    outCOunc = np.zeros([ny,nx])
    outSample = np.zeros([ny, nx], dtype=int)
    outhours = np.zeros([ny, nx])
    outu = np.zeros([ny, nx])
    outv = np.zeros([ny, nx])

    files = glob.glob(Datadir + filekey + strdate + '*.nc')

    if (len(files) < 1):
        print("wrong number of files for date " + strdate + ': ', len(files))
        return [0,0,0,0,0,0]

    for COfile in files:
        [CO, COunc, hours, lat, lon]=ReadTROPOMI(COfile,dataname,datauncname,minqa,maxrelerr)
        if (np.all(CO <= 0.)) | (np.all(np.isnan(CO))):
            continue

        inds=((CO>0)&(lat>=(y0-dy))&(lat<=(y0+dy))&(lon>=(x0-dx))&(lon<=(x0+dx))).nonzero()

        if np.array(inds).flatten().size<=0:
            continue

        hvnx, hvny = CO.shape
        COsample = np.zeros([hvnx, hvny], dtype=int)

        COsample[CO > 0] = 1
        hour=np.mean(hours[inds])

        strhour=('{:10.0f}'.format(np.round(hour)+100).strip())[1:3]
        obt=strdate+strhour+'00A'


        [metlat, metlon, u, v] = \
            GetWindSpeed(np.array([[x0]]), np.array([[y0]]), obt, bufferzone=kwargs['bufferzone'],datetype='YYMM')

        # for each orbit, regrid to the regular grids (the next orbit will overwrite the previous one but minor effects)



        COdata = np.stack((CO, COunc, hours), axis=2)
        uvdata = np.stack((u, v), axis=2)

        [rgCO, rgsample] = SpatialFunctions.Regrid3Ddata(COdata, \
                                                         COsample, lat, lon, y0 - dy, y0 + dy, x0 - dx, x0 + dx,res)

        [rguv, uvsample] = SpatialFunctions.Regrid3Ddata(uvdata, (u - u + 1).astype(int), \
                                                         metlat, metlon, y0 - dy, y0 + dy, x0 - dx, x0 + dx, res)

        rgu = rguv[:, :, 0]
        rgv = rguv[:, :, 1]

        rgu[uvsample <= 0] = np.nan
        rgv[uvsample <= 0] = np.nan

        inds = (rgsample > 0).nonzero()
        outCO[inds] = (rgCO[:, :, 0])[inds]
        outCOunc[inds]=(rgCO[:, :, 1])[inds]
        outSample[inds] = rgsample[inds]
        outhours[inds] = (rgCO[:, :, 2])[inds]

        outu[inds] = rgu[inds]
        outv[inds] = rgv[inds]

        obno = obno + 1

    outu[outu == 0.] = np.nan
    outv[outv == 0] = np.nan

    if obno==0:
        return [0,0,0,0,0,0]

    return [outCO,outCOunc,outSample, outhours,outu, outv]



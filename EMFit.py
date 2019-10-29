# Exponentially modified (convoluted) function fitting, including EMG and Non-Gaussian source functions
from scipy.special import erfc as serfc
import numpy as np
from lmfit import minimize,  Parameters, report_fit
from scipy.stats.stats import pearsonr
from pyDOE import lhs
from scipy.stats.distributions import norm
from scipy.stats import t
import mmap
from scipy.optimize import minimize as scipyminimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, leastsq
import numdifftools as nd
#xW: windy coordinates
#xC: Calm coordinates
#AvgW(C): concentration under Windy (Calm) conditions
def EMFFit (xW,xC,AvgC,AvgW,xmax,minx0):

    if (len(AvgC) >= xmax) & (len(AvgW) >= xmax):

        [nhf, CMatrix, startind, endind] = PrepareEMF(xW, xC, AvgC, xmax)

        if nhf==None:
            return [None, None, None, None]
        #testings for different initial values, we found that it is not sensitive once the shape of source function (Calm conditions) is determined
        #initials=np.array([0.1,0.2, 0.5, 1., 2., 5.,10.])
        inixs=[minx0 * 3.]
        inias=[1.]
        minchisquare = -1

        for inix in inixs:
            for inia in inias:

                params = Parameters()
                params.add('x0', value=inix, min=1./3)
                params.add('a', value=inia, min=0.0)
                params.add('b', value=0.)

                try:
                    out = minimize(residualEMF, params, args=(xW, AvgW, nhf, CMatrix, startind, endind))

                    FitAvgWEMF = EMF(xW, nhf, CMatrix, startind, endind, out.params['a'], out.params['b'],
                                     out.params['x0'])
                    corvalue = (pearsonr(AvgW[startind:endind + 1], FitAvgWEMF))[0]
                    #print outstr

                    if (minchisquare == -1) | (out.chisqr < minchisquare):
                        EMFout = out
                        minchisquare = out.chisqr

                except:
                    continue

        if minchisquare==-1:
            return [None,None,None,None]
        else:
            return [EMFout, xW[startind:endind + 1], AvgW[startind:endind + 1], FitAvgWEMF]


    else:
        return [None,None,None,None]

def EMGFit(x,data,samplewd,minx0,nSample,**kwargs):

    # initial guess of each parameter
    nW = len(x)
    ini_a = np.sum((data[1:] + data[0:nW - 1]) / 2. * (x[1:] - x[0:nW - 1]))
    maxxW = x[np.nanargmax(data)]
    # print maxxW,x[0]
    # if maxxW <= x[0]:
    #     return [False,False]

    HM = (np.max(data) - np.min(data)) / 2.+np.min(data)
    FWHM = np.absolute(x[np.nanargmin(np.absolute(data - HM))] - maxxW)*2./2.355

    minchisquare = -1

    design=lhs(3,samples=nSample)
    means=np.array([ini_a,minx0*3.,FWHM])
    stds=means


    DoFixSource=False

    if 'fixSource' in kwargs:
        if kwargs['fixSource']==True:
            fixxsc=0.
            if 'xsc' in kwargs:
                fixxsc = kwargs['xsc']

            DoFixSource=True



    if 'solver' in kwargs:
        solver=kwargs['solver']
    else:
        solver='trust-constr'

    for i in np.arange(3):
        design[:,i]=norm(loc=means[i],scale=stds[i]).ppf(design[:,i])

    for jfit in np.arange(nSample+1):

        if jfit < nSample:
            fitpars = design[jfit, :]
        else:
            fitpars = means

        if np.min(fitpars)<0:
            continue
        try:

            if solver == 'lmfit':

                params = Parameters()
                params.add('a', value=fitpars[0], min=0.)  # x-direction integration of total burden
                params.add('x0', value=fitpars[1], min=1. / 3)  # x-direction distance in one-lifetime

                if DoFixSource == True:
                    params.add('xsc', value=fixxsc, vary=False)  # # x0+ux
                else:
                    params.add('xsc', value=maxxW, min=x[0], max=x[-1])

                params.add('sigma', value=fitpars[2], min=0.5 / samplewd,
                           max=np.max([maxxW - x[0], x[-1] - maxxW]))  # # standard deviation
                params.add('b', value=np.min(data), min=0., max=np.max(data))  # background concentration

                out = minimize(residualEMG, params, args=(x, data), iter_cb=EMGCall)

                chisqr = np.sum(residualEMG(out.params, x, data) ** 2)

                if (minchisquare == -1) | (chisqr < minchisquare):
                    EMGout = out.params
                    minchisquare = chisqr

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    break

            elif solver == 'trust-constr':  # Do the scipy.optimize fit using "trust_constr" method

                x0 = np.zeros(5)
                x0[0:2] = fitpars[0:2]
                x0[3] = fitpars[2]
                x0[2] = maxxW  # , min = x[0], max = x[-1])
                x0[4] = np.min(data)


                #we now limit the source location to be at most 50 km (rotation will make sense when you are close).
                bounds = Bounds([0., 1. / 3, -50./samplewd, 0.5, 0.], \
                                [np.inf, np.inf, 50./samplewd, np.max([maxxW - x[0], x[-1] - maxxW]), np.max(data)])

                # test if solutions are similar for different "fixsource" handling

                if DoFixSource:
                    bounds.lb[2] = fixxsc
                    bounds.ub[2] = fixxsc
		
		#we set x0>=sigma in the fitting, and reject results when x0==sigma
                #lconstr = LinearConstraint([[0, 2, 1, 2, 0],[0,1,0,-2,0]], [-np.inf,0.], [x[-1],np.inf])
                lconstr = LinearConstraint([[0,1,1,1,0],[0,1,0,-1,0]],[-np.inf,0.],[x[-1],np.inf])
                nlconstr = NonlinearConstraint(EMGNLConstr1, -np.inf, 20.)
                # args = {'x0': x[0], 'xca': fixxca, 'xcc': fixxcc}
                # nlconstr = NonlinearConstraint(EMANLConstr2, -np.inf, 0.)
                res = scipyminimize(EMGchisqr1, x0, args=(x, data), method='trust-constr', \
                                    constraints=[lconstr,nlconstr], options={'verbose': 0},
                                    bounds=bounds)  # constraints=[lconstr, nlconstr],
                Hess = nd.Hessian(EMGchisqr1)(res.x, x, data)

                pars = res.x

                chisqr = EMGchisqr1(pars, x, data)
                sigmas = np.sqrt(np.linalg.inv(Hess) * chisqr / (data.size - pars.size))

                params = Parameters()
                params.add('a', value=pars[0], brute_step=sigmas[0, 0])  # x-direction integration of total burden
                params.add('x0', value=pars[1], brute_step=sigmas[1, 1])

                params.add('xsc', value=pars[2], brute_step=sigmas[2, 2])
                params.add('sigma', value=pars[3], brute_step=sigmas[3, 3])
                params.add('b', value=pars[4], brute_step=sigmas[4, 4])

                if (minchisquare == -1) | (chisqr < minchisquare):
                    EMGout = params
                    minchisquare = chisqr

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    break

            else:

                print ("solver wrong! specify one of these: pyipm, trust-constr, lmfit !")
                return [False, False]

        except:
            continue

    if minchisquare == -1:
        return [False,False]
    else:

        return [EMGout,True]


def EMAFit(x,data,samplewd,minx0,nSample,**kwargs):

    # initial guess of each parameter
    nW = len(x)
    ini_a = np.sum((data[1:] + data[0:nW - 1]) / 2. * (x[1:] - x[0:nW - 1]))
    maxxW = x[np.nanargmax(data)]

    # if maxxW <= x[0]:
    #     return [False,False]

    HM = (np.max(data) - np.min(data)) / 2.+np.min(data)
    FWHM = np.absolute(x[np.nanargmin(np.absolute(data - HM))] - maxxW)*2./2.355
    minchisquare=-1

    design = lhs(6, samples=nSample)
    means = np.array([ini_a/2./(minx0*30.),ini_a/2./(minx0*3.),minx0 * 30., minx0*3.,FWHM,FWHM/10.])
    stds = means

    DoFixSource=False
    if 'fixSource' in kwargs:
        if kwargs['fixSource']==True:
            fixxsca=0.
            fixxscc=0.
            if 'xsca' in kwargs:
                fixxsca = kwargs['xsca']
            if 'xscc' in kwargs:
                fixxscc = kwargs['xscc']

            DoFixSource=True

    # DoSameSource = False
    # if 'sameSource' in kwargs:
    #     if kwargs['sameSource'] == True:
    #         DoSameSource = True

    PDom = False    #no precursor source
    SDom = False    #no primary source
    SameSource = False  #source location is the same for aerosols and precursors
    SameSigma=False

    if 'sameSource' in kwargs:
        if kwargs['sameSource'] == True:
            SameSource = True

    if 'sameSigma' in kwargs:
        if kwargs['sameSigma'] == True:
            SameSigma = True

    if 'pDom' in kwargs:
        if kwargs['pDom'] == True:
            PDom = True

    if 'sDom' in kwargs:
        if kwargs['sDom'] == True:
            SDom = True

    if 'solver' in kwargs:
        solver=kwargs['solver']
    else:
        solver='trust-constr'

    for i in np.arange(6):
        design[:, i] = norm(loc=means[i], scale=stds[i]).ppf(design[:, i])


    for jfit in np.arange(nSample+1):
        if jfit<nSample:
            fitpars = design[jfit, :]
        else:
            fitpars = means
        if np.min(fitpars) < 0:
            continue




        try:
            if solver == 'lmfit':

                params = Parameters()
                params.add('a', value=fitpars[0], min=0.)  # x-direction integration of total burden
                params.add('xa', value=fitpars[2], min=1. / 3)
                params.add('xc', value=fitpars[3], min=1. / 3)
                # params.add('fc', value=inifc, min=0., max=1.)
                params.add('c', value=fitpars[1], min=0.)

                if DoFixSource == True:
                    params.add('xsca', value=fixxsca, vary=False)  # # x0+ux
                    params.add('xscc', value=fixxscc, vary=False)  # # x0+ux

                else:
                    params.add('xsca', value=maxxW, min=x[0], max=x[-1])
                    params.add('xscc', value=maxxW, min=x[0], max=x[-1])

                params.add('sigmaa', value=fitpars[4], min=0.5 / samplewd,
                           max=np.max([maxxW - x[0], x[-1] - maxxW]))  # standard deviation
                params.add('sigmac', value=fitpars[5], min=0.5 / samplewd,
                           max=np.max([maxxW - x[0], x[-1] - maxxW]))  # standard deviation
                params.add('b', value=np.min(data), min=0.,
                           max=np.max(data))  # background concentration

                out = minimize(residualEMA, params, args=(x, data), iter_cb=EMACall)
                chisqr = np.sum(residualEMA(out.params, x, data) ** 2)

                if (minchisquare == -1) | (chisqr < minchisquare):
                    EMAout = out.params
                    minchisquare = chisqr

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    break

            elif solver == 'trust-constr':  # Do the scipy.optimize fit using "trust_constr" method


                x0 = np.zeros(9)
                x0[0:4] = fitpars[0:4]
                x0[6:8] = fitpars[4:6]
                x0[8] = np.min(data)
                x0[4] = maxxW  # , min = x[0], max = x[-1])
                x0[5] = maxxW  # , min = x[0], max = x[-1])

                #Now limit the source location to be within 50 km (so that rotation makes sense)
                bounds = Bounds([0., 0., 1. / 3, 1. / 3, -50./samplewd, -50./samplewd, 0.5, 0.5, 0.], \
                                [np.inf, np.inf, np.inf, np.inf, 50./samplewd, 50./samplewd, \
                                 np.max([maxxW - x[0], x[-1] - maxxW]), \
                                 np.max([maxxW - x[0], x[-1] - maxxW]), np.max(data)])

                # test if solutions are similar for different "fixsource" handling
                if DoFixSource==True:
                    bounds.lb[4] = fixxsca
                    bounds.ub[4] = fixxsca
                    bounds.lb[5] = fixxscc
                    bounds.ub[5] = fixxscc

                if PDom==True:
                    bounds.lb[1]=0.
                    bounds.ub[1]=0.


                if SDom==True:
                    bounds.lb[0]=0.
                    bounds.ub[0]=0.

                lconstmat=[[0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1, 0],[0,0,1,0,0,0,-1,0,0],[0,0,0,1,0,0,0,-1,0]]
                lconstmin=[-np.inf, -np.inf,0.,0.]
                lconstmax=[x[-1], x[-1],np.inf,np.inf]



                if SameSource==True:
                    lconstmat=np.append(lconstmat,[[0, 0, 0, 0, 1, -1, 0, 0, 0]],axis=0)
                    lconstmin=np.append(lconstmin,0.)
                    lconstmax=np.append(lconstmax,0.)

                if SameSigma==True:
                    lconstmat = np.append(lconstmat, [[0, 0, 0, 0, 0, 0, 1, -1, 0]], axis=0)
                    lconstmin = np.append(lconstmin, 0.)
                    lconstmax = np.append(lconstmax, 0.)

                lconstr = LinearConstraint(lconstmat,lconstmin,lconstmax)



                nlconstr = NonlinearConstraint(EMANLConstr1, -np.inf, 20.)
                res = scipyminimize(EMAchisqr1, x0, args=(x, data), method='trust-constr', \
                                    constraints=[lconstr,nlconstr], options={'verbose': 0},
                                    bounds=bounds)  # constraints=[lconstr, nlconstr],
                Hess = nd.Hessian(EMAchisqr1)(res.x, x, data)
                    # args = {'x0': x[0]}

                # uncertanties from fitting, following Laughner et al. (2016)
                pars = res.x

                chisqr = EMAchisqr1(pars, x, data)
                sigmas = np.sqrt(np.linalg.inv(Hess) * chisqr / (data.size - pars.size))


                params = Parameters()
                if SDom:
                    params.add('a', value=0., brute_step=0.)
                else:
                    params.add('a', value=pars[0], brute_step=sigmas[0, 0])

                if PDom:
                    params.add('c', value=0., brute_step=0.)
                else:
                    params.add('c', value=pars[1], brute_step=sigmas[1, 1])

                params.add('xscc', value=pars[5], brute_step=sigmas[5, 5])
                params.add('xsca', value=pars[4], brute_step=sigmas[4, 4])
                params.add('xa', value=pars[2], brute_step=sigmas[2, 2])
                params.add('xc', value=pars[3], brute_step=sigmas[3, 3])
                params.add('sigmaa', value=pars[6], brute_step=sigmas[6, 6])
                params.add('sigmac', value=pars[7], brute_step=sigmas[7, 7])
                params.add('b', value=pars[8], brute_step=sigmas[8, 8])


                if (minchisquare == -1) | (chisqr < minchisquare):
                    EMAout = params
                    minchisquare = chisqr

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    break

            else:

                print ("solver wrong! specify one of these: pyipm, trust-constr, lmfit !")
                return [False,False]
        except:
            continue

    if minchisquare == -1:
        return [False,False]
    else:

        return [EMAout,True]

#Exponentially modified gaussian function.
#x,xsc,x0 should have the same units.
#xsc=u+x0
def EMG (x,a,x0,xsc,sigma,b,**kwargs):

    sqrt2=np.sqrt(2.)

    return 0.5 * a / x0 * np.exp((xsc - x) / x0 + (sigma ** 2) / 2. / (x0 ** 2)) \
           * serfc((xsc - x) /sqrt2 / sigma + sigma /sqrt2 / x0) + b

#Not fix source
def EMGchisqr1(x,*args):

    return np.sum((EMGres1(x,*args))**2)

#fix source

def EMAchisqr1(x,*args):

    return np.sum((EMAres1(x,*args))**2)

def EMGres1(x,*args):

    return EMG(args[0],x[0],x[1],x[2],x[3],x[4])-args[1]


def EMAres1(x,*args):

    return EMA(args[0],x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])-args[1]


def EMANLConstr1(x):

    return [np.exp(x[4]/x[2]+x[6] ** 2 / (2. * (x[2] ** 2))),     #(x[4] - args['x0']) / x[2] +
            np.exp(x[5]/x[3]+x[7] ** 2 / (2. * (x[3] ** 2))),     #(x[5] - args['x0']) / x[3] +
            np.exp(x[5]/x[2]+x[7] ** 2 / (2. * (x[2] ** 2)))]       #(x[5] - args['x0']) / x[2] +

def EMGNLConstr1(x):

    return np.exp(x[2]/x[1]+x[3] ** 2 / (2. * (x[1] ** 2)))


#Exponentially modified aerosol function.
#x,xa,xc,xsc should have the same units.
#assume same source location (xsc, suitable for point-like sources) for c and a
def EMA (x,a,c,xa,xc,xsca,xscc,sigmaa,sigmac,b,**kwargs):

    #c=f*c
    #assume that xsca=xscc and sigmaa=sigmac

    sqrt2=np.sqrt(2.)

    #print xa,xc,a,c,scdtm,fc
    #print (xsc - x[0]) / xa, (sigmaa ** 2) / 2. / (xa ** 2), (xsc - x[0]) / xc, (sigmac ** 2) / 2. / (xc ** 2)
    if (np.absolute(xc - xa) / xa < 0.0001):
        scdtm = 0.
    else:
        scdtm = c * xa / (xa - xc)

    return 0.5 * a * np.exp((xsca - x) / xa + (sigmaa ** 2) / 2. / (xa ** 2)) \
           * serfc((xsca - x) /sqrt2 / sigmaa + sigmaa /sqrt2 / xa) + \
           0.5 * scdtm * np.exp((xscc - x) / xa + (sigmac ** 2) / 2. / (xa ** 2)) \
           * serfc((xscc - x) /sqrt2 / sigmac + sigmac /sqrt2 / xa)- \
           0.5 * scdtm * np.exp((xscc - x) / xc + (sigmac ** 2) / 2. / (xc ** 2)) \
           * serfc((xscc - x) /sqrt2 / sigmac + sigmac /sqrt2 / xc) + b


def EMALeft (x,a,xa,xsc,sigmaa):

    sqrt2=np.sqrt(2.)

    #print xa,xc,a,c,scdtm,fc
    #print (xsc - x[0]) / xa, (sigmaa ** 2) / 2. / (xa ** 2), (xsc - x[0]) / xc, (sigmac ** 2) / 2. / (xc ** 2)

    return 0.5 * a* np.exp((xsc - x) / xa + (sigmaa ** 2) / 2. / (xa ** 2)) \
           * serfc((xsc - x) /sqrt2 / sigmaa + sigmaa /sqrt2 / xa)

def EMAMiddle(x, c, xa, xc, xsc, sigmac):

        sqrt2 = np.sqrt(2.)

        if (np.absolute(xc - xa) / xa < 0.0001):
            xt = 0.
        else:
            xt = 1. / (1. / xc - 1. / xa)

        scdtm = c * xt / xc

        # print xa,xc,a,c,scdtm,fc
        # print (xsc - x[0]) / xa, (sigmaa ** 2) / 2. / (xa ** 2), (xsc - x[0]) / xc, (sigmac ** 2) / 2. / (xc ** 2)

        return 0.5 * scdtm * np.exp((xsc - x) / xa + (sigmac ** 2) / 2. / (xa ** 2)) \
           * serfc((xsc - x) /sqrt2 / sigmac + sigmac /sqrt2 / xa)

def EMARight(x,c,xa,xc,xsc,sigmac):

    sqrt2 = np.sqrt(2.)

    if (np.absolute(xc - xa) / xa < 0.0001):
        xt = 0.
    else:
        xt = 1. / (1. / xc - 1. / xa)

    scdtm = c * xt / xc

    #return \
        #0.5*scdtm* np.exp((xsc - x) / xa + (sigmaa ** 2) / 2. / (xa ** 2)) \
        #* erfc((xsc - x) /sqrt2 / sigmaa + sigmaa /sqrt2 / xa) \
    return -0.5 * scdtm * np.exp((xsc - x) / xc + (sigmac ** 2) / 2. / (xc ** 2))\
           * serfc((xsc - x) /sqrt2 / sigmac + sigmac /sqrt2 / xc)


def EMACall(params, iter, resid, *fcn_args, **fcn_kws):


    xW=fcn_args[0]
    xsca=params['xsca']
    xscc = params['xscc']
    sigmaa=params['sigmaa']
    sigmac=params['sigmac']

    params['xa'].set(min=np.max([(xsca-xW[0])/20.,(xscc-xW[0])/20.,np.sqrt((sigmaa**2)/40.),np.sqrt((sigmac**2)/40.)]))
    params['xc'].set(min=np.max([(xscc-xW[0])/20.,np.sqrt((sigmac**2)/40.)]))


#To avoid large exponential terms during the iteration, we further constrain the two terms in exp(...) to be less than 20...
def EMGCall(params, iter, resid, *fcn_args, **fcn_kws):

    xsc=params['xsc']
    sigma = params['sigma']
    xW=fcn_args[0]
    params['x0'].set(min=np.max([(xsc-xW[0])/20.,np.sqrt((sigma**2)/40.)]))
    #params['xsc'].set(min=-0.1*params['x0'],max=0.1*params['x0'])

#Calculate CMatrx
def PrepareEMF (xW,xF,F,xmax):

    nhf = np.round(xmax * 2. / 3.).astype(int)
    xstart = -1 * xmax + nhf

    xFill = np.arange(2 * xmax + 1) - xmax

    FFill = np.zeros(2 * xmax + 1)
    FFill[xF + xmax] = F  # zero otherwise

    validx = xFill[FFill > 0.]
    xleft = validx[0]
    xright = validx[-1]

    if (np.max(xW) < xstart + xleft + xmax) | (np.min(xW) > xright):
        return [None,None,None,None]

    #FFill[FFill <= 0.] = np.interp(xFill[FFill <= 0.], xFill[FFill > 0.], FFill[FFill > 0.])
    FFill[FFill <= 0.] = 0

    CMatrix = np.zeros([FFill.size - nhf, nhf + 1], dtype=float)
    for icol in np.arange(nhf + 1):
        CMatrix[:, icol] = FFill[nhf - icol:3 * nhf - icol + 1]

    startind = (np.argwhere(xW >= (xleft + xstart + xmax)))[0][0]
    endind = (np.argwhere(xW <= xright))[-1][0]

    return [nhf, CMatrix, startind, endind]

#Exponentially modified W from any function F
#nhf is the distance [0,nhf] that the exponential decay function applies to
#xW should be adjusted based on availability of xW (We only do complete convolution),using startind and endind
#xmax: 3*nhf/2
def EMF(xW,nhf,CMatrix,startind,endind,a,b,x0):

    expdata = np.exp(-1. * np.arange(nhf + 1) / x0)
    expMat = np.stack([expdata] * (CMatrix.shape)[0], axis=0)
    expMat[CMatrix==0.]=0.
    model = np.nansum(CMatrix * expMat, axis=1)/np.nansum(expMat,axis=1)*a + b
    return model[xW[startind:endind+1]+nhf/2]

def residualEMF(params,x,data,nhf,CMatrix,startind,endind):

    x0 = params['x0']
    a = params['a']
    b = params['b']
    return EMF(x,nhf,CMatrix,startind,endind,a,b,x0)-data[startind:endind+1]


def residualEMG(params,x,data):

    a=params['a']
    x0=params['x0']
    xsc=params['xsc']
    sigma=params['sigma']
    b=params['b']
    return EMG(x,a,x0,xsc,sigma,b)-data

def residualEMA(params,x,data):

    a=params['a']
    c=params['c']
    xa=params['xa']
    xc=params['xc']
    #fc=params['fc']
    xsca=params['xsca']
    xscc = params['xscc']

    sigmaa=params['sigmaa']
    sigmac = params['sigmac']

    b=params['b']
    return EMA(x,a,c,xa,xc,xsca,xscc,sigmaa,sigmac,b)-data


def mapcount(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    f.close()
    return lines

#calculate 95% CI of fitted parameters
def calcErrIntV(value,stddev,ndata,nfit,othererr):

    fiterr=stddev*t.ppf(0.975, ndata-nfit)/np.sqrt(nfit)

    totalsqerr=(fiterr/value)**2+np.total(othererr**2)


    return value*np.sqrt(totalsqerr)




def ExamFit(file,Pardicts,GoodR):

    Dirs=['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
    ndirs=len(Dirs)
    linNo=mapcount(file)
    nheader=2

    nfit=len(Pardicts.keys())

    totalcount=0
    for key,value in Pardicts.items():

        totalcount=totalcount+len(value)

    xmax=linNo-totalcount-nheader-3


    #Fitdata=np.chararray([linNo-nheader,2*ndirs+1],itemsize=10)
    Fitdata = np.genfromtxt(file,dtype='str',skip_header=2)



    AvgW = Fitdata[0:xmax, :].astype(float)

    startrec=False
    for idir in np.arange(8):

        #one sub-dirctionary for each wind direction
        startdict = False
        xstart = xmax

        for key,value in Pardicts.items():


            npar = len(value)

            Fitone = Fitdata[xstart:xstart + npar + 1, 2*idir+1:2*idir+3]
            xstart=xstart+npar+1

            correlation=np.float(Fitone[len(value),0])

            if correlation<GoodR:
                continue

            if startdict==False:

                dirDict = dict()
                startdict = True

            dirDict['Y'] = AvgW[:, 2 * idir + 1]
            dirDict['xW'] = AvgW[:, 0]

            for ipar in np.arange(npar):
                dirDict[value[ipar] + '_' + key] = np.float(Fitone[ipar, 0])
                dirDict[value[ipar] + 'std_' + key] = np.float(Fitone[ipar, 1])

            dirDict['R_'+key]=correlation




        if startdict==True:
            dirDict['ws'] = np.float(Fitdata[xstart, idir * 2 + 1])
            dirDict['wsstd'] = np.float(Fitdata[xstart, idir * 2 + 2])
            if startrec==False:
                FitDict=dict()
                startrec=True

            FitDict[Dirs[idir]] = dirDict


    if startrec==True:
        return FitDict
    else:
        return None







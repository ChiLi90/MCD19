# Exponentially modified (convoluted) function fitting, including EMG and Non-Gaussian source functions
from scipy.special import erfc as serfc
from scipy.special import erf as serf
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

                    if (minchisquare <0) | (out.chisqr < minchisquare):
                        EMFout = out
                        minchisquare = out.chisqr

                except:
                    continue

        if minchisquare<0:
            return [None,None,None,None]
        else:
            return [EMFout, xW[startind:endind + 1], AvgW[startind:endind + 1], FitAvgWEMF]


    else:
        return [None,None,None,None]

def EMGFit(x,data,nSample,**kwargs):

    maxtry=100
    if "maxtry" in kwargs:
        maxtry=kwargs["maxtry"]
    parnames=['a','x0','xsc','sigma','b']
    # initial guess of each parameter
    nW = len(x)
    ini_a = np.sum((data[1:] + data[0:nW - 1]) / 2. * np.absolute(x[1:] - x[0:nW - 1]))

    #Weights of each point
    DtWeights=data-data+1
    # DtWeights[data>np.nanpercentile(data,75.)]=4.
    # DtWeights[data < np.nanpercentile(data, 25.)] = 4.

    maxxW = x[np.nanargmax(data)]
    # print maxxW,x[0]
    # if maxxW <= x[0]:
    #     return [False,False]
    #resx=np.absolute(x[1]-x[0])

    HM = (np.max(data) - np.min(data)) / 2.+np.min(data)
    FWHM = np.absolute(x[np.nanargmin(np.absolute(data - HM))] - maxxW)*2./2.355

    minchisquare = -1
    means=np.array([ini_a,50.,FWHM])

    Gaussian = False

    GauDecay=False

    if 'GauDecay' in kwargs:
        if kwargs['GauDecay']==True:
            GauDecay = True
            avgw10 = kwargs['minx0'] / 3600.



    if 'fixparam' in kwargs:
        fixpars=kwargs['fixparam']
        fixvals=kwargs['fixvalue']


    if 'solver' in kwargs:
        solver=kwargs['solver']
    else:
        solver='trust-constr'

    FoundSolution = False
    SampleDoubled = False
    ntry = 0

    while (FoundSolution == False) & (ntry < maxtry):

        ntry = ntry + 1
        SolutionUpdated=False
        design = lhs(3, samples=nSample)
        stds = 0.99*means
        for i in np.arange(3):
            design[:, i] = norm(loc=means[i], scale=stds[i]).ppf(design[:, i])

        for jfit in np.arange(nSample + 1):



            if jfit < nSample:
                fitpars = design[jfit, :]
            else:
                fitpars = means

            if np.min(fitpars) < 0:
                continue

            if solver == 'lmfit':
                params = Parameters()
                params.add('a', value=fitpars[0], min=0.)  # x-direction integration of total burden
                params.add('x0', value=fitpars[1], min=resx / 3)  # x-direction distance in one-lifetime

                params.add('xsc', value=maxxW, min=np.min(x), max=np.max(x))

                params.add('sigma', value=fitpars[2], min=0.5,
                           max=np.max([maxxW - np.min(x), np.max(x) - maxxW]))  # # standard deviation
                params.add('b', value=np.min(data), min=0., max=np.max(data))  # background concentration

                out = minimize(residualEMG, params, args=(x, data), iter_cb=EMGCall)

                chisqr = np.sum(residualEMG(out.params, x, data) ** 2)

                if (minchisquare <0) | (chisqr < minchisquare):
                    EMGout = out.params
                    minchisquare = chisqr

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    break

            elif solver == 'trust-constr':  # Do the scipy.optimize fit using "trust_constr" method

                global uknindsg, knindsg, knvalsg

                uknindsg = np.arange(5).astype(int)
                knindsg = np.array([]).astype(int)
                knvalsg = []

                x0 = np.zeros(5)
                x0[0:2] = fitpars[0:2]
                x0[3] = fitpars[2]
                x0[2] = maxxW  # , min = x[0], max = x[-1])
                x0[4] = np.min(data)

                lbds = np.array([0., 0., np.min(x), 0., 0.])
                hbds = np.array(
                    [np.inf, np.inf, np.max(x), np.inf, np.max(data)])  #np.max([maxxW - np.min(x), np.max(x) - maxxW])

                # test if solutions are similar for different "fixsource" handling

                if 'fixparam' in kwargs:
                    for ipar in np.arange(len(fixvals)):

                        if (fixpars[ipar] == 1) & (np.absolute(fixvals[ipar]) <= 1.e-20):
                            Gaussian = True

                        if ~np.isin(fixpars[ipar], uknindsg):
                            continue
                        knindsg = np.append(knindsg, np.int(fixpars[ipar]))
                        knvalsg = np.append(knvalsg, fixvals[ipar])
                        delinds = ~np.isin(uknindsg, fixpars[ipar])
                        uknindsg = uknindsg[delinds]
                        x0 = x0[delinds]

                        # process bounds to be consistent with x0 and ukninds...

                bounds = Bounds(lbds[uknindsg], hbds[uknindsg],keep_feasible=True)

                nlconstr = NonlinearConstraint(EMGNLConstr1, -np.inf, 100.)

                try:

                    if Gaussian == True:
                        res = scipyminimize(EMGchisqr2, x0, args=(x, data, DtWeights), method='trust-constr', \
                                            options={'verbose': 0 ,'maxiter':50000}, \
                                            bounds=bounds)  # constraints=[lconstr, nlconstr],
                        Hess = nd.Hessian(EMGchisqr2)(res.x, x, data)

                        pars = res.x

                        chisqr = EMGchisqr2(pars, x, data)
                    else:

                        res = scipyminimize(EMGchisqr1, x0, args=(x, data, DtWeights), method='trust-constr', \
                                            constraints=[nlconstr],options={'verbose': 0,'maxiter':50000}, \
                                            bounds=bounds)  # constraints=[nlconstr],
                        Hess = nd.Hessian(EMGchisqr1)(res.x, x, data,DtWeights)

                        pars = res.x

                        chisqr = EMGchisqr1(pars, x, data, DtWeights)

                except:
                    continue

                try:
                    invHess=np.linalg.inv(Hess)
                except:
                    invHess = np.linalg.pinv(Hess)
                if np.any(np.isnan(invHess)==True):
                    invHess=np.linalg.pinv(Hess)
                    if np.any(np.isnan(invHess)==True):
                        continue
                sigmas = np.sqrt(invHess * chisqr / (data.size - pars.size))


                params = Parameters()
                for ikn in np.arange(len(knindsg)):
                    params.add(parnames[knindsg[ikn]], value=knvalsg[ikn], brute_step=0.)
                for iukn in np.arange(len(uknindsg)):
                    params.add(parnames[uknindsg[iukn]], value=pars[iukn], brute_step=sigmas[iukn, iukn])


                if (minchisquare < 0) | (chisqr < minchisquare):
                    SolutionUpdated=True

                    EMGout = params
                    minchisquare = chisqr
                    # update initial guess
                    means = np.array([EMGout['a'].value, EMGout['x0'].value, EMGout['sigma'].value])


                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    FoundSolution=True
                    ntry=maxtry
                    break

            else:

                print ("solver wrong! specify one of these: pyipm, trust-constr, lmfit !")
                return [False, False]

        if (SolutionUpdated == False) & (SampleDoubled == False):

            nSample = np.int(nSample * 2)
            SampleDoubled = True
            continue

        if (SolutionUpdated == True) & (SampleDoubled == True):
            nSample = np.int(nSample / 2)
            SampleDoubled = False
            continue

        if (SolutionUpdated == False) & (SampleDoubled == True):
            nSample = np.int(nSample / 2)
            ntry = maxtry
            SampleDoubled = False
            continue

    if minchisquare<0:
        return [False,False]
    else:

        return [EMGout,True]


#Notes: we need to rewrite the fitting code to exactly fit the parameters that we need....
def EMAFit(x,data,nSample,**kwargs):

    maxtry = 100
    if "maxtry" in kwargs:
        maxtry = kwargs["maxtry"]
    # initial guess of each parameter
    nW = len(x)
    ini_a = np.sum((data[1:] + data[0:nW - 1]) / 2. * np.absolute(x[1:] - x[0:nW - 1]))
    maxxW = x[np.nanargmax(data)]

    # if maxxW <= x[0]:
    #     return [False,False]

    # Weights of each point
    DtWeights = data - data + 1
    # DtWeights[data > np.nanpercentile(data, 75.)] = 4.
    # DtWeights[data < np.nanpercentile(data, 25.)] = 4.

    HM = (np.max(data) - np.min(data)) / 2.+np.min(data)
    FWHM = np.absolute(x[np.nanargmin(np.absolute(data - HM))] - maxxW)*2./2.355
    minchisquare=-1
    #resx = np.absolute(x[1] - x[0])

    means = np.array([ini_a/2./50.,ini_a/2./10.,50., 10.,FWHM,FWHM/5.])


    parnames=['a','c','xa','xc','xsca','xscc','sigmaa','sigmac','b']

    if 'fixparam' in kwargs:
        fixpars=kwargs['fixparam']
        fixvals=kwargs['fixvalue']

    if 'consparam' in kwargs:
        conspars=kwargs['consparam']
        consvalsl=kwargs['consvaluel']
        consvalsu = kwargs['consvalueu']

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

    FoundSolution=False
    ntry=0
    SampleDoubled = False

    while (FoundSolution==False) & (ntry<maxtry):

        ntry=ntry+1
        SolutionUpdated=False

        design = lhs(6, samples=nSample)

        stds = 0.99*means

        for i in np.arange(6):
            design[:, i] = norm(loc=means[i], scale=stds[i]).ppf(design[:, i])

        #solution for one sets of initial guess
        for jfit in np.arange(nSample + 1):
            if jfit < nSample:
                fitpars = design[jfit, :]
            else:
                fitpars = means
            if np.min(fitpars) < 0:
                continue

            if solver == 'lmfit':

                params = Parameters()
                params.add('a', value=fitpars[0], min=0.)  # x-direction integration of total burden
                params.add('xa', value=fitpars[2], min=1. / 3)
                params.add('xc', value=fitpars[3], min=1. / 3)
                # params.add('fc', value=inifc, min=0., max=1.)
                params.add('c', value=fitpars[1], min=0.)

                params.add('xsca', value=maxxW, min=np.min(x), max=np.max(x))
                params.add('xscc', value=maxxW, min=np.min(x), max=np.max(x))

                params.add('sigmaa', value=fitpars[4], min=0.5 / samplewd,
                           max=np.max([maxxW - np.min(x), np.max(x) - maxxW]))  # standard deviation
                params.add('sigmac', value=fitpars[5], min=0.5 / samplewd,
                           max=np.max([maxxW - np.min(x), np.max(x) - maxxW]))  # standard deviation
                params.add('b', value=np.min(data), min=0.,
                           max=np.max(data))  # background concentration

                out = minimize(residualEMA, params, args=(x, data), iter_cb=EMACall)
                chisqr = np.sum(residualEMA(out.params, x, data, DtWeights) ** 2)

                if (minchisquare < 0) | (chisqr < minchisquare):
                    EMAout = out.params
                    minchisquare = chisqr

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    break

            elif solver == 'trust-constr':  # Do the scipy.optimize fit using "trust_constr" method

                global ukninds, kninds, knvals

                ukninds = np.arange(9).astype(int)
                kninds = np.array([]).astype(int)
                knvals = []

                x0 = np.zeros(9)
                x0[0:4] = fitpars[0:4]
                x0[6:8] = fitpars[4:6]
                x0[8] = np.min(data)
                x0[4] = maxxW  # , min = x[0], max = x[-1])
                x0[5] = maxxW  # , min = x[0], max = x[-1])

                lowbs = np.array([0., 0., 0., 0., np.min(x), np.min(x), 0., 0., 0.])
                highbs = np.array([np.inf, np.inf, np.inf, np.inf, np.max(x), np.max(x), np.inf, np.inf, np.max(data)])  #np.max([maxxW - np.min(x), np.max(x) - maxxW])

                if 'consparam' in kwargs:
                    for ipar in np.arange(len(consvalsl)):
                        lowbs[conspars[ipar]] = consvalsl[ipar]
                        highbs[conspars[ipar]] = consvalsu[ipar]
                        x0[conspars[ipar]] = np.mean([consvalsl[ipar], consvalsu[ipar]])

                # test if solutions are similar for different "fixsource" handling
                if PDom == True:
                    kninds = np.append(kninds, np.array([1, 3, 5, 7]).astype(int))
                    knvals = np.append(knvals, [0., 0., 0., 0.])

                    delinds = ~np.isin(ukninds, [1, 3, 5, 7])
                    ukninds = ukninds[delinds]
                    x0 = x0[delinds]

                if SDom == True:
                    kninds = np.append(kninds, np.array([0, 4, 6]).astype(int))
                    knvals = np.append(knvals, [0., 0., 0.])

                    delinds = ~np.isin(ukninds, [0, 4, 6])
                    ukninds = ukninds[delinds]
                    x0 = x0[delinds]

                if 'fixparam' in kwargs:
                    for ipar in np.arange(len(fixvals)):
                        if ~np.isin(fixpars[ipar], ukninds):
                            continue
                        kninds = np.append(kninds, np.int(fixpars[ipar]))
                        knvals = np.append(knvals, fixvals[ipar])
                        delinds = ~np.isin(ukninds, fixpars[ipar])
                        ukninds = ukninds[delinds]
                        x0 = x0[delinds]

                # process bounds to be consistent with x0 and ukninds...
                bounds = Bounds(lowbs[ukninds], highbs[ukninds],keep_feasible=True)
                lconstmat = [[]]
                lconstmin = []
                lconstmax = []

                if SameSource == True:

                    if ~np.isin(4, ukninds) & ~np.isin(5, ukninds):
                        lconstmat = np.append(lconstmat, [np.zeros(len(ukninds))], axis=0)

                        lconstmat[:, ukninds == 4] = 1
                        lconstmat[:, ukninds == 5] = 1
                        lconstmin = np.append(lconstmin, 0.)
                        lconstmax = np.append(lconstmax, 0.)

                if SameSigma == True:
                    if ~np.isin(6, ukninds) & ~np.isin(7, ukninds):
                        lconstmat = np.append(lconstmat, [np.zeros(len(ukninds))], axis=0)
                        lconstmat[:, ukninds == 6] = 1
                        lconstmat[:, ukninds == 7] = 1
                        lconstmin = np.append(lconstmin, 0.)
                        lconstmax = np.append(lconstmax, 0.)

                nlconstr = NonlinearConstraint(EMANLConstr1, -np.inf, 0.)

                if (SameSource == True) | (SameSigma == True):
                    lconstr = LinearConstraint(lconstmat, lconstmin, lconstmax)
                    constraints = [lconstr, nlconstr]
                else:
                    constraints = nlconstr

                try:
                    res = scipyminimize(EMAchisqr1, x0, args=(x, data, DtWeights), method='trust-constr', \
                                        constraints=constraints,options={'verbose': 0,'maxiter':50000}, bounds=bounds)  #constraints=constraints,
                except:
                    continue



                pars = res.x
                Hess = nd.Hessian(EMAchisqr1)(pars, x, data,DtWeights)
                # args = {'x0': x[0]}

                # uncertanties from fitting, following Laughner et al. (2016)
                chisqr = EMAchisqr1(pars, x, data,DtWeights)

                try:
                    invHess=np.linalg.inv(Hess)
                except:
                    invHess = np.linalg.pinv(Hess)
                if np.any(np.isnan(invHess)==True):
                    invHess=np.linalg.pinv(Hess)
                    if np.any(np.isnan(invHess)==True):
                        continue
                sigmas = np.sqrt(invHess * chisqr / (data.size - pars.size))

                params = Parameters()

                for ikn in np.arange(len(kninds)):
                    params.add(parnames[kninds[ikn]], value=knvals[ikn], brute_step=0.)
                for iukn in np.arange(len(ukninds)):
                    params.add(parnames[ukninds[iukn]], value=pars[iukn], brute_step=sigmas[iukn, iukn])

                if (minchisquare < 0 ) | (chisqr < minchisquare):
                    SolutionUpdated=True

                    EMAout = params
                    minchisquare = chisqr
                    # update initial guess
                    means = np.array([EMAout['a'].value, EMAout['c'].value, EMAout['xa'].value, EMAout['xc'].value,\
                                      EMAout['sigmaa'].value, EMAout['sigmac'].value])


                if minchisquare<0:
                    continue

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    FoundSolution = True
                    ntry = maxtry
                    break
            else:
                print ("solver wrong! specify one of these: pyipm, trust-constr, lmfit !")
                return [False, False]


        if (SolutionUpdated == False) & (SampleDoubled == False):
            nSample = np.int(nSample * 2)
            SampleDoubled = True
            continue

        if (SolutionUpdated == True) & (SampleDoubled == True):
            nSample = np.int(nSample / 2)
            SampleDoubled = False
            continue

        if (SolutionUpdated == False) & (SampleDoubled == True):
            nSample = np.int(nSample / 2)
            ntry = maxtry
            SampleDoubled = False
            continue
    if minchisquare<0:
        return [False, False]
    else:
        return [EMAout, True]




#Exponentially modified gaussian function.
#x,xsc,x0 should have the same units.
#xsc=u+x0
def EMG (x,a,x0,xsc,sigma,b,**kwargs):

    sqrt2=np.sqrt(2.)
    if 'Gaussian' in kwargs:
        return a/np.sqrt(2*np.pi)/sigma*np.exp(-1*(x**2)/2/(sigma**2))+b
    # else:
    #     efct=sigma**2/(x-xsc)/x0
    #     ld=a / (1 + efct) * np.exp(-1 * (x - xsc) / x0 * (1 - 0.5 * efct)) + b
    #     ld[x<=xsc]=b
    #
    #     return ld

    else:
        expterm=np.exp((xsc - x) / x0 + (sigma ** 2) / 2. / (x0 ** 2))
        expterm[np.isinf(expterm)==True]=0.  #since erfc function in large exp term is very small

        return 0.5 * a / x0 * expterm * serfc((xsc - x) /sqrt2 / sigma + sigma /sqrt2 / x0) + b

#Not fix source
def EMGchisqr1(x,*args):

    return np.sum((EMGres1(x,*(args[0:2]))**2)*args[2])

def EMGchisqr3(x,*args):

    return np.sum((EMGres1(x,*args,avgw=True))**2)

def EMGchisqr2(x, *args):

    return np.sum((EMGres1(x, *args, Gaussian=True)) ** 2)


#fix source

def EMAchisqr1(x,*args):

    return np.sum((EMAres1(x,*(args[0:2]))**2)*args[2])

def EMGres1(x,*args,**kwargs):

    unknowns = uknindsg
    knowns = knindsg
    # value of known parameters
    knownkeys = knvalsg

    constrx = np.zeros(5)

    constrx[knowns] = knownkeys
    constrx[unknowns] = x

    if 'Gaussian' in kwargs:
        return EMG(args[0], *constrx, Gaussian=True) - args[1]
    if 'avgw' in kwargs:
        return EMG(args[0], constrx, avgw=args[2]) - args[1]
    else:

        return EMG(args[0],*constrx)-args[1]


def EMAres1(x,*args):

    #index of parameters and keys
    unknowns = ukninds
    knowns=kninds
    #value of known parameters
    knownkeys=knvals

    constrx=np.zeros(9)

    constrx[knowns]=knownkeys
    constrx[unknowns]=x

    return EMA(args[0],*constrx)-args[1]


def EMANLConstr1(x,*args):
    unknowns = ukninds
    knowns = kninds
    # value of known parameters
    knownkeys = knvals

    constrx = np.zeros(9)

    constrx[knowns] = knownkeys
    constrx[unknowns] = x

    return [np.exp(constrx[4]/constrx[2]+constrx[6] ** 2 / (2. * (constrx[2] ** 2)))-100.,     #(x[4] - args['x0']) / x[2] +
            np.exp(constrx[5]/constrx[3]+constrx[7] ** 2 / (2. * (constrx[3] ** 2)))-100.,     #(x[5] - args['x0']) / x[3] +
            np.exp(constrx[5]/constrx[2]+constrx[7] ** 2 / (2. * (constrx[2] ** 2)))-100.]       #(x[5] - args['x0']) / x[2] +

def EMGNLConstr1(x):

    unknowns = uknindsg
    knowns = knindsg
    # value of known parameters
    knownkeys = knvalsg

    constrx = np.zeros(5)

    constrx[knowns] = knownkeys
    constrx[unknowns] = x

    return np.exp(constrx[2]/constrx[1]+constrx[3] ** 2 / (2. * (constrx[1] ** 2)))


def EMG2DFit(x,y,data,x01h,samplewd,nSample,**kwargs):


    #data is 2-D distribution of columns (e.g. AOD, SO2 mol/m2)
    # initial guess of each parameter
    #x,y always in the grid resolution (dxy==1)
    #7 parameters to fit, similar in two different parameterizations
    #samplewd: resolution, i.e. how many km represented by 1 grid

    npar=7  #(a,x0,ux,uy,sigma,eta,b)

    GauDecay = False
    if 'GauDecay' in kwargs:
        if kwargs['GauDecay'] == True:
            GauDecay = True

    if 'fixparam' in kwargs:
        fixpars=kwargs['fixparam']
        fixvals=kwargs['fixvalue']

    # if GauDecay == False:
    #     eta=kwargs['eta']

    if 'solver' in kwargs:
        solver=kwargs['solver']
    else:
        solver='trust-constr'

    #total emission (kg)
    ini_a = np.sum(data)  #dxy==1
    #location of maxima
    maxxW = x[np.nanargmax(data)]
    maxyW = y[np.nanargmax(data)]


    if np.array(maxxW).size>1:
        maxxW=maxxW[0]
        maxyW = maxyW[0]
    #take the across-wind data at maxima to estimate sigma

    ydata=data[x==maxxW]
    peaky=y[x==maxxW]
    HM = (np.max(ydata) - np.min(ydata)) / 2.+np.min(ydata)
    FWHM = np.absolute(peaky[np.nanargmin(np.absolute(ydata - HM))] - maxyW)*2./2.355

    minchisquare = -1

    #latin hypercube construction of initials
    design=lhs(3,samples=nSample)
    if GauDecay==False:
        means=np.array([ini_a,x01h,FWHM])
    else:
        means=np.array([ini_a/x01h,x01h,FWHM])

    stds=nSample*means

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
            if solver == 'trust-constr':  # Do the scipy.optimize fit using "trust_constr" method

                # initials and bounds
                x0 = np.zeros(npar)  # (a,x0,ux,uy,sigma,eta,b)
                x0[0:2] = fitpars[0:2]
                x0[2] = maxxW
                x0[3] = maxyW
                x0[4] = fitpars[2]
                x0[5] = np.min(data)
                x0[6] = 1.5*samplewd

                bounds = Bounds([0., 1. / 3, np.min(x), np.min(y), 0.5, 0.,0.], \
                                [np.inf, np.inf, np.max(x), np.max(y), np.inf, np.max(data),np.inf])

                if 'fixparam' in kwargs:
                    for ipar in np.arange(len(fixvals)):
                        bounds.lb[fixpars[ipar]] = fixvals[ipar]
                        bounds.ub[fixpars[ipar]] = fixvals[ipar]
                        x0[fixpars[ipar]] = fixvals[ipar]

                if GauDecay == False:
                    # EMG Fitting

                    nlconstr = NonlinearConstraint(EMG2DConstr, -np.inf, 20.)
                    res = scipyminimize(EMG2Dchisqr, x0, args=(x, y, data), method='trust-constr', \
                                        constraints=[nlconstr], options={'verbose': 0,'maxiter':10000}, \
                                        bounds=bounds)  # constraints=[lconstr, nlconstr],
                    pars = res.x
                    Hess = nd.Hessian(EMG2Dchisqr)(pars, x, y, data)

                    chisqr = EMG2Dchisqr(pars, x, y, data)

                elif GauDecay == True:
                    # 2-D gaussian decay fitting
                    res = scipyminimize(GauDecay2Dchisqr, x0, args=(x, y, data), method='trust-constr', \
                                        options={'verbose': 0,'maxiter':10000}, bounds=bounds)  # constraints=[lconstr, nlconstr],
                    pars = res.x
                    Hess = nd.Hessian(GauDecay2Dchisqr)(pars, x, y, data)

                    chisqr = GauDecay2Dchisqr(pars, x, y, data)
                else:
                    print ("Something went wrong, the type of fitting is not determined!")
                    return [False, False]

                sigmas = np.sqrt(np.linalg.inv(Hess) * chisqr / (data.size - pars.size))

                params = Parameters()
                params.add('x0', value=pars[1], brute_step=sigmas[1, 1])
                params.add('ux', value=pars[2], brute_step=sigmas[2, 2])
                params.add('uy', value=pars[3], brute_step=sigmas[3, 3])
                params.add('sigma', value=pars[4], brute_step=sigmas[4, 4])
                if GauDecay == True:
                    params.add('Qu', value=pars[0], brute_step=sigmas[0, 0])  # x-direction integration of total burden
                else:
                    params.add('a', value=pars[0], brute_step=sigmas[0, 0])  # x-direction integration of total burden
                    params.add('eta', value=pars[6], brute_step=sigmas[6, 6])  # x-direction integration of total burden
                params.add('b', value=pars[5], brute_step=sigmas[5, 5])

                if (minchisquare <0) | (chisqr < minchisquare):
                    EMGout = params
                    minchisquare = chisqr

                if np.sqrt(minchisquare) / np.mean(data) < 0.01:
                    break

            else:

                print ("solver wrong! specify one of these: pyipm, trust-constr, lmfit !")
                return [False, False]

        except:
            print ("Unsuccessful Fitting!")
            continue

    if minchisquare <0:
        return [False,False]
    else:

        return [EMGout,True]

def EMG2DConstr(x):
    return np.exp(x[2] / x[1] + x[4] ** 2 / (2. * (x[1] ** 2)))


def EMG2Dchisqr(x,*args):

    return np.sum((EMG2D(args[0],args[1],*x)-args[2])**2)

def GauDecay2Dchisqr(x,*args):

    return np.sum((GauDecay2D(args[0],args[1],*x)-args[2])**2)

#2-D distribution of column (#/m2 kg/m2, mol/mo2..) for a point source with 1st-order decay
#a emission within 1 tau
#sigma is the width close to the source point
def EMG2D(x,y,a,x0,ux,uy,sigma,b,eta):

    xp=x-ux
    yp=y-uy
    sqrt2=np.sqrt(2.)

    sigmay=np.zeros(x.shape)+sigma
    sigmay=np.sqrt(sigma**2+np.absolute(xp)*eta)

    yfactor=np.exp(-0.5*(yp**2)/(sigmay**2))/sigmay
    return a/2/np.sqrt(np.pi)/sqrt2/x0*np.exp(sigma**2/2./(x0**2)-xp/x0)*serfc((sigma/x0-xp/sigma)/sqrt2)*yfactor+b

#qu=Q/u kg/m (a/x0) emission within 1 lifetime...
def GauDecay2D(x,y,qu,x0,ux,uy,sigma,b):

    xp = x - ux
    yp = y - uy
    efct=sigma**2/xp/x0

    xfactor=np.exp(-1*xp/x0*(1.-0.5*efct))
    yfactor=np.exp(-0.5*(1+efct)*((yp/sigma)**2))

    return qu/np.sqrt(2.*np.pi)/sigma/np.sqrt(1+efct)*xfactor*yfactor+b


def EMA (x,a,c,xa,xc,xsca,xscc,sigmaa,sigmac,b):
    #c=f*c
    #assume that xsca=xscc and sigmaa=sigmac

    sqrt2=np.sqrt(2.)
    #print xa,xc,a,c,scdtm,fc
    #print (xsc - x[0]) / xa, (sigmaa ** 2) / 2. / (xa ** 2), (xsc - x[0]) / xc, (sigmac ** 2) / 2. / (xc ** 2)


    aexpterm=np.exp((xsca - x) / xa + (sigmaa ** 2) / 2. / (xa ** 2))
    aexpterm[np.isinf(aexpterm)==True]=0.
    EMA1=0.5 * a * aexpterm \
         * serfc((xsca - x) / sqrt2 / sigmaa + sigmaa / sqrt2 / xa)

    if (np.absolute(a)<1.e-20)|(np.absolute(sigmaa)<1.e-20):
        EMA1[:]=0.

    # in the case of gamma distribution
    if np.absolute(xa - xc) / xa < 1.e-10:
        tsterm = (x-xscc) / (sigmac ** 2) - 1. / xc
        erfterm = serf((sigmac ** 2 - xc * (x - xscc)) / sqrt2 / sigmac / xc)
        gausterm = np.exp(-1 * ((x - xscc) ** 2) / 2. / (sigmac ** 2))
        convterm=sigmac ** 2 / 2. * tsterm * np.exp(((sigmac *tsterm) ** 2) / 2.)*(1.-erfterm)+sigmac/sqrt2/np.sqrt(np.pi)

        outEMA=EMA1 + convterm * gausterm * c / xc + b
        outEMA[np.isnan(outEMA) | np.isinf(outEMA)]=EMA1[np.isnan(outEMA) | np.isinf(outEMA)]+b
        return outEMA

    scdtm = c * xa / (xa - xc)
    bexpterm=np.exp((xscc - x) / xa + (sigmac ** 2) / 2. / (xa ** 2))
    bexpterm[np.isinf(bexpterm)==True]=0.
    EMA2 = 0.5 * scdtm *bexpterm  \
           * serfc((xscc - x) / sqrt2 / sigmac + sigmac / sqrt2 / xa)
    cexpterm=np.exp((xscc - x) / xc + (sigmac ** 2) / 2. / (xc ** 2))
    cexpterm[np.isinf(cexpterm) == True] = 0.
    EMA3 = -0.5 * scdtm *cexpterm  \
           * serfc((xscc - x) / sqrt2 / sigmac + sigmac / sqrt2 / xc)
    if (np.absolute(scdtm)<1.e-20) | (np.absolute(sigmac)<1.e-20):
        EMA2[:]=0.
        EMA3[:]=0.

    return EMA1+EMA2+EMA3+b


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







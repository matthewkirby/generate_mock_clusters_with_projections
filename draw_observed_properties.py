import numpy as np
from time import time
import os
from scipy.interpolate import InterpolatedUnivariateSpline, interp2d
from scipy.optimize import curve_fit
import scipy.special as spc
from scipy.integrate import quad
import cPickle
import matplotlib.pyplot as plt


# Fitting functions
def sigmoid(x, a):
    return 1./(1.+np.exp(-a*x))
def flipped_sigmoid(x, a):
    return 1. - sigmoid(x, a)
def power_law(x, lna, b):
    return lna+b*x
def linear(x, m, b):
    return x*m+b


class Converter(object):
    def __init__(self):
        self.is_setup = False
        self.icdf_is_setup = False
        # Make sure that these names are in the same order as rows in the input
        self.names = ['tauprj', 'muprj', 'sigprj', 'fmask', 'fprj']


    def setup_2dinterp(self, datablock, ltbins, zbins, param):
        """Given a projection model parameter, for lamtrue < 160, extrapolate
        to lamtrue = 300 over the redshift grid [0.1, 0.3] and return a
        2d interpolation over lamtrue and z
        
        Parameters
        ----------
        datablock : np.ndarray
            Projection model data block
        ltbins : np.ndarray
            Lam_true for the projection model grid
        zbins : np.ndarray
            Redshift for the projection model grid
        param : dict
            Name of the parameter and index from file

        Returns
        -------
        2D linear spline of the parameter over lamtrue in [1,300] and
        z in [0.1, 0.3]
        """
        pfit = np.reshape(datablock[param['idx'],:], (len(zbins), len(ltbins)))
        lcut = 100
        ltbins_extra = np.linspace(lcut, 300, 300-lcut+1)
        interp = np.zeros((len(zbins), len(ltbins_extra)+len(ltbins[ltbins<lcut])))

        # Extrapolate model params to lamtrue
        for i in range(len(zbins)):
            if param['name'] == 'fprj':
                interp[i,:] = np.hstack((pfit[i,:][ltbins<lcut], sigmoid(ltbins_extra, curve_fit(sigmoid, ltbins[ltbins<lcut], pfit[i,:][ltbins<lcut])[0])))
            elif param['name'] == 'fmask':
                interp[i,:] = np.hstack((pfit[i,:][ltbins<lcut], flipped_sigmoid(ltbins_extra, curve_fit(flipped_sigmoid, ltbins[ltbins<lcut], pfit[i,:][ltbins<lcut])[0])))
            elif param['name'] == 'tauprj':
                fit = curve_fit(power_law, np.log(ltbins[ltbins<lcut]), np.log(pfit[i,:][ltbins<lcut]))[0]
                interp[i,:] = np.hstack((pfit[i,:][ltbins<lcut], np.exp(power_law(np.log(ltbins_extra), fit[0], fit[1]))))
            else:
                p = np.polyfit(ltbins[ltbins<lcut], pfit[i,:][ltbins<lcut], 1)
                interp[i,:] = np.hstack((pfit[i,:][ltbins<lcut], linear(ltbins_extra, p[0], p[1])))

        # Make sure the two fraction params are between [0,1]
        if param['name'] in ['fmask', 'fprj']:
            interp[interp<0.] = 0.
            interp[interp>1.] = 1.

        # Return the 2d spline
        return interp2d(np.hstack((ltbins[ltbins<lcut], ltbins_extra)), zbins, interp, kind='linear')


    def setup_splines(self):
        """Set up the 2d interpolations for the projection model params"""
        if self.is_setup: return

        zbins = np.linspace(0.10, 0.30, 5)
        # zbins = np.linspace(0.10, 0.80, 15 )####################################
        ltbins = np.array([1.,3.,5.,7.,9.,12.,15.55555534,20.,24.,26.11111069,30.,
                           36.66666412,40.,47.22222137,57.77777863,68.33332825,78.8888855,
                           89.44444275,100.,120.,140.,160.])
        shape1 = (len(zbins), len(ltbins))

        # Load the model, extrapolate, and make 2d interpolation
        # fname = 'projection_model/prj_params_v9_41_lssmock.txt'
        fname = 'projection_model/prj_params_LSSmock_DESY1A_v1.1.txt'
        fit_lssmock = np.loadtxt(os.path.join('grids', fname))
        for name, idx in zip(self.names, range(5)):
            fit2d = self.setup_2dinterp(fit_lssmock, ltbins, zbins, {'name':name, 'idx':idx})
            setattr(self, '{}fit'.format(name), fit2d)

        self.is_setup = True
        return


    def plot_splines(self):
        # Load the data
        # fname = 'projection_model/prj_params_v9_41_lssmock.txt'
        fname = 'projection_model/prj_params_LSSmock_DESY1A_v1.1.txt'
        fit_lssmock = np.loadtxt(os.path.join('grids', fname))
        zbins = np.linspace(0.10, 0.30, 5)
        # zbins = np.linspace(0.10, 0.80, 15)################################################
        ltbins = np.array([1.,3.,5.,7.,9.,12.,15.55555534,20.,24.,26.11111069,30.,
                           36.66666412,40.,47.22222137,57.77777863,68.33332825,78.8888855,
                           89.44444275,100.,120.,140.,160.])
        shape1 = (len(zbins), len(ltbins))

        # Set up the plot
        f, (a0,a1,a2,a3) = plt.subplots(4,1,sharex=True)
        plt.subplots_adjust(hspace=0.001)
        ltruelist = np.arange(1,300.1, 1)
        a3.set_xlabel('lamtrue')
        ztrue = 0.1

        # Fracs
        a0.plot(ltruelist, self.fprjfit(ltruelist, ztrue), '-k')
        a0.plot(ltruelist, self.fmaskfit(ltruelist, ztrue), '-k')
        a0.plot(ltbins, np.reshape(fit_lssmock[3,:], shape1)[0,:], 'rx')
        a0.plot(ltbins, np.reshape(fit_lssmock[4,:], shape1)[0,:], 'rx')
        a0.set_ylabel('fracs')

        # tau
        a1.semilogy(ltruelist, self.tauprjfit(ltruelist, ztrue), '-k')
        a1.semilogy(ltbins, np.reshape(fit_lssmock[0,:], shape1)[0,:], 'rx')
        a1.set_ylabel('tau')

        # mu
        a2.plot(ltruelist, self.muprjfit(ltruelist, ztrue), '-k')
        a2.plot(ltbins, np.reshape(fit_lssmock[1,:], shape1)[0,:], 'rx')
        a2.set_ylabel('mu')

        # sigma
        a3.plot(ltruelist, self.sigprjfit(ltruelist, ztrue), '-k')
        a3.plot(ltbins, np.reshape(fit_lssmock[2,:], shape1)[0,:], 'rx')
        a3.set_ylabel('sigma')
        plt.show()


    def find_model_params(self, ltrue, z):
        """Find the projection model params given ltrue and z"""
        if not self.is_setup: self.setup_splines()
        model = {}
        for name in self.names:
            model[name] = getattr(self, '{}fit'.format(name))(ltrue, z)
        return model


    def p_lambda_obs_true(self, ltrue, lobs, z):
        """Given a lamtrue and z, compute the prob of measuring lamobs
        
        Parameters
        ----------
        ltrue : float
            True richness
        lobs : float
            Observed richness
        z : float
            Redshift

        Returns
        -------
        Probability of measuring lamobs given lamtrue and z
        """
        if not self.is_setup: self.setup_splines()

        model = self.find_model_params(ltrue, z)

        fprj, fmsk, mu = model['fprj'], model['fmask'], model['muprj']
        sigma, tau = model['sigprj'], model['tauprj']

        # Some stuff to make the calculation easier
        sig2 = sigma*sigma
        A = np.exp(0.5*tau*(2.*mu + tau*sig2 - 2.*lobs))
        root2siginv = 1./(np.sqrt(2.)*sigma)

        # The 4 terms in the model
        t1 = (1.-fmsk)*(1.-fprj)*np.exp(-(lobs - mu)*(lobs- mu)/(2*sig2))/np.sqrt(2.*np.pi*sig2)
        t2 = 0.5*((1.-fmsk)*fprj*tau + fmsk*fprj/ltrue)*A*spc.erfc((mu + tau*sig2 - lobs)*root2siginv)
        t3 = (fmsk*0.5/ltrue)*(spc.erfc((mu - lobs - ltrue)*root2siginv)
                                 - spc.erfc((mu - lobs)*root2siginv))
        t4 = (fmsk*fprj*0.5/ltrue)*np.exp(-1.*tau*ltrue)*A*spc.erfc((mu + tau*sig2 - ltrue
                                                                     - lobs)*root2siginv)
        pdf = t1+t2+t3-t4
        pdf[pdf < 0] = 0
        return pdf


    def cdf_lambda_obs_true(self, ltrue, lobs, z):
        """Compute the CDF of P(lamobs|lamtrue)

        Parameters
        ----------
        ltrue : float
            True richness
        lobs : array like
            Observed richness grid
        z : float
            Redshift

        Returns
        -------
        CDF for lamobs given lamtrue, z
        """
        if not self.is_setup: self.setup_splines()
        def integrand(lobs_x):
            return self.p_lambda_obs_true(ltrue, lobs_x, z)

        newparts = np.vectorize(quad)(integrand, lobs[:-1], lobs[1:], limit=50)[0]
        cdf = np.zeros(len(newparts))

        for i in range(1,len(cdf)):
            cdf[i] = cdf[i-1] + newparts[i]

        return cdf


    def compute_iCDF(self, ltrue, z):
        """Given a ltrue and z, compute the inverse CDF"""
        lamobs_grid = np.hstack(( \
                np.linspace(-10, ltrue/2.5, 100), 
                np.linspace(ltrue/2.5, 3.*ltrue, 500)[1:]))
        cdf4interp = self.cdf_lambda_obs_true(ltrue, lamobs_grid, z)
        cdf4interp = np.clip(cdf4interp, 0.0, 1.0)

        # Find rightmost 0 and leftmost 1 and clip
        rev = cdf4interp[::-1]
        i0 = len(rev) - np.argmax(-rev) - 1
        i1 = np.argmax(cdf4interp)
        cdf4interp = cdf4interp[i0:i1+1]
        lamobs_grid = lamobs_grid[i0:i1+1]

        return lamobs_grid, cdf4interp


    def setup_iCDF_grid(self):
        """Decide to setup or load the grid of iCDFs"""
        # Try to load cdfs or create them
        try:
            with open(r"grids/projection_model.pkl", "rb") as input_file:
                output_obj = cPickle.load(input_file)
            print("Loaded pretabulated projection cdfs")
            icdfgrid = output_obj['icdfgrid']
            self.ltlist, self.zlist = output_obj['ltgrid'], output_obj['zgrid']
            self.lt0, self.dlt = self.ltlist[0], self.ltlist[1]-self.ltlist[0]
            self.z0, self.dz = self.zlist[0], self.zlist[1]-self.zlist[0]
        except IOError:
            print("Computing projection cdfs")
            icdfgrid = self.build_iCDF_grid()

        # Interpolate each of the CDFs
        icdfgrid_interp = np.zeros((len(icdfgrid), len(icdfgrid[0]))).tolist()
        for j in range(len(icdfgrid)):
            for i in range(len(icdfgrid[0])):
                cdf = icdfgrid[j][i]['cdf']
                lobsgrid = icdfgrid[j][i]['lobs']
                icdfgrid_interp[j][i] = {
                        'icdf': InterpolatedUnivariateSpline(cdf, lobsgrid, k=1),
                        'rmin': np.min(cdf), 'rmax': np.max(cdf)}

        self.icdfgrid = icdfgrid_interp
        self.icdf_is_setup = True
        return


    def build_iCDF_grid(self):
        """Set up a grid of inverse CDFs. (very slow!)"""
        ltgrid = np.linspace(1, 300, 1+(300-1)/2)
        self.lt0, self.dlt = ltgrid[0], ltgrid[1]-ltgrid[0]
        # zgrid = np.linspace(0.1, 0.3, 9) # 9 / 3
        zgrid = np.linspace(0.1, 0.8, 29) # 9 / 3
        self.z0, self.dz = zgrid[0], zgrid[1]-zgrid[0]
        self.ltlist, self.zlist = ltgrid, zgrid

        # Make the empty grid
        icdfgrid = np.zeros((len(zgrid), len(ltgrid))).tolist()
        icdfsave = []

        # Loop over the grid, filling in the inverse cdfs
        for j in range(len(zgrid)):
            for i in range(len(ltgrid)):
                print ltgrid[i], zgrid[j]
                lobsgrid, cdf = self.compute_iCDF(ltgrid[i], zgrid[j])
                icdfgrid[j][i] = {'lobs':lobsgrid, 'cdf':cdf, 'z':zgrid[j], 'lt':ltgrid[i]}

        # Save the grid to a file for later
        output_obj = {'icdfgrid': icdfgrid,
                      'ltgrid': ltgrid, 'zgrid': zgrid}
        with open(r"grids/projection_model.pkl", "wb") as output_file:
            cPickle.dump(output_obj, output_file)
        return icdfgrid


    def draw_from_cdf(self, ltrue, z):
        """Now that the iCDFs are set up, draw lamobs given a 
        list of lamtrues and redshifts
        """
        if not self.icdf_is_setup: self.setup_iCDF_grid()

        idx = np.floor((ltrue-self.lt0)/self.dlt).astype(int)
        idy = np.floor((z-self.z0)/self.dz).astype(int)

        # 2 grid points around ltrue and z
        lobsl0z0 = np.zeros(len(ltrue)).tolist()
        lobsl0z1 = np.zeros(len(ltrue)).tolist()
        lobsl1z0 = np.zeros(len(ltrue)).tolist()
        lobsl1z1 = np.zeros(len(ltrue)).tolist()

        # For each lamtrue, draw the four lobs
        for i in range(len(ltrue)):
            rmin = max([self.icdfgrid[idy[i]][idx[i]]['rmin'], self.icdfgrid[idy[i]+1][idx[i]]['rmin'], 
                        self.icdfgrid[idy[i]][idx[i]+1]['rmin'], self.icdfgrid[idy[i]+1][idx[i]+1]['rmin']])
            rmax = min([self.icdfgrid[idy[i]][idx[i]]['rmax'], self.icdfgrid[idy[i]+1][idx[i]]['rmax'], 
                        self.icdfgrid[idy[i]][idx[i]+1]['rmax'], self.icdfgrid[idy[i]+1][idx[i]+1]['rmax']])
            r = np.random.uniform(low=rmin, high=rmax)

            lobsl0z0[i] = self.icdfgrid[idy[i]][idx[i]]['icdf'](r)
            lobsl1z0[i] = self.icdfgrid[idy[i]][idx[i]+1]['icdf'](r)
            lobsl0z1[i] = self.icdfgrid[idy[i]+1][idx[i]]['icdf'](r)
            lobsl1z1[i] = self.icdfgrid[idy[i]+1][idx[i]+1]['icdf'](r)

        # Linear interpolation over z 00+01, 10+11
        z1, z2 = self.zlist[idy], self.zlist[idy+1]
        wz = (z-z1)/(z2-z1)
        lobsl0 = lobsl0z0*(1.-wz) + lobsl0z1*wz
        lobsl1 = lobsl1z0*(1.-wz) + lobsl1z1*wz

        # Linear interpolation over ltrue
        lt1, lt2 = self.ltlist[idx], self.ltlist[idx+1]
        wlt = (ltrue-lt1)/(lt2-lt1)
        lobs = lobsl0*(1.-wlt) + lobsl1*wlt

        return lobs



def draw_observed_richness(catalog):
    """ Given a catalog with true richness, draw observed richness
    using the redmapper projection model.

    Parameters
    ----------
    catalog : DataFrame
        Dataframe with a row per cluster

    Returns
    -------
    Dataframe with observed richness included.
    """

    # Make cuts 13<lamtrue<1200
    catalog = catalog[catalog['true_richness'] > 1.0 ]
    catalog = catalog[catalog['true_richness'] < 300.]
    if max(catalog['true_richness']) > 300:
        print("CDFs do not go high enough for lamtrue={}".format(max(catalog['true_richness'])))

    drawer = Converter()
    lobs_list = drawer.draw_from_cdf(catalog['true_richness'].values, catalog['z'].values)
    catalog['obs_richness'] = lobs_list
    catalog['obs_richness_std'] = np.ones(len(catalog))

    return catalog




if __name__ == "__main__":
    conv = Converter()

    ltruelist = np.linspace(1, 299.99)
    zlist = np.linspace(0.10, 0.29999)

    lobslist = conv.draw_from_cdf(ltruelist, zlist)
    print(lobslist)








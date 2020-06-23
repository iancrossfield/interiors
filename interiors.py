"""Perform basic inference on the interior composition of a planet.


Given a mass and radius, interpolate over previously-calculated
interior models for various assumptions about its interior and report
back the results.

E.g., :func:`h2o_model` assumes a planet is made of an Earthlike rocky
interior with a large H2O mass fraction.  Alternatively,
:func:`hhe_model` assumes a planet is made of an Earthlike rocky
interior with the rest of its mass in hydrogen and helium.

*REQUIREMENTS*
--pandas, numpy, scipy
--You must set the correct path to the interior model files ("_modelfile")


    :EXAMPLE1:
      ::
       
         import interiors
         import pylab as py

         # Define your system:
         planetname = 'GJ 1214 b'
         mass, u_mass, radius, u_radius = 6.43, 0.86, 2.27, 0.08

         # Run the code:
         hhe_fractions = interiors.hhe_model(mass, u_mass, radius, u_radius, nsamp=1000)
         h2o_fractions = interiors.h2o_model(mass, u_mass, radius, u_radius, nsamp=1000)

         # Make some pretty plots:
         titlevals = planetname, mass, u_mass, radius, u_radius
         fig=py.figure()
         fig.text(.5, .92, '%s: Mp=%1.2f+/-%1.2f Me, Rp=%1.2f+/-%1.2f Re' % titlevals, horizontalalignment='center', fontsize=16)
         py.subplot(121)
         py.hist((h2o_fractions[np.isfinite(h2o_fractions)]), 50)
         py.xlabel('H2O mass fraction', fontsize=13)
         py.subplot(122)
         py.hist(np.log10(hhe_fractions[np.isfinite(hhe_fractions)]), 50)
         py.xlabel('log10(H/He mass fraction)', fontsize=13)



# 2020-06-16 16:04 IJMC: Commented code and put online.
"""

import pandas as pd
from scipy import interpolate
import numpy as np



#from scipy.interpolate import griddata

_modelfile = 'valencia_2011_planetmodels.csv'

rearth = 6378.136 # km
tab = pd.read_csv(_modelfile) 
ntab = len(tab)
allrad_h2o = np.array([tab[col].values/(rearth) for col in tab[['earth', '3_water', '10_water', '20_water', 'halfwater', 'allwater']]])
allrad_hhe = np.array([tab[col].values/(rearth) for col in tab[['earth', '0.01_hhe', '0.1_hhe', '1_hhe', '10_hhe']]])
allh2o = np.tile(np.array([[8.7e-7*0.02, .03, .1, .2, .5, 1]]).T, (ntab))
allhhe = np.tile(np.array([[8.7e-7*0.5e-6, .0001, .001, .01, .1]]).T, (ntab))
allmass_h2o = np.tile((tab.mearth), (allh2o.shape[0],1))
allmass_hhe = np.tile((tab.mearth), (allhhe.shape[0],1))

valid_h2o = np.isfinite(allrad_h2o*allh2o*allmass_h2o)
valid_hhe = np.isfinite(allrad_hhe*allhhe*allmass_hhe)



def h2o_model(mp=None, ump=None, rp=None, urp=None, nsamp=1000, nmass=1000, nrad=900, verbose=True):
    """Infer the H2O fraction of a planet with an Earth-composition core.

    :INPUTS:
      mp, ump -- mass  of planet (and its uncertainty) in Earth units

      rp, urp -- radius of planet (and its uncertainty) in Earth units

      nsamp -- number of Monte Carlo trials for assessing uncertainties

    :OUTPUTS:
      If no planet parameters are input, returned (mass, radius, H2O
      fraction) where mass and radius are 1D vectors of length nmass
      and nrad, and H2o fraction is a 2D array.

      If no planet *uncertainties* are input, returns a single number
      indicating the H2O bulk mass fraction for the given mass and radius.

      If mp,rp,ump,urp are all given, return 'nsamp' number of samples
      of H2O bulk mass fraction, assuming Gaussian errors on mass and
      radius.
    """
    # 2019-12-12 IJMC: Created.
    # 2020-06-23 15:23 IJMC: Fixed printout units
    # Generate a 2
    #if mp is None or rp is None: # just generate a grid
    gridmass = np.linspace(1, 20, nmass)
    gridrad = np.linspace(1, 4, nrad)
    gmass, grad = np.meshgrid(gridmass, gridrad)
    grid_z0 = interpolate.griddata((allmass_h2o[valid_h2o], allrad_h2o[valid_h2o]), allh2o[valid_h2o], (gmass, grad), method='linear', fill_value=np.nan)

    lo_limit = np.interp(gridmass, tab.mearth, tab['earth']/(rearth))
    hi_limit = np.interp(gridmass, tab.mearth, tab['allwater']/(rearth))
    invalid = np.logical_or(grad < lo_limit, grad > hi_limit)
    grid_z0[invalid] = np.nan
    ret = gridmass, gridrad, grid_z0
    
    if mp is not None and rp is not None:
        nangrid = np.logical_not(np.isfinite(grid_z0))
        grid_z0[nangrid] = -999
        h2o_spline = interpolate.RectBivariateSpline(gridrad, gridmass, grid_z0, kx=1, ky=1, s=0)
        #thisvalue = interpolate.griddata((allmass_h2o[valid_h2o], allrad_h2o[valid_h2o]), allh2o[valid_h2o], (mp, rp), method='linear', fill_value=np.nan)
        if ump is not None and urp is not None:
            rps = np.random.normal(rp, urp, size=int(nsamp))
            mps = np.random.normal(mp, ump, size=int(nsamp))
            h2ofracs = h2o_spline(rps, mps, grid=False)
            h2ofracs[h2ofracs<0] = np.nan
            ret = h2ofracs
            if verbose:
                validfrac = 100.0*np.isfinite(h2ofracs).sum()/h2ofracs.size
                h2ovalues = np.log10(np.nanmedian(h2ofracs)), \
                    np.std(np.log10(h2ofracs[np.isfinite(h2ofracs)]))
                h2ovalues2 = (100*np.nanmedian(h2ofracs)), \
                    100*np.std((h2ofracs[np.isfinite(h2ofracs)]))
                print('%1.3f%% of samples are consistent with a Rock+H/He composition.' % validfrac)
                print('H2O mass fraction is roughly (%1.2f +/- %1.2f) dex' % h2ovalues)
                print('                 or, roughly (%1.2f +/- %1.2f) %%' % h2ovalues2)
        else:
            h2ofrac = h2o_spline(rp, mp)
            h2ofracs[h2ofracs<0] = np.nan
            ret = h2ofrac
            
    return ret


def hhe_model(mp=None, ump=None, rp=None, urp=None, nsamp=1000, nmass=1000, nrad=900, verbose=True):
    """Infer the HHE fraction of a planet with an Earth-composition core.

    :INPUTS:
      mp, rp -- mass and radius of planet in Earth units

      ump, urp -- uncertainties on mass and radius, in Earth units

      nsamp -- number of Monte Carlo trials for assessing uncertainties

    :OUTPUTS:
      If no planet parameters are input, returned (mass, radius, HHE
      fraction) where mass and radius are 1D vectors of length nmass
      and nrad, and Hhe fraction is a 2D array.

      If no planet *uncertainties* are input, returns a single number
      indicating the HHE bulk mass fraction for the given mass and radius.

      If mp,rp,ump,urp are all given, return 'nsamp' number of samples
      of HHE bulk mass fraction, assuming Gaussian errors on mass and
      radius.
    """
    # 2019-12-12 IJMC: Created.
    # 2020-04-13: Fixed hi_limit (was 'allwater')
    
    # Generate a 2
    #if mp is None or rp is None: # just generate a grid
    gridmass = np.linspace(1, 20, nmass)
    gridrad = np.linspace(1, 6.5, nrad)
    gmass, grad = np.meshgrid(gridmass, gridrad)
    grid_z0 = interpolate.griddata((allmass_hhe[valid_hhe], allrad_hhe[valid_hhe]), np.log10(allhhe[valid_hhe]), (gmass, grad), method='linear', fill_value=np.nan)

    lo_limit = np.interp(gridmass, tab.mearth, tab['earth']/(rearth))
    hi_limit = np.interp(gridmass, tab.mearth, tab['10_hhe']/(rearth))
    invalid = np.logical_or(grad < lo_limit, grad > hi_limit)
    grid_z0[invalid] = np.nan
    ret = gridmass, gridrad, 10**grid_z0
    
    if mp is not None and rp is not None:
        nangrid = np.logical_not(np.isfinite(grid_z0))
        grid_z0[nangrid] = -999
        hhe_spline = interpolate.RectBivariateSpline(gridrad, gridmass, grid_z0, kx=1, ky=1, s=0)
        #thisvalue = interpolate.griddata((allmass_hhe[valid_hhe], allrad_hhe[valid_hhe]), np.log10(allhhe[valid_hhe]), (mp, rp), method='linear', fill_value=np.nan)
        if ump is not None and urp is not None:
            rps = np.random.normal(rp, urp, size=int(nsamp))
            mps = np.random.normal(mp, ump, size=int(nsamp))
            hhefracs = 10**hhe_spline(rps, mps, grid=False)
            hhefracs[hhefracs<=0] = np.nan
            ret = hhefracs
            if verbose:
                validfrac = 100.0*np.isfinite(hhefracs).sum()/hhefracs.size
                hhevalues = np.log10(np.nanmedian(hhefracs)), \
                    np.std(np.log10(hhefracs[np.isfinite(hhefracs)]))
                print('%1.3f%% of samples are consistent with a Rock+H/He composition.' % validfrac)
                print('H/He mass fraction is roughly (%1.2f +/- %1.2f) dex' % hhevalues)
        else:
            hhefrac = 10**hhe_spline(rp, mp)
            hhefracs[hhefracs<0] = np.nan
            ret = hhefrac


            
    return ret

    



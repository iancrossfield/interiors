# WARNING:
Apparently the interior (Mass/Radius) models used in this code are somewhat out of date. You may want to instead try out the https://github.com/cpiaulet/smint tool.

# interiors
Perform basic inference on the interior composition of a planet.


If you use this code please cite Ian Crossfield, as well as the interior models on which this code relies: [Valencia, D. 2011, IAUS 276, p.181](https://ui.adsabs.harvard.edu/abs/2011IAUS..276..181V/abstract) and [Lopez & Fortney 2014, ApJ 792](https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/)


Given a mass and radius, interpolate over previously-calculated
interior models for various assumptions about its interior and report
back the results.

E.g., :func:`h2o_model` assumes a planet is made of an Earthlike rocky
interior with a large H2O mass fraction.  Alternatively,
:func:`hhe_model` assumes a planet is made of an Earthlike rocky
interior with the rest of its mass in hydrogen and helium.

*REQUIREMENTS*:

--pandas, numpy, scipy

--You must set the correct path to the interior model files ("_modelfile")


    :EXAMPLE:
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



 2020-06-16 16:04 IJMC: Commented code and put online.
 2020-06-24 17:02 IJMC: Updates: new H/He model from Lopez+2014, other minor fixes.

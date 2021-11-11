import numpy as np
from gce_model import gce
from gce_model import gce_params
from scipy.integrate import simps

mg_index = 4; si_index = 5; ca_index = 6; ti_index = 7; fe_index = 8

#Constant predictions from WMAP
bdm_ratio_wmap = 0.169
bdm_ratio_wmap_err = 0.021

def lnlike(pars, data, massdata=None, bulk_alpha=False, include_ti=False):

    """
    Adapted into Python from IDL, original code written by ENK (Kirby et al. 2011b).
    Calculates the negative log-likelihood function for a given set of GCE model
    parameters, based on the Eq. 18 in Kirby et al. 2008
    The negative log-likelihood function is constructed based on the model output
    abundance trajectories for the abundances, the probability of a star forming at a
    point in time in the model, the observed abundances and associated erros, and the
    observed and model gas and stellar masses of the dwarf galaxy (Sec 3. of K11b)

    Parameters
    ----------
    pars: array-like: input parameters for the GCE model: [A_in/1e6, tau_in, A_out/1e3,
                       A_star/1e6, alpha, M_gas_0]
    data: dict: dictionary containing observational abundance information for the given
                dwarf galaxy
    massdata: array-like: array containing data for the relevant observed galaxy: [mgas, mstar,
                   mstarerr] in units of Msun

    Returns:
    -------
    likelihood: float: negative log-likelihood value for the given GCE model parameters
                       based on the given observational data set
    """

    pars = np.asarray(pars)

    if massdata is not None:
        mgas, mstar, mstarerr = massdata

    #Observational data to use in the likelihood determination
    n_obs = len(data['fe_h'])

    #Based on the GCE model used, check if the number of parameters is correct
    n_pars = len(pars)
    w = np.where(pars < 0.0)[0]
    if len(w) > 0: return np.inf
    if gce_params.name == 'Zwind':
        if n_pars == 7:
            if pars[6] > 1.0: return np.inf
        else:
            if pars[5] > 1.0: return np.inf

    #Run the GCE model for this given set of model parameters
    model,_ = gce.gce_model(pars)
    n_model = len(model['t'])

    #Initialize variables to be used in the likelihood calculation
    norm = np.sqrt(2*np.pi)
    likelihood = 0.

    #Consider only timesteps after 7 Myr in the galaxy's evolution, and check that
    #there are a sufficient number of finite values in the abundances returned by the
    #GCE model
    where_greater = model['t'] > 0.007

    where_finite = np.isfinite(model['eps'][:,mg_index]) &\
                   np.isfinite(model['eps'][:,si_index]) &\
                   np.isfinite(model['eps'][:,ca_index]) &\
                   np.isfinite(model['eps'][:,fe_index])
    if include_ti:
        where_finite = where_finite & np.isfinite(model['eps'][:,ti_index])

    where_good = where_greater & where_finite
    if len(model['t'][where_good]) < 10:
        return np.inf

    #If the model passes the above criteria, then construct the abundance ratios for
    #[Fe/H], [Mg/Fe], [Si/Fe], [Ca/Fe]
    feh = model['eps'][:,fe_index][where_good]
    mgfe = model['eps'][:,mg_index][where_good] - model['eps'][:,fe_index][where_good]
    sife = model['eps'][:,si_index][where_good] - model['eps'][:,fe_index][where_good]
    cafe = model['eps'][:,ca_index][where_good] - model['eps'][:,fe_index][where_good]
    tife = model['eps'][:,ti_index][where_good] - model['eps'][:,ti_index][where_good]

    #Timesteps to consider from the model
    t = model['t'][where_good]

    #Calculate the probability with respect to time that a star forms
    dp_dt = model['mdot'][where_good]/model['mstar'][n_model-1]
    dp_dt = dp_dt[dp_dt > 0.]
    dp_dt /= simps(dp_dt, t) #note that the original IDL uses Netwon-Cotes integration here

    """
    #Do not consider the last 3 Myr of the evolution of the galaxy
    maxt = n_model - 3

    #Calculate the integrated ejecta in each element
    ejecta = np.zeros(len(model['eps'][0]))
    for j in range(len(model['eps'][0])):

        w = np.where((model['mout'][:,j][7:maxt] > 0.)&(np.isfinite(model['mout'][:,j][7:maxt])))[0] + 7
        if len(w) > 50:
            ejecta[j] = simps(model['mout'][:,j][w], model['t'][w])
        else: ejecta[j] = -999.

    #Calculate the total outflow based on the ejecta in each element
    total_outflow = np.sum(ejecta[0:1]) + np.sum(ejecta[3:6])*10**(1.31) + ejecta[7]*10**(0.03)

    #Now calculate the integrated mass influx
    w = np.where((model['f_in'][7:maxt] > 0.)&(np.isfinite(model['f_in'][7:maxt])))[0] + 7
    if len(w) > 50:
        integrated_mass = simps(model['f_in'][w], model['t'][w]) + model['mgas'][0]
    else: integrated_mass = -999
    """

    #Calculate the total mass based on the luminosity and the mass-to-light ratio
    #assumed for the observed dwarf galaxies
    #bdm_ratio = integrated_mass/total_mass

    #Now loop through the observational data dictionary provided in the input
    #parameters, where i is the index for a given red giant star in the galaxy,
    #and determine the likelihood as compared to the model for each
    #instance based on simultaneously using [Fe/H], [Mg/Fe], [Si/Fe], and [Ca/Fe]

    print(massdata, bulk_alpha, include_ti)
    for i in range(n_obs):

        feh_dist = ((feh - data['fe_h'][i])/data['e_fe_h'][i])**2.

        if bulk_alpha:

            if include_ti:
                alphafe = (mgfe + cafe + sife + tife)/4.
            else:
                alphafe = (mgfe + cafe + sife)/3.

            alphafe_dist = ((alphafe - data['alpha_fe'][i])/data['e_alpha_fe'][i])**2.

            dist_arr = [feh_dist, alphafe_dist]
            err_arr = [data['e_fe_h'][i], data['e_alpha_fe'][i]]

        else:

            mgfe_dist = ((mgfe - data['mg_fe'][i])/data['e_mg_fe'][i])**2.
            cafe_dist = ((cafe - data['ca_fe'][i])/data['e_ca_fe'][i])**2.
            sife_dist = ((sife - data['si_fe'][i])/data['e_si_fe'][i])**2.

            dist_arr = [feh_dist, mgfe_dist, sife_dist, cafe_dist]

            err_arr = [data['e_fe_h'][i], data['e_mg_fe'][i], data['e_si_fe'][i],
                       data['e_ca_fe'][i]]

            if include_ti:
                tife_dist = ((tife - data['ti_fe'][i])/data['e_ti_fe'][i])**2.
                dist_arr.append(tife_dist)
                err_arr.append(data['e_ti_fe'][i])

        err_mult = np.nanprod(err_arr)
        dist = np.nansum(dist_arr, axis=0)

        lfunc = np.exp(-0.5*dist) / (err_mult * norm**4.)

        int_i = simps(lfunc*dp_dt, t)
        if (np.isfinite(int_i)) and (int_i >= 0.):
            likelihood -= np.log(int_i)
        else:
            likelihood += 5.

    #Now add terms to the likelihood based on the stellar mass in the GCE model and the
    #observed gas mass as described in Eq. 17 of Kirby et al 2011b
    if massdata is not None:

        #Now calculate the amount of gas remaining in the galaxy at the end of the GCE model
        remaining_gas = model['mgas'][n_model-1]
        if np.abs(model['mgas'][n_model-2] - remaining_gas) > 0.5*remaining_gas:
            remaining_gas = 0.
        if remaining_gas < 0.: remaining_gas = 0.

        z0_mstar_term = 0.5*((model['mstar'][n_model-1] - mstar)/mstarerr)**2.
        delta_mgas = 1.e3
        z0_mgas_term = 0.5*((remaining_gas - mgas)/delta_mgas)**2. #assuming observed gas mass of 0 Msun at z = 0

        likelihood += 0.1*n_obs*(z0_mstar_term + z0_mgas_term + np.log(norm**2. * mstarerr * delta_mgas))

        if (model['mstar'][n_model-1] < delta_mgas): likelihood += 3.e6

    return likelihood

def lnprior(pars):
    """
    Definite the a priori probability distribution for the parameters of the GCE model
    Assume a uniform distribution over a reasonable range of parameter values

    Parameters:
    ----------
    pars: array-like: [A_in/1e6, tau_in, A_out/1e3, A_star/1e6, alpha, M_gas_0]

    Returns:
    --------
    Value of the negative log likelihood for the a priori probability distribution
    """

    A_in, tau_in, A_out, A_star, alpha, M_gas_0 = pars

    #Define the upper and lower limits for the aforementioned reasonable ranges
    ulim_Ain, ulim_tauin, ulim_Aout = 4., 0.5, 13.
    ulim_Astar, ulim_alpha, ulim_Mgas0 = 8., 1.5, 20.
    llim = 0.1

    if  (llim*1.e3 < A_in < ulim_Ain*1.e3) and (llim < tau_in < ulim_tauin) and\
        (llim < A_out < ulim_Aout) and (llim < A_star < ulim_Astar) and\
        (llim < alpha < ulim_alpha) and (llim  < M_gas_0 < ulim_Mgas0):
        return 0.
        #return np.log(1./(ulim_A_in*ulim_tau_in*ulim_A_out*ulim_A_star*ulim_alpha*ulim_M_gas_0))
    else:
        return np.inf

def lnprob(pars, data, massdata=None, include_ti=False, bulk_alpha=False):
#Define the full NEGATIVE log-probability distribution for the GCE model fitted against
#the observational data for minimization

    lp = lnprior(pars)
    if not np.isfinite(lp):
        return np.inf
    else:
        return lp + lnlike(pars, data, massdata=massdata, include_ti=include_ti,
                           bulk_alpha=bulk_alpha)

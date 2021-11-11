"""
This program is a Python version wrriten by I.Escala (Caltech) of the GCE model code
from Kirby et al. 2011b. The original version of the code was written by E.N.K in IDL.
"""
import numpy as np
import scipy
import scipy.integrate

from gce_model import initialize_gce_data
from gce_model import dtd_ia
from gce_model import gce_params
from gce_model import imf

"""
Mass ejected by supernova (courtesy Hai Fu):

Thornton et al (1998) calculated that only 10% of the initial SN
energy (1E51 erg) is released in the form of kinetic energy (8.5E49
erg). From virial theorem, M = 5 r sigma**2/G, so the escape velocity,
v_e = sqrt(2 G M / r) = sqrt(10 sigma**2). So the mass ejected equals
M_eject = E/v_e**2 = ((8.5e49 erg) / ((10 km (s**(-1)))**2)) / 10 = 4274 M_sun
"""

def idl_tabulate(x, f, p=5) :
    def newton_cotes(x, f) :
        if x.shape[0] < 2 :
            return 0.
        rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
        weights = scipy.integrate.newton_cotes(rn)[0]
        return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
    ret = 0
    for idx in xrange(0, x.shape[0], p - 1):
        ret += newton_cotes(x[idx:idx + p], f[idx:idx + p])
    return ret

def interp_func(x,y,xinterp):
    index = np.argsort(x)
    if len(xinterp.shape) !=0:
        indexinterp = np.argsort(xinterp)
        yinterp = np.interp(xinterp[indexinterp],x[index],y[index])[indexinterp]
    else: yinterp = np.interp(xinterp,x[index],y[index])
    return yinterp

def gce_model(pars):  #SFR fixed to gas mass

    # for name = 'sfr-law' and npars=6:
    #pars = [f_in_norm0, f_in_t0, f_out/1e3, sfr_norm, sfr_exp, model[0]['mgas']/1e6]
    #using names used in the paper:
    #pars = [A_in/1e6, tau_in, A_out/1e3, A_star/1e6, alpha, M_gas_0]

    z_II, nel, nom06, kar10, eps_sun, M_SN, M_HN, M_lims, z_lims = initialize_gce_data.initialize()

    name = gce_params.name
    ia_model = gce_params.ia_model
    imf_model = gce_params.imf_model

    npars = len(pars)

    #no gas infall
    if npars == 4:
        ipar = 2
    #with gas infall
    else:
        ipar = 0
        #f_in_norm: Normalization of gas inflow rate (10**6 M_sun Gyr**-1)
        f_in_norm0 = pars[0]
        #f_in_t0: Exponential decay time for gas inflow (Gyr)
        f_in_t0 = pars[1]

    #f_out: Strength of gas outflow due to supernovae (M_sun SN**-1)
    f_out = pars[2-ipar]*1.e3
    sfr_norm = pars[3-ipar]
    sfr_exp = pars[4-ipar]

    #Number of timesteps in the model.
    #For delta_t = 0.001, the model runs for 13.6 Gyr
    n = 13600

    #Define all of the data to be saved for the model for each timestep
    model = np.zeros(n, dtype=[('t','float64'),('f_in','float64'),('Ia_rate','float64'),\
    ('II_rate','float64'),('de_dt','float64',(nel)),('dstar_dt','float64',(nel)),\
    ('abund','float64',(nel)),('eps','float64',(nel)),('mout','float64',(nel)),\
    ('z','float64'),('mdot','float64'),('mgal','float64'),('mstar','float64'),\
    ('mgas','float64')])

    delta_t = 0.001  #time step (Gyr)
    model['t'] = np.arange(n)*delta_t #time passed in model array -- age universe (Gyr)

    #gas inflow rate (M_sun  Gyr**-1), Eq. 14, Weschler et al. 2002
    model['f_in'] = 1.e6*f_in_norm0*model['t']*np.exp(-1.0*model['t']/f_in_t0)

    #constants for Newton-Cotes integration
    int2 = delta_t * np.array([1., 1.]) / 2. #trapezoid rule
    int3 = delta_t * np.array([1., 4., 1.]) / 3. #simpsons rule
    int4 = delta_t * np.array([1., 3., 3., 1.]) * 3. / 8. #simpsons 3/8 rule
    int5 = delta_t * np.array([7., 32., 12., 32., 7.]) * 2. / 45. #boole's rule

    #initial star formation rate (M_sun Gyr**-1)
    #model[0]['mdot'] = 0.0

    #initial gas mass (M_sun) for no gas infall
    if npars == 4: model[0]['mgas'] = 1.e6*pars[3]
    #initial gas mass (M_sun) for gas infall
    if npars == 6: model[0]['mgas'] = 1.e6*pars[5]
    #Otherwise assume no initial gas mass (M_sun)
    else: model[0]['mgas'] = 0.

    #pristine element fractions by mass (dimensionless)
    pristine = np.zeros(nel)
    pristine[0] = 0.7514 #Hydrogen from BBN
    pristine[1] = 0.2486 #Helium from BBN
    #Set the abundances for H and He, where abundance is the gas mass in the element
    model[0]['abund'][0] = model[0]['mgas']*pristine[0]
    model[0]['abund'][1] = model[0]['mgas']*pristine[1]
    #Set the initial galaxy mass to the initial gas mass
    model[0]['mgal'] = model[0]['mgas']

    #hypernovae fraction for stars with M >= 20 M_sun (dimensionless)
    epsilon_HN = 0.0
    wint = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6] #Index for redefinition of SN masses

    #Note that the following are the masses for which we have supernovae models
    #M_SN = [13, 15, 18, 20, 25, 30, 40]
    #redefine SN masses to cover full SN mass range and to turn on hypernovae at M = 20 M_sun
    M_SN_ej = M_SN[wint]
    M_SN_ej[0] = 10.0   #lower mass limit for Type II SN to explode (M_sun)
    M_SN_ej[9] = 100.0  #upper mass limit for Type II SN to explode (M_sun)
    #tiny mass to subtract from 20 M_sun to define an endpoint for SN and a start point for HN (M_sun)
    M_SN_ej[4] -= 0.01
    #M_SN_ej = [10, 13, 15, 18, 19.99, 20, 25, 30, 40, 100]

    #dummy integration array (M_sun), 50 steps btw 10 and 100 for massive stars
    m_int1 = np.logspace(np.log10(10),np.log10(100),50)
    #dummy integration array, corresponding to total stellar lifetime for massive stars
    #in (Gyr), Eq. 6
    t_int1 = 1.2*m_int1**(-1.85) + 0.003

    #dummy integration array (M_sun), 50 steps between 0.865 and 10
    #For low and intermediate mass stars
    #m_int2 = np.logspace(np.log10(0.865),np.log10(10),50) #G.D.
    m_int2 = np.logspace(np.log10(0.865), np.log10(10.), 50) #Up to 8 in Evan's GCE code

    wlo = m_int2 < 6.6   #Define threshold for "low" mass stars
    whi = m_int2 >= 6.6  #Define threshold for intermediate mass stars
    t_int2 = np.zeros(len(m_int2))

    #dummy integration array, corresponding to total stellar lifetime for
    #low- and intermediate-mass stars (Gyr)
    t_int2[whi] = 1.2*m_int2[whi]**(-1.85) + 0.003  #same as for the massive stars, Eq. 6
    #Define the lower mass limits for contributions from certain types of ejecta
    M_lims_ej = np.concatenate(([0.865],M_lims,[10.0])) #SNeIa, AGB, SNe II
    # Eq. 12 for "low" mass stars
    t_int2[wlo] = 10**((0.334-np.sqrt(1.790-0.2232*(7.764-np.log10(m_int2[wlo]))))/0.1116)

    #Define mass-dependent lifetime for AGB stars
    t_agb = 10**((0.334-np.sqrt(1.790-0.2232*(7.764-np.log10(M_lims))))/0.1116)

    #Define the arrays for the ejected gas mass as a function of SN/HN/AGB
    #progenitor mass
    x_sn = np.zeros(len(M_SN))
    x_hn = np.zeros(len(M_HN))
    x_agb = np.zeros(len(M_lims)+2)

    #Define arrays for the mass of each tracked element contributed by each source
    M_II_arr = np.zeros(nel)
    M_agb_arr = np.zeros(nel)
    M_Ia_arr = np.zeros(nel)

    #set atomic indices and check that they are found properly
    #kar10['atomic'] =  1       2       6       8      12      14      26
    #nom06['atomic'] =  1       2       6       8      12      14      20      22      26

    #Set the index for each tracked element that has contributions
    #to its yield from SNe II
    h_sn_index = np.where(nom06['atomic'] == 1)[0]
    he_sn_index = np.where(nom06['atomic'] == 2)[0]
    c_sn_index = np.where(nom06['atomic'] == 6)[0]
    o_sn_index = np.where(nom06['atomic'] == 8)[0]
    mg_sn_index = np.where(nom06['atomic'] == 12)[0]
    si_sn_index = np.where(nom06['atomic'] == 14)[0]
    ca_sn_index = np.where(nom06['atomic'] == 20)[0]
    ti_sn_index = np.where(nom06['atomic'] == 22)[0]
    fe_sn_index = np.where(nom06['atomic'] == 26)[0]

    if ((len(h_sn_index)==0) or (len(he_sn_index)==0) or (len(c_sn_index)==0) or
    (len(o_sn_index)==0) or (len(mg_sn_index)==0) or (len(si_sn_index)==0) or
    (len(ca_sn_index)==0) or (len(ti_sn_index)==0) or (len(fe_sn_index)==0)):
        print("Error with nom06['atomic'] values")

    #Similarly define indices for tracked elements with contribution from AGB winds
    h_agb_index = np.where(kar10['atomic'] == 1)[0]
    he_agb_index = np.where(kar10['atomic'] == 2)[0]
    c_agb_index = np.where(kar10['atomic'] == 6)[0]
    o_agb_index = np.where(kar10['atomic'] == 8)[0]
    mg_agb_index = np.where(kar10['atomic'] == 12)[0]
    si_agb_index = np.where(kar10['atomic'] == 14)[0]
    fe_agb_index = np.where(kar10['atomic'] == 26)[0]

    if ((len(h_agb_index)==0) or (len(he_agb_index)==0) or (len(c_agb_index)==0) or
    (len(o_agb_index)==0) or (len(mg_agb_index)==0) or (len(si_agb_index)==0) or
    (len(fe_agb_index)==0)):
        print("Error with kar10['atomic'] values")

    ########## NOW STEP THROUGH TIME ##########

    j = 0

    #While the age of the universe at a given timestep is less than
    #the age of the universe at z = 0, AND more than 10 Myr has not passed
    #OR gas remains within the galaxy at a given time step
    #AND the gas mass in iron at the previous timestep is subsolar
    while ((j < (n - 1)) and ((j <= 10) or ( (model[j]['mgas'] > 0.0)
    and (model[j-1]['eps'][fe_sn_index] < 0.0) ) )):

        #model['abund']=[H, He, C, O, Mg, Si, Ca, Ti, Fe]

        #If we are at the initial timestep in the model (t = 0)
        if j == 0:
            #gas mass (M_sun), determined from individual element gas masses Eq. 2
            model[j]['mgas'] = model[j-1]['abund'][h_sn_index] + model[j-1]['abund'][he_sn_index] + \
                (model[j-1]['abund'][mg_sn_index] + model[j-1]['abund'][si_sn_index] + \
                 model[j-1]['abund'][ca_sn_index] + model[j-1]['abund'][ti_sn_index])*10.0**(1.31) + \
                 model[j-1]['abund'][fe_sn_index]*10.0**(0.03)

        #SFR for power law model (M_sun Gyr**-1), Eq. 5
        model[j]['mdot'] = sfr_norm*model[j]['mgas']**sfr_exp / 1.e6**(sfr_exp-1.0)

        """
        if (j > 10) and ((model[j]['mgas'] <= 0.) or model[j-1]['eps'][fe_sn_index] >= 0.):
            model = model[0:j-1]
            break
        """

        #Define the gas phase absolute metal mass fraction (dimensionless), Eq. 3
        #If, at the previous timestep, the gas mass was nonzero
        if model[j-1]['mgas'] > 0.0:

            #Subtract off contributions from H and He to the total gas mass
            model[j]['z'] = (model[j]['mgas'] - model[j-1]['abund'][h_sn_index] - \
            model[j-1]['abund'][he_sn_index])/model[j]['mgas']

        #Otherwise, if the gas is depleted, the gas mass is zero
        else:
            model[j]['z'] =  0.0

        #SN Ia delay time distribution (SNe Gyr**-1 (M_sun)**-1)
        #returns Phi_Ia, or the SNe Ia rate, for each t (t_delay in paper), Eq 9
        dtd = dtd_ia.dtd_ia(model[0:j+1]['t'], ia_model)[::-1]

        #If some time has passed in the universe, such that SNe Ia might go off
        if j > 1:

            #Integrate to determine the instantaneous SN Ia rate (SN Gyr**-1), Eq. 10
            model[j]['Ia_rate'] = scipy.integrate.simps(model[0:j+1]['mdot']*dtd,dx=delta_t)
            #model[j]['Ia_rate'] = idl_tabulate(model[0:j+1]['t'], model[0:j+1]['mdot']*dtd)

        #Otherwise, not enough time passed for SN Ia
        else:
            model[j]['Ia_rate'] = 0.0

        #If the lifetime of massive stars at a given timestep is less than the current
        #age of the universe in the model, then determine which stars will explode
        wexplode = np.where(t_int1 < model[j]['t'])[0]

        #If there are massive stars to explode
        if len(wexplode) > 0:

            #SFR interpolated onto t_int1 grid (M_sun Gyr**-1)
            mdot_int1 = interp_func(model[j]['t']-model[0:j+1]['t'], model[0:j+1]['mdot'],
            t_int1[wexplode])

            #instantaneous SN II rate (SN Gyr**-1), Eq. 8
            R_II = scipy.integrate.simps(mdot_int1*imf.imf(m_int1[wexplode], imf_model),
            m_int1[wexplode])
            #imf_dat = imf.imf(m_int1[wexplode], imf_model)
            #R_II = idl_tabulate(m_int1[wexplode], mdot_int1*imf_dat)

        #Otherwise the SNe II rate is zero
        else:
            R_II = 0.0
        model[j]['II_rate'] = R_II

        #where the time array includes AGB stars for M_init <= 8 M_sun
        wagb = np.where(t_int2 < model[j]['t'])[0]
        if len(wagb) > 0:
            #SFR interpolated onto t_int2 grid (M_sun Gyr**-1)
            mdot_int2 = interp_func(model[j]['t']-model[0:j+1]['t'], model[0:j+1]['mdot'],
            t_int2[wagb])

        ########## NOW STEP THROUGH ELEMENTS ##########

        for i in range(nel):

            ##### SN II #####
            if len(wexplode) > 0:

                #gas phase metal fraction at time the supernova progenitor was born
                z_sn = interp_func(model[j]['t']-model[0:j+1]['t'], model[0:j+1]['z'],
                1.2*M_SN**(-1.85)+0.003)
                #ejected mass as a function of SN mass (M_sun)
                for m in range(7):
                    x_sn[m] = interp_func(z_II, nom06[i]['II'][:,m],
                    min(max(z_sn[m], min(z_II)), max(z_II)))

                #gas phase metal fraction at time the hypernova progenitor was born
                z_hn = interp_func(model[j]['t']-model[0:j+1]['t'], model[0:j+1]['z'],
                1.2*M_HN**(-1.85)+0.003)
                #ejected mass as a function of HN mass (M_sun)
                for m in range(4):
                    x_hn[m] =  interp_func(z_II, nom06[i]['HN'][:,m],
                    min(max(z_hn[m],min(z_II)),max(z_II)))

                x_ej = x_sn[wint]  #ejected mass for SN+HN (M_sun)
                #ejected mass for 10 M_sun SN (M_sun)
                x_ej[0] = x_ej[1]*M_SN_ej[0]/M_SN_ej[1]
                #ejected mass for 100 M_sun SN (M_sun)
                x_ej[len(x_ej)-1] *= M_SN_ej[len(M_SN_ej)-1]/M_SN_ej[len(M_SN_ej)-2]
                #decreased SN ejecta by HN fraction (M_sun)
                x_ej[5:] = x_ej[5:]*(1.0 - epsilon_HN)
                #augment HN ejecta by HN fraction (M_sun)
                x_ej[5:] = x_ej[5:] + epsilon_HN*(np.concatenate((x_hn, [x_hn[3]*100.0/M_HN[3]])))
                #interpolate ejected mass onto dummy integration mass variable (M_sun)
                x_ej = interp_func(M_SN_ej,x_ej, m_int1[wexplode])

                #If assuming a certain SFR law and a certain abundance for metallicity
                #dependent models
                if ((name == 'sfr-law_perets1') and (model[i]['z'] < 0.019*10.0**(-2.0))):
                    x_ej = 0.996*x_ej + 0.004*perets[i].dotIa_1*m_int1[wexplode]
                if ((name == 'sfr-law_perets2') and (model[i]['z'] < 0.019*10.0**(-2.0))):
                    x_ej = 0.996*x_ej + 0.004*perets[i].dotIa_2*m_int1[wexplode]

                #mass ejected from SN II at this time step (M_sun Gyr**-1), Eq. 7
                M_II_arr[i] = scipy.integrate.simps(x_ej*mdot_int1*imf.imf(m_int1[wexplode],imf_model),
                m_int1[wexplode])
                #imf_dat = imf.imf(m_int1[wexplode],imf_model)
                #M_II_arr[i] = idl_tabulate(m_int1[wexplode], x_ej*mdot_int1*imf_dat)

            #Otherwise, if there are no contributions from SNe II at this timestep
            else:
                M_II_arr[i] = 0.0

            ##### AGB #####
            #If there are contributions from AGB stars at this timestep
            if len(wagb) > 0:

                #metallicity at the time low- and intermediate-mass stars in AGB phase
                #were born
                z_agb = interp_func(model[j]['t']-model[0:j+1]['t'], model[0:j+1]['z'],
                t_agb)

                #Based on the element currently considered, determine if
                #there are contributions to the stars from AGB yields
                if i <= 5: ik = i
                if i == 6 or i == 7: ik = 0
                if i == 8: ik = 6

                for m in range(kar10[ik]['AGB'].shape[1]):
                    #ejected mass as a function of LIMS mass (M_sun)
                    x_agb[m+1] = interp_func(z_lims, kar10[ik]['AGB'][:,m],
                    min(max(z_agb[m],min(z_lims)),max(z_lims)))

                x_agb[0] = x_agb[1]*M_lims_ej[0]/M_lims_ej[1]
                x_agb[len(x_agb)-1] = x_agb[len(x_agb)-2]*M_lims_ej[len(M_lims_ej)-1]/M_lims_ej[len(M_lims_ej)-2]
                #interpolate ejected mass onto dummy integration mass variable (M_sun)
                x_agb_ej = interp_func(M_lims_ej, x_agb, m_int2[wagb])

                if i <= 5 or i == 8:
                    #mass ejected from AGB stars at this time step (M_sun Gyr**-1), Eq. 13
                    #imf_dat = imf.imf(m_int2[wagb], imf_model)
                    #M_agb_arr[i] = idl_tabulate(m_int2[wagb], x_agb_ej*mdot_int2*imf_dat)
                    M_agb_arr[i] = scipy.integrate.simps(x_agb_ej*mdot_int2*imf.imf(m_int2[wagb],imf_model),
                    m_int2[wagb])

                else:
                    whgt0 = np.where(model['abund'][0:j+1,0] > 0)[0]
                    #abundance interpolated onto t_int2 grid (M_sun Gyr**-1)
                    x_i_int2 = interp_func(model[j]['t']-model[whgt0]['t'],
                    model[whgt0]['abund'][:,i]/model[whgt0]['abund'][:,0], t_int2[wagb])

                    #mass ejected from AGB stars at this time step (M_sun Gyr**-1)
                    M_agb_arr[i] = scipy.integrate.simps(x_agb_ej*x_i_int2*mdot_int2*imf.imf(m_int2[wagb],imf_model),
                    m_int2[wagb])
                    #imf_dat = imf.imf(m_int2[wagb], imf_model)
                    #M_agb_arr[i] = idl_tabulate(m_int2[wagb], x_agb_ej*x_i_int2*mdot_int2*imf_dat)

            #Otherwise, there is no contribution from AGB stars
            else:
                M_agb_arr[i] = 0.0

            f_Ia = nom06[i]['Ia'] #mass ejected from each SN Ia (M_sun SN**-1)
            M_Ia_arr[i] = f_Ia*model[j]['Ia_rate'] #mass returned to the ISM, Eq. 11

        ########## NOW STEP THROUGH ELEMENTS AGAIN ##########

        for i in range(nel):

            #If, at a given timestep, there is gas within the galaxy
            if model[j]['mgas'] > 0.0:
                #Then the gas mass fraction for a given element is calculated based on the
                #previous timestep
                x_el = model[j-1]['abund'][i]/model[j]['mgas']
            #Otherwise, if there is no gas, then the gas mass fraction is zero
            else: x_el = 0.0

            #mass loading formulation of supernova ejecta
            #(SN ejecta is not instantaneously mixed)

            ##### CONSIDER OTHER GCE MODELS #####

            if name == 'massloading':
                chi = pars[2]  #fraction of SN ejected mass that escapes the galaxy
                #amount of ISM entrained in SN wind relative to escaping SN ejecta

                if npars == 6: eta = pars[5]
                else: eta = pars[4]

                #SN ejecta that escapes the galaxy (M_sun)
                sn_ejecta = chi*(M_II_arr[i]+M_Ia_arr[i])

                #total mass in Type II SN ejecta
                M_II_tot = np.sum(M_II_arr[0:1]) + np.sum(M_II_arr[3:6])*10.0**(1.31) + \
                M_II_arr[7]*10.0**(0.03)
                #total mass in Type Ia SN ejecta
                M_Ia_tot = np.sum(M_Ia_arr[0:1]) + np.sum(M_Ia_arr[3:6])*10.0**(1.31) + \
                M_Ia_arr[7]*10.0**(0.03)
                #total mass in Type II SN ejecta
                M_SN_tot = M_II_tot + M_Ia_tot

                #ISM entrained in SN wind (M_sun)
                entrained_ism = eta*x_el*M_SN_tot
                #gas outflow, equal to SN ejecta and entrained ISM (M_sun)
                model[j]['mout'][i] = sn_ejecta + entrained_ism

            #Define the metal enhancement of the SN winds (dimensionless)
            else:
                if name == 'Zwind':
                    if len(pars) == 7: z_enhance = pars[6]
                    else: z_enhance = pars[5]

                elif name == 'sfr-law_Zwind':
                    z_enhance = 0.01

                else:
                    z_enhance = 0.0

                #Now determine the gas outflow rate for the given element
                #Include some metallicity dependence based on which element is
                #being considered

                if i <= 1:
                    #gas outflow, proportional to Types II and Ia SNe rate (M_sun Gyr**-1)
                    #Eq. 15
                    model[j]['mout'][i] = f_out*(1.0-z_enhance)*x_el*(model[j]['II_rate'] + \
                    model[j]['Ia_rate'])
                elif i > 1 and model[j]['z'] > 0:
                    model[j]['mout'][i] = f_out*(z_enhance*((model[j]['z'])**(-1.0)-1.0)+1.0)*x_el*(model[j]['II_rate'] + \
                    model[j]['Ia_rate'])
                else:
                    model[j]['mout'][i] = 0.0

            #Now determine the star formation rate for a given element
            #(The rate at which the element is locked up in stars)
            #At the given time step
            #gas locked into stars (M_sun Gyr**-1) minus
            #gas returned from Type II SNe, AGB stars, and Type Ia SNe (M_sun Gyr**-1)
            model[j]['dstar_dt'][i] = (x_el)*model[j]['mdot'] - M_II_arr[i] - \
            M_agb_arr[i] - M_Ia_arr[i]

            #change in gas phase abundance, owing to star formation (M_sun Gyr**-1)
            #minus gas outflow from SN winds (M_sun Gyr**-1)
            #plus PRISTINE gas infall, constant rate (M_sun Gyr**-1)
            model[j]['de_dt'][i] = -1.0*model[j]['dstar_dt'][i] - model[j]['mout'][i] + \
            model[j]['f_in']*pristine[i]

            #Calculate the gas phase mass fraction (M_sun) according to timestep
            #If it is not the first timestep
            if j > 0:
                """
                mod = (j+3)%4
                if mod == 0:
                    model[j]['abund'][i] = model[j-1]['abund'][i] + np.sum(int2*model['de_dt'][j-1:j+1,i])
                if mod == 1:
                    model[j]['abund'][i] = model[j-2]['abund'][i] + np.sum(int3*model['de_dt'][j-2:j+1,i])
                if mod == 2:
                    model[j]['abund'][i] = model[j-3]['abund'][i] + np.sum(int4*model['de_dt'][j-3:j+1,i])
                if mod == 3:
                    model[j]['abund'][i] = model[j-4]['abund'][i] + np.sum(int5*model['de_dt'][j-4:j+1,i])
                """
                if j < 4:
                    if j-1 == 0: model[j]['abund'][i] = model[j-1]['abund'][i] + np.sum(int2*model['de_dt'][j-1:j+1,i])
                    elif j-1 == 1: model[j]['abund'][i] = model[j-2]['abund'][i] + np.sum(int3*model['de_dt'][j-2:j+1,i])
                    elif j-1 == 2: model[j]['abund'][i] = model[j-3]['abund'][i] + np.sum(int4*model['de_dt'][j-3:j+1,i])
                else: model[j]['abund'][i] = model[j-4]['abund'][i] + np.sum(int5*model['de_dt'][j-4:j+1,i])

            if model[j]['abund'][i] < 0.0:
                model[j]['abund'][i] = 0
                model[j]['eps'][i] = np.nan
            else:
                #gas phase abundance (number of atoms in units of M_sun/amu = 1.20d57)
                model[j]['eps'][i] = np.log10(model[j]['abund'][i]/interp_func(z_II,
                nom06[i]['weight_II'][:,3], model[j]['z']))

        ##### NOW BACK INTO LARGER TIMESTEP FOR LOOP #####

        model[j]['eps'] -= model[j]['eps'][0] + (eps_sun - 12.0)

        #Calculate the stellar mass of the galaxy at a given timestep
        #If more than 1 Myr has passed
        if j > 0:
            """
            mod = (j+3)%4 #If less than 4 Myr has passed
            if mod == 0:
                model[j]['mstar'] = model[j-1]['mstar'] + np.sum(int2*np.sum(model['dstar_dt'][j-1:j+1], 1))
            if mod == 1:
                model[j]['mstar'] = model[j-2]['mstar'] + np.sum(int3*np.sum(model['dstar_dt'][j-2:j+1], 1))
            if mod == 2:
                model[j]['mstar'] = model[j-3]['mstar'] + np.sum(int4*np.sum(model['dstar_dt'][j-3:j+1], 1))
            if mod == 3:
                model[j]['mstar'] = model[j-4]['mstar'] + np.sum(int5*np.sum(model['dstar_dt'][j-4:j+1], 1))
            """
            if j < 4:
                if j-1 == 0: model[j]['mstar'] = model[j-1]['mstar'] + np.sum(int2*np.sum(model[j-1:j+1]['dstar_dt'], 1))
                elif j-1 == 1: model[j]['mstar'] = model[j-2]['mstar'] + np.sum(int3*np.sum(model[j-2:j+1]['dstar_dt'], 1))
                elif j-1 == 2: model[j]['mstar'] = model[j-3]['mstar'] + np.sum(int4*np.sum(model[j-3:j+1]['dstar_dt'], 1))
            else: model[j]['mstar'] = model[j-4]['mstar'] + np.sum(int5*np.sum(model[j-4:j+1]['dstar_dt'], 1))

        #If it is the first timestep, there is no stellar mass
        else: model[j]['mstar'] = 0.0

        #total galaxy mass (M_sun) at this timestep
        model[j]['mgal'] = model[j]['mgas'] + model[j]['mstar']

        #Increment in time
        j = j + 1

        if j < n:

            #Update the gas mass for the following timestep
            model[j]['mgas'] = model[j-1]['abund'][h_sn_index] + model[j-1]['abund'][he_sn_index] + \
            (model[j-1]['abund'][mg_sn_index] + model[j-1]['abund'][si_sn_index] + \
            model[j-1]['abund'][ca_sn_index]+model[j-1]['abund'][ti_sn_index])*10.0**(1.31) + \
            model[j-1]['abund'][fe_sn_index]*10.0**(0.03)

            #If somehow the galaxy has negative gas mass, it actually has zero gas mass
            if model[j]['mgas'] < 0.:
                model[j]['mgas'] = 0.0

    ##### OUTSIDE OF ALL FOR LOOPS #####

    #Now return only the model data up to the point at which the galaxy ran out of gas
    model = model[0:j]

    return model, nom06['atomic']

import numpy as np

def imf(mass_array, imf_model):

    if imf_model == 'kroupa93':

        if len(mass_array)>0:
            dN_dM = np.zeros(len(mass_array)) #dN/dm
            #wmed = np.where((mass_array >= 0.5) & (mass_array < 1))[0]
            whi = np.where(mass_array >= 1.0)[0]
            #wlo = np.where(mass_array < 0.5)[0]
            wlo = np.where(mass_array < 1.0)[0]
            coeff = 0.309866
            #Kroupa IMF coefficient for high-mass stars (M > 1.0 M_sun)
            if len(whi)>0:
                dN_dM[whi] = coeff*mass_array[whi]**-2.7
            #Kroupa IMF coefficient for low-mass stars (0.5 < M/M_sun < 1.0)
            #if len(wmed)>0: dN_dM[wmed] = coeff*mass_array[wmed]**-2.2
            #if len(wlo)>0: dN_dM[wlo] = 0.619732*mass_array[wlo]**-1.2
            if len(wlo) > 0:
                dN_dM[wlo] = coeff*mass_array[wlo]**(-2.2)
        else:
            dN_dM = []
        return dN_dM

    elif imf_model == 'kroupa01':
        #Define the properties characteristic of a Kroupa 2001 IMF
        #Note: option in Kroupa 2001 IMF for m/Msun > 1 s.t. alpha = 2.7 \pm 0.3
        #For information concerning the IMF constants, see Table 1 of Kroupa 2002

        alpha_lo, alpha_med, alpha_hi = 0.3, 1.3, 2.3 #(\pm 0.7, 0.5, 0.3)
        m1, m2, m3 = 0.08, 0.50, 1.
        k = 0.877 #\(pm 0.045 stars pc^(-3) Msun^(-1))

        if len(mass_array) > 0:

            dN_dM = np.zeros(len(mass_array)) #dN/dm

            wlo = np.where(mass_array < m1)[0]
            wmed = np.where((mass_array >= m1)&(mass_array < m2))[0]
            whi = np.where(mass_array >= m2)[0]

            if len(wlo) > 0:
                dN_dM[wlo] = k*(mass_array/m1)**(-1.*alpha_lo)

            if len(wmed) > 0:
                dN_dM[wmed] = k*(mass_array/m1)**(-1.*alpha_med)

            if len(whi) > 0:
                dN_dM[whi] = k*((m2/m1)**(-1.*alpha_med))*(mass_array/m2)**(-1.*alpha_hi)

        else: dN_dM = []
        return dN_dM

    elif imf_model == 'chabrier03':

        if len(mass_array)>0:
            dN_dM = np.zeros(len(mass_array)) #dN/dm
            whi = np.where(mass_array >= 1.0)[0]
            wlo = np.where(mass_array < 1.0)[0]
            if len(whi)>0:
                dN_dM[whi] = 0.232012599*mass_array[whi]**-2.3
            if len(wlo)>0:
                dN_dM[wlo] = 1.8902/(mass_array*np.log(10))*np.exp(-(np.log10(mass_array)-np.log10(0.08))**2/(2*0.69**2))
        else: dN_dM = []
        return dN_dM


    elif imf_model == 'salpeter55':

        if len(mass_array)>0:
            dN_dM = 0.156713*mass_array**-2.35
        else: dN_dM = []
        return dN_dM

    else:
        sys.stderr.write('IMF model '+imf_model+' not recognized')
        sys.exit()

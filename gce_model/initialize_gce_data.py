import numpy as np
import os
import re

def check_array(array):
    nan = np.where(np.isnan(array))
    if nan[0].size != 0:
        print(f"ERROR: not available (nan in weight) metallicities, masses, elements: {nan}")
        array[nan] = 0
    return array

def initialize(yield_path='yields'):
    #z_II, nel, nom06, kar10, eps_sun, M_SN, M_HN, M_lims, z_lims, perets

    yield_path = os.path.join(os.getcwd(), yield_path)

    #Solar abundances from Anders & Grevesse 1989 and Sneden 1992
    eps_sun = np.array([12.00, 10.99, 3.31, 1.42, 2.88, 8.56, 8.05, 8.93, 4.56, 8.09, 6.33,
                        7.58, 6.47, 7.55, 5.45, 7.21, 5.5 , 6.56, 5.12, 6.36, 3.10, 4.99,
                        4.00, 5.67, 5.39, 7.52, 4.92, 6.25, 4.21, 4.60, 2.88, 3.41])
    #atomic_weight = [1.00794, 4.00602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994,
    #18.9984032, 20.1797, 22.98976928, 24.3050, 26.9815386, 28.0355, 30.973762, 32.065,
    #35.453, 39.948, 39.0983, 40.078, 44.955912, 47.957, 50.9415, 51.9951, 54.9438045,
    #55.845, 58.933195, 58.6934, 63.546, 65.38, 69.723, 72.64]

    #Data for the spacing in terms of mass for AGB yields
    M_lims = np.array([1.00, 1.25, 1.50, 1.75, 1.90, 2.25, 2.50, 3.00, 3.50, 4.00, 4.50,
                       5.00, 5.50, 6.00])
    #Data for the spacing in terms of metallicities for the AGB yield models
    z_lims = np.array([0.0001, 0.004, 0.008, 0.02])

    #Atomic numbers for the chemical species for which we have yields from models
    atomic = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 27, 28])

    #kar10 files contain model data during the AGB phase for models described in Karakas+09.
    #Each file has an associated mass and metallicity for the model, including the final
    #core mass, for the yields (files 2 - 6)
    sub_string = ['2', '3', '4', '5', '6'] #the substring indicates which yield file to use
    kar10files = [yield_path+'/kar10/table_a'+ sub_string[i] +'.txt' for i in range(len(sub_string))]
    nkar10 = len(kar10files)

    #Information concerning the species names and atomic numbers for the yields
    iso_string = np.array(['d','he','li','be','b','c','n','o','f','ne','na','mg','al',
                           'si','p','s','fe','co','ni'])
    iso_atomic = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26,27,28])

    #Define the array that will contain the read in information on AGB yields
    kar10 = np.zeros(len(atomic),dtype=[('atomic','float64'),
                    ('AGB','float64',(len(z_lims),len(M_lims))),
                    ('weight','float64',(len(z_lims),len(M_lims)))])
    kar10['atomic'] = atomic

    #Read in the files
    for i in range(nkar10):
        with open(kar10files[i],'r') as fi:
            for ln in fi:

                ln_list = ln.split() #Split a line according to whitespace
                if ln.startswith("#"): #If a given line starts with #

                    m_init = float(ln_list[3]) #initial stellar mass
                    z_in = float(ln_list[7][:-1]) #initial stellar metallicity
                    wm = np.where(M_lims == m_init)[0]
                    wz = np.where(z_lims == z_in)[0]

                elif ln_list[0] == 'species': #do not read this file
                    pass

                else: #For a standard line containing data in the file:

                    #Read in species, mass number, and mass loss in this model
                    iso_in, isotope, masslost_in = ln_list[0], int(ln_list[1]), float(ln_list[3])
                    #strip the isotope of additional numbers to get just species name
                    elname = iso_in.strip('0123456789-*^')
                    el_index = np.where(iso_string == elname)[0]

                    if (len(el_index) == 1) & (len(wm) == 1) & (len(wz) == 1):

                        atom = iso_atomic[el_index[0]]
                        if isotope == 1: atom = 1
                        wa = np.where(kar10['atomic'] == atom)[0][0]
                        #add isotope mass * yield to kar10[atomic]['weight'][metallicity,mass]
                        kar10[wa]['weight'][wz[0],wm[0]] += isotope*masslost_in
                        #add yield to kar10[atomic]['AGB'][metallicity,mass]
                        kar10[wa]['AGB'][wz[0],wm[0]] += masslost_in

    #divide (isotope mass * yield) by yield
    for j in range(len(kar10['atomic'])):
        kar10[j]['weight'] /= kar10[j]['AGB']

    #cut down the number of elements
    wel = [0, 1, 5, 7, 11, 13, 16]
    #[H,He,C,O,Mg,Si,Fe]
    kar10 = kar10[wel]

    check_array(kar10['weight'])
    if np.where(kar10['weight'] == 0.0)[0].size != 0:
        print(f"ERROR: not available (0.0 yield) metallicities, masses, elements: {np.where(kar10['weight']==0.0)}")

    #Set up in order to read in SN II yield files
    M_SN = np.array([13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0])
    M_HN = np.array([20.0, 25.0, 30.0, 40.0])
    z_II = np.array([0.0, 0.001, 0.004, 0.02])

    iso_string = np.array(['p','d','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',
                           'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe',
                           'Co','Ni','Cu','Zn','Ga','Ge'])
    iso_atomic = np.array([1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                           24,25,26,27,28,29,30,31,32])

    atomic = np.unique(iso_atomic)
    nom06 = np.zeros(len(atomic),dtype=[('atomic','float64'),
                    ('II','float64',(len(z_II),len(M_SN))),
                    ('weight_II','float64',(len(z_II),len(M_SN))),
                    ('HN','float64',(len(z_II),len(M_HN))),
                    ('weight_HN','float64',(len(z_II),len(M_HN))),
                    ('Ia','float64'),('weight_Ia','float64')])
    nom06['atomic'] = atomic

    #load SN II data
    nom06snfile = yield_path+'/nom06/tab1.txt'
    #Metallicity, element, solar mass progenitor models
    data = np.genfromtxt(nom06snfile, names = ['Metallicity','isotope','II_13', 'II_15',
    'II_18', 'II_20', 'II_25', 'II_30', 'II_40'],dtype=None,skip_header=24,
    encoding='utf-8')

    mass_array = np.array(data[['II_13', 'II_15', 'II_18', 'II_20', 'II_25', 'II_30', 'II_40']].tolist())

    #Formatting for species name, array where the atomic name only remains
    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])
    mask = (elname != 'M_final_') & (elname != 'M_cut_') #mask these lines

    #Now read in the specified lines with data
    for i in np.arange(len(elname))[mask]:

        wz = np.where(z_II == data['Metallicity'][i])[0]
        el_index = np.where(iso_string == elname[i])[0]

        if (len(el_index) == 1) & (len(wz) == 1):

            atom = iso_atomic[el_index[0]]
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])

            wa = np.where(nom06['atomic'] == atom)[0][0]
            for mass_i in range(len(M_SN)):
                #add isotope mass * yield to nom06 data
                nom06[wa]['weight_II'][wz[0],mass_i] += isotope*mass_array[i,mass_i]
                # add yield to nom06 data
                nom06[wa]['II'][wz[0],mass_i] += mass_array[i,mass_i]

    #load HN data (contains M_SN >= 20 Msun models)
    nom06hnfile = yield_path+'/nom06/tab2.txt'
    data = np.genfromtxt(nom06hnfile, names = ['Metallicity','isotope','II_20', 'II_25',
    'II_30', 'II_40'],dtype=None,skip_header=21, encoding='utf-8')
    mass_array = np.array(data[['II_20', 'II_25', 'II_30', 'II_40']].tolist())

    #array where the atomic name only remains
    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])
    mask = (elname != 'M_final_') & (elname != 'M_cut_')

    for i in np.arange(len(elname))[mask]:

        wz = np.where(z_II == data['Metallicity'][i])[0]
        el_index = np.where(iso_string == elname[i])[0]

        if (len(el_index) == 1) & (len(wz) == 1):

            atom = iso_atomic[el_index[0]]
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])

            wa = np.where(nom06['atomic'] == atom)[0][0]
            for mass_i in range(len(M_HN)):
                #add isotope mass * yield to nom06 data array
                nom06[wa]['weight_HN'][wz[0],mass_i] += isotope*mass_array[i,mass_i]
                #add yield to nom06 data array
                nom06[wa]['HN'][wz[0],mass_i] += mass_array[i,mass_i]

    #load He yields from /nom06/tab3.txt - He yield is not included in iwa99

    nom06Iafile = yield_path+'/nom06/tab3.txt'
    data = np.genfromtxt(nom06Iafile, names = ['isotope', 'yield'],dtype=None,
    skip_header=16,usecols=(0,5), encoding='utf-8')

    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])

    for i in np.arange(len(elname)):

        if elname[i] == 'He':

            atom = 2
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])

            wa = np.where(nom06['atomic'] == atom)[0][0]
            nom06[wa]['weight_Ia'] += isotope*data['yield'][i]
            nom06[wa]['Ia'] += data['yield'][i]

    #then add yields with iwa99/tab3.txt
    iwa99file = yield_path+'/iwa99/tab3.txt'
    data = np.genfromtxt(iwa99file, names = ['isotope','yield'],dtype=None,usecols=(0,2),
    encoding='utf-8')

    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])

    for i in np.arange(len(elname)):

        el_index = np.where(iso_string == elname[i])[0]
        if (len(el_index) == 1):

            atom = iso_atomic[el_index[0]]
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])

            wa = np.where(nom06['atomic'] == atom)[0][0]
            nom06[wa]['weight_Ia'] += isotope*data['yield'][i]
            nom06[wa]['Ia'] += data['yield'][i]

    #Normalize the weights
    nom06['weight_II'][nom06['II']>0] /= nom06['II'][nom06['II']>0]
    nom06['weight_HN'][nom06['HN']>0] /= nom06['HN'][nom06['HN']>0]
    nom06['weight_Ia'][nom06['Ia']>0] /= nom06['Ia'][nom06['Ia']>0]

    wel = [0, 1, 5, 7, 11, 13, 19, 21, 25]
    #[H,He,C,O,Mg,Si,Ca,Ti,Fe]
    eps_sun = eps_sun[wel]

    nel = len(wel)
    nom06 = nom06[wel]

    check_array(nom06['weight_II'])
    check_array(nom06['weight_HN'])
    check_array(nom06['weight_Ia'])

    perets = np.zeros(8,dtype=[('dotIa_1','float64'),('dotIa_2','float64')])
    perets[3:8]['dotIa_1'] = [4.4e-5, 0.001, 0.07, 0.0059, 0.00025]
    perets[3:8]['dotIa_2'] = [2.7e-5, 0.00058, 0.052, 0.021, 0.0025]

    return z_II, nel, nom06, kar10, eps_sun, M_SN, M_HN, M_lims, z_lims

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


# Assign labels used in eep conversion
eep_params = dict(
    age = 'Age_gyr',
    log_central_temp = 'LogT_cen',
    core_hydrogen_frac = 'X_cen',
#    hydrogen_lum = 'H lum (Lsun)',
    lum = 'LogL_lsun',
    logg = 'Log_g',
    log_teff = 'log_Teff',
    core_helium_frac = 'Y_cen',
    L3a = '3a_lsun',
    core_z_frac = 'Z_cen',
    mass = 'Mass_msun',
    teff_scale = 5, # used in metric function
    lum_scale = 1, # used in metric function
    # `intervals` is a list containing the number of secondary Equivalent
    # Evolutionary Phases (EEPs) between each pair of primary EEPs.
    intervals = [200, # Between PreMS and ZAMS
                 250, # ZAMS-TAMS
                 150, # TAMS-TRGB
                  50] # TRGB - ZAHB/ZACHeB
                 
)

def my_PreMS(track, eep_params, i0=None):
    '''
    Let the first point be the PreMS.
    '''
    return 0

def my_ZAMS(track, eep_params, i0):
    '''
    YREC 25 release defines ZAMS point as the point where X_cen drops 0.001 from initial X_cen.
    '''
    core_hydrogen_frac = eep_params['core_hydrogen_frac']
    Xc_tr = track.loc[i0:, core_hydrogen_frac]
    X0 = Xc_tr.iloc[0]                 # initial hydrogen fraction
    condition = Xc_tr <= (X0 - 0.001) 
    if not condition.any():
        return -1  
    return condition.idxmax()
    
    
def my_TAMS(track, eep_params, i0, Xmin=1e-4):
    '''
    YREC25 defines the TAMS as the first point in the track where Xcen
    drops below 1e-4. 
    '''
    core_hydrogen_frac = eep_params['core_hydrogen_frac']
    Xc_tr = track.loc[i0:, core_hydrogen_frac]
    below_crit = Xc_tr <= Xmin
    if not below_crit.any():
        return -1
    return below_crit.idxmax()

 
def my_RGBTip(track, eep_params, i0):
    """
    RGB Tip:
      - For M < 2 Msun: absolute luminosity maximum (He-flash tip)
      - For M >= 2 Msun: where Y_cen is highest (He max), otherwise luminosity max
    """
    mass = eep_params["mass"]
    Yc   = eep_params["core_helium_frac"]
    lum  = eep_params["lum"]

    mass_tr = track[mass].iloc[0]
    Yc_tr  = track.loc[i0:, Yc]
    lum_tr = track.loc[i0:, lum]
    

    return lum_tr.idxmax()


def my_ZAHB(track, eep_params, i0=None):
    """
    YREC25 defines as the first point where trialpha luminosity exceeds 0.001
    while Y_cen > 0.5 and Y_cen < 1 - Z_cen0 - 0.04.
    """
    L3a = eep_params['L3a']
    Yc  = eep_params['core_helium_frac']
    Zc  = eep_params['core_z_frac']

    L3a_tr = track.loc[i0:, L3a]
    Yc_tr  = track.loc[i0:, Yc]
    Zc_tr  = track.loc[i0:, Zc]

    # Before core He ignition
    heburn = (L3a_tr > 0.001) & (Yc_tr > 0.5) & (Yc_tr < (1.0 - Zc_tr.iloc[0] - 0.04))

    if not heburn.any():
        return -1
    return heburn.idxmax()


def my_HRD(track, eep_params):
    '''
    Adapted from eep._HRD_distance to fix lum logarithm
    '''

    # Allow for scaling to make changes in Teff and L comparable
    Tscale = eep_params['teff_scale']
    Lscale = eep_params['lum_scale']

    log_teff = eep_params['log_teff']
    lum = eep_params['lum']

    logTeff = track[log_teff]
    logLum = track[lum]

    N = len(track)
    dist = np.zeros(N)
    for i in range(1, N):
        temp_dist = (((logTeff.iloc[i] - logTeff.iloc[i-1])*Tscale)**2
                    + ((logLum.iloc[i] - logLum.iloc[i-1])*Lscale)**2)
        dist[i] = dist[i-1] + np.sqrt(temp_dist)

    return dist


def read_columns(path):
    with open(path, 'r') as f:
        columns = [l.strip() for l in f]
    
    return columns

def parse_filename(filename):
    file_str = filename.replace('.track', '')

    # Mass
    # New format: 'm0500' -> 0.500
    mass_str = file_str[1:5]  
    mass = float(mass_str)/1000

    # Files are in [Fe/H]
    # Look for 'fehm' or 'fehp'
    met_i = file_str.find('feh') + 3   
    met_sign = -1 if file_str[met_i] == 'm' else 1
    met_str = file_str[met_i+1:met_i+4]  
    met = met_sign * float(met_str)/100

    # Tracks are not alpha enhanced 
    alpha = 0.0

    return mass, met, alpha


def from_yrec(path, columns=None):
    if columns is None:
        raw_grids_path = os.path.dirname(path)
        columns = read_columns(os.path.join(raw_grids_path, 'column_labels.txt'))

    fname = os.path.basename(path)
    initial_mass, initial_met, initial_alpha = parse_filename(fname)
    df = pd.read_csv(path, skiprows=1, header=None)
    df.columns = columns
    
    s = np.arange(len(df))
    m = np.ones_like(s) * initial_mass
    z = np.ones_like(s) * initial_met

    # Build multi-indexed DataFrame, dropping unwanted columns
    multi_index = pd.MultiIndex.from_tuples(zip(m, z, s),
        names=['initial_mass', 'initial_met', 'step'])
    df.index = multi_index
    
    return df


def all_from_yrec(raw_grids_path, progress=True):
    df_list = []
    filelist = [f for f in os.listdir(raw_grids_path) if '.track' in f]
    columns = read_columns(os.path.join(raw_grids_path, 'column_labels.txt'))

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    for fname in file_iter:
        fpath = os.path.join(raw_grids_path, fname)
        df_list.append(from_yrec(fpath, columns))

    dfs = pd.concat(df_list).sort_index()
    # If you want to compute a total hydrogen luminosity, uncomment the next line
    #dfs[eep_params['hydrogen lum']] = dfs[['ppI', 'ppII', 'ppIII']].sum(axis=1)

    return dfs 


def install(
    raw_grids_path,
    name=None,
    eep_params=eep_params,
    eep_functions={'prems': my_PreMS,'zams':my_ZAMS, 'eams': 'skip',
    'iams': 'skip', 'tams': my_TAMS, 'rgbump': 'skip', 'trgb': my_RGBTip, 'zahb': my_ZAHB},
    metric_function=my_HRD,
    ):
    '''
    The main method to install grids that are output of the `rotevol` rotational
    evolution tracer code.

    Parameters
    ----------
    raw_grids_path (str): the path to the folder containing the raw model grids.

    name (str, optional): the name of the grid you're installing. By default,
        the basename of the `raw_grids_path` will be used.

    eep_params (dict, optional): contains a mapping from your grid's specific
        column names to the names used by kiauhoku's default EEP functions.
        It also contains 'eep_intervals', the number of secondary EEPs
        between each consecutive pair of primary EEPs. By default, the params
        defined at the top of this script will be used, but users may specify
        their own.

    eep_functions (dict, optional): if the default EEP functions won't do the
        job, you can specify your own and supply them in a dictionary.
        EEP functions must have the call signature
        function(track, eep_params), where `track` is a single track.
        If none are supplied, the default functions will be used.

    metric_function (callable, None): the metric function is how the EEP
        interpolator spaces the secondary EEPs. By default, the path
        length along the evolution track on the H-R diagram (luminosity vs.
        Teff) is used, but you can specify your own if desired.
        metric_function must have the call signature
        function(track, eep_params), where `track` is a single track.
        If no function is supplied, defaults to yrec.my_HRD.

    Returns None
    '''
    from kiauhoku.stargrid import from_pandas
    from kiauhoku.config import grids_path as install_path

    if name is None:
        name = os.path.basename(raw_grids_path)

    # Create cache directories
    path = os.path.join(install_path, name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Cache eep parameters
    with open(os.path.join(path, f'{name}_eep_params.pkl'), 'wb') as f:
        pickle.dump(eep_params, f)

    print('Reading and combining grid files')
    grids = all_from_yrec(raw_grids_path)
    grids = from_pandas(grids, name=name)

    # Save full grid to file
    full_save_path = os.path.join(path, f'{name}_full.pqt')
    print(f'Saving to {full_save_path}')
    grids.to_parquet(full_save_path)

    print(f'Converting to eep-based tracks')
    eeps = grids.to_eep(eep_params, eep_functions, metric_function)

    # Save EEP grid to file
    eep_save_path = os.path.join(path, f'{name}_eep.pqt')
    print(f'Saving to {eep_save_path}')
    eeps.to_parquet(eep_save_path)

    # Create and save interpolator to file
    interp = eeps.to_interpolator()
    interp_save_path = os.path.join(path, f'{name}_interpolator.pkl')
    print(f'Saving interpolator to {interp_save_path}')
    interp.to_pickle(path=interp_save_path)

    print(f'Model grid "{name}" installed.')   


if __name__ == "__main__":
    install("/home/zach/repositories/kiauhoku/grids/yrec", name="yrec")

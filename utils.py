import pickle
import numpy as np

from mpi4py import MPI

# Transfsorm Velocity Vectors in Cartenian to Polar Coordiantes
def _cart_to_polar(x, y):
    # Vector Modulus
    psi_ = np.nan_to_num(np.sqrt(x**2 + y**2))
    # Vector Direction
    phi_ = np.nan_to_num(np.arctan2(y, x))
    # Correct for domain -2pi to 2pi
    phi_[phi_ < 0.] += 2*np.pi
    return psi_, phi_

# Load all variable in a pickle file
def _load_file(name):
    def __load_variable(files = []):
        while True:
            try:
                files.append(pickle.load(f))
            except:
                return files
    with open(name, 'rb') as f:
        files = __load_variable()
    return files

# Group down together the entired dataset in predictions and covariates
def _save_file(X_, name):
    with open(name, 'wb') as f:
        pickle.dump(X_, f)
    print(name)

# Get MPI node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print('>> MPI: Name: {} Rank: {} Size: {}'.format(name, rank, size) )
    return int(rank), int(size), comm

# Define a euclidian frame of coordenate s
def _euclidian_coordiantes(N_x, N_y):
    return np.meshgrid(np.linspace(0, N_x - 1, N_x), np.linspace(0, N_y - 1, N_y))

# Path names and software directories
def _load_segm_models(segm_models_folder):

    # Generative Models
    nbc_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'nbc_030.pkl') )
    kms_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'kms_021.pkl') )
    gmm_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'gmm_030.pkl') )
    gda_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'gda_032.pkl') )

    # Discriminative Models
    rrc_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'rrc_131.pkl') )
    svc_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'svc_130.pkl') )
    gpc_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'gpc_130.pkl') )

    # MRF Models
    mrf_        = _load_file(name = '{}/{}'.format(segm_models_folder, r'mrf_132.pkl') )
    icm_mrf_    = _load_file(name = '{}/{}'.format(segm_models_folder, r'icm-mrf_121.pkl') )
    sa_mrf_     = _load_file(name = '{}/{}'.format(segm_models_folder, r'sa-mrf_132.pkl') )
    sa_icm_mrf_ = _load_file(name = '{}/{}'.format(segm_models_folder, r'sa-icm-mrf_221.pkl') )

    names_  = [r'NBC', r'k-means', r'GMM', r'GDA', r'RRC', r'SVC', r'GPC', r'MRF', r'ICM-MRF']
    models_ = [nbc_, kms_, gmm_, gda_, rrc_, svc_, gpc_, mrf_, icm_mrf_]
    return names_, models_

__all__ = ['_load_file', '_save_file', '_get_node_info', '_euclidian_coordiantes', '_cart_to_polar', '_load_segm_models']

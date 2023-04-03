import random
import numpy as np
import hmdl
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import bz2
import _pickle as cPickle
import time 
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
import sklearn.model_selection
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from keras.datasets import cifar10, mnist
import keras.models
import keras.layers
import keras.layers.convolutional
import keras.layers.pooling
import keras.regularizers
from numba import cuda

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN SETTINGS FOR AN EXPERIMENT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
EXPERIMENT_ID = None # specify a known experiment's id as string (to reproduce it) or None to run a new experiment
UNPICKLE_DATA = True
PICKLE_DATA = False # set True if given data set is read and generated for the first time 

UNPICKLE_HMDL_CLF = False
COPY_INITIAL_WEIGHTS_KERAS_TO_HMDL = False
FIT_OR_REFIT_HMDL_CLF = True
PICKLE_HMDL_CLF = True

LOAD_KERAS_CLF = False
COPY_INITIAL_WEIGHTS_HMDL_TO_KERAS = False
FIT_OR_REFIT_KERAS_CLF = True
SAVE_KERAS_CLF = True

SHOW_SOME_DATA_IMAGES = False   
SHOW_SOME_DATA_IMAGES_COUNT = 25
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# END OF MAIN SETTINGS AN EXPERIMENT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


FOLDER_DATA = "./data/"
FOLDER_HMDL_CLFS = "./hmdl_clfs/"
FOLDER_EXPERIMENTS = "./experiments/"
FOLDER_KERAS_CLFS = "./keras_clfs/"

SUFFIX_DATA_BIN = "_data.bin"
SUFFIX_EXPERIMENT_BIN = "_experiment.bin"
SUFFIX_EXPERIMENT_LOG = "_experiment.log"
SUFFIX_HMDL_CLF_BIN = "_hmdl_clf.bin"
SUFFIX_KERAS_CLF_BIN = "_keras_clf"

VERBOSE_LAYERS = False

def set_seeds(seed):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_tf_global_determinism(seed):
    set_seeds(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def hash_experiment(data_description, hmdl_clf_description, digits=10):
    all_str = dict_to_str(data_description) + "\n" + list_to_str(hmdl_clf_description)    
    return str((hash_function(all_str) & ((1 << 32) - 1)) % 10**digits).rjust(digits, "0")

def hash_function(s):
    h = 0
    for c in s:
        h *= 31 
        h += ord(c)
    return h

def list_to_str(list):
    list_str = ""
    for i, elem in enumerate(list):
        list_str += "[" if i == 0 else " "  
        list_str += str(elem) + (",\n" if i < len(list) - 1 else "]")
    return list_str 

def dict_to_str(dict):
    dict_str = ""
    for i, key in enumerate(dict):
        dict_str += "{" if i == 0 else " "  
        dict_str += str(key) + ": " + str(dict[key]) + (",\n" if i < len(dict) - 1 else "}")
    return dict_str

def do_pickle(file_path, some_list):
    print(f"DO PICKLE... [to: {file_path}]")
    t1 = time.time()
    with bz2.BZ2File(file_path, "w") as f: 
        cPickle.dump(some_list, f)
    t2 = time.time()
    print(f"DO PICKLE DONE. [time: {t2 - t1} s]")

def do_unpickle(file_path):
    print(f"DO UNPICKLE... [from: {file_path}]")
    t1 = time.time()    
    some_list = bz2.BZ2File(file_path, "rb")
    some_list = cPickle.load(some_list)        
    t2 = time.time()
    print(f"DO UNPICKLE DONE. [time: {t2 - t1} s]")
    return some_list

def show_some_data_images(images, indexes=None, as_grid=True, title=None, subtitles=None):
    print("SHOW SOME DATA IMAGES... [close images window to continue]")        
    if indexes is None:
        indexes = np.arange(images.shape[0])
    qx = 1
    qy = len(indexes)    
    if as_grid:
        qx = int(np.ceil(np.sqrt(len(indexes))))
        qy = qx
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    min_value = np.min(images)
    max_value = np.max(images)
    if images.shape[-1] == 1: # single channel
        plt.gray()
    for i in range(len(indexes)):
        ax = plt.subplot(qx, qy, i + 1)
        ax.axis("off")
        if subtitles is not None:
            ax.title.set_text(subtitles[i])
        plt.imshow((images[indexes[i]] - min_value) / (max_value - min_value), interpolation="none")    
    plt.tight_layout()        
    plt.show()
    print("SHOW SOME DATA IMAGES DONE.") 

class Logger(object):
    def __init__(self, log_file):
        self.terminal_ = sys.stdout
        self.log_file_ = log_file
   
    def write(self, message):
        self.terminal_.write(message)
        self.terminal_.flush()
        self.log_file_.write(message)
        self.log_file_.flush()  
        
    def write_to_file_only(self, message):
        self.log_file_.write(message)
        self.log_file_.flush()        

    def flush(self):
        pass # needed for python 3 compatibility
        
def prepare_data(description, seed):
    print("PREPARE DATA...")        
    t1 = time.time()        
    data_name = description["name"]
    m_train_per_class_limit = description["m_train_per_class_limit"]
    m_test_per_class_limit = description["m_test_per_class_limit"]
    uniform_scaling = description["uniform_scaling"]
    data_name_full = data_name + "_" + (str(m_train_per_class_limit) if m_train_per_class_limit is not None else "all") + "_" + (str(m_test_per_class_limit) if m_test_per_class_limit is not None else "all") \
        + "_" + ("u" if uniform_scaling else "s") + "_" + str(seed)
    print(f"[data: {data_name_full}]")
    file_path = FOLDER_DATA + data_name_full + SUFFIX_DATA_BIN
    if UNPICKLE_DATA:        
        [(X_train, y_train), (X_test, y_test)] = do_unpickle(file_path)        
    else:
        if description["name"] == "mnist":
            (X_train, y_train), (X_test, y_test) = mnist.load_data()  
        elif description["name"] == "olivetti":
            of = fetch_olivetti_faces() 
            X = of["data"] # 400 x 4096
            y = of["target"]            
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.7, stratify=y)
            X_train = np.reshape(X_train, (X_train.shape[0], 64, 64))
            X_test = np.reshape(X_test, (X_test.shape[0], 64, 64))              
        elif description["name"] == "cifar-10":
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()                                                    
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)        
        print(f"[shapes of source data -> train: {X_train.shape}, test: {X_test.shape}]")
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        indexes_train = []
        indexes_test = []
        n_classes = np.unique(y_train).size
        for label in range(n_classes):
            indexes = np.where(y_train == label)[0] 
            if m_train_per_class_limit is not None:
                indexes = indexes[:m_train_per_class_limit]
            indexes_train.append(indexes)
        indexes_train = np.concatenate(indexes_train)    
        indexes_train = indexes_train[np.random.permutation(indexes_train.size)]
        for label in range(n_classes):
            indexes = np.where(y_test == label)[0] 
            if m_test_per_class_limit is not None:
                indexes = indexes[:m_test_per_class_limit]
            indexes_test.append(indexes)
        indexes_test = np.concatenate(indexes_test)                    
        X_train = np.ascontiguousarray(X_train[indexes_train])
        y_train = np.ascontiguousarray(y_train[indexes_train])
        X_test = np.ascontiguousarray(X_test[indexes_test])
        y_test = np.ascontiguousarray(y_test[indexes_test])
        if (len(X_train.shape) != 4): # when single channel
            m, h, w = X_train.shape
            X_train = np.reshape(X_train, (m, h, w, 1))
            m, h, w = X_test.shape
            X_test = np.reshape(X_test, (m, h, w, 1))
        if uniform_scaling:
            min_val = np.min(X_train)
            max_val = np.max(X_train)
            X_train = (X_train - min_val) / (max_val - min_val) * 2.0 - 1.0
            X_test = (X_test - min_val) / (max_val - min_val) * 2.0 - 1.0
        else:
            means = np.mean(X_train, axis=0)
            stds = np.std(X_train, axis=0)
            if np.sum(stds == 0) > 0:
                raise Exception("Exception: standard scaling impossible for given data, standard deviations are equal to zero for some pixels; try uniform scaling instead.")
            X_train = (X_train - means) / stds
            X_test = (X_test - means) / stds
        print(f"[shapes of produced data -> train: {X_train.shape}, test: {X_test.shape}]")                    
    if PICKLE_DATA:        
        do_pickle(file_path, [(X_train, y_train), (X_test, y_test)])     
    t2 = time.time()    
    print(f"PREPARE DATA DONE. [time: {t2 - t1} s]")
    return X_train, y_train, X_test, y_test

def gpu_props():
    gpu = cuda.get_current_device()
    props = {}
    props["name"] = gpu.name.decode("ASCII")
    props["max_threads_per_block"] = gpu.MAX_THREADS_PER_BLOCK
    props["max_block_dim_x"] = gpu.MAX_BLOCK_DIM_X
    props["max_block_dim_y"] = gpu.MAX_BLOCK_DIM_Y
    props["max_block_dim_z"] = gpu.MAX_BLOCK_DIM_Z
    props["max_grid_dim_x"] = gpu.MAX_GRID_DIM_X
    props["max_grid_dim_y"] = gpu.MAX_GRID_DIM_Y
    props["max_grid_dim_z"] = gpu.MAX_GRID_DIM_Z    
    props["max_shared_memory_per_block"] = gpu.MAX_SHARED_MEMORY_PER_BLOCK
    props["async_engine_count"] = gpu.ASYNC_ENGINE_COUNT
    props["can_map_host_memory"] = gpu.CAN_MAP_HOST_MEMORY
    props["multiprocessor_count"] = gpu.MULTIPROCESSOR_COUNT
    props["warp_size"] = gpu.WARP_SIZE
    props["unified_addressing"] = gpu.UNIFIED_ADDRESSING
    props["pci_bus_id"] = gpu.PCI_BUS_ID
    props["pci_device_id"] = gpu.PCI_DEVICE_ID
    props["compute_capability"] = gpu.compute_capability            
    CC_CORES_PER_SM_DICT = {
        (2,0) : 32,
        (2,1) : 48,
        (3,0) : 192,
        (3,5) : 192,
        (3,7) : 192,
        (5,0) : 128,
        (5,2) : 128,
        (6,0) : 64,
        (6,1) : 128,
        (7,0) : 64,
        (7,5) : 64,
        (8,0) : 64,
        (8,6) : 128
        }
    props["cores_per_SM"] = CC_CORES_PER_SM_DICT.get(gpu.compute_capability)
    props["cores_total"] = props["cores_per_SM"] * gpu.MULTIPROCESSOR_COUNT
    return props    

def clean_gpu_name(name):
    name = name.lower()
    name = name.replace(" ", "_")
    name = name.replace(".", "_")    
    return name

def keras_weights_l1_norm(keras_clf):
    norm = 0.0
    for i, l in enumerate(keras_clf.layers):
        weights = l.get_weights()
        if weights is not None:
            for k in range(len(weights)):
                keras_weights = keras_clf.layers[i].get_weights()[k]
                norm += np.sum(np.abs(keras_weights))
    return norm

def keras_weights_l2_norm(keras_clf):
    norm = 0.0
    for i, l in enumerate(keras_clf.layers):
        weights = l.get_weights()
        if weights is not None:
            for k in range(len(weights)):
                keras_weights = keras_clf.layers[i].get_weights()[k]
                norm += np.sum(np.square(keras_weights))
    return np.sqrt(norm)
                
def build_hmdl_clf(description):
    HmdlClfClass, params = description[0]
    hmdl_clf = HmdlClfClass(**params)     
    for LayerClass, params in description[1:]:
        layer = LayerClass(**params)
        hmdl_clf.add(layer)
    return hmdl_clf        

def build_keras_clf(hmdl_clf):
    keras_clf = keras.models.Sequential()
    for l in hmdl_clf.layers_:
        if isinstance(l, hmdl.Conv2D):
            ki = tf.keras.initializers.HeUniform() if l.activation_name_ == "relu" else tf.keras.initializers.GlorotUniform()
            kr, br = keras_kernel_and_bias_regularizers(l.l1_penalties_, l.l2_penalties_)            
            keras_clf.add(keras.layers.convolutional.Conv2D(name=l.name_, input_shape=l.input_shape_, filters=l.n_kernels_, kernel_size=(l.kernel_size_, l.kernel_size_), padding="same", activation=l.activation_name_, \
                                                            kernel_initializer=ki, kernel_regularizer=kr, bias_regularizer=br))
        elif isinstance(l, hmdl.MaxPool2D):
            keras_clf.add(keras.layers.pooling.MaxPool2D(name=l.name_, input_shape=l.input_shape_, pool_size=l.pool_size_))
        elif isinstance(l, hmdl.Dropout):
            keras_clf.add(keras.layers.Dropout(name=l.name_, input_shape=l.input_shape_, rate=l.rate_))
        elif isinstance(l, hmdl.Flatten):
            keras_clf.add(keras.layers.Flatten(name=l.name_, input_shape=l.input_shape_))
        elif isinstance(l, hmdl.Dense):
            ki = tf.keras.initializers.HeUniform() if l.activation_name_ == "relu" else tf.keras.initializers.GlorotUniform()
            kr, br = keras_kernel_and_bias_regularizers(l.l1_penalties_, l.l2_penalties_)
            keras_clf.add(keras.layers.Dense(name=l.name_, input_shape=l.input_shape_, units=l.n_neurons_, activation=l.activation_name_, kernel_initializer=ki, kernel_regularizer=kr, bias_regularizer=br))    
    optimizer = None
    if hmdl_clf.use_adam_:
        optimizer = tf.keras.optimizers.Adam(learning_rate=hmdl_clf.learning_rate_, beta_1=hmdl_clf.adam_beta_1_, beta_2=hmdl_clf.adam_beta_1_, epsilon=hmdl_clf.adam_epsilon_, amsgrad=False)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=hmdl_clf.learning_rate_, momentum=hmdl_clf.momentum_rate_, nesterov=False)
    keras_clf.compile(optimizer=optimizer, loss=hmdl_clf.loss_name_, metrics=["accuracy"]) 
    return keras_clf

def keras_kernel_and_bias_regularizers(l1_penalties, l2_penalties):
    l1_kernel = 0.0 if not l1_penalties else l1_penalties[0] 
    l2_kernel = 0.0 if not l2_penalties else l2_penalties[0]
    kernel_regularizer = keras.regularizers.l1_l2(l1_kernel, l2_kernel)
    l1_bias = 0.0 if not l1_penalties else l1_penalties[1] 
    l2_bias = 0.0 if not l2_penalties else l2_penalties[1]
    bias_regularizer = keras.regularizers.l1_l2(l1_bias, l2_bias)
    return kernel_regularizer, bias_regularizer     

def keras_weights_dict(keras_clf):
    weights_dict = {} 
    for l in keras_clf.layers:
        weights = l.get_weights()
        if weights is not None:
            weights_copy = [] 
            for w in range(len(weights)):
                weights_copy.append(np.copy(weights[w]))
            weights_dict[l.name] = weights_copy
    return weights_dict

def hmdl_weights_dict(hmdl_clf):
    weights_dict = {} 
    for l in hmdl_clf.layers_:
        weights = l.weights_
        if weights is not None:
            weights_copy = [] 
            for w in range(len(weights)):
                weights_copy.append(np.copy(weights[w]))
            weights_dict[l.name_] = weights_copy
    return weights_dict

def copy_weights_keras_to_hmdl(keras_weights, hmdl_clf): 
    for l in hmdl_clf.layers_:
        if l.weights_ is not None:          
            keras_weights_l = keras_weights[l.name_]
            for w in range(len(l.weights_)):                
                l.weights_[w] = np.copy(keras_weights_l[w].T if isinstance(l, hmdl.Dense) and w == 0 else keras_weights_l[w])

def copy_initial_weights_hmdl_to_keras(hmdl_weights, keras_clf): 
    for l in keras_clf.layers:
        if l.get_weights() is not None:
            hmdl_weights_l = hmdl_weights[l.name]
            for w in range(len(l.get_weights())):
                l.get_weights()[w] = np.copy(hmdl_weights_l[w].T if isinstance(l, keras.layers.Dense) and w == 0 else hmdl_weights_l[w])
                
def hmdl_clf_to_structure_str(hmdl_clf):
    structure_str = ""
    if len(hmdl_clf.layers_) > 0:
        h, w, c = hmdl_clf.layers_[0].input_shape_
        structure_str += f"{h}x{w}x{c}->"
    for l in hmdl_clf.layers_:
        if isinstance(l, hmdl.Conv2D):
            structure_str += f"C({l.kernel_size_},{l.n_kernels_}"            
        elif isinstance(l, hmdl.MaxPool2D):
            structure_str += f"M({l.pool_size_}"
        elif isinstance(l, hmdl.Dropout):
            structure_str += f"DR({l.rate_}"            
        elif isinstance(l, hmdl.Flatten):
            structure_str += f"F("
        elif isinstance(l, hmdl.Dense):
            structure_str += f"D({l.n_neurons_}"
        structure_str += f",{l.activation_name_}" if l.activation_name_ else ""            
        structure_str += ");"
    return structure_str
    
def hmdl_clf_to_structure_str_mathematica(hmdl_clf):
    structure_str = "{"
    if len(hmdl_clf.layers_) > 0:
        h, w, c = hmdl_clf.layers_[0].input_shape_
        structure_str += "{" + f"{h},{w},{c}" + "}"
    for l in hmdl_clf.layers_:
        structure_str += ",{"
        if isinstance(l, hmdl.Conv2D):
            structure_str += f"\"C\",{l.kernel_size_},{l.n_kernels_}"            
        elif isinstance(l, hmdl.MaxPool2D):
            structure_str += f"\"M\",{l.pool_size_}"
        elif isinstance(l, hmdl.Dropout):
            structure_str += f"\"DR\",{l.rate_}"            
        elif isinstance(l, hmdl.Flatten):
            structure_str += f"\"F\""
        elif isinstance(l, hmdl.Dense):
            structure_str += f"\"D\",{l.n_neurons_}"
        structure_str += f",\"{l.activation_name_}\"" if l.activation_name_ else ""
        structure_str += "}"
    structure_str += "}"
    return structure_str


if __name__ == "__main__":
    
    set_tf_global_determinism(seed=0) # this is just a 'reset' seed; actual seed for experiment specified later on, in data description
    
    gpu_name_clean = clean_gpu_name(cuda.get_current_device().name.decode("ASCII"))
    print(f"GPU NAME (CLEAN): {gpu_name_clean}")
    FOLDER_GPU_NAME = gpu_name_clean + "/" 
    if not os.path.exists(FOLDER_EXPERIMENTS + FOLDER_GPU_NAME):
        os.makedirs(FOLDER_EXPERIMENTS + FOLDER_GPU_NAME)
    if not os.path.exists(FOLDER_HMDL_CLFS + FOLDER_GPU_NAME):        
        os.makedirs(FOLDER_HMDL_CLFS + FOLDER_GPU_NAME)    
    if not os.path.exists(FOLDER_KERAS_CLFS + FOLDER_GPU_NAME):
        os.makedirs(FOLDER_KERAS_CLFS + FOLDER_GPU_NAME)        
    
    log_file = None
    if EXPERIMENT_ID is not None:
        log_file = open(FOLDER_EXPERIMENTS + FOLDER_GPU_NAME + EXPERIMENT_ID + SUFFIX_EXPERIMENT_LOG, "w+")
        logger = Logger(log_file)
        sys.stdout = logger    
    print("EXPERIMENT ID: " + ("None [new experiment]" if EXPERIMENT_ID is None else EXPERIMENT_ID + " [to be reproduced]"))
    
    t1 = time.time()
    
    n_repetitions = 1 # default, possibly enlarged later by the settings
    r = 0
    hmdl_fit_times = []
    hmdl_test_accs = []
    hmdl_test_losses = []
    keras_fit_times = []    
    keras_test_accs = []
    keras_test_losses = []
    while r < n_repetitions: 
        if r == 0:
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # EXPERIMENT SETTINGS
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
            if EXPERIMENT_ID is None: # new experiment
                data_description = {
                    "name": "olivetti", # current possibilities: "mnist", "olivetti", "cifar-10"
                    "m_train_per_class_limit": None, # specify None if unlimitied
                    "m_test_per_class_limit": None, # specify None if unlimitied
                    "uniform_scaling": True,
                    "seed": 0,
                    "n_repetitions": 10
                    }
                print(f"\nDATA DESCRIPTION:\n{dict_to_str(data_description)}\n")
                seed = data_description["seed"]
                np.random.seed(seed)
                n_repetitions = data_description["n_repetitions"]
                X_train, y_train, X_test, y_test = prepare_data(data_description, seed)
                m, height, width, n_channels = X_train.shape
                n_classes = np.unique(y_train).size
                hmdl_clf_description = [
                    (hmdl.SequentialClassifier, {"n_epochs": 10**2, "n_batches": 10, "loss": "categorical_crossentropy", "learning_rate": 1e-3, "decay_rate": 0.0, "use_adam": True, "momentum_rate": 0.0, "gradient_clip": None}),
                    # (hmdl.Conv2D, {"input_shape": (height, width, n_channels), "kernel_size": 5, "n_kernels": 32, "activation": "relu"}),
                    # (hmdl.Conv2D, {"kernel_size": 5, "n_kernels": 32, "activation": "relu"}),
                    # (hmdl.MaxPool2D, {"pool_size": 2}),
                    # (hmdl.Dropout, {"rate": 0.125}),
                    # (hmdl.Conv2D, {"kernel_size": 3, "n_kernels": 64, "activation": "relu"}),
                    # (hmdl.Conv2D, {"kernel_size": 3, "n_kernels": 64, "activation": "relu"}),
                    # (hmdl.MaxPool2D, {"pool_size": 2}),
                    # (hmdl.Dropout, {"rate": 0.125}),                                        
                    (hmdl.Flatten, {"input_shape": (height, width, n_channels)}),
                    (hmdl.Dense, {"n_neurons": 8, "activation": "relu"}),
                    (hmdl.Dropout, {"rate": 0.25}),                                        
                    (hmdl.Dense, {"n_neurons": n_classes, "activation": "softmax"})            
                    ]
                print(f"\nHMDL CLF DESCRIPTION:\n{list_to_str(hmdl_clf_description)}\n")
                EXPERIMENT_ID = hash_experiment(data_description, hmdl_clf_description)
                do_pickle(FOLDER_EXPERIMENTS + FOLDER_GPU_NAME + EXPERIMENT_ID + SUFFIX_EXPERIMENT_BIN, [data_description, hmdl_clf_description])
                log_file = open(FOLDER_EXPERIMENTS + FOLDER_GPU_NAME + EXPERIMENT_ID + SUFFIX_EXPERIMENT_LOG, "w+")
                logger = Logger(log_file)
                logger.write_to_file_only(f"GPU NAME (CLEAN): {gpu_name_clean}\n")
                logger.write_to_file_only("EXPERIMENT ID: None [new experiment]\n")
                logger.write_to_file_only(f"\nDATA DESCRIPTION:\n{dict_to_str(data_description)}\n")
                logger.write_to_file_only(f"\nHMDL CLF DESCRIPTION:\n{list_to_str(hmdl_clf_description)}\n")
                sys.stdout = logger
                print(f"\nABOUT TO RUN NEW EXPERIMENT WITH ID: {EXPERIMENT_ID} [generated]")          
            else:
                [data_description, hmdl_clf_description] = do_unpickle(FOLDER_EXPERIMENTS + FOLDER_GPU_NAME + EXPERIMENT_ID + SUFFIX_EXPERIMENT_BIN) 
                print(f"\nDATA DESCRIPTION:\n{dict_to_str(data_description)}")
                seed = data_description["seed"]
                np.random.seed(seed)
                n_repetitions = data_description["n_repetitions"]
                X_train, y_train, X_test, y_test = prepare_data(data_description, seed)
                n_classes = np.unique(y_train).size        
                print(f"\nHMDL CLF DESCRIPTION:\n{list_to_str(hmdl_clf_description)}")
                print(f"\nABOUT TO REPRODUCE EXPERIMENT WITH ID: {EXPERIMENT_ID}")        
            if SHOW_SOME_DATA_IMAGES:
                show_some_data_images(X_train[:SHOW_SOME_DATA_IMAGES_COUNT], title=f"SOME TRAIN IMAGES FROM DATA SET: {data_description['name']}", subtitles=y_train[:SHOW_SOME_DATA_IMAGES_COUNT])    
            print(f"GPU PROPERTIES: {gpu_props()}")
        print("\n***")
        seed = data_description["seed"] + r
        print(f"REPETITION: {r + 1}/{n_repetitions}... [seed now: {seed}]")
        if r > 0:            
            np.random.seed(seed)
            X_train, y_train, X_test, y_test = prepare_data(data_description, seed)    
            
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # HMDL CLF
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                
        hmdl_clf = build_hmdl_clf(hmdl_clf_description)
        print(f"\nHMDL MODEL SUMMARY:\n{hmdl_clf.summary()}")
        print(f"HMDL MODEL STRUCTURE STRING: {hmdl_clf_to_structure_str(hmdl_clf)}")
        print(f"HMDL MODEL STRUCTURE STRING FOR MATHEMATICA: {hmdl_clf_to_structure_str_mathematica(hmdl_clf)}\n")
        
        keras_clf = None
        if COPY_INITIAL_WEIGHTS_KERAS_TO_HMDL:
            print("[copying initial weights from keras to hmdl]")
            keras_clf = build_keras_clf(hmdl_clf)
            keras_initial_weights = keras_weights_dict(keras_clf)
            copy_weights_keras_to_hmdl(keras_initial_weights, hmdl_clf)
        hmdl_initial_weights = hmdl_weights_dict(hmdl_clf)
    
        y_train_one_hot = hmdl.to_one_hot(y_train, n_classes)
        y_test_one_hot = hmdl.to_one_hot(y_test, n_classes)
        if FIT_OR_REFIT_HMDL_CLF:                
            print("HMDL CLF ACC AND LOSS... [before fit]")
            t1_eval = time.time()
            hmdl_y_pred_train_before = hmdl_clf.forward(X_train, verbose_layers=VERBOSE_LAYERS)            
            print(f"[train acc: {hmdl_clf.acc(hmdl_y_pred_train_before, y_train)}, train loss: {hmdl_clf.loss(hmdl_y_pred_train_before, y_train_one_hot)}]")      
            hmdl_y_pred_test_before = hmdl_clf.forward(X_test, verbose_layers=VERBOSE_LAYERS)
            print(f"[test acc: {hmdl_clf.acc(hmdl_y_pred_test_before, y_test)}, test loss: {hmdl_clf.loss(hmdl_y_pred_test_before, y_test_one_hot)}]")
            t2_eval = time.time()
            print(f"HMDL CLF ACC AND LOSS DONE. [time: {t2_eval - t1_eval} s]\n")
        
        hmdl_clf_file_path = FOLDER_HMDL_CLFS + FOLDER_GPU_NAME + EXPERIMENT_ID + SUFFIX_HMDL_CLF_BIN
        if UNPICKLE_HMDL_CLF:
            hmdl_clf = build_hmdl_clf(hmdl_clf_description)                
            hmdl_clf.set_weights(do_unpickle(hmdl_clf_file_path))
            
        if FIT_OR_REFIT_HMDL_CLF:
            print("HMDL FIT...")
            t1_fit = time.time()
            hmdl_clf.fit(X_train, y_train, verbose_layers=VERBOSE_LAYERS, verbose_fit_info=True)
            t2_fit = time.time()
            print(f"HMDL FIT DONE. [time: {t2_fit - t1_fit} s]")
            hmdl_fit_times.append(t2_fit - t1_fit)     
            if PICKLE_HMDL_CLF:
                do_pickle(hmdl_clf_file_path, hmdl_clf.get_weights())
                
        if UNPICKLE_HMDL_CLF or FIT_OR_REFIT_HMDL_CLF:
            print("\nHMDL CLF ACC AND LOSS... [after fit]")
            t1_eval = time.time()
            hmdl_y_pred_train_after = hmdl_clf.forward(X_train)        
            hmdl_y_pred_test_after = hmdl_clf.forward(X_test)
            print(f"[train acc: {hmdl_clf.acc(hmdl_y_pred_train_after, y_train)}, train loss: {hmdl_clf.loss(hmdl_y_pred_train_after, y_train_one_hot)}]")
            hmdl_test_acc_after = hmdl_clf.acc(hmdl_y_pred_test_after, y_test)
            hmdl_test_loss_after = hmdl_clf.loss(hmdl_y_pred_test_after, y_test_one_hot)
            print(f"[test acc: {hmdl_test_acc_after}, test loss: {hmdl_test_loss_after}]")
            t2_eval = time.time()
            print(f"HMDL CLF ACC AND LOSS DONE. [time: {t2_eval - t1_eval} s]\n")
            hmdl_test_accs.append(hmdl_test_acc_after)
            hmdl_test_losses.append(hmdl_test_loss_after)
    
    
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # KERAS CLF
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
        if LOAD_KERAS_CLF or FIT_OR_REFIT_KERAS_CLF:        
            if keras_clf is None:            
                keras_clf = build_keras_clf(hmdl_clf)
            print("\nKERAS MODEL SUMMARY:")
            keras_clf.summary()        
            
            if COPY_INITIAL_WEIGHTS_HMDL_TO_KERAS:
                    print("\n[copying initial weights from hmdl to keras]")
                    copy_initial_weights_hmdl_to_keras(hmdl_initial_weights, keras_clf)
            
            print("\nKERAS CLF ACC AND LOSS... [before fit]")
            t1_eval = time.time()
            keras_train_loss_before, keras_train_acc_before = keras_clf.evaluate(X_train, y_train_one_hot)        
            keras_test_loss_before, keras_test_acc_before = keras_clf.evaluate(X_test, y_test_one_hot)
            print(f"\n[train acc: {keras_train_acc_before}, train loss: {keras_train_loss_before}]")
            print(f"[test acc: {keras_test_acc_before}, test loss: {keras_test_loss_before}]")        
            t2_eval = time.time()
            print(f"KERAS CLF ACC AND LOSS DONE. [time: {t2_eval - t1_eval} s]\n")
            
            if LOAD_KERAS_CLF:
                keras_clf.load_weights(FOLDER_KERAS_CLFS + FOLDER_GPU_NAME + EXPERIMENT_ID + SUFFIX_KERAS_CLF_BIN)            
            if FIT_OR_REFIT_KERAS_CLF:                                    
                print(f"[keras clf norms of weights before fit -> l1: {keras_weights_l1_norm(keras_clf):0.7}, l2: {keras_weights_l2_norm(keras_clf):0.7}]\n")
                n_epochs = hmdl_clf_description[0][1]["n_epochs"]
                n_batches = hmdl_clf_description[0][1]["n_batches"]
                batch_size = X_train.shape[0] // n_batches
                print("KERAS FIT...")
                t1_fit = time.time()            
                keras_clf.fit(X_train, y_train_one_hot, epochs=n_epochs, batch_size=batch_size)
                t2_fit = time.time()
                print(f"KERAS FIT DONE. [time: {t2_fit - t1_fit} s]")
                keras_fit_times.append(t2_fit - t1_fit)
                print(f"\n[keras clf norms of weights after fit -> l1: {keras_weights_l1_norm(keras_clf):0.7}, l2: {keras_weights_l2_norm(keras_clf):0.7}]")
                if SAVE_KERAS_CLF:
                    keras_clf.save(FOLDER_KERAS_CLFS + FOLDER_GPU_NAME + EXPERIMENT_ID + SUFFIX_KERAS_CLF_BIN)
            
            if LOAD_KERAS_CLF or FIT_OR_REFIT_KERAS_CLF:
                print("\nKERAS CLF ACC AND LOSS... [after fit]")
                t1_eval = time.time()
                keras_train_loss_after, keras_train_acc_after = keras_clf.evaluate(X_train, y_train_one_hot)            
                keras_test_loss_after, keras_test_acc_after = keras_clf.evaluate(X_test, y_test_one_hot)
                print(f"\n[train acc: {keras_train_acc_after}, train loss: {keras_train_loss_after}]")
                print(f"[test acc: {keras_test_acc_after}, test loss: {keras_test_loss_after}]")        
                t2_eval = time.time()
                print(f"KERAS CLF ACC AND LOSS DONE. [time: {t2_eval - t1_eval} s]")
                keras_test_accs.append(keras_test_acc_after)
                keras_test_losses.append(keras_test_loss_after)            
        r += 1
    
    print("")
    if hmdl_test_accs:
        print(f"HMDL MEANS -> FIT TIME: {np.mean(hmdl_fit_times) if hmdl_fit_times else 'n.a.'}, TEST ACC: {np.mean(hmdl_test_accs)}, TEST LOSS: {np.mean(hmdl_test_losses)}")
    if keras_test_accs:
        print(f"KERAS MEANS -> FIT TIME: {np.mean(keras_fit_times) if keras_fit_times else 'n.a'}, TEST ACC: {np.mean(keras_test_accs)}, TEST LOSS: {np.mean(keras_test_losses)}")    
    
    t2 = time.time()
    print(f"\nALL DONE FOR EXPERIMENT WITH ID: {EXPERIMENT_ID}. [time: {t2 - t1} s]")
    if log_file:
        log_file.close()    
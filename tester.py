import random
import numpy as np
import hmdl
import os
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tfc
import itertools
import time
from experimenter import set_seeds, set_tf_global_determinism
from experimenter import Logger, hash_function, dict_to_str, do_pickle, do_unpickle
import keras.models
import keras.layers.convolutional
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN SETTINGS FOR A TEST
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SAVE_OR_RESAVE_LOG = False
DO_TEST = False
PICKLE_TEST_RESULTS = False
DO_COMPARISONS = False
DO_PLOTS = True
PLOT_TITLE = "CONV2D FORWARD EXECUTIONS FOR FILTERS: 11x11, 15x15, 19x19"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# END OF MAIN SETTINGS FOR A TEST
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


FOLDER_TESTS = "./tests/"
SUFFIX_TEST_BIN = "_test.bin"
SUFFIX_TEST_LOG = "_test.log"

TIME_EPSILON = 1e-7

PLOT_FONTSIZE_TITLE = 12
PLOT_FONTSIZE_AXES = 12
PLOT_FONTSIZE_LEGEND = 7.5
PLOT_FIGSIZE = (8, 4.5)
PLOT_MARKERSIZE = 4
PLOT_GRID_COLOR = (0.2, 0.2, 0.2) 
PLOT_GRID_DASHES = (0.25, 2.5)
PLOT_LEGEND_LOC = 4
PLOT_LEGEND_HANDLELENGTH = 4

def hash_test(test_description, digits=10):    
    return str((hash_function(dict_to_str(test_description)) & ((1 << 32) - 1)) % 10**digits).rjust(digits, "0")

if __name__ == "__main__":    
                    
    set_tf_global_determinism(seed=0)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    tfc.logging.set_verbosity(tfc.logging.ERROR) 

    # TEST 1
    # test_description = {
    #     "m": 32,
    #     "do_forward_impl": ["numpy_gemm", "numba_jit_direct", "numba_jit_gemm", "cv2", "numba_cuda_direct", "numba_cuda_tiles", "numba_cuda_viafft", "keras"],
    #     "kernel_size": [3, 5, 7],
    #     "n_kernels": [16, 32, 64],
    #     "n_channels": [16, 32, 64],
    #     "img_side": [2, 4, 8, 16, 32, 64],
    #     "n_repetitions": 10
    #     }

    # TEST 2
    test_description = {
        "m": 32,
        "do_forward_impl": ["numba_cuda_direct", "numba_cuda_tiles", "numba_cuda_viafft"],
        "kernel_size": [3, 5, 7, 9, 11, 13, 15, 17, 19],
        "n_kernels": [16, 32, 64],
        "n_channels": [16, 32, 64],
        "img_side": [2, 4, 8, 16, 32, 64],
        "n_repetitions": 10
        }

    TEST_ID = hash_test(test_description)        
    
    log_file = None
    if SAVE_OR_RESAVE_LOG:
        log_file = open(FOLDER_TESTS + TEST_ID + SUFFIX_TEST_LOG, "w+")
        logger = Logger(log_file)
        sys.stdout = logger    
    
    print(f"TEST ID: {TEST_ID}")
    print(f"\nTEST DESCRIPTION:\n{dict_to_str(test_description)}")
    
    t1 = time.time()
    
    test_results = {}
    test_indexer = {}
    if DO_TEST:           
        n_tests = len(test_description["kernel_size"]) * len(test_description["n_kernels"]) * len(test_description["n_channels"]) * len(test_description["img_side"])                        
        for impl in test_description["do_forward_impl"]:
            print("\n" + ("=" * 172))
            print(f"TESTING IMPLEMENTATION {impl}...")
            t1_this_impl = time.time()
            for index, atuple in enumerate(itertools.product(test_description["kernel_size"], test_description["n_kernels"], test_description["n_channels"], test_description["img_side"])):
                (kernel_size, n_kernels, n_channels, img_side) = atuple 
                X = np.random.rand(test_description["m"], img_side, img_side, n_channels)
                X = X.astype(np.float32)
                clf = None  
                if impl != "keras":
                    clf = hmdl.SequentialClassifier()
                    clf.add(hmdl.Conv2D(input_shape=(img_side, img_side, n_channels), n_kernels=n_kernels, kernel_size=kernel_size, do_forward_impl=impl))
                else:                                                                            
                    clf = keras.models.Sequential()                                 
                    clf.add(keras.layers.convolutional.Conv2D(input_shape=(img_side, img_side, n_channels), filters=n_kernels, kernel_size=(kernel_size, kernel_size), padding="same"))                    
                    clf.predict(X) # to "warm up" session                
                print(f"{index + 1}/{n_tests}... [impl: {impl}, kernel_size: {kernel_size}, n_kernels: {n_kernels}, n_channels: {n_channels}, img_side: {img_side}]")
                total = 0.0
                for r in range(test_description["n_repetitions"]):                    
                    t1_test = time.time()
                    if impl != "keras":                            
                        clf.forward(X, verbose_layers=True)
                    else:
                        clf.predict(X)
                    t2_test = time.time()
                    total += t2_test - t1_test
                mean_time = total / test_description["n_repetitions"]
                mean_time = max(mean_time, TIME_EPSILON)       
                test_results[(impl, index)] = mean_time
                test_indexer[(impl, *atuple)] = index 
                print(f"{index + 1}/{n_tests} DONE. [mean time: {mean_time} s]")
                print("-" * 172)
            t2_this_impl = time.time()            
            print(f"TESTING IMPLEMENTATION {impl} DONE. [time: {t2_this_impl - t1_this_impl} s]")
        if PICKLE_TEST_RESULTS:
            do_pickle(FOLDER_TESTS + TEST_ID + SUFFIX_TEST_BIN, [test_results, test_indexer])

    if not test_results:
        [test_results, test_indexer] = do_unpickle(FOLDER_TESTS + TEST_ID + SUFFIX_TEST_BIN)
        n_tests = len(test_description["kernel_size"]) * len(test_description["n_kernels"]) * len(test_description["n_channels"]) * len(test_description["img_side"])

    if DO_COMPARISONS:
        for impl in test_description["do_forward_impl"]:
            print("\n" + ("=" * 172))
            print(f"COMPARISONS FOR IMPLEMENTATION {impl}:")        
            for index, (kernel_size, n_kernels, n_channels, img_side) in enumerate(itertools.product(test_description["kernel_size"], test_description["n_kernels"], test_description["n_channels"], test_description["img_side"])):
                print(f"{index + 1}/{n_tests} [impl: {impl}, kernel_size: {kernel_size}, n_kernels: {n_kernels}, n_channels: {n_channels}, img_side: {img_side}]")
                t_ref = test_results[(impl, index)]
                print(f"T_REF: {t_ref} s [time for this implementation]")
                for other_impl in test_description["do_forward_impl"]:
                    if other_impl != impl:
                        t_other = test_results[(other_impl, index)]
                        print(f"COMPARISON AGAINST {other_impl} -> T_OTHER: {t_other} s, LOG_10(T_REF / T_OTHER): {np.log10(t_ref / t_other)}")
                print("-" * 172)
            print(f"COMPARISONS FOR IMPLEMENTATION {impl} DONE.")
            
    t2 = time.time()
    print(f"\nALL DONE FOR TEST WITH ID: {TEST_ID}. [time: {t2 - t1} s]")        
    if log_file:
        log_file.close()
            
    if DO_PLOTS:
        fig = plt.figure(1, figsize=PLOT_FIGSIZE)
        plt.title(f"{PLOT_TITLE} (BATCH SIZE: {test_description['m']})", fontsize=PLOT_FONTSIZE_TITLE)                
        
        arg_name = "img_side"
        arg_label = "IMAGE SIDE"                
        xs = test_description[arg_name]
        plt.xticks(xs)                
        
        # TEST 1 PLOTS
        # ys_descriptions = [            
        #     ({"do_forward_impl": "numpy_gemm", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "darkgray", "linestyle": "-"}),
        #     ({"do_forward_impl": "numba_jit_direct", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "cyan", "linestyle": "-"}),
        #     ({"do_forward_impl": "numba_jit_gemm", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "green", "linestyle": "-"}),
        #     ({"do_forward_impl": "cv2", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "orange", "linestyle": "-"}),
        #     ({"do_forward_impl": "numba_cuda_direct", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "black", "linestyle": "-"}),
        #     ({"do_forward_impl": "numba_cuda_tiles", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "blue", "linestyle": "-"}),
        #     ({"do_forward_impl": "numba_cuda_viafft", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "red", "linestyle": "-"}),
        #     ({"do_forward_impl": "keras", "kernel_size": 3, "n_kernels": 64, "n_channels": 64}, {"color": "pink", "linestyle": "-"})                        
        #     ]
        
        # TEST 2 PLOTS
        ys_descriptions = [
            ({"do_forward_impl": "numba_cuda_direct", "kernel_size": 11, "n_kernels": 64, "n_channels": 64}, {"color": "black", "linestyle": ":"}),
            ({"do_forward_impl": "numba_cuda_direct", "kernel_size": 15, "n_kernels": 64, "n_channels": 64}, {"color": "black", "linestyle": "--"}),
            ({"do_forward_impl": "numba_cuda_direct", "kernel_size": 19, "n_kernels": 64, "n_channels": 64}, {"color": "black", "linestyle": "-"}),
            ({"do_forward_impl": "numba_cuda_tiles", "kernel_size": 11, "n_kernels": 64, "n_channels": 64}, {"color": "blue", "linestyle": ":"}),
            ({"do_forward_impl": "numba_cuda_tiles", "kernel_size": 15, "n_kernels": 64, "n_channels": 64}, {"color": "blue", "linestyle": "--"}),
            ({"do_forward_impl": "numba_cuda_tiles", "kernel_size": 19, "n_kernels": 64, "n_channels": 64}, {"color": "blue", "linestyle": "-"}),
            ({"do_forward_impl": "numba_cuda_viafft", "kernel_size": 11, "n_kernels": 64, "n_channels": 64}, {"color": "red", "linestyle": ":"}),
            ({"do_forward_impl": "numba_cuda_viafft", "kernel_size": 15, "n_kernels": 64, "n_channels": 64}, {"color": "red", "linestyle": "--"}),
            ({"do_forward_impl": "numba_cuda_viafft", "kernel_size": 19, "n_kernels": 64, "n_channels": 64}, {"color": "red", "linestyle": "-"})                
            ]
        
        params_names = ["do_forward_impl", "kernel_size", "n_kernels", "n_channels", "img_side"]      
        for (params_dict, style) in ys_descriptions:
            params_values = []
            arg_i = None
            for i, pn in enumerate(params_names):
                if pn in params_dict:
                    params_values.append(params_dict[pn])
                else:
                    params_values.append(None)
                    arg_i = i
            ys = []
            for x in xs:
                params_values[arg_i] = x
                index = test_indexer[tuple(params_values)]
                ys.append(test_results[(params_dict["do_forward_impl"], index)])
            params_values_copy = params_values.copy()
            params_values_copy.pop(arg_i)
            plt.plot(xs, ys, label=str(params_values_copy), **{**style, **{"marker": "o", "markersize": PLOT_MARKERSIZE}})                    
        plt.xlabel(arg_label, fontsize=PLOT_FONTSIZE_AXES)            
        plt.ylabel("TIME [s]", fontsize=PLOT_FONTSIZE_AXES)
        plt.yscale("log")    
        plt.legend(loc=PLOT_LEGEND_LOC, prop={"size": PLOT_FONTSIZE_LEGEND}, handlelength=PLOT_LEGEND_HANDLELENGTH)        
        plt.grid(color=PLOT_GRID_COLOR, zorder=0, dashes=PLOT_GRID_DASHES)                  
        plt.show()
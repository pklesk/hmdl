import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import itertools
import time
import cv2
from numba import jit, cuda
from numba import void, int8, int16, int32, float32, complex64, boolean
import math
import cmath
from numba.core.errors import NumbaPerformanceWarning
import warnings
        
warnings.simplefilter("ignore", category=NumbaPerformanceWarning) 

__version__ = "1.0.0"

MAX_EXP_ARG = 0.5 * np.log(np.finfo(np.float32).max)
MIN_NON_ZERO = 1e-7
MAX_IMG_SIDE = 256
MAX_N_CHANNELS = 256
MAX_N_KERNELS = 256
CUDA_MAX_MEMORY_PER_CALL = 8 * 1024**2 # suitable (based on experiments) for: NVIDIA Quadro M4000M
#CUDA_MAX_MEMORY_PER_CALL = 256 * 1024**2 # suitable (based on experiments) for: GRID A100-7-40C

DEBUG_VERBOSE_CONV = False
DEBUG_VERBOSE_MAXPOOL = False
DEBUG_VERBOSE_ACTIVATION = False

def to_one_hot(y, n_classes):
    m = y.shape[0]
    y_one_hot = np.zeros((m, n_classes), dtype=np.float32)
    y_one_hot[np.arange(m), y] = 1.0
    return y_one_hot

def prepare_call_ranges(m, n_calls_min, power_two_sizes=False):
    if n_calls_min > m:                
        print(f"[warning: wanted n_calls_min = {n_calls_min} greater than m = {m} in prepare_call_ranges(...); hence, setting n_calls_min to m]")
        n_calls_min = m
    if not power_two_sizes:
        n_calls = n_calls_min
        call_size = m // n_calls
        call_ranges = call_size * np.ones(n_calls, dtype=np.int32)        
        call_ranges[:m % n_calls] += 1        
    else:
        call_size = 2**int(np.log2(m // n_calls_min))
        n_calls = int(np.ceil(m / call_size))
        call_ranges = call_size * np.ones(n_calls, dtype=np.int32)
        call_ranges[-1] = m % call_size
    call_ranges = np.r_[0, np.cumsum(call_ranges)]        
    return n_calls, call_ranges

@cuda.jit(device=True)
def reverse_bits(index, n_bits):
    result = int16(0)
    for _ in range(n_bits):
        result <<= 1
        result |= int(index & 1)
        index >>= 1
    return result


class Layer():

    def __init__(self, name=None, input_shape=None, activation=None, tunable=True, do_forward_impl=None, do_backward_impl=None, do_backward_output_impl=None):
        self.name_ = name
        self.input_shape_ = input_shape
        self.tunable_ = tunable
        self.activation_name_ = activation
        self.activation_ = activation                
        if self.activation_ is not None:
            self.activation_ =  getattr(self, activation)
            self.activation_d_ = getattr(self, activation + "_d")
        self.do_forward_impl_ = do_forward_impl
        self.do_backward_impl_ = do_backward_impl
        self.do_backward_output_impl_ = do_backward_output_impl                
        self.initial_ = False
        self.output_shape_ = None
        self.input_ = None
        self.weights_ = []
        self.l1_penalties_ = []
        self.l2_penalties_ = []
        self.adam_m_ = []
        self.adam_v_ = []
        self.output_ = None
        self.delta_backward_input_ = None
        self.delta_ = None # (own derivative or 1) * delta_backward_input
        self.gradient_ = [] # must correspond in shape to self.weights_
        self.prev_correction_ = [] # must correspond in shape to self.weights_
        self.delta_backward_output_ = None
        self.do_forward_function_ = None
        self.do_backward_function_ = None
        self.do_backward_output_function_ = None
        self.cuda_available_ = cuda.is_available()
        self.cuda_max_threads_per_block_ = cuda.get_current_device().MAX_THREADS_PER_BLOCK if self.cuda_available_ else None
        self.cuda_n_streams_ = cuda.get_current_device().ASYNC_ENGINE_COUNT if self.cuda_available_ else None
        self.shortname_default_prefix_ = None        
    
    def __compile__(self, layer_prev):
        if self.input_shape_ is None and layer_prev is not None: 
            self.input_shape_ = layer_prev.output_shape_
    
    def __setup_forward_backward_impls__(self, do_forward_impl_default, do_backward_impl_default, do_backward_output_impl_default):
        if self.do_forward_impl_ is None:
            self.do_forward_impl_ = do_forward_impl_default            
        if self.do_forward_impl_ is not None:
            self.do_forward_function_ = getattr(self, "do_forward_" + self.do_forward_impl_) 
        if self.do_backward_impl_ is None:
            self.do_backward_impl_ = do_backward_impl_default
        if self.do_backward_impl_ is not None:
            self.do_backward_function_ = getattr(self, "do_backward_" + self.do_backward_impl_)
        if self.do_backward_output_impl_ is None:
            self.do_backward_output_impl_ = do_backward_output_impl_default
        if self.do_backward_output_impl_ is not None: 
            self.do_backward_output_function_ = getattr(self, "do_backward_output_" + self.do_backward_output_impl_)            

    @staticmethod                
    def weights_conv_glorot_uniform(kernel_size, n_channels, n_kernels, activation_relu=False):
        return (np.random.rand(kernel_size, kernel_size, n_channels, n_kernels).astype(np.float32) * 2.0 - 1.0) * np.sqrt(6.0 / (kernel_size**2 * (n_channels + n_kernels))) * np.sqrt(1.0 + activation_relu)
    
    @staticmethod
    def weights_conv_glorot_normal(kernel_size, n_channels, n_kernels, activation_relu=False):
        return np.random.randn(kernel_size, kernel_size, n_channels, n_kernels).astype(np.float32) * np.sqrt(2.0 / (kernel_size**2 * (n_channels + n_kernels))) * np.sqrt(1.0 + activation_relu)

    @staticmethod                
    def weights_conv_he_uniform(kernel_size, n_channels, n_kernels):
        return (np.random.rand(kernel_size, kernel_size, n_channels, n_kernels).astype(np.float32) * 2.0 - 1.0) * np.sqrt(6.0 / (kernel_size**2 * n_channels))
    
    @staticmethod
    def weights_conv_he_normal(kernel_size, n_channels, n_kernels, activation_relu=False):
        return np.random.randn(kernel_size, kernel_size, n_channels, n_kernels).astype(np.float32) * np.sqrt(2.0 / (kernel_size**2 * n_channels))

    @staticmethod
    def weights_glorot_uniform(n_inputs, n_outputs, activation_relu=False):
        return (np.random.rand(n_outputs, n_inputs).astype(np.float32) * 2.0 - 1.0) * np.sqrt(6.0 / (n_inputs + n_outputs)) * np.sqrt(1.0 + activation_relu)
    
    @staticmethod
    def weights_glorot_normal(n_inputs, n_outputs, activation_relu=False):
        return np.random.randn(n_outputs, n_inputs).astype(np.float32) * np.sqrt(2.0 / (n_inputs + n_outputs)) * np.sqrt(1.0 + activation_relu)
                                                                                                
    @staticmethod
    def weights_he_uniform(n_inputs, n_outputs):
        return (np.random.rand(n_outputs, n_inputs).astype(np.float32) * 2.0 - 1.0) * np.sqrt(6.0 / n_inputs)                                                                                                
                                                                                                                                                                                    
    @staticmethod
    def weights_he_normal(n_inputs, n_outputs):
        return np.random.randn(n_outputs, n_inputs).astype(np.float32) * np.sqrt(2.0 / n_inputs)    
    
    def sigmoid(self):         
        indexes_gz = self.output_ > 0.0
        indexes_leqz = self.output_ <= 0.0
        self.output_[indexes_gz] =  1.0 / (1.0 + np.exp(-self.output_[indexes_gz]))
        self.output_[indexes_leqz] =  np.exp(self.output_[indexes_leqz]) / (1.0 + np.exp(self.output_[indexes_leqz]))
    
    def sigmoid_d(self):
        return self.output_ * (1.0 - self.output_)
        
    def softmax(self):            
        self.output_ = np.clip(self.output_, -np.inf, MAX_EXP_ARG) 
        self.output_ = np.exp(self.output_)
        exps_sums = np.array([np.sum(self.output_, axis=1)]).T
        self.output_ /= np.tile(exps_sums, (1, self.output_.shape[1]))            
        
    def softmax_d(self):        
        d = np.empty((self.output_.shape[0], self.output_.shape[1], self.output_.shape[1]), dtype=np.float32)
        for i in range(self.output_.shape[0]):
            d[i] = np.diag(self.output_[i]) - np.reshape(self.output_[i], (self.output_.shape[1], 1)).dot(np.reshape(self.output_[i], (1, self.output_.shape[1])))
        return d
        
    def relu(self):
        self.output_ *= self.output_ > 0.0        
    
    def relu_d(self):
        return self.output_ > 0.0
                
    def forward(self, the_input):
        if self.input_shape_ is not None and the_input.shape[1:] != self.input_shape_:
            raise Exception("Wrong shape of passed input.")
        self.input_ = the_input        
        self.do_forward()        
        if self.activation_ is not None:
            t1 = time.time()
            self.activation_()
            t2 = time.time()
            if DEBUG_VERBOSE_ACTIVATION:
                print(f"[activation for {self.name_} {self}: {t2 - t1} s]")        

    def backward(self, delta_backward_input):
        if self.output_shape_ is not None and delta_backward_input.shape[1:] != self.output_shape_:
            raise Exception("Wrong shape of passed delta backward input.")
        self.delta_backward_input_ = delta_backward_input
        self.do_backward()        
        if not self.initial_:
            self.do_backward_output()
                
    def do_forward(self):
        pass
        
    def do_backward(self):
        pass
    
    def do_backward_output(self):
        pass

        
class Conv2D(Layer):
              
    def __init__(self, name=None, input_shape=None, activation=None, tunable=True, do_forward_impl=None, do_backward_impl=None, do_backward_output_impl=None, \
                 kernel_size=3, n_kernels=1, \
                 l1_penalty_kernel=0.0, l1_penalty_bias=0.0, l2_penalty_kernel=0.0, l2_penalty_bias=0.0):        
        super().__init__(name, input_shape, activation, tunable, do_forward_impl, do_backward_impl, do_backward_output_impl)
        self.shortname_default_prefix_ = "c"        
        self.n_kernels_ = n_kernels
        self.kernel_size_ = kernel_size
        if self.n_kernels_ > MAX_N_KERNELS:
            raise Exception(f"Wanted number of kernels exceeds allowed limit of {MAX_N_KERNELS}.")
        if self.activation_name_ == "softmax":
            raise Exception("Softmax activation not allowed for Conv2D layer.")
        self.height_ = None 
        self.width_ = None
        self.n_channels_ = None
        self.dev_weights_0_complex_ = None # for FFT purposes (if turned on; memorized result of FFT2 on weights from do_forward pass to be used in do_backward)
        if l1_penalty_kernel > 0.0 or l1_penalty_bias > 0.0:
            self.l1_penalties_.append(max(l1_penalty_kernel, 0.0))
            self.l1_penalties_.append(max(l1_penalty_bias, 0.0))
        if l2_penalty_kernel > 0.0 or l2_penalty_bias > 0.0:                
            self.l2_penalties_.append(max(l2_penalty_kernel, 0.0))
            self.l2_penalties_.append(max(l2_penalty_bias, 0.0))
                
    def __compile__(self, layer_prev):
        super().__compile__(layer_prev)
        self.height_, self.width_, self.n_channels_ = self.input_shape_
        if self.height_ > MAX_IMG_SIDE or self.width_ > MAX_IMG_SIDE:
            raise Exception(f"Image height or width exceeds allowed limit of {MAX_IMG_SIDE}.")
        if self.n_channels_ > MAX_N_CHANNELS:
            raise Exception(f"Wanted number of channels exceeds allowed limit of {MAX_N_CHANNELS}.")
        if self.activation_name_ == "relu":
            self.weights_.append(Layer.weights_conv_he_uniform(self.kernel_size_, self.n_channels_, self.n_kernels_))
        else:
            self.weights_.append(Layer.weights_conv_glorot_uniform(self.kernel_size_, self.n_channels_, self.n_kernels_))
        self.weights_.append(np.zeros(self.n_kernels_, dtype=np.float32))
        self.adam_m_.append(np.zeros(self.weights_[0].shape, dtype=np.float32))
        self.adam_m_.append(np.zeros(self.weights_[1].shape, dtype=np.float32))
        self.adam_v_.append(np.zeros(self.weights_[0].shape, dtype=np.float32))
        self.adam_v_.append(np.zeros(self.weights_[1].shape, dtype=np.float32))       
        self.output_shape_ = (self.height_, self.width_, self.n_kernels_)
        self.gradient_.append(np.zeros(self.weights_[0].shape, dtype=np.float32))
        self.gradient_.append(np.zeros(self.weights_[1].shape, dtype=np.float32))
        self.prev_correction_.append(np.zeros(self.weights_[0].shape, dtype=np.float32))
        self.prev_correction_.append(np.zeros(self.weights_[1].shape, dtype=np.float32))
        pif, pib, pibo = self.rules_for_promising_impls()
        self.__setup_forward_backward_impls__(pif, pib, pibo)         
    
    def rules_for_promising_impls(self):
        impl_forward = None
        impl_backward = None
        impl_backward_output = None
        if self.cuda_available_:
            if self.kernel_size_**2 * self.n_channels_ * self.height_ * self.width_ >= 15**2 * 16 * 32**2: # fft looks promising for forward and backward_output
                impl_forward = "numba_cuda_viafft"                
            else:
                if self.kernel_size_**2 * self.height_ * self.width_ / self.n_channels_ >= 7**2 * 32**2 / 32: # tiles look promising for forward and backward_output
                    impl_forward = "numba_cuda_tiles"
                else: 
                    impl_forward = "numba_cuda_direct"
            if self.kernel_size_**2 * self.n_kernels_ * self.height_ * self.width_ >= 15**2 * 16 * 32**2: # fft looks promising for forward and backward_output
                impl_backward_output = "numba_cuda_viafft"                
            else:
                if self.kernel_size_**2 * self.height_ * self.width_ / self.n_kernels_ >= 7**2 * 32**2 / 128: # tiles look promising for forward and backward_output
                    impl_backward_output = "numba_cuda_tiles"                    
                else: 
                    impl_backward_output = "numba_cuda_direct"                                        
            if self.height_ <= 64 and self.width_ <= 64 and self.kernel_size_**2 * self.height_ * self.width_ * self.n_channels_ * self.n_kernels_ >= 13**2 * 32**2 * 32 * 16: # tiles look promising for backward                            
                impl_backward = "numba_cuda_tiles"
            else:
                impl_backward = "numba_cuda_direct"
        else:
            impl_forward = "cv2"
            impl_backward = "numba_jit"
            impl_backward_output = "numba_jit"
        return impl_forward, impl_backward, impl_backward_output
            
    def do_forward(self):
        if self.output_ is None or self.output_.shape[0] != self.input_.shape[0]:            
            self.output_ = np.empty((self.input_.shape[0], self.height_, self.width_, self.n_kernels_), dtype=np.float32)                
        self.do_forward_function_()
                                                
    def do_forward_numba_cuda_direct(self):
        t0 = time.time()
        tpb =  self.cuda_max_threads_per_block_ // 2
        memory = self.input_.nbytes + self.weights_[0].nbytes + self.weights_[1].nbytes + self.output_.nbytes 
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_CONV:
            print(f"[do_forward_numba_cuda_direct for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())              
        with cuda.pinned(self.weights_[0], self.weights_[1], self.input_, self.output_):
            dev_self_weights_0 = cuda.to_device(self.weights_[0])
            dev_self_weights_1 = cuda.to_device(self.weights_[1])       
            for i in range(n_calls):     
                stream = streams[i % self.cuda_n_streams_]
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_input = self.input_[call_slice]    
                dev_sub_input = cuda.to_device(sub_input, stream=stream)
                dev_sub_output = cuda.device_array((sub_input.shape[0], self.height_, self.width_, self.n_kernels_), dtype=np.float32, stream=stream)
                bpg = (sub_input.shape[0] * self.height_ * self.width_ * self.n_kernels_ + tpb - 1) // tpb
                Conv2D.do_forward_numba_cuda_direct_job[bpg, tpb, stream](dev_sub_input, dev_self_weights_0, dev_self_weights_1, dev_sub_output)                
                dev_sub_output.copy_to_host(ary=self.output_[call_slice], stream=stream)                                        
                stream.synchronize() # remark: without this synchronization in this particular implementation (numba_cuda_direct), occasional errors related to freeing GPU memory occur 
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_forward_numba_cuda_direct for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")            
                        
    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], float32[:, :, :, :], float32[:], float32[:, :, :, :]))
    def do_forward_numba_cuda_direct_job(sub_input_, self_weights_0_, self_weights_1_, sub_output_):        
        m, height, width, n_channels = sub_input_.shape
        n_kernels = self_weights_0_.shape[-1]
        index = cuda.grid(1)
        wk = int(width * n_kernels)
        hwk = int(height * wk)
        if index >= m * hwk:
            return
        i = int(index / hwk)        
        index = int(index % hwk)
        j = int(index / wk)
        index = int(index % wk)
        k = int(index / n_kernels)
        f = int(index % n_kernels)
        kernel_size = self_weights_0_.shape[0]
        ksh = kernel_size >> 1
        temp = float32(0.0)
        for a in range(-ksh, ksh + 1):
            ja = int(j + a)            
            if ja < 0 or ja >= height:
                continue
            ksha = int(ksh + a)
            for b in range(-ksh, ksh + 1):
                kb = int(k + b)                
                if kb < 0 or kb >= width:
                    continue
                kshb = int(ksh + b)
                for c in range(n_channels):                             
                    temp += self_weights_0_[ksha, kshb, c, f] * sub_input_[i, ja, kb, c]
        sub_output_[i, j, k, f] = temp + self_weights_1_[f]        
            
    def do_forward_numba_cuda_tiles(self):
        t0 = time.time()
        tile_size = 16
        tpb =  (1, tile_size, tile_size)
        bpg_h = (self.height_ + tile_size - 1) // tile_size
        bpg_w = (self.width_ + tile_size - 1) // tile_size            
        memory = self.input_.nbytes + self.weights_[0].nbytes + self.weights_[1].nbytes + self.output_.nbytes 
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_CONV:
            print(f"[do_forward_numba_cuda_tiles for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())            
        with cuda.pinned(self.weights_[0], self.weights_[1], self.input_, self.output_):
            dev_self_weights_0 = cuda.to_device(self.weights_[0])
            dev_self_weights_1 = cuda.to_device(self.weights_[1])        
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]                               
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_input = self.input_[call_slice]    
                dev_sub_input = cuda.to_device(sub_input, stream=stream)
                dev_sub_output = cuda.device_array((sub_input.shape[0], self.height_, self.width_, self.n_kernels_), dtype=np.float32, stream=stream)                
                bpg = (sub_input.shape[0] * self.n_kernels_, bpg_h, bpg_w)
                Conv2D.do_forward_numba_cuda_tiles_job[bpg, tpb, stream](dev_sub_input, dev_self_weights_0, dev_self_weights_1, dev_sub_output)
                dev_sub_output.copy_to_host(ary=self.output_[call_slice], stream=stream)            
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_forward_numba_cuda_tiles for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")                                            

    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], float32[:, :, :, :], float32[:], float32[:, :, :, :]))
    def do_forward_numba_cuda_tiles_job(sub_input_, self_weights_0_, self_weights_1_, sub_output_):
        shared_weights = cuda.shared.array(shape=(25, 25), dtype=float32) # assumed maximum filter 25 x 25
        shared_input = cuda.shared.array(shape=(40, 40), dtype=float32) # assumed maximum: 16 x 16 tiles with a total pad of 24 due to assumed maximum filter 25 x 25         
        _, height, width, n_channels = sub_input_.shape
        n_kernels = self_weights_0_.shape[-1]
        i_f, j, k = cuda.grid(3)        
        i = int(i_f / n_kernels)        
        f = int(i_f % n_kernels)
        kernel_size = self_weights_0_.shape[0]
        kernel_size_2 = kernel_size * kernel_size
        ksh = kernel_size >> 1
        tile_size = cuda.blockDim.y
        tile_size_2 = tile_size * tile_size
        tile_size_padded = tile_size + kernel_size - 1
        tile_size_padded_2 = tile_size_padded * tile_size_padded
        ppt = int((tile_size_padded_2 + tile_size_2 - 1) / tile_size_2) # points to read per thread
        wpt = int((kernel_size_2 + tile_size_2 - 1) / tile_size_2) # weights to read per thread
        temp = float32(0.0)
        for c in range(n_channels):
            dest = cuda.threadIdx.y + cuda.threadIdx.z * tile_size            
            for _ in range(ppt): # possibly only one iteration (when more threads in block than input entries to read cooperatively)                              
                dest_j = int(dest / tile_size_padded)
                dest_k = int(dest % tile_size_padded)
                src_j = dest_j + cuda.blockIdx.y * tile_size - ksh
                src_k = dest_k + cuda.blockIdx.z * tile_size - ksh
                if dest_j < tile_size_padded:                
                    if src_j >= 0 and src_j < height and src_k >= 0 and src_k < width:            
                        shared_input[dest_j, dest_k] = sub_input_[i, src_j, src_k, c]                                                                    
                    else:
                        shared_input[dest_j, dest_k] = float32(0.0)                                            
                dest += tile_size_2            
            dest = cuda.threadIdx.y + cuda.threadIdx.z * tile_size            
            for _ in range(wpt): # possibly only one iteration (when more threads in block than weights to read cooperatively)                            
                a = int(dest / kernel_size)
                b = int(dest % kernel_size)
                if a < kernel_size:                
                    shared_weights[a, b] = self_weights_0_[a, b, c, f]                                        
                dest += tile_size_2
            cuda.syncthreads()
            for a in range(kernel_size):
                for b in range(kernel_size):                                                    
                    temp += shared_weights[a, b] * shared_input[cuda.threadIdx.y + a, cuda.threadIdx.z + b]                                        
            cuda.syncthreads()
        if j < height and k < width:
            sub_output_[i, j, k, f] = temp + self_weights_1_[f]
    
    def do_forward_numba_cuda_viafft(self):
        t0 = time.time()
        height_padded = self.height_ + self.kernel_size_ - 1
        height_padded = int(2**int(np.ceil(np.log2(height_padded))))
        width_padded = self.width_ + self.kernel_size_ - 1
        width_padded = int(2**int(np.ceil(np.log2(width_padded))))
        n_channels_2power_ceiled = int(2**int(np.ceil(np.log2(self.n_channels_))))                
        memory = (self.input_.nbytes + self.weights_[0].nbytes + self.weights_[1].nbytes + self.output_.nbytes)   
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))        
        if DEBUG_VERBOSE_CONV:
            print(f"[do_forward_numba_cuda_viafft for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())        
        with cuda.pinned(self.weights_[0], self.weights_[1], self.input_, self.output_):
            dev_self_weights_0 = cuda.to_device(self.weights_[0])
            dev_self_weights_1 = cuda.to_device(self.weights_[1])       
            dev_weights_0_complex = cuda.device_array((height_padded, width_padded, self.n_channels_, self.n_kernels_), dtype=np.complex64)
            bpg = (self.n_kernels_, self.n_channels_, height_padded)
            tpb = (width_padded)
            Conv2D.numba_cuda_job_r2c[bpg, tpb](dev_self_weights_0, 3, 2, 0, 1, 0, 1, True, dev_weights_0_complex) # preparing complex array on device with flipped and padded weights
            tpb_wanted = MAX_IMG_SIDE
            elements_pb = tpb_wanted // width_padded
            elements_blocks = (self.n_kernels_ + elements_pb - 1) // elements_pb
            bpg = (elements_blocks, self.n_channels_, height_padded)
            tpb = (tpb_wanted)            
            Conv2D.numba_cuda_job_fft[bpg, tpb](dev_weights_0_complex, 1.0, 1, 3, 2, 0, 1, elements_pb) # FFT along rows of dev_weights_0_complex
            tpb_wanted = MAX_IMG_SIDE
            elements_pb = tpb_wanted // height_padded
            elements_blocks = (self.n_kernels_ + elements_pb - 1) // elements_pb
            bpg = (elements_blocks, self.n_channels_, width_padded)
            tpb = (tpb_wanted)                                    
            Conv2D.numba_cuda_job_fft[bpg, tpb](dev_weights_0_complex, 1.0, 1, 3, 2, 1, 0, elements_pb) # FFT along columns of dev_weights_0_complex
            cuda.synchronize()             
            self.dev_weights_0_complex_ = dev_weights_0_complex
            for i in range(n_calls):     
                stream = streams[i % self.cuda_n_streams_]
                call_slice = slice(call_ranges[i], call_ranges[i + 1])                
                sub_input = self.input_[call_slice]   
                dev_sub_input = cuda.to_device(sub_input, stream=stream)                
                dev_sub_input_complex = cuda.device_array((sub_input.shape[0], height_padded, width_padded, self.n_channels_), dtype=np.complex64, stream=stream)
                bpg = (sub_input.shape[0], self.n_channels_, height_padded)
                tpb = (width_padded)
                Conv2D.numba_cuda_job_r2c[bpg, tpb, stream](dev_sub_input, 0, 3, 1, 2, 1, 2, False, dev_sub_input_complex) # preparing complex array on device with sub_input
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // width_padded
                elements_blocks = (sub_input.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_channels_, height_padded)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_input_complex, 1.0, 1, 0, 3, 1, 2, elements_pb) # FFT along rows of dev_sub_input_complex
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // height_padded
                elements_blocks = (sub_input.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_channels_, width_padded)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_input_complex, 1.0, 1, 0, 3, 2, 1, elements_pb) # FFT along columns of dev_sub_input_complex
                dev_sub_output_complex = cuda.device_array((sub_input.shape[0], height_padded, width_padded, self.n_kernels_), dtype=np.complex64, stream=stream)
                tpb_wanted = MAX_N_CHANNELS
                elements_pb = tpb_wanted // n_channels_2power_ceiled
                elements_blocks = (sub_input.shape[0] + elements_pb - 1) // elements_pb 
                bpg = (height_padded * width_padded, elements_blocks, self.n_kernels_)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_muladdffts_in_do_forward[bpg, tpb, stream](dev_weights_0_complex, dev_sub_input_complex, n_channels_2power_ceiled, elements_pb, dev_sub_output_complex) # for each kernel (filter): multiply FFTs of weights and data and add them along channels                                  
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // width_padded
                elements_blocks = (sub_input.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_kernels_, height_padded)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_output_complex, -1.0, width_padded, 0, 3, 1, 2, elements_pb) # IFFT along rows of dev_sub_output_complex
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // height_padded
                elements_blocks = (sub_input.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_kernels_, width_padded)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_output_complex, -1.0, height_padded, 0, 3, 2, 1, elements_pb) # IFFT along columns of dev_sub_output_complex
                dev_sub_output = cuda.device_array((sub_input.shape[0], self.height_, self.width_, self.n_kernels_), dtype=np.float32, stream=stream)
                bpg = (sub_input.shape[0], self.n_kernels_, height_padded)
                tpb = (width_padded)                
                Conv2D.numba_cuda_job_c2rsame_in_do_forward[bpg, tpb, stream](dev_sub_output_complex, dev_self_weights_1, 0, 3, 1, 2, 1, 2, 3, self.height_, self.width_, self.kernel_size_, dev_sub_output) # populating output array with real parts only (and "same" mode clip)
                dev_sub_output.copy_to_host(ary=self.output_[call_slice], stream=stream)
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_forward_numba_cuda_viafft for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")

    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], int8, int8, int8, int8, int8, int8, boolean, complex64[:, :, :, :]))
    def numba_cuda_job_r2c(src_, bx_index_, by_index_, bz_index_, tx_index_, height_index_, width_index_, flip_, dest_):
        indexer = cuda.local.array(4, dtype=int32)
        indexer[bx_index_] = cuda.blockIdx.x
        indexer[by_index_] = cuda.blockIdx.y 
        indexer[bz_index_] = cuda.blockIdx.z 
        indexer[tx_index_] = cuda.threadIdx.x
        index_dest = (indexer[0], indexer[1], indexer[2], indexer[3])
        value = complex64(0.0)
        src_shape = src_.shape
        if indexer[height_index_] < src_shape[height_index_] and indexer[width_index_] < src_shape[width_index_]:                        
            if flip_:                                                             
                indexer[height_index_] = src_shape[height_index_] - 1 - indexer[height_index_]
                indexer[width_index_] = src_shape[width_index_] - 1 - indexer[width_index_] 
            index_src = (indexer[0], indexer[1], indexer[2], indexer[3])                                 
            value = complex64(src_[index_src])
        dest_[index_dest] = value
            
    @staticmethod                  
    @cuda.jit(void(complex64[:, :, :, :], float32, int32, int8, int8, int8, int8, int32))
    def numba_cuda_job_fft(f_, sgn_, normalizer_, bx_index_, by_index_, bz_index_, tx_index_, elements_pb_):  
        shared_f = cuda.shared.array(512, dtype=complex64)
        indexer = cuda.local.array(4, dtype=int32)
        m = f_.shape[bx_index_]
        N = f_.shape[tx_index_]
        eb = cuda.blockIdx.x
        e_t = cuda.threadIdx.x
        t = int(e_t / elements_pb_)
        e_in_this_block = int(e_t % elements_pb_)        
        e = eb * elements_pb_ + e_in_this_block        
        if e >= m:
            return 
        t_shift = e_in_this_block * N
        indexer[bx_index_] = e
        indexer[by_index_] = cuda.blockIdx.y 
        indexer[bz_index_] = cuda.blockIdx.z 
        indexer[tx_index_] = t
        index = (indexer[0], indexer[1], indexer[2], indexer[3])                
        n_bits = int(round(math.log2(N)))
        t_rev = reverse_bits(t, n_bits)
        shared_f[t_rev + t_shift] = f_[index]
        cuda.syncthreads()
        stride = 1
        sgn_m2pi = sgn_ * (-2 * cmath.pi)
        while stride < N:
            stride_half = stride
            stride <<= 1    
            arg = sgn_m2pi / stride 
            u = int(t / stride) * stride
            v = int(t % stride)
            if v < stride_half:
                uv = t_shift + u + v
                a = shared_f[uv]
                omega_stride_v = cmath.exp(complex64(v * arg * 1.0j))                
                b = omega_stride_v * shared_f[uv + stride_half] 
                shared_f[uv] = a + b
                shared_f[uv + stride_half] = a - b
            cuda.syncthreads()
        f_[index] = shared_f[t + t_shift] / float32(normalizer_)

    @staticmethod
    @cuda.jit(void(complex64[:, :, :, :], complex64[:, :, :, :], int32, int32, complex64[:, :, :, :]))
    def numba_cuda_job_muladdffts_in_do_forward(weights_0_complex_, sub_input_complex_, n_channels_2power_ceiled, elements_pb_, sub_output_complex_):
        shared_muladds = cuda.shared.array(512, dtype=complex64)
        m, _, width_padded, n_channels = sub_input_complex_.shape
        j_k = cuda.blockIdx.x
        eb = cuda.blockIdx.y
        f = cuda.blockIdx.z
        e_c = cuda.threadIdx.x
        c = int(e_c / elements_pb_)
        e_in_this_block = int(e_c % elements_pb_)
        i = eb * elements_pb_ + e_in_this_block
        if i >= m:
            return 
        c_shift = e_in_this_block * n_channels_2power_ceiled
        j = int(j_k / width_padded)
        k = int(j_k % width_padded)
        shared_muladds[c + c_shift] = weights_0_complex_[j, k, c, f] * sub_input_complex_[i, j, k, c] if c < n_channels else complex64(0.0)
        cuda.syncthreads()
        stride = n_channels_2power_ceiled >> 1 # half of no. of channels
        while stride > 0: # sum-reduction pattern
            if c < stride:
                shared_muladds[c + c_shift] += shared_muladds[c + stride + c_shift]
            cuda.syncthreads()
            stride >>= 1   
        if c == 0:
            sub_output_complex_[i, j, k, f] = shared_muladds[0 + c_shift]

    @staticmethod
    @cuda.jit(void(complex64[:, :, :, :], float32[:], int8, int8, int8, int8, int8, int8, int8, int32, int32, int32, float32[:, :, :, :]))
    def numba_cuda_job_c2rsame_in_do_forward(src_, self_weights_1_, bx_index_, by_index_, bz_index_, tx_index_, height_index_, width_index_, kernel_index_, self_height_, self_width_, self_kernel_size_, dest_):
        indexer = cuda.local.array(4, dtype=int32)
        indexer[bx_index_] = cuda.blockIdx.x
        indexer[by_index_] = cuda.blockIdx.y 
        indexer[bz_index_] = cuda.blockIdx.z 
        indexer[tx_index_] = cuda.threadIdx.x
        index_dest = (indexer[0], indexer[1], indexer[2], indexer[3])
        ksh = self_kernel_size_ >> 1
        if indexer[height_index_] < self_height_ and indexer[width_index_] < self_width_:
            indexer[height_index_] += ksh
            indexer[width_index_] += ksh                     
            index_src = (indexer[0], indexer[1], indexer[2], indexer[3])                                             
            dest_[index_dest] = src_[index_src].real + self_weights_1_[indexer[kernel_index_]]

    def do_forward_numpy_gemm(self):
        ksh = self.kernel_size_ // 2
        kssc = self.kernel_size_**2 * self.n_channels_
        input_padded = np.zeros((self.input_.shape[0], self.height_ + 2 * ksh, self.width_ + 2 * ksh, self.n_channels_), dtype=np.float32)
        input_padded[:, ksh : -ksh, ksh : -ksh, :] = self.input_
        input_image_size = self.height_ * self.width_
        column_tiled_inputs = np.empty((kssc, self.input_.shape[0] * input_image_size), dtype=np.float32)                
        index = 0
        for i in range(self.input_.shape[0]):                             
            for j in range(self.height_):
                for k in range(self.width_):
                    single_input_padded = input_padded[i, j : j + self.kernel_size_, k : k + self.kernel_size_, :]
                    column_tiled_inputs[:, index] = np.reshape(single_input_padded, kssc)
                    index += 1
        row_weights_0 = np.empty((self.n_kernels_, kssc), dtype=np.float32) 
        for f in range(self.n_kernels_):
            row_weights_0[f] = np.reshape(self.weights_[0][:, :, :, f], (1, kssc))
        temp_output = row_weights_0.dot(column_tiled_inputs)
        for f in range(self.n_kernels_):            
            self.output_[:, :, :, f] = np.reshape(temp_output[f, :], (self.input_.shape[0], self.height_, self.width_)) + self.weights_[1][f]

    def do_forward_cv2(self):
        convolutions = np.zeros((self.height_, self.width_, self.n_channels_), dtype=np.float32)
        for i in range(self.input_.shape[0]):
            for f in range(self.n_kernels_):                                             
                for c in range(self.n_channels_):
                    convolutions[:, :, c] = cv2.filter2D(self.input_[i, :, :, c], -1, self.weights_[0][:, :, c, f], borderType=cv2.BORDER_CONSTANT)
                self.output_[i, :, :, f] = convolutions.sum(axis=2) + self.weights_[1][f]
                    
    def do_forward_numba_jit_direct(self):
        Conv2D.do_forward_numba_jit_direct_job(self.input_, self.height_, self.width_, self.n_channels_, self.n_kernels_, self.kernel_size_, self.weights_[0], self.weights_[1], self.output_)
    
    @staticmethod
    @jit(void(float32[:, :, :, :], int32, int32, int32, int32, int32, float32[:, :, :, :], float32[:], float32[:, :, :, :]), nopython=True, cache=True)
    def do_forward_numba_jit_direct_job(self_input_, self_height_, self_width_, self_n_channels_, self_n_kernels_, self_kernel_size_, self_weights_0_, self_weights_1_, self_output_):
        ksh = self_kernel_size_ // 2
        for i in range(self_input_.shape[0]):
            for f in range(self_n_kernels_):
                for j in range(self_height_):
                    for k in range(self_width_):
                        temp = 0.0
                        for a in range(-ksh, ksh + 1):
                            ja = j + a
                            if ja < 0 or ja >= self_height_:
                                continue
                            for b in range(-ksh, ksh + 1):
                                kb = k + b
                                if kb < 0 or kb >= self_width_:
                                    continue
                                for c in range(self_n_channels_):
                                    temp += self_weights_0_[ksh + a, ksh + b, c, f] * self_input_[i, ja, kb, c]                                                       
                        self_output_[i, j, k, f] = temp + self_weights_1_[f]

    def do_forward_numba_jit_gemm(self):
        Conv2D.do_forward_numba_jit_gemm_job(self.input_, self.height_, self.width_, self.n_channels_, self.n_kernels_, self.kernel_size_, self.weights_[0], self.weights_[1], self.output_)
    
    @staticmethod
    @jit(void(float32[:, :, :, :], int32, int32, int32, int32, int32, float32[:, :, :, :], float32[:], float32[:, :, :, :]), nopython=True, cache=True)
    def do_forward_numba_jit_gemm_job(self_input_, self_height_, self_width_, self_n_channels_, self_n_kernels_, self_kernel_size_, self_weights_0_, self_weights_1_, self_output_):
        ksh = self_kernel_size_ // 2
        kssc = self_kernel_size_**2 * self_n_channels_
        input_padded = np.zeros((self_input_.shape[0], self_height_ + 2 * ksh, self_width_ + 2 * ksh, self_n_channels_), dtype=np.float32)
        input_padded[:, ksh : -ksh, ksh : -ksh, :] = self_input_
        input_image_size = self_height_ * self_width_
        column_tiled_inputs = np.empty((kssc, self_input_.shape[0] * input_image_size), dtype=np.float32)                
        index = 0
        for i in range(self_input_.shape[0]):                             
            for j in range(self_height_):
                for k in range(self_width_):
                    single_input_padded = np.copy(input_padded[i, j : j + self_kernel_size_, k : k + self_kernel_size_, :])
                    column_tiled_inputs[:, index] = np.reshape(single_input_padded, kssc)
                    index += 1
        row_weights_0 = np.empty((self_n_kernels_, kssc), dtype=np.float32) 
        for f in range(self_n_kernels_):
            row_weights_0[f] = np.reshape(np.copy(self_weights_0_[:, :, :, f]), (1, kssc))
        temp_output = row_weights_0.dot(column_tiled_inputs)
        for f in range(self_n_kernels_):
            self_output_[:, :, :, f] = np.reshape(temp_output[f], (int32(self_input_.shape[0]), self_height_, self_width_)) + self_weights_1_[f]
        
    def do_backward(self):
        t0 = time.time()
        if self.activation_ is not None:                        
            self.delta_ = self.activation_d_() * self.delta_backward_input_          
        else:                    
            self.delta_ = self.delta_backward_input_
        t1 = time.time()        
        self.gradient_[0] = np.zeros((self.kernel_size_, self.kernel_size_, self.n_channels_, self.n_kernels_), dtype=np.float32)
        self.gradient_[1] = None
        self.do_backward_function_()
        if DEBUG_VERBOSE_ACTIVATION:
            print(f"[d_activation -> t1 - t0: {t1 - t0} s]")
    
    def do_backward_numba_cuda_direct(self):
        t0 = time.time()
        tpb = self.cuda_max_threads_per_block_ // 2
        kss = self.kernel_size_**2
        total_gradient_0_batch_size = self.input_.shape[0] * kss * self.n_channels_ * self.n_kernels_
        memory = self.input_.nbytes + self.delta_.nbytes + total_gradient_0_batch_size * np.dtype(np.float32).itemsize
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1 
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_numba_cuda_direct for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())        
        with cuda.pinned(self.input_, self.delta_):
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_input = self.input_[call_slice]
                sub_delta = self.delta_[call_slice]
                dev_sub_input = cuda.to_device(sub_input, stream=stream)
                dev_sub_delta = cuda.to_device(sub_delta, stream=stream)
                dsg0b_shape = (sub_input.shape[0], self.kernel_size_, self.kernel_size_, self.n_channels_, self.n_kernels_)
                dev_sub_gradient_0_batch = cuda.device_array(dsg0b_shape, dtype=np.float32, stream=stream)
                bpg = (np.prod(dsg0b_shape) + tpb - 1) // tpb
                Conv2D.do_backward_numba_cuda_direct_job[bpg, tpb, stream](dev_sub_input, dev_sub_delta, dev_sub_gradient_0_batch)                
                stream.synchronize()
                self.gradient_[0] += np.sum(dev_sub_gradient_0_batch.copy_to_host(stream=stream), axis=0)        
        t2 = time.time()
        self.gradient_[1] = np.sum(self.delta_, axis=(0, 1, 2))
        t3 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_numba_cuda_direct for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t3 - t2: {t3 - t2}, t3 - t0: {t3 - t0} s]")    
                                
    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :, :]))
    def do_backward_numba_cuda_direct_job(sub_input_, sub_delta_, sub_gradient_0_batch_):
        m, height, width, n_channels = sub_input_.shape
        n_kernels = sub_delta_.shape[-1]
        kernel_size = sub_gradient_0_batch_.shape[1]
        ksh = kernel_size >> 1
        index = cuda.grid(1)        
        ck = n_channels * n_kernels
        ksck = kernel_size * ck
        kssck = kernel_size * ksck
        if index >= m * kssck:
            return 
        i = int(index / kssck)
        index = int(index % kssck)
        a = int(index / ksck)
        a -= ksh
        index = int(index % ksck)
        b = int(index / ck)
        b -= ksh
        index = int(index % ck)
        l = int(index / n_kernels)
        s = int(index % n_kernels)
        temp = float32(0.0)
        for j in range(height):
            ja = int(j + a)
            if ja < 0 or ja >= height:
                continue
            for k in range(width):
                kb = int(k + b)
                if kb < 0 or kb >= width:
                    continue
                temp += sub_delta_[i, j, k, s] * sub_input_[i, ja, kb, l]
        sub_gradient_0_batch_[i, a + ksh, b + ksh, l, s] = temp
                                                                
    def do_backward_numba_cuda_tiles(self):
        t0 = time.time()
        tile_size = 16
        tpb =  (1, tile_size, tile_size)
        bpg_h = (self.kernel_size_ + tile_size - 1) // tile_size
        bpg_w = (self.kernel_size_ + tile_size - 1) // tile_size
        kss = self.kernel_size_**2            
        total_gradient_0_batch_size = self.input_.shape[0] * kss * self.n_channels_ * self.n_kernels_
        memory = self.input_.nbytes + self.delta_.nbytes + total_gradient_0_batch_size * np.dtype(np.float32).itemsize        
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_numba_cuda_tiles for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())
        with cuda.pinned(self.weights_[0], self.weights_[1], self.input_, self.gradient_[0]):        
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_input = self.input_[call_slice]
                sub_delta = self.delta_[call_slice]
                dev_sub_input = cuda.to_device(sub_input, stream=stream)
                dev_sub_delta = cuda.to_device(sub_delta, stream=stream)
                dsg0b_shape = (sub_input.shape[0], self.kernel_size_, self.kernel_size_, self.n_channels_, self.n_kernels_)
                dev_sub_gradient_0_batch = cuda.device_array(dsg0b_shape, dtype=np.float32, stream=stream)                
                bpg = (sub_input.shape[0] * self.n_channels_ * self.n_kernels_, bpg_h, bpg_w)
                Conv2D.do_backward_numba_cuda_tiles_job[bpg, tpb, stream](dev_sub_input, dev_sub_delta, dev_sub_gradient_0_batch)
                stream.synchronize()
                self.gradient_[0] += np.sum(dev_sub_gradient_0_batch.copy_to_host(stream=stream), axis=0)        
        t2 = time.time()
        self.gradient_[1] = np.sum(self.delta_, axis=(0, 1, 2))
        t3 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_numba_cuda_tiles for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t3 - t2: {t3 - t2}, t3 - t0: {t3 - t0} s]")                                    

    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :, :]))
    def do_backward_numba_cuda_tiles_job(sub_input_, sub_delta_, sub_gradient_0_batch_):
        shared_delta = cuda.shared.array(shape=(64, 64), dtype=float32) # for this impl of do_backward assumed maximum layer input image shape: 32 x 32
        shared_input = cuda.shared.array(shape=(80, 80), dtype=float32) # as above + pad of 16 due to maximum tiles 16 x 16            
        _, height, width, n_channels = sub_input_.shape
        n_kernels = sub_delta_.shape[-1]
        i_c_f, a, b = cuda.grid(3)
        ck = n_channels * n_kernels        
        i = int(i_c_f / ck)        
        c_f = int(i_c_f % ck)
        c = int(c_f / n_kernels)
        f = int(c_f % n_kernels)
        kernel_size = sub_gradient_0_batch_.shape[1]
        ksh = kernel_size >> 1
        tile_size = cuda.blockDim.y
        tile_size_2 = tile_size * tile_size
        pad = tile_size
        if kernel_size - 1 < pad:
            pad = kernel_size - 1
        height_padded = height + pad
        width_padded = width + pad
        a0 = cuda.blockIdx.y * tile_size
        b0 = cuda.blockIdx.z * tile_size
        ppt = int((height_padded * width_padded + tile_size_2 - 1) / tile_size_2) # input points to read per thread
        dpt = int((height * width + tile_size_2 - 1) / tile_size_2) # delta points to read per thread        
        dest = cuda.threadIdx.y + cuda.threadIdx.z * tile_size            
        for _ in range(ppt):                              
            dest_j = int(dest / width_padded)
            dest_k = int(dest % width_padded)
            src_j = dest_j -ksh + a0
            src_k = dest_k -ksh + b0
            if dest_j < height_padded:                
                if src_j >= 0 and src_j < height and src_k >= 0 and src_k < width:            
                    shared_input[dest_j, dest_k] = sub_input_[i, src_j, src_k, c]                                                                    
                else:
                    shared_input[dest_j, dest_k] = float32(0.0)                                            
            dest += tile_size_2
        dest = cuda.threadIdx.y + cuda.threadIdx.z * tile_size
        for _ in range(dpt):
            j = int(dest / width)
            k = int(dest % width)
            if j < height:               
                shared_delta[j, k] = sub_delta_[i, j, k, f]                                        
            dest += tile_size_2
        cuda.syncthreads()
        temp = float32(0.0)
        if a < kernel_size and b < kernel_size:
            for j in range(height):
                for k in range(width):                                                    
                    temp += shared_delta[j, k] * shared_input[j + a - a0, k + b - b0]
        if a < kernel_size and b < kernel_size:
            sub_gradient_0_batch_[i, a, b, c, f] = temp
                                                
    def do_backward_numba_jit(self):
        self.gradient_[0] = np.zeros((self.kernel_size_, self.kernel_size_, self.n_channels_, self.n_kernels_), dtype=np.float32)
        Conv2D.do_backward_numba_jit_job(self.input_, self.height_, self.width_, self.n_channels_, self.n_kernels_, self.kernel_size_, self.delta_, self.gradient_[0], self.gradient_[1])            
    
    @staticmethod
    @jit(void(float32[:, :, :, :], int32, int32, int32, int32, int32, float32[:, :, :, :], float32[:, :, :, :], float32[:]), nopython=True, cache=True)
    def do_backward_numba_jit_job(self_input_, self_height_, self_width_, self_n_channels_, self_n_kernels_, self_kernel_size_, self_delta_, self_gradient_0_, self_gradient_1_):
        ksh = self_kernel_size_ // 2
        kssc = self_kernel_size_**2 * self_n_channels_
        input_padded = np.zeros((self_input_.shape[0], self_height_ + 2 * ksh, self_width_ + 2 * ksh, self_n_channels_), dtype=np.float32)
        input_padded[:, ksh : -ksh, ksh : -ksh, :] = self_input_
        input_image_size = self_height_ * self_width_
        column_tiled_inputs = np.empty((input_image_size, kssc), dtype=np.float32)
        input_ones = np.ones((input_image_size, 1), dtype=np.float32)
        for i in range(self_input_.shape[0]):
            for s in range(self_n_kernels_):
                sub_delta_i_s = np.copy(self_delta_[i, :, :, s])
                row_delta_s = np.reshape(sub_delta_i_s, (1, input_image_size))
                index = 0
                for a in range(-ksh, ksh + 1):                    
                    for b in range(-ksh, ksh + 1):
                        for l in range(self_n_channels_):
                            single_input_padded_l = np.copy(input_padded[i, ksh + a : ksh + a + self_height_, ksh + b : ksh + b + self_width_, l])
                            column_tiled_inputs[:, index] = np.reshape(single_input_padded_l, input_image_size)
                            index += 1
                self_gradient_0_[:, :, :, s] += np.reshape(row_delta_s.dot(column_tiled_inputs), (self_kernel_size_, self_kernel_size_, self_n_channels_))
                self_gradient_1_[s] += row_delta_s.dot(input_ones)[0, 0]                
                        
    def do_backward_numpy(self):
        ksh = self.kernel_size_ // 2
        kssc = self.kernel_size_**2 * self.n_channels_
        input_padded = np.zeros((self.input_.shape[0], self.height_ + 2 * ksh, self.width_ + 2 * ksh, self.n_channels_), dtype=np.float32)
        input_padded[:, ksh : -ksh, ksh : -ksh, :] = self.input_
        input_image_size = self.height_ * self.width_
        column_tiled_inputs = np.empty((input_image_size, kssc), dtype=np.float32)
        input_ones = np.ones((input_image_size, 1), dtype=np.float32)
        for i in range(self.input_.shape[0]):
            for s in range(self.n_kernels_):
                row_delta_s = np.reshape(self.delta_[i, :, :, s], (1, input_image_size))                 
                for index, (a, b, l) in enumerate(itertools.product(range(-ksh, ksh + 1, 1), range(-ksh, ksh + 1, 1), range(self.n_channels_))):
                    single_input_padded_l = input_padded[i, ksh + a : ksh + a + self.height_, ksh + b : ksh + b + self.width_, l]
                    column_tiled_inputs[:, index] = np.reshape(single_input_padded_l, input_image_size)
                self.gradient_[0][:, :, :, s] += np.reshape(row_delta_s.dot(column_tiled_inputs), (self.kernel_size_, self.kernel_size_, self.n_channels_))
                self.gradient_[1][s] += row_delta_s.dot(input_ones)[0, 0]

    def do_backward_output(self):
        if self.delta_backward_output_ is None or self.delta_backward_output_.shape[0] != self.input_.shape[0]:            
            self.delta_backward_output_ = np.empty(self.input_.shape, dtype=np.float32)                    
        self.do_backward_output_function_()
                    
    def do_backward_output_numba_cuda_direct(self):
        t0 = time.time()
        tpb = self.cuda_max_threads_per_block_ // 2        
        memory = self.delta_.nbytes + self.weights_[0].nbytes + self.delta_backward_output_.nbytes
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0 
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_output_numba_cuda_direct for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())
        with cuda.pinned(self.delta_, self.weights_[0], self.delta_backward_output_):
            dev_self_weights_0 = cuda.to_device(self.weights_[0])                              
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]                  
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_delta = self.delta_[call_slice]
                dev_sub_delta = cuda.to_device(sub_delta, stream=stream)
                sub_output_shape = (sub_delta.shape[0], self.height_, self.width_, self.n_channels_)
                dev_sub_delta_backward_output = cuda.device_array(sub_output_shape, dtype=np.float32, stream=stream)
                bpg = (np.prod(sub_output_shape) + tpb - 1) // tpb            
                Conv2D.do_backward_output_numba_cuda_direct_job[bpg, tpb, stream](dev_sub_delta, dev_self_weights_0, dev_sub_delta_backward_output)            
                dev_sub_delta_backward_output.copy_to_host(ary=self.delta_backward_output_[call_slice], stream=stream)
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_output_numba_cuda_direct for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")          

    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :]))
    def do_backward_output_numba_cuda_direct_job(sub_delta_, self_weights_0_, sub_delta_backward_output_):
        m, height, width, n_channels = sub_delta_backward_output_.shape
        n_kernels = self_weights_0_.shape[-1]
        kernel_size = self_weights_0_.shape[0]
        ksh = kernel_size >> 1
        index = cuda.grid(1) 
        wc = width * n_channels
        hwc = height * wc
        if index >= m * hwc:
            return 
        i = int(index / hwc) 
        index = int(index % hwc)
        j = int(index / wc)
        index = int(index % wc)
        k = int(index / n_channels)        
        s = int(index % n_channels)
        temp = float32(0.0)        
        for c in range(-ksh, ksh + 1):
            jc = int(j - c)
            if jc < 0 or jc >= height:
                continue            
            for d in range(-ksh, ksh + 1):
                kd = int(k - d)
                if kd < 0 or kd >= width:
                    continue
                for t in range(n_kernels):                
                    temp += self_weights_0_[c + ksh, d + ksh, s, t] * sub_delta_[i, jc, kd, t]
        sub_delta_backward_output_[i, j, k, s] = temp

    def do_backward_output_numba_cuda_tiles(self):
        t0 = time.time()
        tile_size = 16
        tpb =  (1, tile_size, tile_size)
        bpg_h = (self.height_ + tile_size - 1) // tile_size
        bpg_w = (self.width_ + tile_size - 1) // tile_size            
        memory = self.delta_.nbytes + self.weights_[0].nbytes + self.delta_backward_output_.nbytes 
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.delta_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_output_numba_cuda_tiles for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())
        with cuda.pinned(self.weights_[0], self.delta_, self.delta_backward_output_):
            dev_self_weights_0 = cuda.to_device(self.weights_[0])   
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]                     
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_delta = self.delta_[call_slice]
                dev_sub_delta = cuda.to_device(sub_delta, stream=stream)
                dev_sub_delta_backward_output = cuda.device_array((sub_delta.shape[0], self.height_, self.width_, self.n_channels_), dtype=np.float32, stream=stream)
                bpg = (sub_delta.shape[0] * self.n_channels_, bpg_h, bpg_w)      
                Conv2D.do_backward_output_numba_cuda_tiles_job[bpg, tpb, stream](dev_sub_delta, dev_self_weights_0, dev_sub_delta_backward_output)                
                dev_sub_delta_backward_output.copy_to_host(ary=self.delta_backward_output_[call_slice], stream=stream)
            for stream in streams:
                stream.synchronize()
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_output_numba_cuda_tiles for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")                                         

    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :]))
    def do_backward_output_numba_cuda_tiles_job(sub_delta_, self_weights_0_, sub_backward_output_):
        shared_weights = cuda.shared.array(shape=(25, 25), dtype=float32) # assumed maximum filter 25 x 25
        shared_delta = cuda.shared.array(shape=(40, 40), dtype=float32) # assumed maximum: 16 x 16 tiles with a total pad of 24 due to assumed maximum filter 25 x 25         
        _, height, width, n_kernels = sub_delta_.shape
        n_channels = self_weights_0_.shape[-2]
        i_s, j, k = cuda.grid(3)
        i = int(i_s / n_channels)
        s = int(i_s % n_channels)
        kernel_size = self_weights_0_.shape[0]
        kernel_size_2 = kernel_size * kernel_size
        ksh = kernel_size >> 1
        tile_size = cuda.blockDim.y
        tile_size_2 = tile_size * tile_size
        tile_size_padded = tile_size + kernel_size - 1
        tile_size_padded_2 = tile_size_padded * tile_size_padded
        ppt = int((tile_size_padded_2 + tile_size_2 - 1) / tile_size_2) # points to read per thread
        wpt = int((kernel_size_2 + tile_size_2 - 1) / tile_size_2) # weights to read per thread
        temp = float32(0.0)
        for t in range(n_kernels):
            dest = cuda.threadIdx.y + cuda.threadIdx.z * tile_size
            for _ in range(ppt): # possibly only one iteration (when more threads in block than input entries to read cooperatively)                              
                dest_j = int(dest / tile_size_padded)
                dest_k = int(dest % tile_size_padded)
                src_j = dest_j + cuda.blockIdx.y * tile_size - ksh
                src_k = dest_k + cuda.blockIdx.z * tile_size - ksh
                if dest_j < tile_size_padded:
                    if src_j >= 0 and src_j < height and src_k >= 0 and src_k < width:            
                        shared_delta[dest_j, dest_k] = sub_delta_[i, src_j, src_k, t]                                                                    
                    else:
                        shared_delta[dest_j, dest_k] = float32(0.0)
                dest += tile_size_2
            dest = cuda.threadIdx.y + cuda.threadIdx.z * tile_size
            for _ in range(wpt): # possibly only one iteration (when more threads in block than weights to read cooperatively)                            
                c = int(dest / kernel_size)
                d = int(dest % kernel_size)
                if c < kernel_size:
                    shared_weights[c, d] = self_weights_0_[kernel_size - 1 - c, kernel_size - 1 - d, s, t] # weights flip here                                        
                dest += tile_size_2
            cuda.syncthreads()
            for c in range(kernel_size):
                for d in range(kernel_size):                                                    
                    temp += shared_weights[c, d] * shared_delta[cuda.threadIdx.y + c, cuda.threadIdx.z + d]                                        
            cuda.syncthreads()
        if j < height and k < width:
            sub_backward_output_[i, j, k, s] = temp

    def do_backward_output_numba_cuda_viafft(self):
        t0 = time.time()        
        height_padded = self.height_ + self.kernel_size_ - 1
        height_padded = int(2**int(np.ceil(np.log2(height_padded))))
        width_padded = self.width_ + self.kernel_size_ - 1
        width_padded = int(2**int(np.ceil(np.log2(width_padded))))
        n_kernels_2power_ceiled = int(2**int(np.ceil(np.log2(self.n_kernels_))))
        memory = (self.input_.nbytes + self.weights_[0].nbytes + self.output_.nbytes)   
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_output_numba_cuda_viafft for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")                 
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())
        t1 = time.time()   
        with cuda.pinned(self.delta_, self.delta_backward_output_):            
            if self.dev_weights_0_complex_ is None: # in case do_forward was not carried out via FFT
                dev_self_weights_0 = cuda.to_device(self.weights_[0])
                dev_self_weights_1 = cuda.to_device(self.weights_[1])       
                dev_weights_0_complex = cuda.device_array((height_padded, width_padded, self.n_channels_, self.n_kernels_), dtype=np.complex64)
                bpg = (self.n_kernels_, self.n_channels_, height_padded)
                tpb = (width_padded)
                Conv2D.numba_cuda_job_r2c[bpg, tpb](dev_self_weights_0, 3, 2, 0, 1, 0, 1, True, dev_weights_0_complex) # preparing complex array on device with flipped and padded weights
                cuda.synchronize()
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // width_padded
                elements_blocks = (self.n_kernels_ + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_channels_, height_padded)
                tpb = (tpb_wanted)            
                Conv2D.numba_cuda_job_fft[bpg, tpb](dev_weights_0_complex, 1.0, 1, 3, 2, 0, 1, elements_pb) # FFT along rows of dev_weights_0_complex             
                cuda.synchronize()
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // height_padded
                elements_blocks = (self.n_kernels_ + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_channels_, width_padded)
                tpb = (tpb_wanted)                                    
                Conv2D.numba_cuda_job_fft[bpg, tpb](dev_weights_0_complex, 1.0, 1, 3, 2, 1, 0, elements_pb) # FFT along columns of dev_weights_0_complex
                cuda.synchronize()
                self.dev_weights_0_complex_ = dev_weights_0_complex            
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]
                call_slice = slice(call_ranges[i], call_ranges[i + 1])                
                sub_delta = self.delta_[call_slice]
                dev_sub_delta = cuda.to_device(sub_delta, stream=stream)                
                dev_sub_delta_complex = cuda.device_array((sub_delta.shape[0], height_padded, width_padded, self.n_kernels_), dtype=np.complex64, stream=stream)
                bpg = (sub_delta.shape[0], self.n_kernels_, height_padded)
                tpb = (width_padded)
                Conv2D.numba_cuda_job_r2c[bpg, tpb, stream](dev_sub_delta, 0, 3, 1, 2, 1, 2, True, dev_sub_delta_complex) # preparing complex array on device with sub_delta
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // width_padded
                elements_blocks = (sub_delta.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_kernels_, height_padded)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_delta_complex, 1.0, 1, 0, 3, 1, 2, elements_pb) # FFT along rows within dev_sub_delta_complex
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // height_padded
                elements_blocks = (sub_delta.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_kernels_, width_padded)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_delta_complex, 1.0, 1, 0, 3, 2, 1, elements_pb) # FFT along columns within dev_sub_delta_complex
                dev_sub_delta_backward_output_complex = cuda.device_array((sub_delta.shape[0], height_padded, width_padded, self.n_channels_), dtype=np.complex64, stream=stream)
                tpb_wanted = MAX_N_KERNELS
                elements_pb = tpb_wanted // n_kernels_2power_ceiled
                elements_blocks = (sub_delta.shape[0] + elements_pb - 1) // elements_pb 
                bpg = (height_padded * width_padded, elements_blocks, self.n_channels_)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_muladdffts_in_do_backward_output[bpg, tpb, stream](self.dev_weights_0_complex_, dev_sub_delta_complex, n_kernels_2power_ceiled, elements_pb, dev_sub_delta_backward_output_complex)                                   
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // width_padded
                elements_blocks = (sub_delta.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_channels_, height_padded)
                tpb = (tpb_wanted)                
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_delta_backward_output_complex, -1.0, width_padded, 0, 3, 1, 2, elements_pb) # IFFT along rows within dev_sub_delta_backward_output_complex
                tpb_wanted = MAX_IMG_SIDE
                elements_pb = tpb_wanted // height_padded
                elements_blocks = (sub_delta.shape[0] + elements_pb - 1) // elements_pb
                bpg = (elements_blocks, self.n_channels_, width_padded)
                tpb = (tpb_wanted)
                Conv2D.numba_cuda_job_fft[bpg, tpb, stream](dev_sub_delta_backward_output_complex, -1.0, height_padded, 0, 3, 2, 1, elements_pb) # IFFT along columns within dev_sub_delta_backward_output_complex
                sub_output_shape = (sub_delta.shape[0], self.height_, self.width_, self.n_channels_)
                dev_sub_delta_backward_output = cuda.device_array(sub_output_shape, dtype=np.float32, stream=stream)                        
                bpg = (sub_delta.shape[0], self.n_channels_, height_padded)
                tpb = (width_padded)                             
                Conv2D.numba_cuda_job_c2rsame_in_do_backward_output[bpg, tpb, stream](dev_sub_delta_backward_output_complex, 0, 3, 1, 2, 1, 2, self.height_, self.width_, self.kernel_size_, dev_sub_delta_backward_output)
                dev_sub_delta_backward_output.copy_to_host(ary=self.delta_backward_output_[call_slice], stream=stream)             
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_CONV:
            print(f"[do_backward_output_numba_cuda_viafft for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")
        
    @staticmethod
    @cuda.jit(void(complex64[:, :, :, :], complex64[:, :, :, :], int32, int32, complex64[:, :, :, :]))
    def numba_cuda_job_muladdffts_in_do_backward_output(weights_0_complex_, sub_delta_complex_, n_kernels_2power_ceiled, elements_pb_, sub_delta_backward_output_complex_):
        shared_muladds = cuda.shared.array(512, dtype=complex64)
        m, _, width_padded, n_kernels = sub_delta_complex_.shape
        j_k = cuda.blockIdx.x
        eb = cuda.blockIdx.y
        c = cuda.blockIdx.z
        e_f = cuda.threadIdx.x
        f = int(e_f / elements_pb_)
        e_in_this_block = int(e_f % elements_pb_)
        i = eb * elements_pb_ + e_in_this_block
        if i >= m:
            return 
        f_shift = e_in_this_block * n_kernels_2power_ceiled
        j = int(j_k / width_padded)
        k = int(j_k % width_padded)
        shared_muladds[f + f_shift] = weights_0_complex_[j, k, c, f] * sub_delta_complex_[i, j, k, f] if f < n_kernels else complex64(0.0)
        cuda.syncthreads()
        stride = n_kernels_2power_ceiled >> 1 # half of no. of threads
        while stride > 0: # sum-reduction pattern
            if f < stride:
                shared_muladds[f + f_shift] += shared_muladds[f + stride + f_shift]
            cuda.syncthreads()
            stride >>= 1   
        if f == 0:
            sub_delta_backward_output_complex_[i, j, k, c] = shared_muladds[0 + f_shift]
            
    @staticmethod
    @cuda.jit(void(complex64[:, :, :, :], int8, int8, int8, int8, int8, int8, int32, int32, int32, float32[:, :, :, :]))
    def numba_cuda_job_c2rsame_in_do_backward_output(src_, bx_index_, by_index_, bz_index_, tx_index_, height_index_, width_index_, self_height_, self_width_, self_kernel_size_, dest_):
        indexer = cuda.local.array(4, dtype=int32)
        indexer[bx_index_] = cuda.blockIdx.x
        indexer[by_index_] = cuda.blockIdx.y 
        indexer[bz_index_] = cuda.blockIdx.z 
        indexer[tx_index_] = cuda.threadIdx.x
        index_dest = (indexer[0], indexer[1], indexer[2], indexer[3])
        ksh = self_kernel_size_ >> 1
        if indexer[height_index_] < self_height_ and indexer[width_index_] < self_width_:
            indexer[height_index_] = self_height_+ self_kernel_size_ - 2 - (indexer[height_index_] + ksh)
            indexer[width_index_] = self_width_ + self_kernel_size_ - 2 - (indexer[width_index_] + ksh)        
            index_src = (indexer[0], indexer[1], indexer[2], indexer[3])                                             
            dest_[index_dest] = src_[index_src].real                  
                    
    def do_backward_output_numba_jit(self):
        Conv2D.do_backward_output_numba_jit_job(self.height_, self.width_, self.n_channels_, self.n_kernels_, self.kernel_size_, self.delta_, self.weights_[0], self.delta_backward_output_)
    
    @staticmethod
    @jit(void(int32, int32, int32, int32, int32, float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :]), nopython=True, cache=True)
    def do_backward_output_numba_jit_job(self_height_, self_width_, self_n_channels_, self_n_kernels_, self_kernel_size_, self_delta_, self_weights_0_, self_delta_backward_output_):
        ksh = self_kernel_size_ // 2
        kssk = self_kernel_size_**2 * self_n_kernels_
        m = self_delta_backward_output_.shape[0]
        delta_padded = np.zeros((m, self_height_ + 2 * ksh, self_width_ + 2 * ksh, self_n_kernels_), dtype=np.float32)
        delta_padded[:, ksh : -ksh, ksh : -ksh, :] = self_delta_
        input_image_size = self_height_ * self_width_
        weights_flipped_reshaped = np.empty((self_n_channels_, kssk), dtype=np.float32)
        for s in range(self_n_channels_):
            weights_s = np.copy(self_weights_0_[:, :, s, :])
            for t in range(self_n_kernels_):
                    weights_s[:, :, t] = np.fliplr(np.flipud(weights_s[:, :, t]))
            weights_flipped_reshaped[s] = np.reshape(weights_s, kssk)
        column_tiled_deltas = np.empty((kssk, input_image_size), dtype=np.float32)
        for i in range(m):
            for s in range(self_n_channels_):             
                row_w_mirrored_s = weights_flipped_reshaped[s]                 
                index = 0
                for j in range(self_height_):
                    for k in range(self_width_):
                        single_delta_padded_t = np.copy(delta_padded[i, j : j + self_kernel_size_, k : k + self_kernel_size_, :])
                        column_tiled_deltas[:, index] = np.reshape(single_delta_padded_t, kssk)
                        index += 1
                self_delta_backward_output_[i, :, :, s] = np.reshape(row_w_mirrored_s.dot(column_tiled_deltas), (self_height_, self_width_))
                                
    def do_backward_output_numpy(self):
        self.delta_backward_output_ = np.empty(self.input_.shape, dtype=np.float32)
        ksh = self.kernel_size_ // 2
        kssk = self.kernel_size_**2 * self.n_kernels_
        delta_padded = np.zeros((self.input_.shape[0], self.height_ + 2 * ksh, self.width_ + 2 * ksh, self.n_kernels_), dtype=np.float32)
        delta_padded[:, ksh : -ksh, ksh : -ksh, :] = self.delta_
        input_image_size = self.height_ * self.width_
        weights_flipped = np.copy(self.weights_[0])
        for s in range(self.n_channels_):
            for t in range(self.n_kernels_):
                    weights_flipped[:, :, s, t] = np.fliplr(np.flipud(weights_flipped[:, :, s, t]))        
        for i in range(self.input_.shape[0]):
            for s in range(self.n_channels_):                                
                row_w_mirrored_s = np.reshape(weights_flipped[:, :, s, :], (1, kssk)) 
                column_tiled_deltas = np.empty((kssk, input_image_size), dtype=np.float32)
                for index, (j, k) in enumerate(itertools.product(range(self.height_), range(self.width_))):
                    single_delta_padded_t = delta_padded[i, j : j + self.kernel_size_, k : k + self.kernel_size_, :]
                    column_tiled_deltas[:, index] = np.reshape(single_delta_padded_t, kssk)
                self.delta_backward_output_[i, :, :, s] = np.reshape(row_w_mirrored_s.dot(column_tiled_deltas), (self.height_, self.width_))   

            
class MaxPool2D(Layer):

    def __init__(self, name=None, input_shape=None, activation=None, do_forward_impl=None, do_backward_impl=None, do_backward_output_impl=None, pool_size=2):        
        super().__init__(name, input_shape, activation, tunable=False, do_forward_impl=None, do_backward_impl=None, do_backward_output_impl=None)
        if self.activation_name_ == "softmax":
            raise Exception("Softmax activation not allowed for MaxPool2D layer.")
        self.shortname_default_prefix_ = "m"
        self.pool_size_ = pool_size
        self.n_pools_height_ = None
        self.n_pools_width_ = None
        self.argmax_ = None
                 
    def __compile__(self, layer_prev):
        super().__compile__(layer_prev)
        self.height_, self.width_, self.n_kernels_ = self.input_shape_
        self.n_pools_height_ = self.height_ // self.pool_size_
        self.n_pools_width_ = self.width_ // self.pool_size_
        self.output_shape_ = (self.n_pools_height_, self.n_pools_width_, self.n_kernels_)
        if self.cuda_available_:
            self.__setup_forward_backward_impls__("numba_cuda_direct", None, "numba_cuda_direct") # defaults
        else:
            self.__setup_forward_backward_impls__("numba_jit", None, "numba_jit") # defaults        
                                        
    def do_forward(self):
        if self.output_ is None or self.output_.shape[0] != self.input_.shape[0]:            
            self.output_ = np.empty((self.input_.shape[0], self.n_pools_height_, self.n_pools_width_, self.n_kernels_), dtype=np.float32)
            self.argmax_ = np.empty((self.input_.shape[0], self.n_pools_height_, self.n_pools_width_, self.n_kernels_, 2), dtype=np.int32)
        self.do_forward_function_()                

    def do_forward_numba_cuda_direct(self):
        t0 = time.time()
        tpb =  self.cuda_max_threads_per_block_ // 2        
        memory = self.input_.nbytes + self.output_.nbytes + self.argmax_.nbytes
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_MAXPOOL:
            print(f"[do_forward_numba_cuda_direct for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")        
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())
        with cuda.pinned(self.input_, self.output_, self.argmax_):        
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]  
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_input = self.input_[call_slice]
                dev_sub_input = cuda.to_device(sub_input, stream=stream)
                dev_sub_output = cuda.device_array((sub_input.shape[0], self.n_pools_height_, self.n_pools_width_, self.n_kernels_), dtype=np.float32, stream=stream)
                dev_sub_argmax = cuda.device_array((sub_input.shape[0], self.n_pools_height_, self.n_pools_width_, self.n_kernels_, 2), dtype=np.int32, stream=stream)
                bpg = (sub_input.shape[0] * self.n_pools_height_ * self.n_pools_width_ * self.n_kernels_ + tpb - 1) // tpb
                MaxPool2D.do_forward_numba_cuda_direct_job[bpg, tpb, stream](dev_sub_input, self.pool_size_, self.n_pools_height_, self.n_pools_width_, self.n_kernels_, dev_sub_output, dev_sub_argmax)            
                dev_sub_output.copy_to_host(ary=self.output_[call_slice], stream=stream)
                dev_sub_argmax.copy_to_host(ary=self.argmax_[call_slice], stream=stream) 
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_MAXPOOL:
            print(f"[do_forward_numba_cuda_direct for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")        

    @staticmethod
    @cuda.jit(void(float32[:, :, :, :], int32, int32, int32, int32, float32[:, :, :, :], int32[:, :, :, :, :]))
    def do_forward_numba_cuda_direct_job(sub_input_, self_pool_size_, self_n_pools_height_, self_n_pools_width_, self_n_kernels_, sub_output_, sub_argmax_):
        m = sub_input_.shape[0]
        index = cuda.grid(1)
        wk = self_n_pools_width_ * self_n_kernels_
        hwk = self_n_pools_height_ * wk
        if index >= m * hwk:
            return
        i = int(index / hwk)
        index = int(index % hwk)
        p = int(index / wk)
        index = int(index % wk)
        q = int(index / self_n_kernels_)
        f = int(index % self_n_kernels_)
        p1 = p * self_pool_size_
        p2 = p1 + self_pool_size_
        q1 = q * self_pool_size_
        q2 = q1 + self_pool_size_
        thread_input_i_f = sub_input_[i, p1 : p2, q1 : q2, f]
        temp_max = float32(-np.inf)
        temp_argmax_j = int(0)
        temp_argmax_k = int(0)
        for j in range(self_pool_size_):
            for k in range(self_pool_size_):
                if thread_input_i_f[j, k] > temp_max: 
                    temp_max = thread_input_i_f[j, k]
                    temp_argmax_j = j
                    temp_argmax_k = k
        sub_output_[i, p, q, f] = temp_max
        sub_argmax_[i, p, q, f, 0] = p1 + temp_argmax_j
        sub_argmax_[i, p, q, f, 1] = q1 + temp_argmax_k
    
    def do_forward_numba_jit(self):
        MaxPool2D.do_forward_numba_jit_job(self.input_, self.pool_size_, self.n_pools_height_, self.n_pools_width_, self.n_kernels_, self.output_, self.argmax_)
    
    @staticmethod
    @jit(void(float32[:, :, :, :], int32, int32, int32, int32, float32[:, :, :, :], int32[:, :, :, :, :]), nopython=True, cache=True)
    def do_forward_numba_jit_job(self_input_, self_pool_size_, self_n_pools_height_, self_n_pools_width_, self_n_kernels_, self_output_, self_argmax_):
        for i in range(self_input_.shape[0]):
            for p in range(self_n_pools_height_):
                p1 = p * self_pool_size_
                p2 = p1 + self_pool_size_
                for q in range(self_n_pools_width_):
                    q1 = q * self_pool_size_
                    q2 = q1 + self_pool_size_
                    for f in range(self_n_kernels_):
                        index_linear = np.argmax(self_input_[i, p1 : p2, q1 : q2, f])
                        index0 = index_linear // self_pool_size_
                        index1 = index_linear % self_pool_size_
                        self_argmax_[i, p, q, f, 0] = p1 + index0
                        self_argmax_[i, p, q, f, 1] = q1 + index1                   
                        self_output_[i, p, q, f] = self_input_[i, p1 + index0, q1 + index1, f]
        
    def do_forward_numpy(self):
        for i in range(self.input_.shape[0]):
            for p in range(self.n_pools_height_):
                p1 = p * self.pool_size_
                p2 = p1 + self.pool_size_
                for q in range(self.n_pools_width_):
                    q1 = q * self.pool_size_
                    q2 = q1 + self.pool_size_
                    for f in range(self.n_kernels_):
                        index = np.unravel_index(np.argmax(self.input_[i, p1 : p2, q1 : q2, f]), (self.pool_size_, self.pool_size_))
                        self.argmax_[i, p, q, f, 0] = p1 + index[0]
                        self.argmax_[i, p, q, f, 1] = q1 + index[1]
                        self.output_[i, p, q, f] = self.input_[i, p1 + index[0], q1 + index[1], f]
                    
    def do_backward(self):
        if self.activation_ is not None:        
            self.delta_ = self.activation_d_() * self.delta_backward_input_
        else:
            self.delta_ = self.delta_backward_input_

    def do_backward_output(self):
        self.delta_backward_output_ = np.zeros(self.input_.shape, dtype=np.float32)
        self.do_backward_output_function_()             

    def do_backward_output_numba_cuda_direct(self):
        t0 = time.time()
        tpb =  self.cuda_max_threads_per_block_ // 2        
        memory = self.argmax_.nbytes + self.delta_.nbytes + self.delta_backward_output_.nbytes
        ratio = memory / CUDA_MAX_MEMORY_PER_CALL
        if ratio < 1.0:
            ratio = 1.0
        n_calls, call_ranges = prepare_call_ranges(self.input_.shape[0], int(np.ceil(ratio)))
        if DEBUG_VERBOSE_MAXPOOL:
            print(f"[do_backward_output_numba_cuda_direct for {self.name_} {self} -> n_calls: {n_calls} due to {memory / 1024**2:.2f} MB of memory to transfer]")
        t1 = time.time()
        streams = []
        for _ in range(min(self.cuda_n_streams_, n_calls)):
            streams.append(cuda.stream())            
        with cuda.pinned(self.argmax_, self.delta_, self.delta_backward_output_):
            for i in range(n_calls):
                stream = streams[i % self.cuda_n_streams_]
                call_slice = slice(call_ranges[i], call_ranges[i + 1])
                sub_argmax = self.argmax_[call_slice]
                sub_delta = self.delta_[call_slice]
                dev_sub_argmax = cuda.to_device(sub_argmax, stream=stream)
                dev_sub_delta = cuda.to_device(sub_delta, stream=stream)
                dev_sub_delta_backward_output = cuda.device_array((dev_sub_argmax.shape[0], self.height_, self.width_, self.n_kernels_), dtype=np.float32, stream=stream)
                bpg = (sub_argmax.shape[0] * self.n_pools_height_ * self.n_pools_width_ * self.n_kernels_ + tpb - 1) // tpb
                MaxPool2D.do_backward_output_numba_cuda_direct_job[bpg, tpb, stream](dev_sub_argmax, dev_sub_delta, self.pool_size_, dev_sub_delta_backward_output)
                dev_sub_delta_backward_output.copy_to_host(ary=self.delta_backward_output_[call_slice], stream=stream)
            cuda.synchronize()
        t2 = time.time()
        if DEBUG_VERBOSE_MAXPOOL:
            print(f"[do_backward_output_numba_cuda_direct for {self.name_} {self}, times -> t2 - t1: {t2 - t1} s, t2 - t0: {t2 - t0} s]")               

    @staticmethod
    @cuda.jit(void(int32[:, :, :, :, :], float32[:, :, :, :], int32, float32[:, :, :, :]))
    def do_backward_output_numba_cuda_direct_job(sub_argmax_, sub_delta_, self_pool_size_, sub_delta_backward_output_):
        m, n_pools_height, n_pools_width, n_kernels, _ = sub_argmax_.shape
        index = cuda.grid(1)
        wk = n_pools_width * n_kernels
        hwk = n_pools_height * wk
        if index >= m * hwk:
            return
        i = int(index / hwk)
        index = int(index % hwk)
        p = int(index / wk)
        index = int(index % wk)
        q = int(index / n_kernels)
        f = int(index % n_kernels)
        p1 = int(p * self_pool_size_)
        p2 = int(p1 + self_pool_size_)
        q1 = int(q * self_pool_size_)
        q2 = int(q1 + self_pool_size_)
        for j in range(p1, p2):
            for k in range(q1, q2):
                sub_delta_backward_output_[i, j, k, f] = float32(0.0)
        amp, amq = sub_argmax_[i, p, q, f]
        sub_delta_backward_output_[i, amp, amq, f] = sub_delta_[i, p, q, f]
              
    def do_backward_output_numba_jit(self):
        MaxPool2D.do_backward_output_numba_jit_job(self.input_.shape[0], self.n_pools_height_, self.n_pools_width_, self.n_kernels_, self.argmax_, self.delta_, self.delta_backward_output_)
    
    @staticmethod
    @jit(void(int32, int32, int32, int32, int32[:, :, :, :, :], float32[:, :, :, :], float32[:, :, :, :]), nopython=True, cache=True)
    def do_backward_output_numba_jit_job(m, self_n_pools_height_, self_n_pools_width_, self_n_kernels_, self_argmax_, self_delta_, self_delta_backward_output_):
            for i in range(m):
                for p in range(self_n_pools_height_):
                    for q in range(self_n_pools_width_):
                        for f in range(self_n_kernels_):
                            amp, amq = self_argmax_[i, p, q, f]
                            self_delta_backward_output_[i, amp, amq, f] = self_delta_[i, p, q, f]

    def do_backward_output_numpy(self):             
        for i, p, q, f in itertools.product(range(self.input_.shape[0]), range(self.n_pools_height_), range(self.n_pools_width_), range(self.n_kernels_)):
            amp, amq = self.argmax_[i, p, q, f] 
            self.delta_backward_output_[i, amp, amq, f] = self.delta_[i, p, q, f]

            
class Flatten(Layer):
    
    def __init__(self, name=None, input_shape=None, activation=None):
        super().__init__(name, input_shape, activation, tunable=False)
        self.shortname_default_prefix_ = "f"        

    def __compile__(self, layer_prev):
        super().__compile__(layer_prev)
        self.output_shape_ = (np.prod(np.array(self.input_shape_)),)
        
    def do_forward(self):
        self.output_ = self.input_.reshape(self.input_.shape[0], np.prod(self.input_.shape[1:]))
        
    def do_backward(self):
        if self.activation_ is not None:        
            self.delta_ = self.activation_d_() * self.delta_backward_input_
        else:
            self.delta_ = self.delta_backward_input_
        
    def do_backward_output(self):        
        self.delta_backward_output_ = self.delta_.reshape(self.input_.shape)  
        
        
class Dropout(Layer):
    
    def __init__(self, name=None, input_shape=None, rate=0.1):
        super().__init__(name, input_shape, tunable=False, activation=None)
        self.shortname_default_prefix_ = "dr"
        self.rate_ = rate
        self.coeffs_ = None
        self.model_ = None    

    def __compile__(self, layer_prev):
        super().__compile__(layer_prev)
        self.output_shape_ = self.input_shape_
        
    def do_forward(self):
        if self.model_.fit_ongoing_:
            self.coeffs_ = ((np.random.rand(*self.input_shape_) > self.rate_) * 1.0 / (1.0 - self.rate_)).astype(np.float32)  
            self.output_ = self.coeffs_ * self.input_ 
        else:
            self.output_ = self.input_
        
    def do_backward(self):
        self.delta_ = self.coeffs_ * self.delta_backward_input_
        
    def do_backward_output(self):        
        self.delta_backward_output_ = self.delta_

                
class Dense(Layer):
    
    def __init__(self, name=None, input_shape=None, activation="sigmoid", tunable=True, \
                 n_neurons=8, \
                 l1_penalty_kernel=0.0, l1_penalty_bias=0.0, l2_penalty_kernel=0.0, l2_penalty_bias=0.0):              
        super().__init__(name, input_shape, activation, tunable)
        self.shortname_default_prefix_ = "d"
        self.n_neurons_ = n_neurons   
        self.output_shape_ = (self.n_neurons_,)
        if l1_penalty_kernel > 0.0 or l1_penalty_bias > 0.0:
            self.l1_penalties_.append(max(l1_penalty_kernel, 0.0))
            self.l1_penalties_.append(max(l1_penalty_bias, 0.0))
        if l2_penalty_kernel > 0.0 or l2_penalty_bias > 0.0:                
            self.l2_penalties_.append(max(l2_penalty_kernel, 0.0))
            self.l2_penalties_.append(max(l2_penalty_bias, 0.0))        
    
    def __compile__(self, layer_prev):
        super().__compile__(layer_prev)                
        self.input_shape_ = layer_prev.output_shape_
        if self.activation_name_ == "relu":     
            self.weights_.append(Layer.weights_he_uniform(self.input_shape_[0], self.n_neurons_))
        else:
            self.weights_.append(Layer.weights_glorot_uniform(self.input_shape_[0], self.n_neurons_))
        self.weights_.append(np.zeros(self.n_neurons_, dtype=np.float32))
        self.adam_m_.append(np.zeros(self.weights_[0].shape, dtype=np.float32))
        self.adam_m_.append(np.zeros(self.weights_[1].shape, dtype=np.float32))
        self.adam_v_.append(np.zeros(self.weights_[0].shape, dtype=np.float32))
        self.adam_v_.append(np.zeros(self.weights_[1].shape, dtype=np.float32))        
        self.gradient_ = [np.zeros(self.weights_[0].shape, dtype=np.float32), np.zeros(self.weights_[1].shape, dtype=np.float32)]
        self.prev_correction_ = [np.zeros(self.weights_[0].shape, dtype=np.float32), np.zeros(self.weights_[1].shape, dtype=np.float32)]
        self.__setup_forward_backward_impls__("numpy", "numpy", "numpy") # defaults        
    
    def do_forward(self):
        self.do_forward_function_()
    
    def do_forward_numpy(self):
        weighted_sums = self.weights_[0].dot(self.input_.T).T + self.weights_[1]
        self.output_ = weighted_sums # correct temporary assignment, the wrapping forward() function shall call soon self.activation_(self.output_) once do_forward is complete
        
    def do_backward(self):
        self.delta_ = None
        if self.activation_ is None:
            self.delta_ = self.delta_backward_input_        
        elif self.activation_name_ == "softmax":
            self.delta_ = np.empty((self.input_.shape[0], self.n_neurons_), dtype=np.float32)
            ad = self.activation_d_()
            for i in range(self.input_.shape[0]):
                self.delta_[i] = ad[i].dot(self.delta_backward_input_[i])
        else:
            self.delta_ = self.activation_d_() * self.delta_backward_input_
        self.do_backward_function_()
    
    def do_backward_numpy(self):
        self.prev_correction_ = [np.copy(dc) for dc in self.gradient_]
        self.gradient_[0] = self.delta_.T.dot(self.input_) # correction for weights - summation of corrections over the batch takes place here
        self.gradient_[1] = self.delta_.T.dot(np.ones(self.input_.shape[0], dtype=np.float32)) # correction for intercepts - summation of corrections over the batch size place here
        
    def do_backward_output(self):
        self.do_backward_output_function_()
    
    def do_backward_output_numpy(self):
        self.delta_backward_output_ = self.delta_.dot(self.weights_[0])  

        
class SequentialClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_epochs=10**2, n_batches=10, loss="categorical_crossentropy", learning_rate=1e-3, decay_rate=0.0, use_adam=True, momentum_rate=0.0, gradient_clip=None):
        self.n_epochs_ = n_epochs
        self.n_batches_ = n_batches 
        if self.n_batches_ < 1: 
            self.n_batches_ = 1
        self.loss_name_ = loss
        self.loss_ = getattr(self, loss)
        self.loss_d_ = getattr(self, loss + "_d")   
        self.learning_rate_ = learning_rate
        self.learning_rate_now_ = self.learning_rate_ 
        self.use_adam_ = use_adam
        self.adam_beta_1_ = 0.9
        self.adam_beta_2_ = 0.999
        self.adam_epsilon_ = 1e-7
        self.momentum_rate_ = momentum_rate
        self.decay_rate_ = decay_rate
        self.gradient_clip_ = gradient_clip
        self.class_labels_ = None 
        self.layers_ = []
        self.layers_dict_ = {} 
        self.fit_info_ = None
        self.fit_ongoing_ = False        
        
    def add(self, layer):
        self.layers_.append(layer)
        if len(self.layers_) == 1:
            layer.initial_ = True
        if layer.name_ is None:
            layer.name_ = layer.shortname_default_prefix_ + str(len(self.layers_) - 1)
        self.layers_dict_[layer.name_] = layer        
        layer.__compile__(self.layers_[-2] if len(self.layers_) > 1 else None)
        if isinstance(layer, Dropout):
            layer.model_ = self
            
    def fit(self, X, y, verbose_layers=False, verbose_fit_info=True, verbose_fit_time=False):
        if verbose_fit_time:
            print("HMDL FIT...")
        t1 = time.time()
        self.fit_ongoing_ = True
        self.class_labels_ = np.unique(y)
        m = y.shape[0]
        if X.dtype !=  np.float32:
            X = X.astype(np.float32)
        y_ord = np.zeros(m, dtype=np.int16)
        for index, label in enumerate(self.class_labels_):
            y_ord[y == label] = index
        y_one_hot = to_one_hot(y, self.class_labels_.size)        
        _, batch_ranges = prepare_call_ranges(m, self.n_batches_, power_two_sizes=False)        
        self.learning_rate_now_ = self.learning_rate_
        self.fit_info_ = []
        print(f"[norms of weights -> l1: {self.weights_l1_norm():0.7}, l2: {self.weights_l2_norm():0.7}]")
        for t in range(self.n_epochs_):            
            print(f"EPOCH: {t + 1}/{self.n_epochs_}...")
            t1_epoch = time.time()
            self.learning_rate_now_ *= 1.0 / (1.0 + self.decay_rate_ * t)             
            p = np.random.permutation(m)            
            for b in range(self.n_batches_):
                indexes = p[batch_ranges[b] : batch_ranges[b + 1]]
                X_b = np.ascontiguousarray(X[indexes]) # forcing contiguous array for potential cuda purposes
                y_b = y_one_hot[indexes]           
                y_b_ord = y_ord[indexes]
                self.forward(X_b, verbose_layers)
                if verbose_fit_info:
                    self.memorize_fit_info(t, b, self.layers_[-1].output_, y_b, y_b_ord)
                self.backward(y_b, t * self.n_batches_ + b, verbose_layers)
            t2_epoch = time.time()                          
            if verbose_fit_info:
                self.print_fit_info(t)                
                print(f"[norms of weights -> l1: {self.weights_l1_norm():0.8}, l2: {self.weights_l2_norm():0.8}]")
                print(f"[epoch fit time: {t2_epoch - t1_epoch:0.8} s]")
        self.fit_ongoing_ = False
        t2 = time.time()
        self.prune_dev_references()
        if verbose_fit_time:
            print(f"HMDL FIT DONE. [time: {t2 - t1} s]")             

    def prune_dev_references(self):
        for l in self.layers_:
            if isinstance(l, Conv2D):
                l.dev_weights_0_complex_ = None

    def memorize_fit_info(self, t, b, y_pred, y_b, y_b_ord):
        loss_b = np.mean(self.loss_(y_pred, y_b))
        acc_b = np.mean(self.class_labels_[np.argmax(y_pred, axis=1)] == y_b_ord)
        self.fit_info_.append(np.array([t, b, loss_b, acc_b]))
        
    def print_fit_info(self, t=None, b=None):
        COL_EPOCH_LEN = 12
        COL_BATCH_LEN = 12
        COL_LOSS_LEN = 16        
        fit_info_as_array = np.array(self.fit_info_)
        if t is not None:
            fit_info_as_array = fit_info_as_array[fit_info_as_array[:, 0] == t]
        if b is not None:
            fit_info_as_array = fit_info_as_array[fit_info_as_array[:, 1] == b]
        info_str = ""
        for entry_t, entry_b, entry_loss_b, entry_acc_b in fit_info_as_array:         
            info_str += f" epoch: {int(entry_t + 1)}".ljust(COL_EPOCH_LEN)
            info_str += f" batch: {int(entry_b + 1)}".ljust(COL_BATCH_LEN)
            info_str += f" loss: {entry_loss_b:0.4}".ljust(COL_LOSS_LEN)
            info_str += f" acc: {entry_acc_b:0.4}"
            if b is None:
                info_str += "\n"
        if b is None:
            mean_loss = np.mean(fit_info_as_array[:, 2])
            mean_acc = np.mean(fit_info_as_array[:, 3])
            info_str += f" means --> ".ljust(COL_EPOCH_LEN + COL_BATCH_LEN)
            info_str += f" loss: {mean_loss:.4}".ljust(COL_LOSS_LEN)
            info_str += f" acc: {mean_acc:.4}"
            info_str = f"[fit info (epoch {t + 1}):\n" + info_str + "]"            
        else:
            info_str = f"[fit info (batch {b}): " + info_str + "]"
        print(info_str)
                            
    def forward(self, X, verbose_layers=False):
        t1 = time.time()        
        self.layers_[0].forward(X)
        t2 = time.time()
        if verbose_layers:
            print(f"[forward for {self.layers_[0].name_} {self.layers_[0]} done in {t2 - t1} s]")
        for i in range(1, len(self.layers_)):            
            t1 = time.time() 
            self.layers_[i].forward(self.layers_[i - 1].output_)
            t2 = time.time()
            if verbose_layers:
                print(f"[forward for {self.layers_[i].name_} {self.layers_[i]} done in {t2 - t1} s]")
        return self.layers_[-1].output_
            
    def backward(self, y, step, verbose_layers=False):
        y_pred = self.layers_[-1].output_
        t1 = time.time()
        self.layers_[-1].backward(self.loss_d_(y_pred, y))
        t2 = time.time()
        if verbose_layers:
            print(f"[backward for {self.layers_[-1].name_} {self.layers_[-1]} done in {t2 - t1 } s]")
        for i in range(len(self.layers_) - 2, -1, -1):
            t1 = time.time()
            self.layers_[i].backward(self.layers_[i + 1].delta_backward_output_)
            t2 = time.time()
            if verbose_layers:
                print(f"[backward for {self.layers_[i].name_} {self.layers_[i]} done in {t2 - t1} s]")
        for i in range(len(self.layers_) - 1, -1, -1):
            if self.layers_[i].tunable_:
                for w in range(len(self.layers_[i].weights_)):                        
                        gradient = self.layers_[i].gradient_[w]
                        regularization = 0.0
                        if self.layers_[i].l1_penalties_ and self.layers_[i].l1_penalties_[w] > 0.0:
                            regularization += self.layers_[i].l1_penalties_[w] * np.sign(self.layers_[i].weights_[w])
                        if self.layers_[i].l2_penalties_ and self.layers_[i].l2_penalties_[w] > 0.0:
                            regularization += self.layers_[i].l2_penalties_[w] * self.layers_[i].weights_[w]
                        gradient += regularization 
                        if self.gradient_clip_ != None:
                            gradient = np.clip(gradient, -self.gradient_clip_, self.gradient_clip_)
                        if self.use_adam_:
                            self.layers_[i].adam_m_[w] = self.adam_beta_1_ * self.layers_[i].adam_m_[w] + (1.0 - self.adam_beta_1_) * gradient
                            self.layers_[i].adam_v_[w] = self.adam_beta_2_ * self.layers_[i].adam_v_[w] + (1.0 - self.adam_beta_2_) * gradient**2
                            adam_m_hat = self.layers_[i].adam_m_[w] / (1.0 - self.adam_beta_1_**(step + 1)) 
                            adam_v_hat = self.layers_[i].adam_v_[w] / (1.0 - self.adam_beta_2_**(step + 1))
                            gradient = adam_m_hat / (np.sqrt(adam_v_hat) + self.adam_epsilon_)
                        correction = -self.learning_rate_now_ * gradient 
                        if not self.use_adam_ and self.momentum_rate_ > 0.0:
                            correction += self.momentum_rate_ * self.layers_[i].prev_correction_[w] 
                        self.layers_[i].weights_[w] += correction
                        self.layers_[i].prev_correction_[w] = correction
    
    def predict_proba(self, X, verbose=True, verbose_layers=True):
        t1 = time.time()
        self.forward(X, verbose_layers=verbose_layers)
        t2 = time.time()
        if verbose:
            print(f"[predict_proba on X with shape {X.shape} done in {t2 - t1} s]")
        return self.layers_[-1].output_
        
    def predict(self, X, verbose=True):
        return self.class_labels_[np.argmax(self.predict_proba(X, verbose), axis=1)]
    
    def acc(self, y_pred, y):
        if self.class_labels_ is None:
            self.class_labels_ = np.unique(y)
        return np.mean(self.class_labels_[np.argmax(y_pred, axis=1)] == y)
    
    def loss(self, y_pred, y_one_hot):
        return np.mean(self.loss_(y_pred, y_one_hot))    
    
    @staticmethod
    def categorical_crossentropy(y_pred, y):        
        return np.sum(-y * np.log(np.clip(y_pred, MIN_NON_ZERO, np.inf)), axis=1)        
    
    @staticmethod
    def categorical_crossentropy_d(y_pred, y):
        return -y / np.clip(y_pred, MIN_NON_ZERO, np.inf) 
    
    @staticmethod
    def square_loss(y_pred, y):
        return 0.5 * np.sum((y_pred - y)**2, axis=1)        
    
    @staticmethod
    def square_loss_d(y_pred, y):
        return y_pred - y
    
    def weights_l2_norm(self):
        norm = 0.0
        for i in range(len(self.layers_)):
            if self.layers_[i].tunable_:
                for w in range(len(self.layers_[i].weights_)):
                    norm_summand = np.sum(np.square(self.layers_[i].weights_[w]))
                    norm += norm_summand
        return np.sqrt(norm)    

    def weights_l1_norm(self):
        norm = 0.0
        for i in range(len(self.layers_)):
            if self.layers_[i].tunable_:
                for w in range(len(self.layers_[i].weights_)):
                    norm_summand = np.sum(np.abs(self.layers_[i].weights_[w]))
                    norm += norm_summand
        return norm
    
    def get_weights(self):
        weights = []
        for l in self.layers_:
            weights.append(l.weights_)
        return weights
    
    def set_weights(self, weights):
        for i in range(len(weights)):
            self.layers_[i].weights_ = weights[i]
    
    def summary(self):
        COL_NO_LEN = 4
        COL_NAME_LEN = 8
        COL_TYPE_LEN = 12
        COL_INPUT_SHAPE_LEN = 24
        COL_EXTRA_INFO_LEN = 12
        COL_ACTIVATION_LEN = 14
        COL_OUTPUT_SHAPE_LEN = 22
        COL_DO_FORWARD_IMPL_LEN = 22
        COL_DO_BACKWARD_IMPL_LEN = 22
        COL_DO_BACKWARD_OUTPUT_IMPL_LEN = 26
        COL_L1_PENALTIES_LEN = 20
        COL_L2_PENALTIES_LEN = 20
        COL_PARAMS_LEN = 16    
        SEPARATOR = "-" * (COL_NO_LEN + COL_NAME_LEN + COL_TYPE_LEN + COL_INPUT_SHAPE_LEN + COL_EXTRA_INFO_LEN + COL_ACTIVATION_LEN + COL_OUTPUT_SHAPE_LEN \
                           + COL_DO_FORWARD_IMPL_LEN + COL_DO_BACKWARD_IMPL_LEN + COL_DO_BACKWARD_OUTPUT_IMPL_LEN + COL_L1_PENALTIES_LEN + COL_L2_PENALTIES_LEN + COL_PARAMS_LEN) + "\n"                 
        summary_str = SEPARATOR
        summary_str += "NO.".ljust(COL_NO_LEN)
        summary_str += "LAYER".ljust(COL_NAME_LEN)
        summary_str += "TYPE".ljust(COL_TYPE_LEN)
        summary_str += "INPUT SHAPE".ljust(COL_INPUT_SHAPE_LEN)
        summary_str += "EXTRA INFO".ljust(COL_EXTRA_INFO_LEN)
        summary_str += "ACTIVATION".ljust(COL_ACTIVATION_LEN)
        summary_str += "OUTPUT SHAPE".ljust(COL_OUTPUT_SHAPE_LEN)
        summary_str += "DO FORWARD IMPL".ljust(COL_DO_FORWARD_IMPL_LEN)
        summary_str += "DO BACKWARD IMPL".ljust(COL_DO_BACKWARD_IMPL_LEN)
        summary_str += "DO BACKWARD OUTPUT IMPL".ljust(COL_DO_BACKWARD_OUTPUT_IMPL_LEN)
        summary_str += "L1 PENALTIES".ljust(COL_L1_PENALTIES_LEN)
        summary_str += "L2 PENALTIES".ljust(COL_L2_PENALTIES_LEN)        
        summary_str += "PARAMS".ljust(COL_PARAMS_LEN)
        summary_str += "\n" + SEPARATOR
        total_params = 0
        for i, l in enumerate(self.layers_):
            summary_str += str(i)[:COL_NO_LEN].ljust(COL_NO_LEN)
            summary_str += f"{l.name_}"[:COL_NAME_LEN].ljust(COL_NAME_LEN)
            summary_str += f"{l.__class__.__name__}"[:COL_TYPE_LEN].ljust(COL_TYPE_LEN)
            tup = l.input_shape_
            tup_str = str(tup)[1:] if len(tup) > 1 else (str(tup)[1 : -2] + ")") 
            summary_str += ("(None, " + tup_str)[:COL_INPUT_SHAPE_LEN].ljust(COL_INPUT_SHAPE_LEN)
            kernel_or_pool_size_str = ""
            if isinstance(l, Conv2D):
                kernel_or_pool_size_str = f"{l.kernel_size_}x{l.kernel_size_}"
            elif isinstance(l, MaxPool2D):
                kernel_or_pool_size_str = f"{l.pool_size_}x{l.pool_size_}"
            elif isinstance(l, Dropout):
                kernel_or_pool_size_str = f"{l.rate_}"                
            summary_str += kernel_or_pool_size_str[:COL_EXTRA_INFO_LEN].ljust(COL_EXTRA_INFO_LEN)
            activation_str = "" if l.activation_ is None else l.activation_.__name__
            summary_str += activation_str[:COL_ACTIVATION_LEN].ljust(COL_ACTIVATION_LEN)
            tup = l.output_shape_
            tup_str = str(tup)[1:] if len(tup) > 1 else (str(tup)[1 : -2] + ")")           
            summary_str += ("(None, " + tup_str)[:COL_OUTPUT_SHAPE_LEN].ljust(COL_OUTPUT_SHAPE_LEN)            
            do_forward_impl_str = "" if l.do_forward_impl_ is None else l.do_forward_impl_
            summary_str += do_forward_impl_str[:COL_DO_FORWARD_IMPL_LEN].ljust(COL_DO_FORWARD_IMPL_LEN)
            do_backward_impl_str = "" if l.do_backward_impl_ is None else l.do_backward_impl_
            summary_str += do_backward_impl_str[:COL_DO_BACKWARD_IMPL_LEN].ljust(COL_DO_BACKWARD_IMPL_LEN)                
            do_backward_output_impl_str = "" if l.do_backward_output_impl_ is None else l.do_backward_output_impl_             
            summary_str += do_backward_output_impl_str[:COL_DO_BACKWARD_OUTPUT_IMPL_LEN].ljust(COL_DO_BACKWARD_OUTPUT_IMPL_LEN)
            l1_penalties_str = str(l.l1_penalties_) if l.l1_penalties_ else ""
            summary_str += l1_penalties_str[:COL_L1_PENALTIES_LEN].ljust(COL_L1_PENALTIES_LEN)
            l2_penalties_str = str(l.l2_penalties_) if l.l2_penalties_ else ""
            summary_str += l2_penalties_str[:COL_L2_PENALTIES_LEN].ljust(COL_L2_PENALTIES_LEN)            
            params = sum([np.prod(w.shape) for w in l.weights_]) if l.tunable_ else 0
            total_params += params
            summary_str += str(params)[:COL_PARAMS_LEN].ljust(COL_PARAMS_LEN)
            summary_str += "\n"
        summary_str += SEPARATOR
        summary_str += f"TOTAL PARAMS: {total_params}\n"
        summary_str += f"FIT SETTINGS: [n_epochs: {self.n_epochs_}, n_batches: {self.n_batches_}, loss: {self.loss_name_}, learning_rate: {self.learning_rate_}, decay_rate: {self.decay_rate_}, use_adam: {self.use_adam_}, momentum_rate: {self.momentum_rate_}, gradient_clip: {self.gradient_clip_}]\n"
        summary_str += SEPARATOR
        return summary_str
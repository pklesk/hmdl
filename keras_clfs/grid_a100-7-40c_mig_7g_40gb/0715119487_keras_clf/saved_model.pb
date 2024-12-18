��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
v
Adam/v/d15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/v/d15/bias
o
#Adam/v/d15/bias/Read/ReadVariableOpReadVariableOpAdam/v/d15/bias*
_output_shapes
:
*
dtype0
v
Adam/m/d15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/m/d15/bias
o
#Adam/m/d15/bias/Read/ReadVariableOpReadVariableOpAdam/m/d15/bias*
_output_shapes
:
*
dtype0

Adam/v/d15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*"
shared_nameAdam/v/d15/kernel
x
%Adam/v/d15/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d15/kernel*
_output_shapes
:	�
*
dtype0

Adam/m/d15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*"
shared_nameAdam/m/d15/kernel
x
%Adam/m/d15/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d15/kernel*
_output_shapes
:	�
*
dtype0
w
Adam/v/d13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameAdam/v/d13/bias
p
#Adam/v/d13/bias/Read/ReadVariableOpReadVariableOpAdam/v/d13/bias*
_output_shapes	
:�*
dtype0
w
Adam/m/d13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameAdam/m/d13/bias
p
#Adam/m/d13/bias/Read/ReadVariableOpReadVariableOpAdam/m/d13/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/d13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nameAdam/v/d13/kernel
y
%Adam/v/d13/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d13/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/d13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nameAdam/m/d13/kernel
y
%Adam/m/d13/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d13/kernel* 
_output_shapes
:
��*
dtype0
u
Adam/v/c9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/v/c9/bias
n
"Adam/v/c9/bias/Read/ReadVariableOpReadVariableOpAdam/v/c9/bias*
_output_shapes	
:�*
dtype0
u
Adam/m/c9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/m/c9/bias
n
"Adam/m/c9/bias/Read/ReadVariableOpReadVariableOpAdam/m/c9/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/c9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameAdam/v/c9/kernel

$Adam/v/c9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c9/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/c9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameAdam/m/c9/kernel

$Adam/m/c9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c9/kernel*(
_output_shapes
:��*
dtype0
u
Adam/v/c8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/v/c8/bias
n
"Adam/v/c8/bias/Read/ReadVariableOpReadVariableOpAdam/v/c8/bias*
_output_shapes	
:�*
dtype0
u
Adam/m/c8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/m/c8/bias
n
"Adam/m/c8/bias/Read/ReadVariableOpReadVariableOpAdam/m/c8/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/c8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameAdam/v/c8/kernel
~
$Adam/v/c8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c8/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/m/c8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameAdam/m/c8/kernel
~
$Adam/m/c8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c8/kernel*'
_output_shapes
:@�*
dtype0
t
Adam/v/c5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/c5/bias
m
"Adam/v/c5/bias/Read/ReadVariableOpReadVariableOpAdam/v/c5/bias*
_output_shapes
:@*
dtype0
t
Adam/m/c5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/c5/bias
m
"Adam/m/c5/bias/Read/ReadVariableOpReadVariableOpAdam/m/c5/bias*
_output_shapes
:@*
dtype0
�
Adam/v/c5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameAdam/v/c5/kernel
}
$Adam/v/c5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c5/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/m/c5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameAdam/m/c5/kernel
}
$Adam/m/c5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c5/kernel*&
_output_shapes
:@@*
dtype0
t
Adam/v/c4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/c4/bias
m
"Adam/v/c4/bias/Read/ReadVariableOpReadVariableOpAdam/v/c4/bias*
_output_shapes
:@*
dtype0
t
Adam/m/c4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/c4/bias
m
"Adam/m/c4/bias/Read/ReadVariableOpReadVariableOpAdam/m/c4/bias*
_output_shapes
:@*
dtype0
�
Adam/v/c4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameAdam/v/c4/kernel
}
$Adam/v/c4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c4/kernel*&
_output_shapes
: @*
dtype0
�
Adam/m/c4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameAdam/m/c4/kernel
}
$Adam/m/c4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c4/kernel*&
_output_shapes
: @*
dtype0
t
Adam/v/c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/v/c1/bias
m
"Adam/v/c1/bias/Read/ReadVariableOpReadVariableOpAdam/v/c1/bias*
_output_shapes
: *
dtype0
t
Adam/m/c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/m/c1/bias
m
"Adam/m/c1/bias/Read/ReadVariableOpReadVariableOpAdam/m/c1/bias*
_output_shapes
: *
dtype0
�
Adam/v/c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameAdam/v/c1/kernel
}
$Adam/v/c1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c1/kernel*&
_output_shapes
:  *
dtype0
�
Adam/m/c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameAdam/m/c1/kernel
}
$Adam/m/c1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c1/kernel*&
_output_shapes
:  *
dtype0
t
Adam/v/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/v/c0/bias
m
"Adam/v/c0/bias/Read/ReadVariableOpReadVariableOpAdam/v/c0/bias*
_output_shapes
: *
dtype0
t
Adam/m/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/m/c0/bias
m
"Adam/m/c0/bias/Read/ReadVariableOpReadVariableOpAdam/m/c0/bias*
_output_shapes
: *
dtype0
�
Adam/v/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/v/c0/kernel
}
$Adam/v/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c0/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/m/c0/kernel
}
$Adam/m/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c0/kernel*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
h
d15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
d15/bias
a
d15/bias/Read/ReadVariableOpReadVariableOpd15/bias*
_output_shapes
:
*
dtype0
q

d15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_name
d15/kernel
j
d15/kernel/Read/ReadVariableOpReadVariableOp
d15/kernel*
_output_shapes
:	�
*
dtype0
i
d13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
d13/bias
b
d13/bias/Read/ReadVariableOpReadVariableOpd13/bias*
_output_shapes	
:�*
dtype0
r

d13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
d13/kernel
k
d13/kernel/Read/ReadVariableOpReadVariableOp
d13/kernel* 
_output_shapes
:
��*
dtype0
g
c9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	c9/bias
`
c9/bias/Read/ReadVariableOpReadVariableOpc9/bias*
_output_shapes	
:�*
dtype0
x
	c9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_name	c9/kernel
q
c9/kernel/Read/ReadVariableOpReadVariableOp	c9/kernel*(
_output_shapes
:��*
dtype0
g
c8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	c8/bias
`
c8/bias/Read/ReadVariableOpReadVariableOpc8/bias*
_output_shapes	
:�*
dtype0
w
	c8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_name	c8/kernel
p
c8/kernel/Read/ReadVariableOpReadVariableOp	c8/kernel*'
_output_shapes
:@�*
dtype0
f
c5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	c5/bias
_
c5/bias/Read/ReadVariableOpReadVariableOpc5/bias*
_output_shapes
:@*
dtype0
v
	c5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_name	c5/kernel
o
c5/kernel/Read/ReadVariableOpReadVariableOp	c5/kernel*&
_output_shapes
:@@*
dtype0
f
c4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	c4/bias
_
c4/bias/Read/ReadVariableOpReadVariableOpc4/bias*
_output_shapes
:@*
dtype0
v
	c4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_name	c4/kernel
o
c4/kernel/Read/ReadVariableOpReadVariableOp	c4/kernel*&
_output_shapes
: @*
dtype0
f
c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	c1/bias
_
c1/bias/Read/ReadVariableOpReadVariableOpc1/bias*
_output_shapes
: *
dtype0
v
	c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_name	c1/kernel
o
c1/kernel/Read/ReadVariableOpReadVariableOp	c1/kernel*&
_output_shapes
:  *
dtype0
f
c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	c0/bias
_
c0/bias/Read/ReadVariableOpReadVariableOpc0/bias*
_output_shapes
: *
dtype0
v
	c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	c0/kernel
o
c0/kernel/Read/ReadVariableOpReadVariableOp	c0/kernel*&
_output_shapes
: *
dtype0
�
serving_default_c0_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	c8/kernelc8/bias	c9/kernelc9/bias
d13/kerneld13/bias
d15/kerneld15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *,
f'R%
#__inference_signature_wrapper_32703

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
valueֈB҈ Bʈ
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias
 "_jit_compiled_convolution_op*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses* 
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_random_generator* 
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias
 J_jit_compiled_convolution_op*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator* 
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias
 `_jit_compiled_convolution_op*
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias
 i_jit_compiled_convolution_op*
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses
v_random_generator* 
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses* 
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
~
 0
!1
)2
*3
?4
@5
H6
I7
^8
_9
g10
h11
�12
�13
�14
�15*
~
 0
!1
)2
*3
?4
@5
H6
I7
^8
_9
g10
h11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

 0
!1*

 0
!1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

)0
*1*

)0
*1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

?0
@1*

?0
@1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

H0
I1*

H0
I1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

^0
_1*

^0
_1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

g0
h1*

g0
h1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c9/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc9/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ZT
VARIABLE_VALUE
d13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEd13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ZT
VARIABLE_VALUE
d15/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEd15/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
[U
VARIABLE_VALUEAdam/m/c0/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/c0/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/c0/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/c0/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/c1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/c1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/c1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/c1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/c4/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/c4/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/c4/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/c4/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/c5/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/c5/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/c5/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/c5/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/c8/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/c8/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/c8/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/c8/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/c9/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/c9/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/c9/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/c9/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/d13/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/d13/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/d13/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/d13/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/d15/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/d15/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/d15/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/d15/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	c8/kernelc8/bias	c9/kernelc9/bias
d13/kerneld13/bias
d15/kerneld15/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c1/kernelAdam/v/c1/kernelAdam/m/c1/biasAdam/v/c1/biasAdam/m/c4/kernelAdam/v/c4/kernelAdam/m/c4/biasAdam/v/c4/biasAdam/m/c5/kernelAdam/v/c5/kernelAdam/m/c5/biasAdam/v/c5/biasAdam/m/c8/kernelAdam/v/c8/kernelAdam/m/c8/biasAdam/v/c8/biasAdam/m/c9/kernelAdam/v/c9/kernelAdam/m/c9/biasAdam/v/c9/biasAdam/m/d13/kernelAdam/v/d13/kernelAdam/m/d13/biasAdam/v/d13/biasAdam/m/d15/kernelAdam/v/d15/kernelAdam/m/d15/biasAdam/v/d15/biastotal_1count_1totalcountConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *'
f"R 
__inference__traced_save_33823
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	c8/kernelc8/bias	c9/kernelc9/bias
d13/kerneld13/bias
d15/kerneld15/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c1/kernelAdam/v/c1/kernelAdam/m/c1/biasAdam/v/c1/biasAdam/m/c4/kernelAdam/v/c4/kernelAdam/m/c4/biasAdam/v/c4/biasAdam/m/c5/kernelAdam/v/c5/kernelAdam/m/c5/biasAdam/v/c5/biasAdam/m/c8/kernelAdam/v/c8/kernelAdam/m/c8/biasAdam/v/c8/biasAdam/m/c9/kernelAdam/v/c9/kernelAdam/m/c9/biasAdam/v/c9/biasAdam/m/d13/kernelAdam/v/d13/kernelAdam/m/d13/biasAdam/v/d13/biasAdam/m/d15/kernelAdam/v/d15/kernelAdam/m/d15/biasAdam/v/d15/biastotal_1count_1totalcount*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� **
f%R#
!__inference__traced_restore_33995�
�
�
#__inference_signature_wrapper_32703
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *)
f$R"
 __inference__wrapped_model_31867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
Y
=__inference_m6_layer_call_and_return_conditional_losses_33206

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
V
"__inference__update_step_xla_33016
gradient"
variable: @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: @: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
: @
"
_user_specified_name
gradient
�
J
"__inference__update_step_xla_33021
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
�
�
=__inference_c8_layer_call_and_return_conditional_losses_33255

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
,
__inference_loss_fn_10_33451
identity`
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
\
#__inference_dr7_layer_call_fn_33211

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_32011w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�H
�
E__inference_sequential_layer_call_and_return_conditional_losses_32407

inputs"
c0_32342: 
c0_32344: "
c1_32347:  
c1_32349: "
c4_32354: @
c4_32356:@"
c5_32359:@@
c5_32361:@#
c8_32366:@�
c8_32368:	�$
c9_32371:��
c9_32373:	�
	d13_32379:
��
	d13_32381:	�
	d15_32385:	�

	d15_32387:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_32342c0_32344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31920�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_32347c1_32349*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31939�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31873�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_32161�
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_32354c4_32356*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31973�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_32359c5_32361*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31992�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31885�
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_32178�
c8/StatefulPartitionedCallStatefulPartitionedCalldr7/PartitionedCall:output:0c8_32366c8_32368*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_32026�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_32371c9_32373*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_32045�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_31897�
dr11/PartitionedCallPartitionedCallm10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_32195�
f12/PartitionedCallPartitionedCalldr11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_32072�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_32379	d13_32381*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_32087�
dr14/PartitionedCallPartitionedCall$d13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_32207�
d15/StatefulPartitionedCallStatefulPartitionedCalldr14/PartitionedCall:output:0	d15_32385	d15_32387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_32120`
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_32337
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_32302o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
+
__inference_loss_fn_5_33426
identity^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
>__inference_d13_layer_call_and_return_conditional_losses_32087

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_32793

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_32407o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
\
>__inference_dr7_layer_call_and_return_conditional_losses_32178

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_c0_layer_call_and_return_conditional_losses_31920

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
K
"__inference__update_step_xla_33041
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
Y
=__inference_m6_layer_call_and_return_conditional_losses_31885

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
?
#__inference_m10_layer_call_fn_33282

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_31897�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
O
"__inference__update_step_xla_33066
gradient
variable:	�
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�
: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�

"
_user_specified_name
gradient
�
+
__inference_loss_fn_7_33436
identity^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
,
__inference_loss_fn_13_33466
identity_
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
IdentityIdentity#d13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
@
$__inference_dr14_layer_call_fn_33357

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_32207a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
Z
>__inference_m10_layer_call_and_return_conditional_losses_31897

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

]
>__inference_dr7_layer_call_and_return_conditional_losses_33228

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_31873

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
Z
>__inference_f12_layer_call_and_return_conditional_losses_33325

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
?__inference_dr14_layer_call_and_return_conditional_losses_33374

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
,
__inference_loss_fn_12_33461
identitya
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    \
IdentityIdentity%d13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
@
$__inference_dr11_layer_call_fn_33297

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_32195i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_d15_layer_call_fn_33383

inputs
unknown:	�

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_32120o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_c4_layer_call_fn_33161

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31973w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
"__inference_c1_layer_call_fn_33102

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31939w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
?
#__inference_dr7_layer_call_fn_33216

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_32178h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_c0_layer_call_and_return_conditional_losses_33093

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
>__inference_d15_layer_call_and_return_conditional_losses_33396

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
=__inference_c5_layer_call_and_return_conditional_losses_31992

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
\
>__inference_dr7_layer_call_and_return_conditional_losses_33233

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
,
__inference_loss_fn_11_33456
identity^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
=__inference_c4_layer_call_and_return_conditional_losses_33174

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
V
"__inference__update_step_xla_33006
gradient"
variable:  *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:  : *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:  
"
_user_specified_name
gradient
�
]
?__inference_dr11_layer_call_and_return_conditional_losses_33314

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
,
__inference_loss_fn_14_33471
identitya
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    \
IdentityIdentity%d15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
,
__inference_loss_fn_15_33476
identity_
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
IdentityIdentity#d15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
"__inference_c9_layer_call_fn_33264

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_32045x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_33152

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
"__inference_c5_layer_call_fn_33183

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31992w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_32756

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_32302o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
K
"__inference__update_step_xla_33051
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_32161

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
?
#__inference_dr3_layer_call_fn_33135

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_32161h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
+
__inference_loss_fn_0_33401
identity`
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
X
"__inference__update_step_xla_33046
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:($
"
_user_specified_name
variable:R N
(
_output_shapes
:��
"
_user_specified_name
gradient
��
�/
__inference__traced_save_33823
file_prefix:
 read_disablecopyonread_c0_kernel: .
 read_1_disablecopyonread_c0_bias: <
"read_2_disablecopyonread_c1_kernel:  .
 read_3_disablecopyonread_c1_bias: <
"read_4_disablecopyonread_c4_kernel: @.
 read_5_disablecopyonread_c4_bias:@<
"read_6_disablecopyonread_c5_kernel:@@.
 read_7_disablecopyonread_c5_bias:@=
"read_8_disablecopyonread_c8_kernel:@�/
 read_9_disablecopyonread_c8_bias:	�?
#read_10_disablecopyonread_c9_kernel:��0
!read_11_disablecopyonread_c9_bias:	�8
$read_12_disablecopyonread_d13_kernel:
��1
"read_13_disablecopyonread_d13_bias:	�7
$read_14_disablecopyonread_d15_kernel:	�
0
"read_15_disablecopyonread_d15_bias:
-
#read_16_disablecopyonread_iteration:	 1
'read_17_disablecopyonread_learning_rate: D
*read_18_disablecopyonread_adam_m_c0_kernel: D
*read_19_disablecopyonread_adam_v_c0_kernel: 6
(read_20_disablecopyonread_adam_m_c0_bias: 6
(read_21_disablecopyonread_adam_v_c0_bias: D
*read_22_disablecopyonread_adam_m_c1_kernel:  D
*read_23_disablecopyonread_adam_v_c1_kernel:  6
(read_24_disablecopyonread_adam_m_c1_bias: 6
(read_25_disablecopyonread_adam_v_c1_bias: D
*read_26_disablecopyonread_adam_m_c4_kernel: @D
*read_27_disablecopyonread_adam_v_c4_kernel: @6
(read_28_disablecopyonread_adam_m_c4_bias:@6
(read_29_disablecopyonread_adam_v_c4_bias:@D
*read_30_disablecopyonread_adam_m_c5_kernel:@@D
*read_31_disablecopyonread_adam_v_c5_kernel:@@6
(read_32_disablecopyonread_adam_m_c5_bias:@6
(read_33_disablecopyonread_adam_v_c5_bias:@E
*read_34_disablecopyonread_adam_m_c8_kernel:@�E
*read_35_disablecopyonread_adam_v_c8_kernel:@�7
(read_36_disablecopyonread_adam_m_c8_bias:	�7
(read_37_disablecopyonread_adam_v_c8_bias:	�F
*read_38_disablecopyonread_adam_m_c9_kernel:��F
*read_39_disablecopyonread_adam_v_c9_kernel:��7
(read_40_disablecopyonread_adam_m_c9_bias:	�7
(read_41_disablecopyonread_adam_v_c9_bias:	�?
+read_42_disablecopyonread_adam_m_d13_kernel:
��?
+read_43_disablecopyonread_adam_v_d13_kernel:
��8
)read_44_disablecopyonread_adam_m_d13_bias:	�8
)read_45_disablecopyonread_adam_v_d13_bias:	�>
+read_46_disablecopyonread_adam_m_d15_kernel:	�
>
+read_47_disablecopyonread_adam_v_d15_kernel:	�
7
)read_48_disablecopyonread_adam_m_d15_bias:
7
)read_49_disablecopyonread_adam_v_d15_bias:
+
!read_50_disablecopyonread_total_1: +
!read_51_disablecopyonread_count_1: )
read_52_disablecopyonread_total: )
read_53_disablecopyonread_count: 
savev2_const
identity_109��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: r
Read/DisableCopyOnReadDisableCopyOnRead read_disablecopyonread_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp read_disablecopyonread_c0_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: t
Read_1/DisableCopyOnReadDisableCopyOnRead read_1_disablecopyonread_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp read_1_disablecopyonread_c0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_c1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:  t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_c1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_c4_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_c4_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: @t
Read_5/DisableCopyOnReadDisableCopyOnRead read_5_disablecopyonread_c4_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp read_5_disablecopyonread_c4_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_c5_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_c5_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@t
Read_7/DisableCopyOnReadDisableCopyOnRead read_7_disablecopyonread_c5_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp read_7_disablecopyonread_c5_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_c8_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_c8_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0w
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�t
Read_9/DisableCopyOnReadDisableCopyOnRead read_9_disablecopyonread_c8_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp read_9_disablecopyonread_c8_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_c9_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_c9_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*(
_output_shapes
:��v
Read_11/DisableCopyOnReadDisableCopyOnRead!read_11_disablecopyonread_c9_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp!read_11_disablecopyonread_c9_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�y
Read_12/DisableCopyOnReadDisableCopyOnRead$read_12_disablecopyonread_d13_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp$read_12_disablecopyonread_d13_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��w
Read_13/DisableCopyOnReadDisableCopyOnRead"read_13_disablecopyonread_d13_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp"read_13_disablecopyonread_d13_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�y
Read_14/DisableCopyOnReadDisableCopyOnRead$read_14_disablecopyonread_d15_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp$read_14_disablecopyonread_d15_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
w
Read_15/DisableCopyOnReadDisableCopyOnRead"read_15_disablecopyonread_d15_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp"read_15_disablecopyonread_d15_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:
x
Read_16/DisableCopyOnReadDisableCopyOnRead#read_16_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp#read_16_disablecopyonread_iteration^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_learning_rate^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_18/DisableCopyOnReadDisableCopyOnRead*read_18_disablecopyonread_adam_m_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp*read_18_disablecopyonread_adam_m_c0_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_19/DisableCopyOnReadDisableCopyOnRead*read_19_disablecopyonread_adam_v_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp*read_19_disablecopyonread_adam_v_c0_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*&
_output_shapes
: }
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_adam_m_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_adam_m_c0_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_21/DisableCopyOnReadDisableCopyOnRead(read_21_disablecopyonread_adam_v_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp(read_21_disablecopyonread_adam_v_c0_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_22/DisableCopyOnReadDisableCopyOnRead*read_22_disablecopyonread_adam_m_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp*read_22_disablecopyonread_adam_m_c1_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
:  
Read_23/DisableCopyOnReadDisableCopyOnRead*read_23_disablecopyonread_adam_v_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp*read_23_disablecopyonread_adam_v_c1_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*&
_output_shapes
:  }
Read_24/DisableCopyOnReadDisableCopyOnRead(read_24_disablecopyonread_adam_m_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp(read_24_disablecopyonread_adam_m_c1_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_adam_v_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_adam_v_c1_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_26/DisableCopyOnReadDisableCopyOnRead*read_26_disablecopyonread_adam_m_c4_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp*read_26_disablecopyonread_adam_m_c4_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_27/DisableCopyOnReadDisableCopyOnRead*read_27_disablecopyonread_adam_v_c4_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp*read_27_disablecopyonread_adam_v_c4_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
: @}
Read_28/DisableCopyOnReadDisableCopyOnRead(read_28_disablecopyonread_adam_m_c4_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp(read_28_disablecopyonread_adam_m_c4_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_29/DisableCopyOnReadDisableCopyOnRead(read_29_disablecopyonread_adam_v_c4_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp(read_29_disablecopyonread_adam_v_c4_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_30/DisableCopyOnReadDisableCopyOnRead*read_30_disablecopyonread_adam_m_c5_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp*read_30_disablecopyonread_adam_m_c5_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@
Read_31/DisableCopyOnReadDisableCopyOnRead*read_31_disablecopyonread_adam_v_c5_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp*read_31_disablecopyonread_adam_v_c5_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@}
Read_32/DisableCopyOnReadDisableCopyOnRead(read_32_disablecopyonread_adam_m_c5_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp(read_32_disablecopyonread_adam_m_c5_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_33/DisableCopyOnReadDisableCopyOnRead(read_33_disablecopyonread_adam_v_c5_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp(read_33_disablecopyonread_adam_v_c5_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_34/DisableCopyOnReadDisableCopyOnRead*read_34_disablecopyonread_adam_m_c8_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp*read_34_disablecopyonread_adam_m_c8_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�
Read_35/DisableCopyOnReadDisableCopyOnRead*read_35_disablecopyonread_adam_v_c8_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp*read_35_disablecopyonread_adam_v_c8_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�}
Read_36/DisableCopyOnReadDisableCopyOnRead(read_36_disablecopyonread_adam_m_c8_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp(read_36_disablecopyonread_adam_m_c8_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_37/DisableCopyOnReadDisableCopyOnRead(read_37_disablecopyonread_adam_v_c8_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp(read_37_disablecopyonread_adam_v_c8_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_38/DisableCopyOnReadDisableCopyOnRead*read_38_disablecopyonread_adam_m_c9_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp*read_38_disablecopyonread_adam_m_c9_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*(
_output_shapes
:��
Read_39/DisableCopyOnReadDisableCopyOnRead*read_39_disablecopyonread_adam_v_c9_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp*read_39_disablecopyonread_adam_v_c9_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_40/DisableCopyOnReadDisableCopyOnRead(read_40_disablecopyonread_adam_m_c9_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp(read_40_disablecopyonread_adam_m_c9_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_41/DisableCopyOnReadDisableCopyOnRead(read_41_disablecopyonread_adam_v_c9_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp(read_41_disablecopyonread_adam_v_c9_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead+read_42_disablecopyonread_adam_m_d13_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp+read_42_disablecopyonread_adam_m_d13_kernel^Read_42/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_43/DisableCopyOnReadDisableCopyOnRead+read_43_disablecopyonread_adam_v_d13_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp+read_43_disablecopyonread_adam_v_d13_kernel^Read_43/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��~
Read_44/DisableCopyOnReadDisableCopyOnRead)read_44_disablecopyonread_adam_m_d13_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp)read_44_disablecopyonread_adam_m_d13_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_45/DisableCopyOnReadDisableCopyOnRead)read_45_disablecopyonread_adam_v_d13_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp)read_45_disablecopyonread_adam_v_d13_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_46/DisableCopyOnReadDisableCopyOnRead+read_46_disablecopyonread_adam_m_d15_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp+read_46_disablecopyonread_adam_m_d15_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_47/DisableCopyOnReadDisableCopyOnRead+read_47_disablecopyonread_adam_v_d15_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp+read_47_disablecopyonread_adam_v_d15_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
~
Read_48/DisableCopyOnReadDisableCopyOnRead)read_48_disablecopyonread_adam_m_d15_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp)read_48_disablecopyonread_adam_m_d15_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:
~
Read_49/DisableCopyOnReadDisableCopyOnRead)read_49_disablecopyonread_adam_v_d15_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp)read_49_disablecopyonread_adam_v_d15_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_total_1^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_count_1^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_52/DisableCopyOnReadDisableCopyOnReadread_52_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpread_52_disablecopyonread_total^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_53/DisableCopyOnReadDisableCopyOnReadread_53_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpread_53_disablecopyonread_count^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_108Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_109IdentityIdentity_108:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_109Identity_109:output:0*�
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:7

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
=__inference_c1_layer_call_and_return_conditional_losses_33115

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
>
"__inference_m6_layer_call_fn_33201

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31885�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_33995
file_prefix4
assignvariableop_c0_kernel: (
assignvariableop_1_c0_bias: 6
assignvariableop_2_c1_kernel:  (
assignvariableop_3_c1_bias: 6
assignvariableop_4_c4_kernel: @(
assignvariableop_5_c4_bias:@6
assignvariableop_6_c5_kernel:@@(
assignvariableop_7_c5_bias:@7
assignvariableop_8_c8_kernel:@�)
assignvariableop_9_c8_bias:	�9
assignvariableop_10_c9_kernel:��*
assignvariableop_11_c9_bias:	�2
assignvariableop_12_d13_kernel:
��+
assignvariableop_13_d13_bias:	�1
assignvariableop_14_d15_kernel:	�
*
assignvariableop_15_d15_bias:
'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: >
$assignvariableop_18_adam_m_c0_kernel: >
$assignvariableop_19_adam_v_c0_kernel: 0
"assignvariableop_20_adam_m_c0_bias: 0
"assignvariableop_21_adam_v_c0_bias: >
$assignvariableop_22_adam_m_c1_kernel:  >
$assignvariableop_23_adam_v_c1_kernel:  0
"assignvariableop_24_adam_m_c1_bias: 0
"assignvariableop_25_adam_v_c1_bias: >
$assignvariableop_26_adam_m_c4_kernel: @>
$assignvariableop_27_adam_v_c4_kernel: @0
"assignvariableop_28_adam_m_c4_bias:@0
"assignvariableop_29_adam_v_c4_bias:@>
$assignvariableop_30_adam_m_c5_kernel:@@>
$assignvariableop_31_adam_v_c5_kernel:@@0
"assignvariableop_32_adam_m_c5_bias:@0
"assignvariableop_33_adam_v_c5_bias:@?
$assignvariableop_34_adam_m_c8_kernel:@�?
$assignvariableop_35_adam_v_c8_kernel:@�1
"assignvariableop_36_adam_m_c8_bias:	�1
"assignvariableop_37_adam_v_c8_bias:	�@
$assignvariableop_38_adam_m_c9_kernel:��@
$assignvariableop_39_adam_v_c9_kernel:��1
"assignvariableop_40_adam_m_c9_bias:	�1
"assignvariableop_41_adam_v_c9_bias:	�9
%assignvariableop_42_adam_m_d13_kernel:
��9
%assignvariableop_43_adam_v_d13_kernel:
��2
#assignvariableop_44_adam_m_d13_bias:	�2
#assignvariableop_45_adam_v_d13_bias:	�8
%assignvariableop_46_adam_m_d15_kernel:	�
8
%assignvariableop_47_adam_v_d15_kernel:	�
1
#assignvariableop_48_adam_m_d15_bias:
1
#assignvariableop_49_adam_v_d15_bias:
%
assignvariableop_50_total_1: %
assignvariableop_51_count_1: #
assignvariableop_52_total: #
assignvariableop_53_count: 
identity_55��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_c0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_c0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_c1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_c1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_c4_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_c4_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_c5_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_c5_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_c8_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_c8_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_c9_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_c9_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_d13_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_d13_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_d15_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_d15_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterationIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_adam_m_c0_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_v_c0_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_m_c0_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_adam_v_c0_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_adam_m_c1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_v_c1_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_m_c1_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_adam_v_c1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_adam_m_c4_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_v_c4_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_m_c4_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp"assignvariableop_29_adam_v_c4_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_adam_m_c5_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_v_c5_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_m_c5_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp"assignvariableop_33_adam_v_c5_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_adam_m_c8_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp$assignvariableop_35_adam_v_c8_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp"assignvariableop_36_adam_m_c8_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp"assignvariableop_37_adam_v_c8_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp$assignvariableop_38_adam_m_c9_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp$assignvariableop_39_adam_v_c9_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp"assignvariableop_40_adam_m_c9_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp"assignvariableop_41_adam_v_c9_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_m_d13_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adam_v_d13_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp#assignvariableop_44_adam_m_d13_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp#assignvariableop_45_adam_v_d13_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_m_d15_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp%assignvariableop_47_adam_v_d15_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp#assignvariableop_48_adam_m_d15_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp#assignvariableop_49_adam_v_d15_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_totalIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_countIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*�
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
]
$__inference_dr14_layer_call_fn_33352

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_32105p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
=__inference_c8_layer_call_and_return_conditional_losses_32026

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_33147

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
#__inference_d13_layer_call_fn_33334

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_32087p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
>__inference_d13_layer_call_and_return_conditional_losses_33347

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
?__inference_dr11_layer_call_and_return_conditional_losses_32195

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
Z
>__inference_m10_layer_call_and_return_conditional_losses_33287

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
=__inference_c9_layer_call_and_return_conditional_losses_32045

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
K
"__inference__update_step_xla_33061
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
�
"__inference_c0_layer_call_fn_33080

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31920w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
=__inference_c1_layer_call_and_return_conditional_losses_31939

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
=__inference_c4_layer_call_and_return_conditional_losses_31973

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

^
?__inference_dr14_layer_call_and_return_conditional_losses_32105

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_33125

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
+
__inference_loss_fn_8_33441
identity`
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
+
__inference_loss_fn_6_33431
identity`
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
V
"__inference__update_step_xla_33026
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient
�
+
__inference_loss_fn_1_33406
identity^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
"__inference_c8_layer_call_fn_33242

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_32026x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
J
"__inference__update_step_xla_33011
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
: 
"
_user_specified_name
gradient
�
]
?__inference_dr14_layer_call_and_return_conditional_losses_32207

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�V
�
E__inference_sequential_layer_call_and_return_conditional_losses_32991

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource:  0
"c1_biasadd_readvariableop_resource: ;
!c4_conv2d_readvariableop_resource: @0
"c4_biasadd_readvariableop_resource:@;
!c5_conv2d_readvariableop_resource:@@0
"c5_biasadd_readvariableop_resource:@<
!c8_conv2d_readvariableop_resource:@�1
"c8_biasadd_readvariableop_resource:	�=
!c9_conv2d_readvariableop_resource:��1
"c9_biasadd_readvariableop_resource:	�6
"d13_matmul_readvariableop_resource:
��2
#d13_biasadd_readvariableop_resource:	�5
"d15_matmul_readvariableop_resource:	�
1
#d15_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�c4/BiasAdd/ReadVariableOp�c4/Conv2D/ReadVariableOp�c5/BiasAdd/ReadVariableOp�c5/Conv2D/ReadVariableOp�c8/BiasAdd/ReadVariableOp�c8/Conv2D/ReadVariableOp�c9/BiasAdd/ReadVariableOp�c9/Conv2D/ReadVariableOp�d13/BiasAdd/ReadVariableOp�d13/MatMul/ReadVariableOp�d15/BiasAdd/ReadVariableOp�d15/MatMul/ReadVariableOp�
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
g
dr3/IdentityIdentitym2/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
	c4/Conv2DConv2Ddr3/Identity:output:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c4/BiasAdd/ReadVariableOpReadVariableOp"c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c4/BiasAddBiasAddc4/Conv2D:output:0!c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c5/BiasAdd/ReadVariableOpReadVariableOp"c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c5/BiasAddBiasAddc5/Conv2D:output:0!c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
g
dr7/IdentityIdentitym6/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
c8/Conv2D/ReadVariableOpReadVariableOp!c8_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
	c8/Conv2DConv2Ddr7/Identity:output:0 c8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c8/BiasAdd/ReadVariableOpReadVariableOp"c8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c8/BiasAddBiasAddc8/Conv2D:output:0!c8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c8/ReluReluc8/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
c9/Conv2D/ReadVariableOpReadVariableOp!c9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
	c9/Conv2DConv2Dc8/Relu:activations:0 c9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c9/BiasAdd/ReadVariableOpReadVariableOp"c9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c9/BiasAddBiasAddc9/Conv2D:output:0!c9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c9/ReluReluc9/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
m10/MaxPoolMaxPoolc9/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
j
dr11/IdentityIdentitym10/MaxPool:output:0*
T0*0
_output_shapes
:����������Z
	f12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   u
f12/ReshapeReshapedr11/Identity:output:0f12/Const:output:0*
T0*(
_output_shapes
:����������~
d13/MatMul/ReadVariableOpReadVariableOp"d13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�

d13/MatMulMatMulf12/Reshape:output:0!d13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
d13/BiasAdd/ReadVariableOpReadVariableOp#d13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
d13/BiasAddBiasAddd13/MatMul:product:0"d13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Y
d13/ReluRelud13/BiasAdd:output:0*
T0*(
_output_shapes
:����������d
dr14/IdentityIdentityd13/Relu:activations:0*
T0*(
_output_shapes
:����������}
d15/MatMul/ReadVariableOpReadVariableOp"d15_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�

d15/MatMulMatMuldr14/Identity:output:0!d15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
d15/BiasAdd/ReadVariableOpReadVariableOp#d15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
d15/BiasAddBiasAddd15/MatMul:product:0"d15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
^
d15/SoftmaxSoftmaxd15/BiasAdd:output:0*
T0*'
_output_shapes
:���������
`
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentityd15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^c8/BiasAdd/ReadVariableOp^c8/Conv2D/ReadVariableOp^c9/BiasAdd/ReadVariableOp^c9/Conv2D/ReadVariableOp^d13/BiasAdd/ReadVariableOp^d13/MatMul/ReadVariableOp^d15/BiasAdd/ReadVariableOp^d15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
c4/BiasAdd/ReadVariableOpc4/BiasAdd/ReadVariableOp24
c4/Conv2D/ReadVariableOpc4/Conv2D/ReadVariableOp26
c5/BiasAdd/ReadVariableOpc5/BiasAdd/ReadVariableOp24
c5/Conv2D/ReadVariableOpc5/Conv2D/ReadVariableOp26
c8/BiasAdd/ReadVariableOpc8/BiasAdd/ReadVariableOp24
c8/Conv2D/ReadVariableOpc8/Conv2D/ReadVariableOp26
c9/BiasAdd/ReadVariableOpc9/BiasAdd/ReadVariableOp24
c9/Conv2D/ReadVariableOpc9/Conv2D/ReadVariableOp28
d13/BiasAdd/ReadVariableOpd13/BiasAdd/ReadVariableOp26
d13/MatMul/ReadVariableOpd13/MatMul/ReadVariableOp28
d15/BiasAdd/ReadVariableOpd15/BiasAdd/ReadVariableOp26
d15/MatMul/ReadVariableOpd15/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
J
"__inference__update_step_xla_33001
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
: 
"
_user_specified_name
gradient
�
J
"__inference__update_step_xla_33031
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
�
J
"__inference__update_step_xla_33071
gradient
variable:
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:
: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:

"
_user_specified_name
gradient
�
W
"__inference__update_step_xla_33036
gradient#
variable:@�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:@�: *
	_noinline(:($
"
_user_specified_name
variable:Q M
'
_output_shapes
:@�
"
_user_specified_name
gradient
�
?
#__inference_f12_layer_call_fn_33319

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_32072a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
"__inference__update_step_xla_33056
gradient
variable:
��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
��: *
	_noinline(:($
"
_user_specified_name
variable:J F
 
_output_shapes
:
��
"
_user_specified_name
gradient
�s
�
E__inference_sequential_layer_call_and_return_conditional_losses_32906

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource:  0
"c1_biasadd_readvariableop_resource: ;
!c4_conv2d_readvariableop_resource: @0
"c4_biasadd_readvariableop_resource:@;
!c5_conv2d_readvariableop_resource:@@0
"c5_biasadd_readvariableop_resource:@<
!c8_conv2d_readvariableop_resource:@�1
"c8_biasadd_readvariableop_resource:	�=
!c9_conv2d_readvariableop_resource:��1
"c9_biasadd_readvariableop_resource:	�6
"d13_matmul_readvariableop_resource:
��2
#d13_biasadd_readvariableop_resource:	�5
"d15_matmul_readvariableop_resource:	�
1
#d15_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�c4/BiasAdd/ReadVariableOp�c4/Conv2D/ReadVariableOp�c5/BiasAdd/ReadVariableOp�c5/Conv2D/ReadVariableOp�c8/BiasAdd/ReadVariableOp�c8/Conv2D/ReadVariableOp�c9/BiasAdd/ReadVariableOp�c9/Conv2D/ReadVariableOp�d13/BiasAdd/ReadVariableOp�d13/MatMul/ReadVariableOp�d15/BiasAdd/ReadVariableOp�d15/MatMul/ReadVariableOp�
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
V
dr3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?�
dr3/dropout/MulMulm2/MaxPool:output:0dr3/dropout/Const:output:0*
T0*/
_output_shapes
:��������� b
dr3/dropout/ShapeShapem2/MaxPool:output:0*
T0*
_output_shapes
::���
(dr3/dropout/random_uniform/RandomUniformRandomUniformdr3/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0*
seed2����_
dr3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dr3/dropout/GreaterEqualGreaterEqual1dr3/dropout/random_uniform/RandomUniform:output:0#dr3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� X
dr3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr3/dropout/SelectV2SelectV2dr3/dropout/GreaterEqual:z:0dr3/dropout/Mul:z:0dr3/dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� �
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
	c4/Conv2DConv2Ddr3/dropout/SelectV2:output:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c4/BiasAdd/ReadVariableOpReadVariableOp"c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c4/BiasAddBiasAddc4/Conv2D:output:0!c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c5/BiasAdd/ReadVariableOpReadVariableOp"c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c5/BiasAddBiasAddc5/Conv2D:output:0!c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
V
dr7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dr7/dropout/MulMulm6/MaxPool:output:0dr7/dropout/Const:output:0*
T0*/
_output_shapes
:���������@b
dr7/dropout/ShapeShapem6/MaxPool:output:0*
T0*
_output_shapes
::���
(dr7/dropout/random_uniform/RandomUniformRandomUniformdr7/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype0*
seed2_
dr7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dr7/dropout/GreaterEqualGreaterEqual1dr7/dropout/random_uniform/RandomUniform:output:0#dr7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@X
dr7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr7/dropout/SelectV2SelectV2dr7/dropout/GreaterEqual:z:0dr7/dropout/Mul:z:0dr7/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@�
c8/Conv2D/ReadVariableOpReadVariableOp!c8_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
	c8/Conv2DConv2Ddr7/dropout/SelectV2:output:0 c8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c8/BiasAdd/ReadVariableOpReadVariableOp"c8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c8/BiasAddBiasAddc8/Conv2D:output:0!c8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c8/ReluReluc8/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
c9/Conv2D/ReadVariableOpReadVariableOp!c9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
	c9/Conv2DConv2Dc8/Relu:activations:0 c9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c9/BiasAdd/ReadVariableOpReadVariableOp"c9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c9/BiasAddBiasAddc9/Conv2D:output:0!c9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c9/ReluReluc9/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
m10/MaxPoolMaxPoolc9/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
W
dr11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dr11/dropout/MulMulm10/MaxPool:output:0dr11/dropout/Const:output:0*
T0*0
_output_shapes
:����������d
dr11/dropout/ShapeShapem10/MaxPool:output:0*
T0*
_output_shapes
::���
)dr11/dropout/random_uniform/RandomUniformRandomUniformdr11/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0*
seed2`
dr11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dr11/dropout/GreaterEqualGreaterEqual2dr11/dropout/random_uniform/RandomUniform:output:0$dr11/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������Y
dr11/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr11/dropout/SelectV2SelectV2dr11/dropout/GreaterEqual:z:0dr11/dropout/Mul:z:0dr11/dropout/Const_1:output:0*
T0*0
_output_shapes
:����������Z
	f12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   }
f12/ReshapeReshapedr11/dropout/SelectV2:output:0f12/Const:output:0*
T0*(
_output_shapes
:����������~
d13/MatMul/ReadVariableOpReadVariableOp"d13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�

d13/MatMulMatMulf12/Reshape:output:0!d13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
d13/BiasAdd/ReadVariableOpReadVariableOp#d13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
d13/BiasAddBiasAddd13/MatMul:product:0"d13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Y
d13/ReluRelud13/BiasAdd:output:0*
T0*(
_output_shapes
:����������W
dr14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dr14/dropout/MulMuld13/Relu:activations:0dr14/dropout/Const:output:0*
T0*(
_output_shapes
:����������f
dr14/dropout/ShapeShaped13/Relu:activations:0*
T0*
_output_shapes
::���
)dr14/dropout/random_uniform/RandomUniformRandomUniformdr14/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2`
dr14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dr14/dropout/GreaterEqualGreaterEqual2dr14/dropout/random_uniform/RandomUniform:output:0$dr14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������Y
dr14/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr14/dropout/SelectV2SelectV2dr14/dropout/GreaterEqual:z:0dr14/dropout/Mul:z:0dr14/dropout/Const_1:output:0*
T0*(
_output_shapes
:����������}
d15/MatMul/ReadVariableOpReadVariableOp"d15_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�

d15/MatMulMatMuldr14/dropout/SelectV2:output:0!d15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
d15/BiasAdd/ReadVariableOpReadVariableOp#d15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
d15/BiasAddBiasAddd15/MatMul:product:0"d15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
^
d15/SoftmaxSoftmaxd15/BiasAdd:output:0*
T0*'
_output_shapes
:���������
`
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentityd15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^c8/BiasAdd/ReadVariableOp^c8/Conv2D/ReadVariableOp^c9/BiasAdd/ReadVariableOp^c9/Conv2D/ReadVariableOp^d13/BiasAdd/ReadVariableOp^d13/MatMul/ReadVariableOp^d15/BiasAdd/ReadVariableOp^d15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
c4/BiasAdd/ReadVariableOpc4/BiasAdd/ReadVariableOp24
c4/Conv2D/ReadVariableOpc4/Conv2D/ReadVariableOp26
c5/BiasAdd/ReadVariableOpc5/BiasAdd/ReadVariableOp24
c5/Conv2D/ReadVariableOpc5/Conv2D/ReadVariableOp26
c8/BiasAdd/ReadVariableOpc8/BiasAdd/ReadVariableOp24
c8/Conv2D/ReadVariableOpc8/Conv2D/ReadVariableOp26
c9/BiasAdd/ReadVariableOpc9/BiasAdd/ReadVariableOp24
c9/Conv2D/ReadVariableOpc9/Conv2D/ReadVariableOp28
d13/BiasAdd/ReadVariableOpd13/BiasAdd/ReadVariableOp26
d13/MatMul/ReadVariableOpd13/MatMul/ReadVariableOp28
d15/BiasAdd/ReadVariableOpd15/BiasAdd/ReadVariableOp26
d15/MatMul/ReadVariableOpd15/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
\
#__inference_dr3_layer_call_fn_33130

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31958w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�M
�
E__inference_sequential_layer_call_and_return_conditional_losses_32143
c0_input"
c0_31921: 
c0_31923: "
c1_31940:  
c1_31942: "
c4_31974: @
c4_31976:@"
c5_31993:@@
c5_31995:@#
c8_32027:@�
c8_32029:	�$
c9_32046:��
c9_32048:	�
	d13_32088:
��
	d13_32090:	�
	d15_32121:	�

	d15_32123:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�dr11/StatefulPartitionedCall�dr14/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_31921c0_31923*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31920�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_31940c1_31942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31939�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31873�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31958�
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_31974c4_31976*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31973�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_31993c5_31995*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31992�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31885�
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_32011�
c8/StatefulPartitionedCallStatefulPartitionedCall$dr7/StatefulPartitionedCall:output:0c8_32027c8_32029*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_32026�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_32046c9_32048*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_32045�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_31897�
dr11/StatefulPartitionedCallStatefulPartitionedCallm10/PartitionedCall:output:0^dr7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_32064�
f12/PartitionedCallPartitionedCall%dr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_32072�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_32088	d13_32090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_32087�
dr14/StatefulPartitionedCallStatefulPartitionedCall$d13/StatefulPartitionedCall:output:0^dr11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_32105�
d15/StatefulPartitionedCallStatefulPartitionedCall%dr14/StatefulPartitionedCall:output:0	d15_32121	d15_32123*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_32120`
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall^dr11/StatefulPartitionedCall^dr14/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall2<
dr11/StatefulPartitionedCalldr11/StatefulPartitionedCall2<
dr14/StatefulPartitionedCalldr14/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr7/StatefulPartitionedCalldr7/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�

^
?__inference_dr14_layer_call_and_return_conditional_losses_33369

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
+
__inference_loss_fn_2_33411
identity`
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�

]
>__inference_dr7_layer_call_and_return_conditional_losses_32011

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_c9_layer_call_and_return_conditional_losses_33277

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
+
__inference_loss_fn_9_33446
identity^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
+
__inference_loss_fn_4_33421
identity`
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�M
�
E__inference_sequential_layer_call_and_return_conditional_losses_32302

inputs"
c0_32237: 
c0_32239: "
c1_32242:  
c1_32244: "
c4_32249: @
c4_32251:@"
c5_32254:@@
c5_32256:@#
c8_32261:@�
c8_32263:	�$
c9_32266:��
c9_32268:	�
	d13_32274:
��
	d13_32276:	�
	d15_32280:	�

	d15_32282:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�dr11/StatefulPartitionedCall�dr14/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_32237c0_32239*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31920�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_32242c1_32244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31939�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31873�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31958�
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_32249c4_32251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31973�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_32254c5_32256*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31992�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31885�
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_32011�
c8/StatefulPartitionedCallStatefulPartitionedCall$dr7/StatefulPartitionedCall:output:0c8_32261c8_32263*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_32026�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_32266c9_32268*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_32045�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_31897�
dr11/StatefulPartitionedCallStatefulPartitionedCallm10/PartitionedCall:output:0^dr7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_32064�
f12/PartitionedCallPartitionedCall%dr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_32072�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_32274	d13_32276*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_32087�
dr14/StatefulPartitionedCallStatefulPartitionedCall$d13/StatefulPartitionedCall:output:0^dr11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_32105�
d15/StatefulPartitionedCallStatefulPartitionedCall%dr14/StatefulPartitionedCall:output:0	d15_32280	d15_32282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_32120`
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall^dr11/StatefulPartitionedCall^dr14/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall2<
dr11/StatefulPartitionedCalldr11/StatefulPartitionedCall2<
dr14/StatefulPartitionedCalldr14/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr7/StatefulPartitionedCalldr7/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
]
$__inference_dr11_layer_call_fn_33292

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_32064x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�H
�
E__inference_sequential_layer_call_and_return_conditional_losses_32231
c0_input"
c0_32146: 
c0_32148: "
c1_32151:  
c1_32153: "
c4_32163: @
c4_32165:@"
c5_32168:@@
c5_32170:@#
c8_32180:@�
c8_32182:	�$
c9_32185:��
c9_32187:	�
	d13_32198:
��
	d13_32200:	�
	d15_32209:	�

	d15_32211:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_32146c0_32148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31920�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_32151c1_32153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31939�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31873�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_32161�
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_32163c4_32165*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31973�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_32168c5_32170*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31992�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31885�
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_32178�
c8/StatefulPartitionedCallStatefulPartitionedCalldr7/PartitionedCall:output:0c8_32180c8_32182*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_32026�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_32185c9_32187*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_32045�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_31897�
dr11/PartitionedCallPartitionedCallm10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_32195�
f12/PartitionedCallPartitionedCalldr11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_32072�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_32198	d13_32200*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_32087�
dr14/PartitionedCallPartitionedCall$d13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_32207�
d15/StatefulPartitionedCallStatefulPartitionedCalldr14/PartitionedCall:output:0	d15_32209	d15_32211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_32120`
c0/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c0/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
>
"__inference_m2_layer_call_fn_33120

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31873�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
=__inference_c5_layer_call_and_return_conditional_losses_33196

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
c5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�Z
�
 __inference__wrapped_model_31867
c0_inputF
,sequential_c0_conv2d_readvariableop_resource: ;
-sequential_c0_biasadd_readvariableop_resource: F
,sequential_c1_conv2d_readvariableop_resource:  ;
-sequential_c1_biasadd_readvariableop_resource: F
,sequential_c4_conv2d_readvariableop_resource: @;
-sequential_c4_biasadd_readvariableop_resource:@F
,sequential_c5_conv2d_readvariableop_resource:@@;
-sequential_c5_biasadd_readvariableop_resource:@G
,sequential_c8_conv2d_readvariableop_resource:@�<
-sequential_c8_biasadd_readvariableop_resource:	�H
,sequential_c9_conv2d_readvariableop_resource:��<
-sequential_c9_biasadd_readvariableop_resource:	�A
-sequential_d13_matmul_readvariableop_resource:
��=
.sequential_d13_biasadd_readvariableop_resource:	�@
-sequential_d15_matmul_readvariableop_resource:	�
<
.sequential_d15_biasadd_readvariableop_resource:

identity��$sequential/c0/BiasAdd/ReadVariableOp�#sequential/c0/Conv2D/ReadVariableOp�$sequential/c1/BiasAdd/ReadVariableOp�#sequential/c1/Conv2D/ReadVariableOp�$sequential/c4/BiasAdd/ReadVariableOp�#sequential/c4/Conv2D/ReadVariableOp�$sequential/c5/BiasAdd/ReadVariableOp�#sequential/c5/Conv2D/ReadVariableOp�$sequential/c8/BiasAdd/ReadVariableOp�#sequential/c8/Conv2D/ReadVariableOp�$sequential/c9/BiasAdd/ReadVariableOp�#sequential/c9/Conv2D/ReadVariableOp�%sequential/d13/BiasAdd/ReadVariableOp�$sequential/d13/MatMul/ReadVariableOp�%sequential/d15/BiasAdd/ReadVariableOp�$sequential/d15/MatMul/ReadVariableOp�
#sequential/c0/Conv2D/ReadVariableOpReadVariableOp,sequential_c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential/c0/Conv2DConv2Dc0_input+sequential/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
$sequential/c0/BiasAdd/ReadVariableOpReadVariableOp-sequential_c0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential/c0/BiasAddBiasAddsequential/c0/Conv2D:output:0,sequential/c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   t
sequential/c0/ReluRelusequential/c0/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
#sequential/c1/Conv2D/ReadVariableOpReadVariableOp,sequential_c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
sequential/c1/Conv2DConv2D sequential/c0/Relu:activations:0+sequential/c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
$sequential/c1/BiasAdd/ReadVariableOpReadVariableOp-sequential_c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential/c1/BiasAddBiasAddsequential/c1/Conv2D:output:0,sequential/c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   t
sequential/c1/ReluRelusequential/c1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
sequential/m2/MaxPoolMaxPool sequential/c1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
}
sequential/dr3/IdentityIdentitysequential/m2/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
#sequential/c4/Conv2D/ReadVariableOpReadVariableOp,sequential_c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
sequential/c4/Conv2DConv2D sequential/dr3/Identity:output:0+sequential/c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
$sequential/c4/BiasAdd/ReadVariableOpReadVariableOp-sequential_c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/c4/BiasAddBiasAddsequential/c4/Conv2D:output:0,sequential/c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@t
sequential/c4/ReluRelusequential/c4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
#sequential/c5/Conv2D/ReadVariableOpReadVariableOp,sequential_c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
sequential/c5/Conv2DConv2D sequential/c4/Relu:activations:0+sequential/c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
$sequential/c5/BiasAdd/ReadVariableOpReadVariableOp-sequential_c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/c5/BiasAddBiasAddsequential/c5/Conv2D:output:0,sequential/c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@t
sequential/c5/ReluRelusequential/c5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
sequential/m6/MaxPoolMaxPool sequential/c5/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
}
sequential/dr7/IdentityIdentitysequential/m6/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
#sequential/c8/Conv2D/ReadVariableOpReadVariableOp,sequential_c8_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
sequential/c8/Conv2DConv2D sequential/dr7/Identity:output:0+sequential/c8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$sequential/c8/BiasAdd/ReadVariableOpReadVariableOp-sequential_c8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/c8/BiasAddBiasAddsequential/c8/Conv2D:output:0,sequential/c8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������u
sequential/c8/ReluRelusequential/c8/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
#sequential/c9/Conv2D/ReadVariableOpReadVariableOp,sequential_c9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
sequential/c9/Conv2DConv2D sequential/c8/Relu:activations:0+sequential/c9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$sequential/c9/BiasAdd/ReadVariableOpReadVariableOp-sequential_c9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/c9/BiasAddBiasAddsequential/c9/Conv2D:output:0,sequential/c9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������u
sequential/c9/ReluRelusequential/c9/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
sequential/m10/MaxPoolMaxPool sequential/c9/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
sequential/dr11/IdentityIdentitysequential/m10/MaxPool:output:0*
T0*0
_output_shapes
:����������e
sequential/f12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
sequential/f12/ReshapeReshape!sequential/dr11/Identity:output:0sequential/f12/Const:output:0*
T0*(
_output_shapes
:�����������
$sequential/d13/MatMul/ReadVariableOpReadVariableOp-sequential_d13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential/d13/MatMulMatMulsequential/f12/Reshape:output:0,sequential/d13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%sequential/d13/BiasAdd/ReadVariableOpReadVariableOp.sequential_d13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/d13/BiasAddBiasAddsequential/d13/MatMul:product:0-sequential/d13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
sequential/d13/ReluRelusequential/d13/BiasAdd:output:0*
T0*(
_output_shapes
:����������z
sequential/dr14/IdentityIdentity!sequential/d13/Relu:activations:0*
T0*(
_output_shapes
:�����������
$sequential/d15/MatMul/ReadVariableOpReadVariableOp-sequential_d15_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
sequential/d15/MatMulMatMul!sequential/dr14/Identity:output:0,sequential/d15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
%sequential/d15/BiasAdd/ReadVariableOpReadVariableOp.sequential_d15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential/d15/BiasAddBiasAddsequential/d15/MatMul:product:0-sequential/d15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
sequential/d15/SoftmaxSoftmaxsequential/d15/BiasAdd:output:0*
T0*'
_output_shapes
:���������
o
IdentityIdentity sequential/d15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/c1/BiasAdd/ReadVariableOp$^sequential/c1/Conv2D/ReadVariableOp%^sequential/c4/BiasAdd/ReadVariableOp$^sequential/c4/Conv2D/ReadVariableOp%^sequential/c5/BiasAdd/ReadVariableOp$^sequential/c5/Conv2D/ReadVariableOp%^sequential/c8/BiasAdd/ReadVariableOp$^sequential/c8/Conv2D/ReadVariableOp%^sequential/c9/BiasAdd/ReadVariableOp$^sequential/c9/Conv2D/ReadVariableOp&^sequential/d13/BiasAdd/ReadVariableOp%^sequential/d13/MatMul/ReadVariableOp&^sequential/d15/BiasAdd/ReadVariableOp%^sequential/d15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 2L
$sequential/c0/BiasAdd/ReadVariableOp$sequential/c0/BiasAdd/ReadVariableOp2J
#sequential/c0/Conv2D/ReadVariableOp#sequential/c0/Conv2D/ReadVariableOp2L
$sequential/c1/BiasAdd/ReadVariableOp$sequential/c1/BiasAdd/ReadVariableOp2J
#sequential/c1/Conv2D/ReadVariableOp#sequential/c1/Conv2D/ReadVariableOp2L
$sequential/c4/BiasAdd/ReadVariableOp$sequential/c4/BiasAdd/ReadVariableOp2J
#sequential/c4/Conv2D/ReadVariableOp#sequential/c4/Conv2D/ReadVariableOp2L
$sequential/c5/BiasAdd/ReadVariableOp$sequential/c5/BiasAdd/ReadVariableOp2J
#sequential/c5/Conv2D/ReadVariableOp#sequential/c5/Conv2D/ReadVariableOp2L
$sequential/c8/BiasAdd/ReadVariableOp$sequential/c8/BiasAdd/ReadVariableOp2J
#sequential/c8/Conv2D/ReadVariableOp#sequential/c8/Conv2D/ReadVariableOp2L
$sequential/c9/BiasAdd/ReadVariableOp$sequential/c9/BiasAdd/ReadVariableOp2J
#sequential/c9/Conv2D/ReadVariableOp#sequential/c9/Conv2D/ReadVariableOp2N
%sequential/d13/BiasAdd/ReadVariableOp%sequential/d13/BiasAdd/ReadVariableOp2L
$sequential/d13/MatMul/ReadVariableOp$sequential/d13/MatMul/ReadVariableOp2N
%sequential/d15/BiasAdd/ReadVariableOp%sequential/d15/BiasAdd/ReadVariableOp2L
$sequential/d15/MatMul/ReadVariableOp$sequential/d15/MatMul/ReadVariableOp:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
Z
>__inference_f12_layer_call_and_return_conditional_losses_32072

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
V
"__inference__update_step_xla_32996
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
: 
"
_user_specified_name
gradient
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_31958

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
+
__inference_loss_fn_3_33416
identity^
c1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
*__inference_sequential_layer_call_fn_32442
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_32407o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�

^
?__inference_dr11_layer_call_and_return_conditional_losses_33309

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
>__inference_d15_layer_call_and_return_conditional_losses_32120

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

^
?__inference_dr11_layer_call_and_return_conditional_losses_32064

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
c0_input9
serving_default_c0_input:0���������  7
d150
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias
 "_jit_compiled_convolution_op"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_random_generator"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias
 J_jit_compiled_convolution_op"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias
 `_jit_compiled_convolution_op"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias
 i_jit_compiled_convolution_op"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses
v_random_generator"
_tf_keras_layer
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
 0
!1
)2
*3
?4
@5
H6
I7
^8
_9
g10
h11
�12
�13
�14
�15"
trackable_list_wrapper
�
 0
!1
)2
*3
?4
@5
H6
I7
^8
_9
g10
h11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
*__inference_sequential_layer_call_fn_32337
*__inference_sequential_layer_call_fn_32442
*__inference_sequential_layer_call_fn_32756
*__inference_sequential_layer_call_fn_32793�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_32143
E__inference_sequential_layer_call_and_return_conditional_losses_32231
E__inference_sequential_layer_call_and_return_conditional_losses_32906
E__inference_sequential_layer_call_and_return_conditional_losses_32991�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_31867c0_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c0_layer_call_fn_33080�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_c0_layer_call_and_return_conditional_losses_33093�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:! 2	c0/kernel
: 2c0/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c1_layer_call_fn_33102�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_c1_layer_call_and_return_conditional_losses_33115�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!  2	c1/kernel
: 2c1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_m2_layer_call_fn_33120�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_m2_layer_call_and_return_conditional_losses_33125�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_dr3_layer_call_fn_33130
#__inference_dr3_layer_call_fn_33135�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_dr3_layer_call_and_return_conditional_losses_33147
>__inference_dr3_layer_call_and_return_conditional_losses_33152�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c4_layer_call_fn_33161�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_c4_layer_call_and_return_conditional_losses_33174�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:! @2	c4/kernel
:@2c4/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c5_layer_call_fn_33183�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_c5_layer_call_and_return_conditional_losses_33196�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!@@2	c5/kernel
:@2c5/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_m6_layer_call_fn_33201�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_m6_layer_call_and_return_conditional_losses_33206�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_dr7_layer_call_fn_33211
#__inference_dr7_layer_call_fn_33216�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_dr7_layer_call_and_return_conditional_losses_33228
>__inference_dr7_layer_call_and_return_conditional_losses_33233�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c8_layer_call_fn_33242�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_c8_layer_call_and_return_conditional_losses_33255�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"@�2	c8/kernel
:�2c8/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c9_layer_call_fn_33264�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
=__inference_c9_layer_call_and_return_conditional_losses_33277�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#��2	c9/kernel
:�2c9/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_m10_layer_call_fn_33282�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_m10_layer_call_and_return_conditional_losses_33287�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_dr11_layer_call_fn_33292
$__inference_dr11_layer_call_fn_33297�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_dr11_layer_call_and_return_conditional_losses_33309
?__inference_dr11_layer_call_and_return_conditional_losses_33314�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_f12_layer_call_fn_33319�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_f12_layer_call_and_return_conditional_losses_33325�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_d13_layer_call_fn_33334�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_d13_layer_call_and_return_conditional_losses_33347�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:
��2
d13/kernel
:�2d13/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_dr14_layer_call_fn_33352
$__inference_dr14_layer_call_fn_33357�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_dr14_layer_call_and_return_conditional_losses_33369
?__inference_dr14_layer_call_and_return_conditional_losses_33374�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_d15_layer_call_fn_33383�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_d15_layer_call_and_return_conditional_losses_33396�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�
2
d15/kernel
:
2d15/bias
�
�trace_02�
__inference_loss_fn_0_33401�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_33406�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_33411�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_33416�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_33421�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_33426�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_33431�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_33436�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_8_33441�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_9_33446�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_10_33451�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_11_33456�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_12_33461�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_13_33466�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_14_33471�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_15_33476�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_sequential_layer_call_fn_32337c0_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_32442c0_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_32756inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_32793inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_32143c0_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_32231c0_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_32906inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_32991inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�	
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_152�
"__inference__update_step_xla_32996
"__inference__update_step_xla_33001
"__inference__update_step_xla_33006
"__inference__update_step_xla_33011
"__inference__update_step_xla_33016
"__inference__update_step_xla_33021
"__inference__update_step_xla_33026
"__inference__update_step_xla_33031
"__inference__update_step_xla_33036
"__inference__update_step_xla_33041
"__inference__update_step_xla_33046
"__inference__update_step_xla_33051
"__inference__update_step_xla_33056
"__inference__update_step_xla_33061
"__inference__update_step_xla_33066
"__inference__update_step_xla_33071�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11z�trace_12z�trace_13z�trace_14z�trace_15
�B�
#__inference_signature_wrapper_32703c0_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c0_layer_call_fn_33080inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_c0_layer_call_and_return_conditional_losses_33093inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c1_layer_call_fn_33102inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_c1_layer_call_and_return_conditional_losses_33115inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_m2_layer_call_fn_33120inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_m2_layer_call_and_return_conditional_losses_33125inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_dr3_layer_call_fn_33130inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_dr3_layer_call_fn_33135inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_dr3_layer_call_and_return_conditional_losses_33147inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_dr3_layer_call_and_return_conditional_losses_33152inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c4_layer_call_fn_33161inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_c4_layer_call_and_return_conditional_losses_33174inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c5_layer_call_fn_33183inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_c5_layer_call_and_return_conditional_losses_33196inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_m6_layer_call_fn_33201inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_m6_layer_call_and_return_conditional_losses_33206inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_dr7_layer_call_fn_33211inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_dr7_layer_call_fn_33216inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_dr7_layer_call_and_return_conditional_losses_33228inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_dr7_layer_call_and_return_conditional_losses_33233inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c8_layer_call_fn_33242inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_c8_layer_call_and_return_conditional_losses_33255inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c9_layer_call_fn_33264inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_c9_layer_call_and_return_conditional_losses_33277inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_m10_layer_call_fn_33282inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_m10_layer_call_and_return_conditional_losses_33287inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_dr11_layer_call_fn_33292inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_dr11_layer_call_fn_33297inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dr11_layer_call_and_return_conditional_losses_33309inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dr11_layer_call_and_return_conditional_losses_33314inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_f12_layer_call_fn_33319inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_f12_layer_call_and_return_conditional_losses_33325inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_d13_layer_call_fn_33334inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_d13_layer_call_and_return_conditional_losses_33347inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_dr14_layer_call_fn_33352inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_dr14_layer_call_fn_33357inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dr14_layer_call_and_return_conditional_losses_33369inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dr14_layer_call_and_return_conditional_losses_33374inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_d15_layer_call_fn_33383inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_d15_layer_call_and_return_conditional_losses_33396inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_33401"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_33406"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_33411"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_33416"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_33421"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_33426"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_6_33431"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_7_33436"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_8_33441"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_9_33446"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_10_33451"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_11_33456"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_12_33461"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_13_33466"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_14_33471"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_15_33476"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
(:& 2Adam/m/c0/kernel
(:& 2Adam/v/c0/kernel
: 2Adam/m/c0/bias
: 2Adam/v/c0/bias
(:&  2Adam/m/c1/kernel
(:&  2Adam/v/c1/kernel
: 2Adam/m/c1/bias
: 2Adam/v/c1/bias
(:& @2Adam/m/c4/kernel
(:& @2Adam/v/c4/kernel
:@2Adam/m/c4/bias
:@2Adam/v/c4/bias
(:&@@2Adam/m/c5/kernel
(:&@@2Adam/v/c5/kernel
:@2Adam/m/c5/bias
:@2Adam/v/c5/bias
):'@�2Adam/m/c8/kernel
):'@�2Adam/v/c8/kernel
:�2Adam/m/c8/bias
:�2Adam/v/c8/bias
*:(��2Adam/m/c9/kernel
*:(��2Adam/v/c9/kernel
:�2Adam/m/c9/bias
:�2Adam/v/c9/bias
#:!
��2Adam/m/d13/kernel
#:!
��2Adam/v/d13/kernel
:�2Adam/m/d13/bias
:�2Adam/v/d13/bias
": 	�
2Adam/m/d15/kernel
": 	�
2Adam/v/d15/kernel
:
2Adam/m/d15/bias
:
2Adam/v/d15/bias
�B�
"__inference__update_step_xla_32996gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33001gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33006gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33011gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33016gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33021gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33026gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33031gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33036gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33041gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33046gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33051gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33056gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33061gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33066gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__update_step_xla_33071gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__update_step_xla_32996~x�u
n�k
!�
gradient 
<�9	%�"
� 
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_33001f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_33006~x�u
n�k
!�
gradient  
<�9	%�"
�  
�
p
` VariableSpec 
`ౣ���?
� "
 �
"__inference__update_step_xla_33011f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_33016~x�u
n�k
!�
gradient @
<�9	%�"
� @
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_33021f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`�����?
� "
 �
"__inference__update_step_xla_33026~x�u
n�k
!�
gradient@@
<�9	%�"
�@@
�
p
` VariableSpec 
`�����?
� "
 �
"__inference__update_step_xla_33031f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`�����?
� "
 �
"__inference__update_step_xla_33036�z�w
p�m
"�
gradient@�
=�:	&�#
�@�
�
p
` VariableSpec 
`�˶���?
� "
 �
"__inference__update_step_xla_33041hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�ȧ���?
� "
 �
"__inference__update_step_xla_33046�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�ܪ���?
� "
 �
"__inference__update_step_xla_33051hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�����?
� "
 �
"__inference__update_step_xla_33056rl�i
b�_
�
gradient
��
6�3	�
�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_33061hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_33066pj�g
`�]
�
gradient	�

5�2	�
�	�

�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_33071f`�]
V�S
�
gradient

0�-	�
�

�
p
` VariableSpec 
`������?
� "
 �
 __inference__wrapped_model_31867| !)*?@HI^_gh����9�6
/�,
*�'
c0_input���������  
� ")�&
$
d15�
d15���������
�
=__inference_c0_layer_call_and_return_conditional_losses_33093s !7�4
-�*
(�%
inputs���������  
� "4�1
*�'
tensor_0���������   
� �
"__inference_c0_layer_call_fn_33080h !7�4
-�*
(�%
inputs���������  
� ")�&
unknown���������   �
=__inference_c1_layer_call_and_return_conditional_losses_33115s)*7�4
-�*
(�%
inputs���������   
� "4�1
*�'
tensor_0���������   
� �
"__inference_c1_layer_call_fn_33102h)*7�4
-�*
(�%
inputs���������   
� ")�&
unknown���������   �
=__inference_c4_layer_call_and_return_conditional_losses_33174s?@7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������@
� �
"__inference_c4_layer_call_fn_33161h?@7�4
-�*
(�%
inputs��������� 
� ")�&
unknown���������@�
=__inference_c5_layer_call_and_return_conditional_losses_33196sHI7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
"__inference_c5_layer_call_fn_33183hHI7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
=__inference_c8_layer_call_and_return_conditional_losses_33255t^_7�4
-�*
(�%
inputs���������@
� "5�2
+�(
tensor_0����������
� �
"__inference_c8_layer_call_fn_33242i^_7�4
-�*
(�%
inputs���������@
� "*�'
unknown�����������
=__inference_c9_layer_call_and_return_conditional_losses_33277ugh8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
"__inference_c9_layer_call_fn_33264jgh8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
>__inference_d13_layer_call_and_return_conditional_losses_33347g��0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
#__inference_d13_layer_call_fn_33334\��0�-
&�#
!�
inputs����������
� ""�
unknown�����������
>__inference_d15_layer_call_and_return_conditional_losses_33396f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������

� �
#__inference_d15_layer_call_fn_33383[��0�-
&�#
!�
inputs����������
� "!�
unknown���������
�
?__inference_dr11_layer_call_and_return_conditional_losses_33309u<�9
2�/
)�&
inputs����������
p
� "5�2
+�(
tensor_0����������
� �
?__inference_dr11_layer_call_and_return_conditional_losses_33314u<�9
2�/
)�&
inputs����������
p 
� "5�2
+�(
tensor_0����������
� �
$__inference_dr11_layer_call_fn_33292j<�9
2�/
)�&
inputs����������
p
� "*�'
unknown�����������
$__inference_dr11_layer_call_fn_33297j<�9
2�/
)�&
inputs����������
p 
� "*�'
unknown�����������
?__inference_dr14_layer_call_and_return_conditional_losses_33369e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
?__inference_dr14_layer_call_and_return_conditional_losses_33374e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
$__inference_dr14_layer_call_fn_33352Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
$__inference_dr14_layer_call_fn_33357Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
>__inference_dr3_layer_call_and_return_conditional_losses_33147s;�8
1�.
(�%
inputs��������� 
p
� "4�1
*�'
tensor_0��������� 
� �
>__inference_dr3_layer_call_and_return_conditional_losses_33152s;�8
1�.
(�%
inputs��������� 
p 
� "4�1
*�'
tensor_0��������� 
� �
#__inference_dr3_layer_call_fn_33130h;�8
1�.
(�%
inputs��������� 
p
� ")�&
unknown��������� �
#__inference_dr3_layer_call_fn_33135h;�8
1�.
(�%
inputs��������� 
p 
� ")�&
unknown��������� �
>__inference_dr7_layer_call_and_return_conditional_losses_33228s;�8
1�.
(�%
inputs���������@
p
� "4�1
*�'
tensor_0���������@
� �
>__inference_dr7_layer_call_and_return_conditional_losses_33233s;�8
1�.
(�%
inputs���������@
p 
� "4�1
*�'
tensor_0���������@
� �
#__inference_dr7_layer_call_fn_33211h;�8
1�.
(�%
inputs���������@
p
� ")�&
unknown���������@�
#__inference_dr7_layer_call_fn_33216h;�8
1�.
(�%
inputs���������@
p 
� ")�&
unknown���������@�
>__inference_f12_layer_call_and_return_conditional_losses_33325i8�5
.�+
)�&
inputs����������
� "-�*
#� 
tensor_0����������
� �
#__inference_f12_layer_call_fn_33319^8�5
.�+
)�&
inputs����������
� ""�
unknown����������@
__inference_loss_fn_0_33401!�

� 
� "�
unknown A
__inference_loss_fn_10_33451!�

� 
� "�
unknown A
__inference_loss_fn_11_33456!�

� 
� "�
unknown A
__inference_loss_fn_12_33461!�

� 
� "�
unknown A
__inference_loss_fn_13_33466!�

� 
� "�
unknown A
__inference_loss_fn_14_33471!�

� 
� "�
unknown A
__inference_loss_fn_15_33476!�

� 
� "�
unknown @
__inference_loss_fn_1_33406!�

� 
� "�
unknown @
__inference_loss_fn_2_33411!�

� 
� "�
unknown @
__inference_loss_fn_3_33416!�

� 
� "�
unknown @
__inference_loss_fn_4_33421!�

� 
� "�
unknown @
__inference_loss_fn_5_33426!�

� 
� "�
unknown @
__inference_loss_fn_6_33431!�

� 
� "�
unknown @
__inference_loss_fn_7_33436!�

� 
� "�
unknown @
__inference_loss_fn_8_33441!�

� 
� "�
unknown @
__inference_loss_fn_9_33446!�

� 
� "�
unknown �
>__inference_m10_layer_call_and_return_conditional_losses_33287�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
#__inference_m10_layer_call_fn_33282�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
=__inference_m2_layer_call_and_return_conditional_losses_33125�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
"__inference_m2_layer_call_fn_33120�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
=__inference_m6_layer_call_and_return_conditional_losses_33206�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
"__inference_m6_layer_call_fn_33201�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
E__inference_sequential_layer_call_and_return_conditional_losses_32143� !)*?@HI^_gh����A�>
7�4
*�'
c0_input���������  
p

 
� ",�)
"�
tensor_0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_32231� !)*?@HI^_gh����A�>
7�4
*�'
c0_input���������  
p 

 
� ",�)
"�
tensor_0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_32906� !)*?@HI^_gh����?�<
5�2
(�%
inputs���������  
p

 
� ",�)
"�
tensor_0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_32991� !)*?@HI^_gh����?�<
5�2
(�%
inputs���������  
p 

 
� ",�)
"�
tensor_0���������

� �
*__inference_sequential_layer_call_fn_32337| !)*?@HI^_gh����A�>
7�4
*�'
c0_input���������  
p

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_32442| !)*?@HI^_gh����A�>
7�4
*�'
c0_input���������  
p 

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_32756z !)*?@HI^_gh����?�<
5�2
(�%
inputs���������  
p

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_32793z !)*?@HI^_gh����?�<
5�2
(�%
inputs���������  
p 

 
� "!�
unknown���������
�
#__inference_signature_wrapper_32703� !)*?@HI^_gh����E�B
� 
;�8
6
c0_input*�'
c0_input���������  ")�&
$
d15�
d15���������

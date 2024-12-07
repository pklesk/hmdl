��
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
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
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
t
Adam/v/d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/v/d7/bias
m
"Adam/v/d7/bias/Read/ReadVariableOpReadVariableOpAdam/v/d7/bias*
_output_shapes
:
*
dtype0
t
Adam/m/d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/m/d7/bias
m
"Adam/m/d7/bias/Read/ReadVariableOpReadVariableOpAdam/m/d7/bias*
_output_shapes
:
*
dtype0
}
Adam/v/d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*!
shared_nameAdam/v/d7/kernel
v
$Adam/v/d7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d7/kernel*
_output_shapes
:	�
*
dtype0
}
Adam/m/d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*!
shared_nameAdam/m/d7/kernel
v
$Adam/m/d7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d7/kernel*
_output_shapes
:	�
*
dtype0
u
Adam/v/d5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/v/d5/bias
n
"Adam/v/d5/bias/Read/ReadVariableOpReadVariableOpAdam/v/d5/bias*
_output_shapes	
:�*
dtype0
u
Adam/m/d5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/m/d5/bias
n
"Adam/m/d5/bias/Read/ReadVariableOpReadVariableOpAdam/m/d5/bias*
_output_shapes	
:�*
dtype0
~
Adam/v/d5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
� �*!
shared_nameAdam/v/d5/kernel
w
$Adam/v/d5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d5/kernel* 
_output_shapes
:
� �*
dtype0
~
Adam/m/d5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
� �*!
shared_nameAdam/m/d5/kernel
w
$Adam/m/d5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d5/kernel* 
_output_shapes
:
� �*
dtype0
t
Adam/v/c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/c1/bias
m
"Adam/v/c1/bias/Read/ReadVariableOpReadVariableOpAdam/v/c1/bias*
_output_shapes
:*
dtype0
t
Adam/m/c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/c1/bias
m
"Adam/m/c1/bias/Read/ReadVariableOpReadVariableOpAdam/m/c1/bias*
_output_shapes
:*
dtype0
�
Adam/v/c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/v/c1/kernel
}
$Adam/v/c1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c1/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/m/c1/kernel
}
$Adam/m/c1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c1/kernel*&
_output_shapes
: *
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
f
d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d7/bias
_
d7/bias/Read/ReadVariableOpReadVariableOpd7/bias*
_output_shapes
:
*
dtype0
o
	d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_name	d7/kernel
h
d7/kernel/Read/ReadVariableOpReadVariableOp	d7/kernel*
_output_shapes
:	�
*
dtype0
g
d5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d5/bias
`
d5/bias/Read/ReadVariableOpReadVariableOpd5/bias*
_output_shapes	
:�*
dtype0
p
	d5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
� �*
shared_name	d5/kernel
i
d5/kernel/Read/ReadVariableOpReadVariableOp	d5/kernel* 
_output_shapes
:
� �*
dtype0
f
c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	c1/bias
_
c1/bias/Read/ReadVariableOpReadVariableOpc1/bias*
_output_shapes
:*
dtype0
v
	c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	c1/kernel
o
c1/kernel/Read/ReadVariableOpReadVariableOp	c1/kernel*&
_output_shapes
: *
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c1/kernelc1/bias	d5/kerneld5/bias	d7/kerneld7/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8� *,
f'R%
#__inference_signature_wrapper_31105

NoOpNoOp
�J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�I
value�IB�I B�I
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
 #_jit_compiled_convolution_op*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator* 
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
E_random_generator* 
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias*
<
0
1
!2
"3
=4
>5
L6
M7*
<
0
1
!2
"3
=4
>5
L6
M7*
:
N0
O1
P2
Q3
R4
S5
T6
U7* 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
[trace_0
\trace_1
]trace_2
^trace_3* 
6
_trace_0
`trace_1
atrace_2
btrace_3* 
* 
�
c
_variables
d_iterations
e_learning_rate
f_index_dict
g
_momentums
h_velocities
i_update_step_xla*

jserving_default* 

0
1*

0
1*

N0
O1* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

!0
"1*

!0
"1*

P0
Q1* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
YS
VARIABLE_VALUE	c1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

~trace_0* 

trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

=0
>1*

=0
>1*

R0
S1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	d5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

L0
M1*

L0
M1*

T0
U1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	d7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
<
0
1
2
3
4
5
6
7*

�0
�1*
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
�
d0
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
�15
�16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
r
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7* 
* 
* 
* 
* 

N0
O1* 
* 
* 
* 
* 
* 
* 

P0
Q1* 
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

R0
S1* 
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

T0
U1* 
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
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
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
VARIABLE_VALUEAdam/m/d5/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/d5/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/d5/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/d5/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/d7/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/d7/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/d7/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/d7/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	d5/kerneld5/bias	d7/kerneld7/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c1/kernelAdam/v/c1/kernelAdam/m/c1/biasAdam/v/c1/biasAdam/m/d5/kernelAdam/v/d5/kernelAdam/m/d5/biasAdam/v/d5/biasAdam/m/d7/kernelAdam/v/d7/kernelAdam/m/d7/biasAdam/v/d7/biastotal_1count_1totalcountConst*+
Tin$
"2 *
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
__inference__traced_save_31705
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	d5/kerneld5/bias	d7/kerneld7/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c1/kernelAdam/v/c1/kernelAdam/m/c1/biasAdam/v/c1/biasAdam/m/d5/kernelAdam/v/d5/kernelAdam/m/d5/biasAdam/v/d5/biasAdam/m/d7/kernelAdam/v/d7/kernelAdam/m/d7/biasAdam/v/d7/biastotal_1count_1totalcount**
Tin#
!2*
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
!__inference__traced_restore_31805��
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_30819

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_31375

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
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
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
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
=__inference_c0_layer_call_and_return_conditional_losses_30692

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
�
�
"__inference_c0_layer_call_fn_31308

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
=__inference_c0_layer_call_and_return_conditional_losses_30692w
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
=__inference_d7_layer_call_and_return_conditional_losses_30786

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
`
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
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
�
+
__inference_loss_fn_0_31467
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
�
�
=__inference_d5_layer_call_and_return_conditional_losses_30753

inputs2
matmul_readvariableop_resource:
� �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
� �*
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
:����������`
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
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
:���������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
=__inference_d7_layer_call_and_return_conditional_losses_31462

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
`
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
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
�(
�
E__inference_sequential_layer_call_and_return_conditional_losses_30801
c0_input"
c0_30693: 
c0_30695: "
c1_30712: 
c1_30714:
d5_30754:
� �
d5_30756:	�
d7_30787:	�

d7_30789:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�d5/StatefulPartitionedCall�d7/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr6/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_30693c0_30695*
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
=__inference_c0_layer_call_and_return_conditional_losses_30692�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_30712c1_30714*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_30711�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_30669�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30730�
f4/PartitionedCallPartitionedCall$dr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_f4_layer_call_and_return_conditional_losses_30738�
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_30754d5_30756*
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
GPU(2*0J 8� *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_30753�
dr6/StatefulPartitionedCallStatefulPartitionedCall#d5/StatefulPartitionedCall:output:0^dr3/StatefulPartitionedCall*
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
GPU(2*0J 8� *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_30771�
d7/StatefulPartitionedCallStatefulPartitionedCall$dr6/StatefulPartitionedCall:output:0d7_30787d7_30789*
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
GPU(2*0J 8� *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_30786`
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
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr6/StatefulPartitionedCalldr6/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�	
�
*__inference_sequential_layer_call_fn_31134

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
� �
	unknown_4:	�
	unknown_5:	�

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30886o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
+
__inference_loss_fn_7_31502
identity^
d7/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"d7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
Y
=__inference_f4_layer_call_and_return_conditional_losses_30738

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:���������� Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
#__inference_signature_wrapper_31105
c0_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
� �
	unknown_4:	�
	unknown_5:	�

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8� *)
f$R"
 __inference__wrapped_model_30663o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
K
"__inference__update_step_xla_31289
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
�
+
__inference_loss_fn_5_31492
identity^
d5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"d5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
Y
=__inference_f4_layer_call_and_return_conditional_losses_31391

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:���������� Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_30730

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
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
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
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
E__inference_sequential_layer_call_and_return_conditional_losses_30847
c0_input"
c0_30804: 
c0_30806: "
c1_30809: 
c1_30811:
d5_30822:
� �
d5_30824:	�
d7_30833:	�

d7_30835:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�d5/StatefulPartitionedCall�d7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_30804c0_30806*
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
=__inference_c0_layer_call_and_return_conditional_losses_30692�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_30809c1_30811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_30711�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_30669�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30819�
f4/PartitionedCallPartitionedCalldr3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_f4_layer_call_and_return_conditional_losses_30738�
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_30822d5_30824*
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
GPU(2*0J 8� *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_30753�
dr6/PartitionedCallPartitionedCall#d5/StatefulPartitionedCall:output:0*
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
GPU(2*0J 8� *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_30831�
d7/StatefulPartitionedCallStatefulPartitionedCalldr6/PartitionedCall:output:0d7_30833d7_30835*
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
GPU(2*0J 8� *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_30786`
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
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_30669

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
�
?
#__inference_dr6_layer_call_fn_31423

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
GPU(2*0J 8� *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_30831a
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
�
�
"__inference_c1_layer_call_fn_31330

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_30711w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
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
�	
�
*__inference_sequential_layer_call_fn_31155

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
� �
	unknown_4:	�
	unknown_5:	�

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30943o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
"__inference_d5_layer_call_fn_31400

inputs
unknown:
� �
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
GPU(2*0J 8� *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_30753p
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
:���������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
=__inference_c1_layer_call_and_return_conditional_losses_31343

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  `
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
:���������  w
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
�
+
__inference_loss_fn_4_31487
identity`
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$d5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
+
__inference_loss_fn_1_31472
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
�
�
=__inference_c1_layer_call_and_return_conditional_losses_30711

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  `
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
:���������  w
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
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_31380

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
+
__inference_loss_fn_3_31482
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
�
=__inference_c0_layer_call_and_return_conditional_losses_31321

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
��
�
__inference__traced_save_31705
file_prefix:
 read_disablecopyonread_c0_kernel: .
 read_1_disablecopyonread_c0_bias: <
"read_2_disablecopyonread_c1_kernel: .
 read_3_disablecopyonread_c1_bias:6
"read_4_disablecopyonread_d5_kernel:
� �/
 read_5_disablecopyonread_d5_bias:	�5
"read_6_disablecopyonread_d7_kernel:	�
.
 read_7_disablecopyonread_d7_bias:
,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: D
*read_10_disablecopyonread_adam_m_c0_kernel: D
*read_11_disablecopyonread_adam_v_c0_kernel: 6
(read_12_disablecopyonread_adam_m_c0_bias: 6
(read_13_disablecopyonread_adam_v_c0_bias: D
*read_14_disablecopyonread_adam_m_c1_kernel: D
*read_15_disablecopyonread_adam_v_c1_kernel: 6
(read_16_disablecopyonread_adam_m_c1_bias:6
(read_17_disablecopyonread_adam_v_c1_bias:>
*read_18_disablecopyonread_adam_m_d5_kernel:
� �>
*read_19_disablecopyonread_adam_v_d5_kernel:
� �7
(read_20_disablecopyonread_adam_m_d5_bias:	�7
(read_21_disablecopyonread_adam_v_d5_bias:	�=
*read_22_disablecopyonread_adam_m_d7_kernel:	�
=
*read_23_disablecopyonread_adam_v_d7_kernel:	�
6
(read_24_disablecopyonread_adam_m_d7_bias:
6
(read_25_disablecopyonread_adam_v_d7_bias:
+
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_c1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_d5_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_d5_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
� �*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
� �e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
� �t
Read_5/DisableCopyOnReadDisableCopyOnRead read_5_disablecopyonread_d5_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp read_5_disablecopyonread_d5_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_d7_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_d7_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
t
Read_7/DisableCopyOnReadDisableCopyOnRead read_7_disablecopyonread_d7_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp read_7_disablecopyonread_d7_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_adam_m_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_adam_m_c0_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_11/DisableCopyOnReadDisableCopyOnRead*read_11_disablecopyonread_adam_v_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp*read_11_disablecopyonread_adam_v_c0_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
: }
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_adam_m_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_adam_m_c0_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_adam_v_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_adam_v_c0_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_adam_m_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_adam_m_c1_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_adam_v_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_adam_v_c1_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
: }
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_adam_m_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_adam_m_c1_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_17/DisableCopyOnReadDisableCopyOnRead(read_17_disablecopyonread_adam_v_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp(read_17_disablecopyonread_adam_v_c1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_18/DisableCopyOnReadDisableCopyOnRead*read_18_disablecopyonread_adam_m_d5_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp*read_18_disablecopyonread_adam_m_d5_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
� �*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
� �g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
� �
Read_19/DisableCopyOnReadDisableCopyOnRead*read_19_disablecopyonread_adam_v_d5_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp*read_19_disablecopyonread_adam_v_d5_kernel^Read_19/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
� �*
dtype0q
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
� �g
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
� �}
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_adam_m_d5_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_adam_m_d5_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_21/DisableCopyOnReadDisableCopyOnRead(read_21_disablecopyonread_adam_v_d5_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp(read_21_disablecopyonread_adam_v_d5_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_22/DisableCopyOnReadDisableCopyOnRead*read_22_disablecopyonread_adam_m_d7_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp*read_22_disablecopyonread_adam_m_d7_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	�

Read_23/DisableCopyOnReadDisableCopyOnRead*read_23_disablecopyonread_adam_v_d7_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp*read_23_disablecopyonread_adam_v_d7_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
}
Read_24/DisableCopyOnReadDisableCopyOnRead(read_24_disablecopyonread_adam_m_d7_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp(read_24_disablecopyonread_adam_m_d7_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:
}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_adam_v_d7_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_adam_v_d7_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_31353

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
P
"__inference__update_step_xla_31284
gradient
variable:
� �*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
� �: *
	_noinline(:($
"
_user_specified_name
variable:J F
 
_output_shapes
:
� �
"
_user_specified_name
gradient
�	
�
*__inference_sequential_layer_call_fn_30905
c0_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
� �
	unknown_4:	�
	unknown_5:	�

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30886o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
?
#__inference_dr3_layer_call_fn_31363

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
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30819h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
\
#__inference_dr3_layer_call_fn_31358

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
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30730w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
E__inference_sequential_layer_call_and_return_conditional_losses_31259

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource: 0
"c1_biasadd_readvariableop_resource:5
!d5_matmul_readvariableop_resource:
� �1
"d5_biasadd_readvariableop_resource:	�4
!d7_matmul_readvariableop_resource:	�
0
"d7_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�d5/BiasAdd/ReadVariableOp�d5/MatMul/ReadVariableOp�d7/BiasAdd/ReadVariableOp�d7/MatMul/ReadVariableOp�
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
: *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
g
dr3/IdentityIdentitym2/MaxPool:output:0*
T0*/
_output_shapes
:���������Y
f4/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   r

f4/ReshapeReshapedr3/Identity:output:0f4/Const:output:0*
T0*(
_output_shapes
:���������� |
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource* 
_output_shapes
:
� �*
dtype0}
	d5/MatMulMatMulf4/Reshape:output:0 d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d5/ReluRelud5/BiasAdd:output:0*
T0*(
_output_shapes
:����������b
dr6/IdentityIdentityd5/Relu:activations:0*
T0*(
_output_shapes
:����������{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0~
	d7/MatMulMatMuldr6/Identity:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
\

d7/SoftmaxSoftmaxd7/BiasAdd:output:0*
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
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
d5/BiasAdd/ReadVariableOpd5/BiasAdd/ReadVariableOp24
d5/MatMul/ReadVariableOpd5/MatMul/ReadVariableOp26
d7/BiasAdd/ReadVariableOpd7/BiasAdd/ReadVariableOp24
d7/MatMul/ReadVariableOpd7/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

]
>__inference_dr6_layer_call_and_return_conditional_losses_30771

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?e
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
 *  �>�
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
�
\
>__inference_dr6_layer_call_and_return_conditional_losses_30831

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
V
"__inference__update_step_xla_31274
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
: 
"
_user_specified_name
gradient
�
J
"__inference__update_step_xla_31299
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
�
�
"__inference_d7_layer_call_fn_31449

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
GPU(2*0J 8� *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_30786o
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
�
+
__inference_loss_fn_6_31497
identity`
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$d7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�:
�
E__inference_sequential_layer_call_and_return_conditional_losses_31214

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource: 0
"c1_biasadd_readvariableop_resource:5
!d5_matmul_readvariableop_resource:
� �1
"d5_biasadd_readvariableop_resource:	�4
!d7_matmul_readvariableop_resource:	�
0
"d7_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�d5/BiasAdd/ReadVariableOp�d5/MatMul/ReadVariableOp�d7/BiasAdd/ReadVariableOp�d7/MatMul/ReadVariableOp�
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
: *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:���������*
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
:���������b
dr3/dropout/ShapeShapem2/MaxPool:output:0*
T0*
_output_shapes
::���
(dr3/dropout/random_uniform/RandomUniformRandomUniformdr3/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
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
:���������X
dr3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr3/dropout/SelectV2SelectV2dr3/dropout/GreaterEqual:z:0dr3/dropout/Mul:z:0dr3/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������Y
f4/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z

f4/ReshapeReshapedr3/dropout/SelectV2:output:0f4/Const:output:0*
T0*(
_output_shapes
:���������� |
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource* 
_output_shapes
:
� �*
dtype0}
	d5/MatMulMatMulf4/Reshape:output:0 d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d5/ReluRelud5/BiasAdd:output:0*
T0*(
_output_shapes
:����������V
dr6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?|
dr6/dropout/MulMuld5/Relu:activations:0dr6/dropout/Const:output:0*
T0*(
_output_shapes
:����������d
dr6/dropout/ShapeShaped5/Relu:activations:0*
T0*
_output_shapes
::���
(dr6/dropout/random_uniform/RandomUniformRandomUniformdr6/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2_
dr6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dr6/dropout/GreaterEqualGreaterEqual1dr6/dropout/random_uniform/RandomUniform:output:0#dr6/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������X
dr6/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr6/dropout/SelectV2SelectV2dr6/dropout/GreaterEqual:z:0dr6/dropout/Mul:z:0dr6/dropout/Const_1:output:0*
T0*(
_output_shapes
:����������{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
	d7/MatMulMatMuldr6/dropout/SelectV2:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
\

d7/SoftmaxSoftmaxd7/BiasAdd:output:0*
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
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
d5/BiasAdd/ReadVariableOpd5/BiasAdd/ReadVariableOp24
d5/MatMul/ReadVariableOpd5/MatMul/ReadVariableOp26
d7/BiasAdd/ReadVariableOpd7/BiasAdd/ReadVariableOp24
d7/MatMul/ReadVariableOpd7/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
V
"__inference__update_step_xla_31264
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
>__inference_dr6_layer_call_and_return_conditional_losses_31435

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?e
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
 *  �>�
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
J
"__inference__update_step_xla_31279
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
�	
�
*__inference_sequential_layer_call_fn_30962
c0_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
� �
	unknown_4:	�
	unknown_5:	�

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30943o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
O
"__inference__update_step_xla_31294
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
__inference_loss_fn_2_31477
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
�
\
>__inference_dr6_layer_call_and_return_conditional_losses_31440

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
�}
�
!__inference__traced_restore_31805
file_prefix4
assignvariableop_c0_kernel: (
assignvariableop_1_c0_bias: 6
assignvariableop_2_c1_kernel: (
assignvariableop_3_c1_bias:0
assignvariableop_4_d5_kernel:
� �)
assignvariableop_5_d5_bias:	�/
assignvariableop_6_d7_kernel:	�
(
assignvariableop_7_d7_bias:
&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: >
$assignvariableop_10_adam_m_c0_kernel: >
$assignvariableop_11_adam_v_c0_kernel: 0
"assignvariableop_12_adam_m_c0_bias: 0
"assignvariableop_13_adam_v_c0_bias: >
$assignvariableop_14_adam_m_c1_kernel: >
$assignvariableop_15_adam_v_c1_kernel: 0
"assignvariableop_16_adam_m_c1_bias:0
"assignvariableop_17_adam_v_c1_bias:8
$assignvariableop_18_adam_m_d5_kernel:
� �8
$assignvariableop_19_adam_v_d5_kernel:
� �1
"assignvariableop_20_adam_m_d5_bias:	�1
"assignvariableop_21_adam_v_d5_bias:	�7
$assignvariableop_22_adam_m_d7_kernel:	�
7
$assignvariableop_23_adam_v_d7_kernel:	�
0
"assignvariableop_24_adam_m_d7_bias:
0
"assignvariableop_25_adam_v_d7_bias:
%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
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
AssignVariableOp_4AssignVariableOpassignvariableop_4_d5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_d5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_d7_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_d7_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_adam_m_c0_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_adam_v_c0_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_adam_m_c0_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_adam_v_c0_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_adam_m_c1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_v_c1_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_m_c1_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_adam_v_c1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_adam_m_d5_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_v_d5_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_m_d5_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_adam_v_d5_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_adam_m_d7_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_v_d7_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_m_d7_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_adam_v_d7_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
�
>
"__inference_f4_layer_call_fn_31385

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
:���������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_f4_layer_call_and_return_conditional_losses_30738a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
\
#__inference_dr6_layer_call_fn_31418

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
GPU(2*0J 8� *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_30771p
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
�%
�
E__inference_sequential_layer_call_and_return_conditional_losses_30943

inputs"
c0_30910: 
c0_30912: "
c1_30915: 
c1_30917:
d5_30923:
� �
d5_30925:	�
d7_30929:	�

d7_30931:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�d5/StatefulPartitionedCall�d7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_30910c0_30912*
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
=__inference_c0_layer_call_and_return_conditional_losses_30692�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_30915c1_30917*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_30711�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_30669�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30819�
f4/PartitionedCallPartitionedCalldr3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_f4_layer_call_and_return_conditional_losses_30738�
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_30923d5_30925*
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
GPU(2*0J 8� *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_30753�
dr6/PartitionedCallPartitionedCall#d5/StatefulPartitionedCall:output:0*
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
GPU(2*0J 8� *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_30831�
d7/StatefulPartitionedCallStatefulPartitionedCalldr6/PartitionedCall:output:0d7_30929d7_30931*
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
GPU(2*0J 8� *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_30786`
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
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
=__inference_d5_layer_call_and_return_conditional_losses_31413

inputs2
matmul_readvariableop_resource:
� �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
� �*
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
:����������`
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
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
:���������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
J
"__inference__update_step_xla_31269
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
�(
�
E__inference_sequential_layer_call_and_return_conditional_losses_30886

inputs"
c0_30853: 
c0_30855: "
c1_30858: 
c1_30860:
d5_30866:
� �
d5_30868:	�
d7_30872:	�

d7_30874:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�d5/StatefulPartitionedCall�d7/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr6/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_30853c0_30855*
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
=__inference_c0_layer_call_and_return_conditional_losses_30692�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_30858c1_30860*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_30711�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_30669�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30730�
f4/PartitionedCallPartitionedCall$dr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_f4_layer_call_and_return_conditional_losses_30738�
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_30866d5_30868*
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
GPU(2*0J 8� *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_30753�
dr6/StatefulPartitionedCallStatefulPartitionedCall#d5/StatefulPartitionedCall:output:0^dr3/StatefulPartitionedCall*
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
GPU(2*0J 8� *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_30771�
d7/StatefulPartitionedCallStatefulPartitionedCall$dr6/StatefulPartitionedCall:output:0d7_30872d7_30874*
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
GPU(2*0J 8� *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_30786`
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
d5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d5/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d7/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr6/StatefulPartitionedCalldr6/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
>
"__inference_m2_layer_call_fn_31348

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
=__inference_m2_layer_call_and_return_conditional_losses_30669�
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
�.
�
 __inference__wrapped_model_30663
c0_inputF
,sequential_c0_conv2d_readvariableop_resource: ;
-sequential_c0_biasadd_readvariableop_resource: F
,sequential_c1_conv2d_readvariableop_resource: ;
-sequential_c1_biasadd_readvariableop_resource:@
,sequential_d5_matmul_readvariableop_resource:
� �<
-sequential_d5_biasadd_readvariableop_resource:	�?
,sequential_d7_matmul_readvariableop_resource:	�
;
-sequential_d7_biasadd_readvariableop_resource:

identity��$sequential/c0/BiasAdd/ReadVariableOp�#sequential/c0/Conv2D/ReadVariableOp�$sequential/c1/BiasAdd/ReadVariableOp�#sequential/c1/Conv2D/ReadVariableOp�$sequential/d5/BiasAdd/ReadVariableOp�#sequential/d5/MatMul/ReadVariableOp�$sequential/d7/BiasAdd/ReadVariableOp�#sequential/d7/MatMul/ReadVariableOp�
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
: *
dtype0�
sequential/c1/Conv2DConv2D sequential/c0/Relu:activations:0+sequential/c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
$sequential/c1/BiasAdd/ReadVariableOpReadVariableOp-sequential_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/c1/BiasAddBiasAddsequential/c1/Conv2D:output:0,sequential/c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  t
sequential/c1/ReluRelusequential/c1/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
sequential/m2/MaxPoolMaxPool sequential/c1/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
}
sequential/dr3/IdentityIdentitysequential/m2/MaxPool:output:0*
T0*/
_output_shapes
:���������d
sequential/f4/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
sequential/f4/ReshapeReshape sequential/dr3/Identity:output:0sequential/f4/Const:output:0*
T0*(
_output_shapes
:���������� �
#sequential/d5/MatMul/ReadVariableOpReadVariableOp,sequential_d5_matmul_readvariableop_resource* 
_output_shapes
:
� �*
dtype0�
sequential/d5/MatMulMatMulsequential/f4/Reshape:output:0+sequential/d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$sequential/d5/BiasAdd/ReadVariableOpReadVariableOp-sequential_d5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/d5/BiasAddBiasAddsequential/d5/MatMul:product:0,sequential/d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
sequential/d5/ReluRelusequential/d5/BiasAdd:output:0*
T0*(
_output_shapes
:����������x
sequential/dr6/IdentityIdentity sequential/d5/Relu:activations:0*
T0*(
_output_shapes
:�����������
#sequential/d7/MatMul/ReadVariableOpReadVariableOp,sequential_d7_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
sequential/d7/MatMulMatMul sequential/dr6/Identity:output:0+sequential/d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
$sequential/d7/BiasAdd/ReadVariableOpReadVariableOp-sequential_d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential/d7/BiasAddBiasAddsequential/d7/MatMul:product:0,sequential/d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
sequential/d7/SoftmaxSoftmaxsequential/d7/BiasAdd:output:0*
T0*'
_output_shapes
:���������
n
IdentityIdentitysequential/d7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/c1/BiasAdd/ReadVariableOp$^sequential/c1/Conv2D/ReadVariableOp%^sequential/d5/BiasAdd/ReadVariableOp$^sequential/d5/MatMul/ReadVariableOp%^sequential/d7/BiasAdd/ReadVariableOp$^sequential/d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2L
$sequential/c0/BiasAdd/ReadVariableOp$sequential/c0/BiasAdd/ReadVariableOp2J
#sequential/c0/Conv2D/ReadVariableOp#sequential/c0/Conv2D/ReadVariableOp2L
$sequential/c1/BiasAdd/ReadVariableOp$sequential/c1/BiasAdd/ReadVariableOp2J
#sequential/c1/Conv2D/ReadVariableOp#sequential/c1/Conv2D/ReadVariableOp2L
$sequential/d5/BiasAdd/ReadVariableOp$sequential/d5/BiasAdd/ReadVariableOp2J
#sequential/d5/MatMul/ReadVariableOp#sequential/d5/MatMul/ReadVariableOp2L
$sequential/d7/BiasAdd/ReadVariableOp$sequential/d7/BiasAdd/ReadVariableOp2J
#sequential/d7/MatMul/ReadVariableOp#sequential/d7/MatMul/ReadVariableOp:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input"�
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
serving_default_c0_input:0���������  6
d70
StatefulPartitionedCall:0���������
tensorflow/serving/predict:Ǐ
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
 #_jit_compiled_convolution_op"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
E_random_generator"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
X
0
1
!2
"3
=4
>5
L6
M7"
trackable_list_wrapper
X
0
1
!2
"3
=4
>5
L6
M7"
trackable_list_wrapper
X
N0
O1
P2
Q3
R4
S5
T6
U7"
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
[trace_0
\trace_1
]trace_2
^trace_32�
*__inference_sequential_layer_call_fn_30905
*__inference_sequential_layer_call_fn_30962
*__inference_sequential_layer_call_fn_31134
*__inference_sequential_layer_call_fn_31155�
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
 z[trace_0z\trace_1z]trace_2z^trace_3
�
_trace_0
`trace_1
atrace_2
btrace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_30801
E__inference_sequential_layer_call_and_return_conditional_losses_30847
E__inference_sequential_layer_call_and_return_conditional_losses_31214
E__inference_sequential_layer_call_and_return_conditional_losses_31259�
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
 z_trace_0z`trace_1zatrace_2zbtrace_3
�B�
 __inference__wrapped_model_30663c0_input"�
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
c
_variables
d_iterations
e_learning_rate
f_index_dict
g
_momentums
h_velocities
i_update_step_xla"
experimentalOptimizer
,
jserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
"__inference_c0_layer_call_fn_31308�
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
 zptrace_0
�
qtrace_02�
=__inference_c0_layer_call_and_return_conditional_losses_31321�
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
 zqtrace_0
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
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
wtrace_02�
"__inference_c1_layer_call_fn_31330�
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
 zwtrace_0
�
xtrace_02�
=__inference_c1_layer_call_and_return_conditional_losses_31343�
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
 zxtrace_0
#:! 2	c1/kernel
:2c1/bias
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
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
"__inference_m2_layer_call_fn_31348�
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
 z~trace_0
�
trace_02�
=__inference_m2_layer_call_and_return_conditional_losses_31353�
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
 ztrace_0
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
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_dr3_layer_call_fn_31358
#__inference_dr3_layer_call_fn_31363�
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
>__inference_dr3_layer_call_and_return_conditional_losses_31375
>__inference_dr3_layer_call_and_return_conditional_losses_31380�
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
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_f4_layer_call_fn_31385�
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
=__inference_f4_layer_call_and_return_conditional_losses_31391�
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
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_d5_layer_call_fn_31400�
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
=__inference_d5_layer_call_and_return_conditional_losses_31413�
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
:
� �2	d5/kernel
:�2d5/bias
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
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_dr6_layer_call_fn_31418
#__inference_dr6_layer_call_fn_31423�
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
>__inference_dr6_layer_call_and_return_conditional_losses_31435
>__inference_dr6_layer_call_and_return_conditional_losses_31440�
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
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_d7_layer_call_fn_31449�
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
=__inference_d7_layer_call_and_return_conditional_losses_31462�
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
:	�
2	d7/kernel
:
2d7/bias
�
�trace_02�
__inference_loss_fn_0_31467�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_31472�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_31477�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_31482�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_31487�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_31492�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_31497�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_31502�
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
annotations� *� z�trace_0
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_sequential_layer_call_fn_30905c0_input"�
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
*__inference_sequential_layer_call_fn_30962c0_input"�
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
*__inference_sequential_layer_call_fn_31134inputs"�
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
*__inference_sequential_layer_call_fn_31155inputs"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_30801c0_input"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_30847c0_input"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_31214inputs"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_31259inputs"�
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
�
d0
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
�15
�16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_72�
"__inference__update_step_xla_31264
"__inference__update_step_xla_31269
"__inference__update_step_xla_31274
"__inference__update_step_xla_31279
"__inference__update_step_xla_31284
"__inference__update_step_xla_31289
"__inference__update_step_xla_31294
"__inference__update_step_xla_31299�
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
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7
�B�
#__inference_signature_wrapper_31105c0_input"�
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
.
N0
O1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c0_layer_call_fn_31308inputs"�
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
=__inference_c0_layer_call_and_return_conditional_losses_31321inputs"�
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
.
P0
Q1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c1_layer_call_fn_31330inputs"�
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
=__inference_c1_layer_call_and_return_conditional_losses_31343inputs"�
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
"__inference_m2_layer_call_fn_31348inputs"�
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
=__inference_m2_layer_call_and_return_conditional_losses_31353inputs"�
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
#__inference_dr3_layer_call_fn_31358inputs"�
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
#__inference_dr3_layer_call_fn_31363inputs"�
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
>__inference_dr3_layer_call_and_return_conditional_losses_31375inputs"�
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
>__inference_dr3_layer_call_and_return_conditional_losses_31380inputs"�
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
"__inference_f4_layer_call_fn_31385inputs"�
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
=__inference_f4_layer_call_and_return_conditional_losses_31391inputs"�
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
.
R0
S1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_d5_layer_call_fn_31400inputs"�
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
=__inference_d5_layer_call_and_return_conditional_losses_31413inputs"�
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
#__inference_dr6_layer_call_fn_31418inputs"�
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
#__inference_dr6_layer_call_fn_31423inputs"�
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
>__inference_dr6_layer_call_and_return_conditional_losses_31435inputs"�
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
>__inference_dr6_layer_call_and_return_conditional_losses_31440inputs"�
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
.
T0
U1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_d7_layer_call_fn_31449inputs"�
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
=__inference_d7_layer_call_and_return_conditional_losses_31462inputs"�
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
__inference_loss_fn_0_31467"�
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
__inference_loss_fn_1_31472"�
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
__inference_loss_fn_2_31477"�
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
__inference_loss_fn_3_31482"�
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
__inference_loss_fn_4_31487"�
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
__inference_loss_fn_5_31492"�
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
__inference_loss_fn_6_31497"�
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
__inference_loss_fn_7_31502"�
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
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
(:& 2Adam/m/c0/kernel
(:& 2Adam/v/c0/kernel
: 2Adam/m/c0/bias
: 2Adam/v/c0/bias
(:& 2Adam/m/c1/kernel
(:& 2Adam/v/c1/kernel
:2Adam/m/c1/bias
:2Adam/v/c1/bias
": 
� �2Adam/m/d5/kernel
": 
� �2Adam/v/d5/kernel
:�2Adam/m/d5/bias
:�2Adam/v/d5/bias
!:	�
2Adam/m/d7/kernel
!:	�
2Adam/v/d7/kernel
:
2Adam/m/d7/bias
:
2Adam/v/d7/bias
�B�
"__inference__update_step_xla_31264gradientvariable"�
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
"__inference__update_step_xla_31269gradientvariable"�
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
"__inference__update_step_xla_31274gradientvariable"�
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
"__inference__update_step_xla_31279gradientvariable"�
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
"__inference__update_step_xla_31284gradientvariable"�
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
"__inference__update_step_xla_31289gradientvariable"�
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
"__inference__update_step_xla_31294gradientvariable"�
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
"__inference__update_step_xla_31299gradientvariable"�
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
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__update_step_xla_31264~x�u
n�k
!�
gradient 
<�9	%�"
� 
�
p
` VariableSpec 
`��ʧ��?
� "
 �
"__inference__update_step_xla_31269f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`��Χ��?
� "
 �
"__inference__update_step_xla_31274~x�u
n�k
!�
gradient 
<�9	%�"
� 
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_31279f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_31284rl�i
b�_
�
gradient
� �
6�3	�
�
� �
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_31289hb�_
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
"__inference__update_step_xla_31294pj�g
`�]
�
gradient	�

5�2	�
�	�

�
p
` VariableSpec 
`శ���?
� "
 �
"__inference__update_step_xla_31299f`�]
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
 __inference__wrapped_model_30663n!"=>LM9�6
/�,
*�'
c0_input���������  
� "'�$
"
d7�
d7���������
�
=__inference_c0_layer_call_and_return_conditional_losses_31321s7�4
-�*
(�%
inputs���������  
� "4�1
*�'
tensor_0���������   
� �
"__inference_c0_layer_call_fn_31308h7�4
-�*
(�%
inputs���������  
� ")�&
unknown���������   �
=__inference_c1_layer_call_and_return_conditional_losses_31343s!"7�4
-�*
(�%
inputs���������   
� "4�1
*�'
tensor_0���������  
� �
"__inference_c1_layer_call_fn_31330h!"7�4
-�*
(�%
inputs���������   
� ")�&
unknown���������  �
=__inference_d5_layer_call_and_return_conditional_losses_31413e=>0�-
&�#
!�
inputs���������� 
� "-�*
#� 
tensor_0����������
� �
"__inference_d5_layer_call_fn_31400Z=>0�-
&�#
!�
inputs���������� 
� ""�
unknown�����������
=__inference_d7_layer_call_and_return_conditional_losses_31462dLM0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������

� 
"__inference_d7_layer_call_fn_31449YLM0�-
&�#
!�
inputs����������
� "!�
unknown���������
�
>__inference_dr3_layer_call_and_return_conditional_losses_31375s;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
>__inference_dr3_layer_call_and_return_conditional_losses_31380s;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
#__inference_dr3_layer_call_fn_31358h;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
#__inference_dr3_layer_call_fn_31363h;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
>__inference_dr6_layer_call_and_return_conditional_losses_31435e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
>__inference_dr6_layer_call_and_return_conditional_losses_31440e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
#__inference_dr6_layer_call_fn_31418Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
#__inference_dr6_layer_call_fn_31423Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
=__inference_f4_layer_call_and_return_conditional_losses_31391h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0���������� 
� �
"__inference_f4_layer_call_fn_31385]7�4
-�*
(�%
inputs���������
� ""�
unknown���������� @
__inference_loss_fn_0_31467!�

� 
� "�
unknown @
__inference_loss_fn_1_31472!�

� 
� "�
unknown @
__inference_loss_fn_2_31477!�

� 
� "�
unknown @
__inference_loss_fn_3_31482!�

� 
� "�
unknown @
__inference_loss_fn_4_31487!�

� 
� "�
unknown @
__inference_loss_fn_5_31492!�

� 
� "�
unknown @
__inference_loss_fn_6_31497!�

� 
� "�
unknown @
__inference_loss_fn_7_31502!�

� 
� "�
unknown �
=__inference_m2_layer_call_and_return_conditional_losses_31353�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
"__inference_m2_layer_call_fn_31348�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
E__inference_sequential_layer_call_and_return_conditional_losses_30801{!"=>LMA�>
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
E__inference_sequential_layer_call_and_return_conditional_losses_30847{!"=>LMA�>
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
E__inference_sequential_layer_call_and_return_conditional_losses_31214y!"=>LM?�<
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
E__inference_sequential_layer_call_and_return_conditional_losses_31259y!"=>LM?�<
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
*__inference_sequential_layer_call_fn_30905p!"=>LMA�>
7�4
*�'
c0_input���������  
p

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_30962p!"=>LMA�>
7�4
*�'
c0_input���������  
p 

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_31134n!"=>LM?�<
5�2
(�%
inputs���������  
p

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_31155n!"=>LM?�<
5�2
(�%
inputs���������  
p 

 
� "!�
unknown���������
�
#__inference_signature_wrapper_31105z!"=>LME�B
� 
;�8
6
c0_input*�'
c0_input���������  "'�$
"
d7�
d7���������

��
��
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
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758ہ
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
Adam/v/d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/v/d3/bias
m
"Adam/v/d3/bias/Read/ReadVariableOpReadVariableOpAdam/v/d3/bias*
_output_shapes
:(*
dtype0
t
Adam/m/d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/m/d3/bias
m
"Adam/m/d3/bias/Read/ReadVariableOpReadVariableOpAdam/m/d3/bias*
_output_shapes
:(*
dtype0
|
Adam/v/d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/v/d3/kernel
u
$Adam/v/d3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d3/kernel*
_output_shapes

:@(*
dtype0
|
Adam/m/d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/m/d3/kernel
u
$Adam/m/d3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d3/kernel*
_output_shapes

:@(*
dtype0
t
Adam/v/d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/d1/bias
m
"Adam/v/d1/bias/Read/ReadVariableOpReadVariableOpAdam/v/d1/bias*
_output_shapes
:@*
dtype0
t
Adam/m/d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/d1/bias
m
"Adam/m/d1/bias/Read/ReadVariableOpReadVariableOpAdam/m/d1/bias*
_output_shapes
:@*
dtype0
}
Adam/v/d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� @*!
shared_nameAdam/v/d1/kernel
v
$Adam/v/d1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d1/kernel*
_output_shapes
:	� @*
dtype0
}
Adam/m/d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� @*!
shared_nameAdam/m/d1/kernel
v
$Adam/m/d1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d1/kernel*
_output_shapes
:	� @*
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
d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name	d3/bias
_
d3/bias/Read/ReadVariableOpReadVariableOpd3/bias*
_output_shapes
:(*
dtype0
n
	d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*
shared_name	d3/kernel
g
d3/kernel/Read/ReadVariableOpReadVariableOp	d3/kernel*
_output_shapes

:@(*
dtype0
f
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	d1/bias
_
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes
:@*
dtype0
o
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� @*
shared_name	d1/kernel
h
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel*
_output_shapes
:	� @*
dtype0
�
serving_default_f0_inputPlaceholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_f0_input	d1/kerneld1/bias	d3/kerneld3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *,
f'R%
#__inference_signature_wrapper_74744

NoOpNoOp
�+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator* 
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
 
0
1
)2
*3*
 
0
1
)2
*3*

+0
,1
-2
.3* 
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
4trace_0
5trace_1
6trace_2
7trace_3* 
6
8trace_0
9trace_1
:trace_2
;trace_3* 
* 
�
<
_variables
=_iterations
>_learning_rate
?_index_dict
@
_momentums
A_velocities
B_update_step_xla*

Cserving_default* 
* 
* 
* 
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Itrace_0* 

Jtrace_0* 

0
1*

0
1*

+0
,1* 
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ptrace_0* 

Qtrace_0* 
YS
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

Wtrace_0
Xtrace_1* 

Ytrace_0
Ztrace_1* 
* 

)0
*1*

)0
*1*

-0
.1* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
YS
VARIABLE_VALUE	d3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

btrace_0* 

ctrace_0* 

dtrace_0* 

etrace_0* 
* 
 
0
1
2
3*

f0
g1*
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
C
=0
h1
i2
j3
k4
l5
m6
n7
o8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
h0
j1
l2
n3*
 
i0
k1
m2
o3*
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
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
+0
,1* 
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
-0
.1* 
* 
* 
* 
* 
* 
* 
* 
8
t	variables
u	keras_api
	vtotal
	wcount*
H
x	variables
y	keras_api
	ztotal
	{count
|
_fn_kwargs*
[U
VARIABLE_VALUEAdam/m/d1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/d1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/d1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/d1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/d3/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/d3/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/d3/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/d3/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

v0
w1*

t	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

z0
{1*

x	variables*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d3/kerneld3/bias	iterationlearning_rateAdam/m/d1/kernelAdam/v/d1/kernelAdam/m/d1/biasAdam/v/d1/biasAdam/m/d3/kernelAdam/v/d3/kernelAdam/m/d3/biasAdam/v/d3/biastotal_1count_1totalcountConst*
Tin
2*
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
__inference__traced_save_75084
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d3/kerneld3/bias	iterationlearning_rateAdam/m/d1/kernelAdam/v/d1/kernelAdam/m/d1/biasAdam/v/d1/biasAdam/m/d3/kernelAdam/v/d3/kernelAdam/m/d3/biasAdam/v/d3/biastotal_1count_1totalcount*
Tin
2*
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
!__inference__traced_restore_75148ު
�

]
>__inference_dr2_layer_call_and_return_conditional_losses_74906

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
+
__inference_loss_fn_3_74953
identity^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"d3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
"__inference_d1_layer_call_fn_74871

inputs
unknown:	� @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_74524o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
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
�
�
G__inference_sequential_9_layer_call_and_return_conditional_losses_74593
f0_input
d1_74572:	� @
d1_74574:@
d3_74583:@(
d3_74585:(
identity��d1/StatefulPartitionedCall�d3/StatefulPartitionedCall�
f0/PartitionedCallPartitionedCallf0_input*
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
=__inference_f0_layer_call_and_return_conditional_losses_74509�
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_74572d1_74574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_74524�
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_74581�
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_74583d3_74585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_74557`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
f0_input
�
+
__inference_loss_fn_0_74938
identity`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
+
__inference_loss_fn_2_74948
identity`
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$d3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
G__inference_sequential_9_layer_call_and_return_conditional_losses_74649

inputs
d1_74633:	� @
d1_74635:@
d3_74639:@(
d3_74641:(
identity��d1/StatefulPartitionedCall�d3/StatefulPartitionedCall�
f0/PartitionedCallPartitionedCallinputs*
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
=__inference_f0_layer_call_and_return_conditional_losses_74509�
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_74633d1_74635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_74524�
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_74581�
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_74639d3_74641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_74557`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
Y
=__inference_f0_layer_call_and_return_conditional_losses_74862

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
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
,__inference_sequential_9_layer_call_fn_74761

inputs
unknown:	� @
	unknown_0:@
	unknown_1:@(
	unknown_2:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_74616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
"__inference_d3_layer_call_fn_74920

inputs
unknown:@(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_74557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_d1_layer_call_and_return_conditional_losses_74884

inputs1
matmul_readvariableop_resource:	� @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
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
=__inference_d3_layer_call_and_return_conditional_losses_74933

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������(`
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
N
"__inference__update_step_xla_74846
gradient
variable:@(*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@(: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:@(
"
_user_specified_name
gradient
�
+
__inference_loss_fn_1_74943
identity^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"d1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
,__inference_sequential_9_layer_call_fn_74660
f0_input
unknown:	� @
	unknown_0:@
	unknown_1:@(
	unknown_2:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_74649o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
f0_input
�
�
 __inference__wrapped_model_74499
f0_inputA
.sequential_9_d1_matmul_readvariableop_resource:	� @=
/sequential_9_d1_biasadd_readvariableop_resource:@@
.sequential_9_d3_matmul_readvariableop_resource:@(=
/sequential_9_d3_biasadd_readvariableop_resource:(
identity��&sequential_9/d1/BiasAdd/ReadVariableOp�%sequential_9/d1/MatMul/ReadVariableOp�&sequential_9/d3/BiasAdd/ReadVariableOp�%sequential_9/d3/MatMul/ReadVariableOpf
sequential_9/f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   
sequential_9/f0/ReshapeReshapef0_inputsequential_9/f0/Const:output:0*
T0*(
_output_shapes
:���������� �
%sequential_9/d1/MatMul/ReadVariableOpReadVariableOp.sequential_9_d1_matmul_readvariableop_resource*
_output_shapes
:	� @*
dtype0�
sequential_9/d1/MatMulMatMul sequential_9/f0/Reshape:output:0-sequential_9/d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&sequential_9/d1/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_9/d1/BiasAddBiasAdd sequential_9/d1/MatMul:product:0.sequential_9/d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
sequential_9/d1/ReluRelu sequential_9/d1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
sequential_9/dr2/IdentityIdentity"sequential_9/d1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
%sequential_9/d3/MatMul/ReadVariableOpReadVariableOp.sequential_9_d3_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0�
sequential_9/d3/MatMulMatMul"sequential_9/dr2/Identity:output:0-sequential_9/d3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
&sequential_9/d3/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
sequential_9/d3/BiasAddBiasAdd sequential_9/d3/MatMul:product:0.sequential_9/d3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(v
sequential_9/d3/SoftmaxSoftmax sequential_9/d3/BiasAdd:output:0*
T0*'
_output_shapes
:���������(p
IdentityIdentity!sequential_9/d3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp'^sequential_9/d1/BiasAdd/ReadVariableOp&^sequential_9/d1/MatMul/ReadVariableOp'^sequential_9/d3/BiasAdd/ReadVariableOp&^sequential_9/d3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 2P
&sequential_9/d1/BiasAdd/ReadVariableOp&sequential_9/d1/BiasAdd/ReadVariableOp2N
%sequential_9/d1/MatMul/ReadVariableOp%sequential_9/d1/MatMul/ReadVariableOp2P
&sequential_9/d3/BiasAdd/ReadVariableOp&sequential_9/d3/BiasAdd/ReadVariableOp2N
%sequential_9/d3/MatMul/ReadVariableOp%sequential_9/d3/MatMul/ReadVariableOp:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
f0_input
�
�
=__inference_d3_layer_call_and_return_conditional_losses_74557

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������(`
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
?
#__inference_dr2_layer_call_fn_74894

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_74581`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_d1_layer_call_and_return_conditional_losses_74524

inputs1
matmul_readvariableop_resource:	� @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
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
�
�
,__inference_sequential_9_layer_call_fn_74627
f0_input
unknown:	� @
	unknown_0:@
	unknown_1:@(
	unknown_2:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_74616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
f0_input
�M
�

!__inference__traced_restore_75148
file_prefix-
assignvariableop_d1_kernel:	� @(
assignvariableop_1_d1_bias:@.
assignvariableop_2_d3_kernel:@((
assignvariableop_3_d3_bias:(&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: 6
#assignvariableop_6_adam_m_d1_kernel:	� @6
#assignvariableop_7_adam_v_d1_kernel:	� @/
!assignvariableop_8_adam_m_d1_bias:@/
!assignvariableop_9_adam_v_d1_bias:@6
$assignvariableop_10_adam_m_d3_kernel:@(6
$assignvariableop_11_adam_v_d3_kernel:@(0
"assignvariableop_12_adam_m_d3_bias:(0
"assignvariableop_13_adam_v_d3_bias:(%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_d1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_d1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_d3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_d3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_adam_m_d1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_adam_v_d1_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_adam_m_d1_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_adam_v_d1_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_adam_m_d3_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_adam_v_d3_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_adam_m_d3_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_adam_v_d3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_1AssignVariableOp_12(
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
�
�
G__inference_sequential_9_layer_call_and_return_conditional_losses_74568
f0_input
d1_74525:	� @
d1_74527:@
d3_74558:@(
d3_74560:(
identity��d1/StatefulPartitionedCall�d3/StatefulPartitionedCall�dr2/StatefulPartitionedCall�
f0/PartitionedCallPartitionedCallf0_input*
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
=__inference_f0_layer_call_and_return_conditional_losses_74509�
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_74525d1_74527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_74524�
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_74542�
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_74558d3_74560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_74557`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^dr2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
f0_input
�
�
__inference__traced_save_75084
file_prefix3
 read_disablecopyonread_d1_kernel:	� @.
 read_1_disablecopyonread_d1_bias:@4
"read_2_disablecopyonread_d3_kernel:@(.
 read_3_disablecopyonread_d3_bias:(,
"read_4_disablecopyonread_iteration:	 0
&read_5_disablecopyonread_learning_rate: <
)read_6_disablecopyonread_adam_m_d1_kernel:	� @<
)read_7_disablecopyonread_adam_v_d1_kernel:	� @5
'read_8_disablecopyonread_adam_m_d1_bias:@5
'read_9_disablecopyonread_adam_v_d1_bias:@<
*read_10_disablecopyonread_adam_m_d3_kernel:@(<
*read_11_disablecopyonread_adam_v_d3_kernel:@(6
(read_12_disablecopyonread_adam_m_d3_bias:(6
(read_13_disablecopyonread_adam_v_d3_bias:(+
!read_14_disablecopyonread_total_1: +
!read_15_disablecopyonread_count_1: )
read_16_disablecopyonread_total: )
read_17_disablecopyonread_count: 
savev2_const
identity_37��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnRead read_disablecopyonread_d1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp read_disablecopyonread_d1_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� @*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� @b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	� @t
Read_1/DisableCopyOnReadDisableCopyOnRead read_1_disablecopyonread_d1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp read_1_disablecopyonread_d1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_d3_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_d3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@(t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_d3_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_d3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:(v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_iteration^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_learning_rate^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_adam_m_d1_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_adam_m_d1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� @*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� @f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	� @}
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_adam_v_d1_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_adam_v_d1_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� @*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� @f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	� @{
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_adam_m_d1_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_adam_m_d1_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_adam_v_d1_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_adam_v_d1_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_adam_m_d3_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_adam_m_d3_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:@(
Read_11/DisableCopyOnReadDisableCopyOnRead*read_11_disablecopyonread_adam_v_d3_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp*read_11_disablecopyonread_adam_v_d3_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@(}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_adam_m_d3_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_adam_m_d3_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:(}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_adam_v_d3_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_adam_v_d3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:(v
Read_14/DisableCopyOnReadDisableCopyOnRead!read_14_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp!read_14_disablecopyonread_total_1^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_15/DisableCopyOnReadDisableCopyOnRead!read_15_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp!read_15_disablecopyonread_count_1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_16/DisableCopyOnReadDisableCopyOnReadread_16_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpread_16_disablecopyonread_total^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_17/DisableCopyOnReadDisableCopyOnReadread_17_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpread_17_disablecopyonread_count^Read_17/DisableCopyOnRead"/device:CPU:0*
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_36Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_37IdentityIdentity_36:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2(
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
Read_17/ReadVariableOpRead_17/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
G__inference_sequential_9_layer_call_and_return_conditional_losses_74831

inputs4
!d1_matmul_readvariableop_resource:	� @0
"d1_biasadd_readvariableop_resource:@3
!d3_matmul_readvariableop_resource:@(0
"d3_biasadd_readvariableop_resource:(
identity��d1/BiasAdd/ReadVariableOp�d1/MatMul/ReadVariableOp�d3/BiasAdd/ReadVariableOp�d3/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:���������� {
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource*
_output_shapes
:	� @*
dtype0|
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
d1/ReluRelud1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@a
dr2/IdentityIdentityd1/Relu:activations:0*
T0*'
_output_shapes
:���������@z
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0~
	d3/MatMulMatMuldr2/Identity:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(x
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(\

d3/SoftmaxSoftmaxd3/BiasAdd:output:0*
T0*'
_output_shapes
:���������(`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
\
>__inference_dr2_layer_call_and_return_conditional_losses_74911

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
J
"__inference__update_step_xla_74851
gradient
variable:(*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:(: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:(
"
_user_specified_name
gradient
�
�
#__inference_signature_wrapper_74744
f0_input
unknown:	� @
	unknown_0:@
	unknown_1:@(
	unknown_2:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *)
f$R"
 __inference__wrapped_model_74499o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
f0_input
�
Y
=__inference_f0_layer_call_and_return_conditional_losses_74509

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
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_sequential_9_layer_call_and_return_conditional_losses_74806

inputs4
!d1_matmul_readvariableop_resource:	� @0
"d1_biasadd_readvariableop_resource:@3
!d3_matmul_readvariableop_resource:@(0
"d3_biasadd_readvariableop_resource:(
identity��d1/BiasAdd/ReadVariableOp�d1/MatMul/ReadVariableOp�d3/BiasAdd/ReadVariableOp�d3/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:���������� {
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource*
_output_shapes
:	� @*
dtype0|
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
d1/ReluRelud1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@V
dr2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?{
dr2/dropout/MulMuld1/Relu:activations:0dr2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@d
dr2/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
::���
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed2����_
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@X
dr2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr2/dropout/SelectV2SelectV2dr2/dropout/GreaterEqual:z:0dr2/dropout/Mul:z:0dr2/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@z
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0�
	d3/MatMulMatMuldr2/dropout/SelectV2:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(x
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(\

d3/SoftmaxSoftmaxd3/BiasAdd:output:0*
T0*'
_output_shapes
:���������(`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
\
>__inference_dr2_layer_call_and_return_conditional_losses_74581

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
O
"__inference__update_step_xla_74836
gradient
variable:	� @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	� @: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	� @
"
_user_specified_name
gradient
�
\
#__inference_dr2_layer_call_fn_74889

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_74542o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_sequential_9_layer_call_fn_74774

inputs
unknown:	� @
	unknown_0:@
	unknown_1:@(
	unknown_2:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_74649o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_sequential_9_layer_call_and_return_conditional_losses_74616

inputs
d1_74600:	� @
d1_74602:@
d3_74606:@(
d3_74608:(
identity��d1/StatefulPartitionedCall�d3/StatefulPartitionedCall�dr2/StatefulPartitionedCall�
f0/PartitionedCallPartitionedCallinputs*
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
=__inference_f0_layer_call_and_return_conditional_losses_74509�
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_74600d1_74602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_74524�
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_74542�
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_74606d3_74608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_74557`
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d1/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^dr2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
>
"__inference_f0_layer_call_fn_74856

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
=__inference_f0_layer_call_and_return_conditional_losses_74509a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
J
"__inference__update_step_xla_74841
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

]
>__inference_dr2_layer_call_and_return_conditional_losses_74542

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
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
f0_input9
serving_default_f0_input:0���������@@6
d30
StatefulPartitionedCall:0���������(tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
<
0
1
)2
*3"
trackable_list_wrapper
<
0
1
)2
*3"
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
4trace_0
5trace_1
6trace_2
7trace_32�
,__inference_sequential_9_layer_call_fn_74627
,__inference_sequential_9_layer_call_fn_74660
,__inference_sequential_9_layer_call_fn_74761
,__inference_sequential_9_layer_call_fn_74774�
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
 z4trace_0z5trace_1z6trace_2z7trace_3
�
8trace_0
9trace_1
:trace_2
;trace_32�
G__inference_sequential_9_layer_call_and_return_conditional_losses_74568
G__inference_sequential_9_layer_call_and_return_conditional_losses_74593
G__inference_sequential_9_layer_call_and_return_conditional_losses_74806
G__inference_sequential_9_layer_call_and_return_conditional_losses_74831�
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
 z8trace_0z9trace_1z:trace_2z;trace_3
�B�
 __inference__wrapped_model_74499f0_input"�
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
<
_variables
=_iterations
>_learning_rate
?_index_dict
@
_momentums
A_velocities
B_update_step_xla"
experimentalOptimizer
,
Cserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Itrace_02�
"__inference_f0_layer_call_fn_74856�
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
 zItrace_0
�
Jtrace_02�
=__inference_f0_layer_call_and_return_conditional_losses_74862�
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
 zJtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ptrace_02�
"__inference_d1_layer_call_fn_74871�
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
 zPtrace_0
�
Qtrace_02�
=__inference_d1_layer_call_and_return_conditional_losses_74884�
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
 zQtrace_0
:	� @2	d1/kernel
:@2d1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
Wtrace_0
Xtrace_12�
#__inference_dr2_layer_call_fn_74889
#__inference_dr2_layer_call_fn_74894�
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
 zWtrace_0zXtrace_1
�
Ytrace_0
Ztrace_12�
>__inference_dr2_layer_call_and_return_conditional_losses_74906
>__inference_dr2_layer_call_and_return_conditional_losses_74911�
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
 zYtrace_0zZtrace_1
"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
`trace_02�
"__inference_d3_layer_call_fn_74920�
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
 z`trace_0
�
atrace_02�
=__inference_d3_layer_call_and_return_conditional_losses_74933�
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
 zatrace_0
:@(2	d3/kernel
:(2d3/bias
�
btrace_02�
__inference_loss_fn_0_74938�
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
annotations� *� zbtrace_0
�
ctrace_02�
__inference_loss_fn_1_74943�
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
annotations� *� zctrace_0
�
dtrace_02�
__inference_loss_fn_2_74948�
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
annotations� *� zdtrace_0
�
etrace_02�
__inference_loss_fn_3_74953�
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
annotations� *� zetrace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_9_layer_call_fn_74627f0_input"�
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
,__inference_sequential_9_layer_call_fn_74660f0_input"�
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
,__inference_sequential_9_layer_call_fn_74761inputs"�
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
,__inference_sequential_9_layer_call_fn_74774inputs"�
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_74568f0_input"�
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_74593f0_input"�
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_74806inputs"�
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_74831inputs"�
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
_
=0
h1
i2
j3
k4
l5
m6
n7
o8"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
h0
j1
l2
n3"
trackable_list_wrapper
<
i0
k1
m2
o3"
trackable_list_wrapper
�
ptrace_0
qtrace_1
rtrace_2
strace_32�
"__inference__update_step_xla_74836
"__inference__update_step_xla_74841
"__inference__update_step_xla_74846
"__inference__update_step_xla_74851�
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
 0zptrace_0zqtrace_1zrtrace_2zstrace_3
�B�
#__inference_signature_wrapper_74744f0_input"�
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_f0_layer_call_fn_74856inputs"�
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
=__inference_f0_layer_call_and_return_conditional_losses_74862inputs"�
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
+0
,1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_d1_layer_call_fn_74871inputs"�
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
=__inference_d1_layer_call_and_return_conditional_losses_74884inputs"�
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
#__inference_dr2_layer_call_fn_74889inputs"�
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
#__inference_dr2_layer_call_fn_74894inputs"�
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
>__inference_dr2_layer_call_and_return_conditional_losses_74906inputs"�
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
>__inference_dr2_layer_call_and_return_conditional_losses_74911inputs"�
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
-0
.1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_d3_layer_call_fn_74920inputs"�
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
=__inference_d3_layer_call_and_return_conditional_losses_74933inputs"�
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
__inference_loss_fn_0_74938"�
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
__inference_loss_fn_1_74943"�
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
__inference_loss_fn_2_74948"�
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
__inference_loss_fn_3_74953"�
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
N
t	variables
u	keras_api
	vtotal
	wcount"
_tf_keras_metric
^
x	variables
y	keras_api
	ztotal
	{count
|
_fn_kwargs"
_tf_keras_metric
!:	� @2Adam/m/d1/kernel
!:	� @2Adam/v/d1/kernel
:@2Adam/m/d1/bias
:@2Adam/v/d1/bias
 :@(2Adam/m/d3/kernel
 :@(2Adam/v/d3/kernel
:(2Adam/m/d3/bias
:(2Adam/v/d3/bias
�B�
"__inference__update_step_xla_74836gradientvariable"�
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
"__inference__update_step_xla_74841gradientvariable"�
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
"__inference__update_step_xla_74846gradientvariable"�
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
"__inference__update_step_xla_74851gradientvariable"�
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
.
v0
w1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
:  (2total
:  (2count
.
z0
{1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__update_step_xla_74836pj�g
`�]
�
gradient	� @
5�2	�
�	� @
�
p
` VariableSpec 
`��毧�?
� "
 �
"__inference__update_step_xla_74841f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`�կ��?
� "
 �
"__inference__update_step_xla_74846nh�e
^�[
�
gradient@(
4�1	�
�@(
�
p
` VariableSpec 
`��ʁ��?
� "
 �
"__inference__update_step_xla_74851f`�]
V�S
�
gradient(
0�-	�
�(
�
p
` VariableSpec 
`�ʁ��?
� "
 �
 __inference__wrapped_model_74499j)*9�6
/�,
*�'
f0_input���������@@
� "'�$
"
d3�
d3���������(�
=__inference_d1_layer_call_and_return_conditional_losses_74884d0�-
&�#
!�
inputs���������� 
� ",�)
"�
tensor_0���������@
� 
"__inference_d1_layer_call_fn_74871Y0�-
&�#
!�
inputs���������� 
� "!�
unknown���������@�
=__inference_d3_layer_call_and_return_conditional_losses_74933c)*/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������(
� ~
"__inference_d3_layer_call_fn_74920X)*/�,
%�"
 �
inputs���������@
� "!�
unknown���������(�
>__inference_dr2_layer_call_and_return_conditional_losses_74906c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
>__inference_dr2_layer_call_and_return_conditional_losses_74911c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� 
#__inference_dr2_layer_call_fn_74889X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@
#__inference_dr2_layer_call_fn_74894X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
=__inference_f0_layer_call_and_return_conditional_losses_74862h7�4
-�*
(�%
inputs���������@@
� "-�*
#� 
tensor_0���������� 
� �
"__inference_f0_layer_call_fn_74856]7�4
-�*
(�%
inputs���������@@
� ""�
unknown���������� @
__inference_loss_fn_0_74938!�

� 
� "�
unknown @
__inference_loss_fn_1_74943!�

� 
� "�
unknown @
__inference_loss_fn_2_74948!�

� 
� "�
unknown @
__inference_loss_fn_3_74953!�

� 
� "�
unknown �
G__inference_sequential_9_layer_call_and_return_conditional_losses_74568w)*A�>
7�4
*�'
f0_input���������@@
p

 
� ",�)
"�
tensor_0���������(
� �
G__inference_sequential_9_layer_call_and_return_conditional_losses_74593w)*A�>
7�4
*�'
f0_input���������@@
p 

 
� ",�)
"�
tensor_0���������(
� �
G__inference_sequential_9_layer_call_and_return_conditional_losses_74806u)*?�<
5�2
(�%
inputs���������@@
p

 
� ",�)
"�
tensor_0���������(
� �
G__inference_sequential_9_layer_call_and_return_conditional_losses_74831u)*?�<
5�2
(�%
inputs���������@@
p 

 
� ",�)
"�
tensor_0���������(
� �
,__inference_sequential_9_layer_call_fn_74627l)*A�>
7�4
*�'
f0_input���������@@
p

 
� "!�
unknown���������(�
,__inference_sequential_9_layer_call_fn_74660l)*A�>
7�4
*�'
f0_input���������@@
p 

 
� "!�
unknown���������(�
,__inference_sequential_9_layer_call_fn_74761j)*?�<
5�2
(�%
inputs���������@@
p

 
� "!�
unknown���������(�
,__inference_sequential_9_layer_call_fn_74774j)*?�<
5�2
(�%
inputs���������@@
p 

 
� "!�
unknown���������(�
#__inference_signature_wrapper_74744v)*E�B
� 
;�8
6
f0_input*�'
f0_input���������@@"'�$
"
d3�
d3���������(
ô£
éº
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ñ¯
o
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *
shared_name	d1/kernel
h
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel*
_output_shapes
:	 *
dtype0
f
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d1/bias
_
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes
:*
dtype0
n
	d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_name	d3/kernel
g
d3/kernel/Read/ReadVariableOpReadVariableOp	d3/kernel*
_output_shapes

:(*
dtype0
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
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
}
Adam/d1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *!
shared_nameAdam/d1/kernel/m
v
$Adam/d1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d1/kernel/m*
_output_shapes
:	 *
dtype0
t
Adam/d1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d1/bias/m
m
"Adam/d1/bias/m/Read/ReadVariableOpReadVariableOpAdam/d1/bias/m*
_output_shapes
:*
dtype0
|
Adam/d3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_nameAdam/d3/kernel/m
u
$Adam/d3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d3/kernel/m*
_output_shapes

:(*
dtype0
t
Adam/d3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/d3/bias/m
m
"Adam/d3/bias/m/Read/ReadVariableOpReadVariableOpAdam/d3/bias/m*
_output_shapes
:(*
dtype0
}
Adam/d1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *!
shared_nameAdam/d1/kernel/v
v
$Adam/d1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d1/kernel/v*
_output_shapes
:	 *
dtype0
t
Adam/d1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d1/bias/v
m
"Adam/d1/bias/v/Read/ReadVariableOpReadVariableOpAdam/d1/bias/v*
_output_shapes
:*
dtype0
|
Adam/d3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_nameAdam/d3/kernel/v
u
$Adam/d3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d3/kernel/v*
_output_shapes

:(*
dtype0
t
Adam/d3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/d3/bias/v
m
"Adam/d3/bias/v/Read/ReadVariableOpReadVariableOpAdam/d3/bias/v*
_output_shapes
:(*
dtype0

NoOpNoOp
Ä*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÿ)
valueõ)Bò) Bë)
´
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¥
	variables
trainable_variables
regularization_losses
	keras_api
 _random_generator
!__call__
*"&call_and_return_all_conditional_losses* 
¦

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*

+iter

,beta_1

-beta_2
	.decay
/learning_ratemYmZ#m[$m\v]v^#v_$v`*
 
0
1
#2
$3*
 
0
1
#2
$3*

00
11
22
33* 
°
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

9serving_default* 
* 
* 
* 

:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

00
11* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*

20
31* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
 
0
1
2
3*

N0
O1*
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
00
11* 
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
20
31* 
* 
8
	Ptotal
	Qcount
R	variables
S	keras_api*
H
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

R	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

T0
U1*

W	variables*
|v
VARIABLE_VALUEAdam/d1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_f0_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ@@
é
StatefulPartitionedCallStatefulPartitionedCallserving_default_f0_input	d1/kerneld1/bias	d3/kerneld3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *,
f'R%
#__inference_signature_wrapper_64583
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ï
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd3/kernel/Read/ReadVariableOpd3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/d1/kernel/m/Read/ReadVariableOp"Adam/d1/bias/m/Read/ReadVariableOp$Adam/d3/kernel/m/Read/ReadVariableOp"Adam/d3/bias/m/Read/ReadVariableOp$Adam/d1/kernel/v/Read/ReadVariableOp"Adam/d1/bias/v/Read/ReadVariableOp$Adam/d3/kernel/v/Read/ReadVariableOp"Adam/d3/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
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
GPU(2*0J 8 *'
f"R 
__inference__traced_save_64775
Æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d3/kerneld3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/d1/kernel/mAdam/d1/bias/mAdam/d3/kernel/mAdam/d3/bias/mAdam/d1/kernel/vAdam/d1/bias/vAdam/d3/kernel/vAdam/d3/bias/v*!
Tin
2*
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
GPU(2*0J 8 **
f%R#
!__inference__traced_restore_64848 Ð
Ø
§
G__inference_sequential_9_layer_call_and_return_conditional_losses_64411

inputs
d1_64395:	 
d1_64397:
d3_64401:(
d3_64403:(
identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCall¢dr2/StatefulPartitionedCall³
f0/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_64264ñ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_64395d1_64397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_64279á
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_64357ú
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_64401d3_64403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_64305`
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
:ÿÿÿÿÿÿÿÿÿ(
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^dr2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ñ
\
>__inference_dr2_layer_call_and_return_conditional_losses_64633

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
+
__inference_loss_fn_0_64674
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
¥0

__inference__traced_save_64775
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop(
$savev2_d3_kernel_read_readvariableop&
"savev2_d3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop/
+savev2_adam_d1_kernel_m_read_readvariableop-
)savev2_adam_d1_bias_m_read_readvariableop/
+savev2_adam_d3_kernel_m_read_readvariableop-
)savev2_adam_d3_bias_m_read_readvariableop/
+savev2_adam_d1_kernel_v_read_readvariableop-
)savev2_adam_d1_bias_v_read_readvariableop/
+savev2_adam_d3_kernel_v_read_readvariableop-
)savev2_adam_d3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: »
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ä

valueÚ
B×
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d3_kernel_read_readvariableop"savev2_d3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_d1_kernel_m_read_readvariableop)savev2_adam_d1_bias_m_read_readvariableop+savev2_adam_d3_kernel_m_read_readvariableop)savev2_adam_d3_bias_m_read_readvariableop+savev2_adam_d1_kernel_v_read_readvariableop)savev2_adam_d1_bias_v_read_readvariableop+savev2_adam_d3_kernel_v_read_readvariableop)savev2_adam_d3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes{
y: :	 ::(:(: : : : : : : : : :	 ::(:(:	 ::(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	 : 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 : 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:%!

_output_shapes
:	 : 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:

_output_shapes
: 
Ú
ï
=__inference_d1_layer_call_and_return_conditional_losses_64618

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û
î
=__inference_d3_layer_call_and_return_conditional_losses_64305

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
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
:ÿÿÿÿÿÿÿÿÿ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
Ò
,__inference_sequential_9_layer_call_fn_64327
f0_input
unknown:	 
	unknown_0:
	unknown_1:(
	unknown_2:(
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_64316o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"
_user_specified_name
f0_input
Û
î
=__inference_d3_layer_call_and_return_conditional_losses_64669

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
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
:ÿÿÿÿÿÿÿÿÿ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

"__inference_d1_layer_call_fn_64605

inputs
unknown:	 
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_64279o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

?
#__inference_dr2_layer_call_fn_64623

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_64290`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

G__inference_sequential_9_layer_call_and_return_conditional_losses_64316

inputs
d1_64280:	 
d1_64282:
d3_64306:(
d3_64308:(
identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCall³
f0/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_64264ñ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_64280d1_64282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_64279Ñ
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_64290ò
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_64306d3_64308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_64305`
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
:ÿÿÿÿÿÿÿÿÿ(
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
»

"__inference_d3_layer_call_fn_64656

inputs
unknown:(
	unknown_0:(
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_64305o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý	
]
>__inference_dr2_layer_call_and_return_conditional_losses_64357

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed2ÿÿÿÿ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
+
__inference_loss_fn_2_64684
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
Î

G__inference_sequential_9_layer_call_and_return_conditional_losses_64455
f0_input
d1_64439:	 
d1_64441:
d3_64445:(
d3_64447:(
identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCallµ
f0/PartitionedCallPartitionedCallf0_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_64264ñ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_64439d1_64441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_64279Ñ
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_64290ò
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_64445d3_64447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_64305`
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
:ÿÿÿÿÿÿÿÿÿ(
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"
_user_specified_name
f0_input
¿
Y
=__inference_f0_layer_call_and_return_conditional_losses_64594

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
º
Ò
,__inference_sequential_9_layer_call_fn_64435
f0_input
unknown:	 
	unknown_0:
	unknown_1:(
	unknown_2:(
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_64411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"
_user_specified_name
f0_input
Ú
ï
=__inference_d1_layer_call_and_return_conditional_losses_64279

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿
Y
=__inference_f0_layer_call_and_return_conditional_losses_64264

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Â
£
G__inference_sequential_9_layer_call_and_return_conditional_losses_64536

inputs4
!d1_matmul_readvariableop_resource:	 0
"d1_biasadd_readvariableop_resource:3
!d3_matmul_readvariableop_resource:(0
"d3_biasadd_readvariableop_resource:(
identity¢d1/BiasAdd/ReadVariableOp¢d1/MatMul/ReadVariableOp¢d3/BiasAdd/ReadVariableOp¢d3/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0|
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
d1/ReluRelud1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dr2/IdentityIdentityd1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0~
	d3/MatMulMatMuldr2/Identity:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\

d3/SoftmaxSoftmaxd3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
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
:ÿÿÿÿÿÿÿÿÿ(´
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_3_64689
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

É
#__inference_signature_wrapper_64583
f0_input
unknown:	 
	unknown_0:
	unknown_1:(
	unknown_2:(
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *)
f$R"
 __inference__wrapped_model_64251o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"
_user_specified_name
f0_input
ý	
]
>__inference_dr2_layer_call_and_return_conditional_losses_64645

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed2ÿÿÿÿ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
>
"__inference_f0_layer_call_fn_64588

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_64264a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¯
æ
 __inference__wrapped_model_64251
f0_inputA
.sequential_9_d1_matmul_readvariableop_resource:	 =
/sequential_9_d1_biasadd_readvariableop_resource:@
.sequential_9_d3_matmul_readvariableop_resource:(=
/sequential_9_d3_biasadd_readvariableop_resource:(
identity¢&sequential_9/d1/BiasAdd/ReadVariableOp¢%sequential_9/d1/MatMul/ReadVariableOp¢&sequential_9/d3/BiasAdd/ReadVariableOp¢%sequential_9/d3/MatMul/ReadVariableOpf
sequential_9/f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
sequential_9/f0/ReshapeReshapef0_inputsequential_9/f0/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%sequential_9/d1/MatMul/ReadVariableOpReadVariableOp.sequential_9_d1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0£
sequential_9/d1/MatMulMatMul sequential_9/f0/Reshape:output:0-sequential_9/d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_9/d1/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
sequential_9/d1/BiasAddBiasAdd sequential_9/d1/MatMul:product:0.sequential_9/d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
sequential_9/d1/ReluRelu sequential_9/d1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_9/dr2/IdentityIdentity"sequential_9/d1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%sequential_9/d3/MatMul/ReadVariableOpReadVariableOp.sequential_9_d3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0¥
sequential_9/d3/MatMulMatMul"sequential_9/dr2/Identity:output:0-sequential_9/d3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
&sequential_9/d3/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0¦
sequential_9/d3/BiasAddBiasAdd sequential_9/d3/MatMul:product:0.sequential_9/d3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v
sequential_9/d3/SoftmaxSoftmax sequential_9/d3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(p
IdentityIdentity!sequential_9/d3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(è
NoOpNoOp'^sequential_9/d1/BiasAdd/ReadVariableOp&^sequential_9/d1/MatMul/ReadVariableOp'^sequential_9/d3/BiasAdd/ReadVariableOp&^sequential_9/d3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 2P
&sequential_9/d1/BiasAdd/ReadVariableOp&sequential_9/d1/BiasAdd/ReadVariableOp2N
%sequential_9/d1/MatMul/ReadVariableOp%sequential_9/d1/MatMul/ReadVariableOp2P
&sequential_9/d3/BiasAdd/ReadVariableOp&sequential_9/d3/BiasAdd/ReadVariableOp2N
%sequential_9/d3/MatMul/ReadVariableOp%sequential_9/d3/MatMul/ReadVariableOp:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"
_user_specified_name
f0_input
Ñ
\
>__inference_dr2_layer_call_and_return_conditional_losses_64290

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
©
G__inference_sequential_9_layer_call_and_return_conditional_losses_64475
f0_input
d1_64459:	 
d1_64461:
d3_64465:(
d3_64467:(
identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCall¢dr2/StatefulPartitionedCallµ
f0/PartitionedCallPartitionedCallf0_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_64264ñ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_64459d1_64461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_64279á
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_64357ú
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_64465d3_64467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_64305`
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
:ÿÿÿÿÿÿÿÿÿ(
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^dr2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"
_user_specified_name
f0_input
üS
Ô
!__inference__traced_restore_64848
file_prefix-
assignvariableop_d1_kernel:	 (
assignvariableop_1_d1_bias:.
assignvariableop_2_d3_kernel:((
assignvariableop_3_d3_bias:(&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: 7
$assignvariableop_13_adam_d1_kernel_m:	 0
"assignvariableop_14_adam_d1_bias_m:6
$assignvariableop_15_adam_d3_kernel_m:(0
"assignvariableop_16_adam_d3_bias_m:(7
$assignvariableop_17_adam_d1_kernel_v:	 0
"assignvariableop_18_adam_d1_bias_v:6
$assignvariableop_19_adam_d3_kernel_v:(0
"assignvariableop_20_adam_d3_bias_v:(
identity_22¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¾
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ä

valueÚ
B×
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_d3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_d3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp$assignvariableop_13_adam_d1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_adam_d1_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_d3_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_d3_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_d1_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_d1_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_d3_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_d3_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
´
Ð
,__inference_sequential_9_layer_call_fn_64498

inputs
unknown:	 
	unknown_0:
	unknown_1:(
	unknown_2:(
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_64316o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ê
\
#__inference_dr2_layer_call_fn_64628

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_64357o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_1_64679
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
º
£
G__inference_sequential_9_layer_call_and_return_conditional_losses_64568

inputs4
!d1_matmul_readvariableop_resource:	 0
"d1_biasadd_readvariableop_resource:3
!d3_matmul_readvariableop_resource:(0
"d3_biasadd_readvariableop_resource:(
identity¢d1/BiasAdd/ReadVariableOp¢d1/MatMul/ReadVariableOp¢d3/BiasAdd/ReadVariableOp¢d3/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0|
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
d1/ReluRelud1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
dr2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?{
dr2/dropout/MulMuld1/Relu:activations:0dr2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
dr2/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:¥
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed2ÿÿÿÿ_
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >²
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
dr2/dropout/CastCastdr2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
dr2/dropout/Mul_1Muldr2/dropout/Mul:z:0dr2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0~
	d3/MatMulMatMuldr2/dropout/Mul_1:z:0 d3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\

d3/SoftmaxSoftmaxd3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
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
:ÿÿÿÿÿÿÿÿÿ(´
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
´
Ð
,__inference_sequential_9_layer_call_fn_64511

inputs
unknown:	 
	unknown_0:
	unknown_1:(
	unknown_2:(
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_64411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¯
serving_default
E
f0_input9
serving_default_f0_input:0ÿÿÿÿÿÿÿÿÿ@@6
d30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ(tensorflow/serving/predict:Ïb
Î
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
	variables
trainable_variables
regularization_losses
	keras_api
 _random_generator
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
»

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer

+iter

,beta_1

-beta_2
	.decay
/learning_ratemYmZ#m[$m\v]v^#v_$v`"
	optimizer
<
0
1
#2
$3"
trackable_list_wrapper
<
0
1
#2
$3"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
Ê
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_9_layer_call_fn_64327
,__inference_sequential_9_layer_call_fn_64498
,__inference_sequential_9_layer_call_fn_64511
,__inference_sequential_9_layer_call_fn_64435À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_9_layer_call_and_return_conditional_losses_64536
G__inference_sequential_9_layer_call_and_return_conditional_losses_64568
G__inference_sequential_9_layer_call_and_return_conditional_losses_64455
G__inference_sequential_9_layer_call_and_return_conditional_losses_64475À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
 __inference__wrapped_model_64251f0_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
9serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_f0_layer_call_fn_64588¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ç2ä
=__inference_f0_layer_call_and_return_conditional_losses_64594¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 2	d1/kernel
:2d1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_d1_layer_call_fn_64605¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ç2ä
=__inference_d1_layer_call_and_return_conditional_losses_64618¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr2_layer_call_fn_64623
#__inference_dr2_layer_call_fn_64628´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·
>__inference_dr2_layer_call_and_return_conditional_losses_64633
>__inference_dr2_layer_call_and_return_conditional_losses_64645´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:(2	d3/kernel
:(2d3/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_d3_layer_call_fn_64656¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ç2ä
=__inference_d3_layer_call_and_return_conditional_losses_64669¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
²2¯
__inference_loss_fn_0_64674
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_1_64679
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_2_64684
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_3_64689
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
#__inference_signature_wrapper_64583f0_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ptotal
	Qcount
R	variables
S	keras_api"
_tf_keras_metric
^
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
!:	 2Adam/d1/kernel/m
:2Adam/d1/bias/m
 :(2Adam/d3/kernel/m
:(2Adam/d3/bias/m
!:	 2Adam/d1/kernel/v
:2Adam/d1/bias/v
 :(2Adam/d3/kernel/v
:(2Adam/d3/bias/v
 __inference__wrapped_model_64251j#$9¢6
/¢,
*'
f0_inputÿÿÿÿÿÿÿÿÿ@@
ª "'ª$
"
d3
d3ÿÿÿÿÿÿÿÿÿ(
=__inference_d1_layer_call_and_return_conditional_losses_64618]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 v
"__inference_d1_layer_call_fn_64605P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ
=__inference_d3_layer_call_and_return_conditional_losses_64669\#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 u
"__inference_d3_layer_call_fn_64656O#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ(
>__inference_dr2_layer_call_and_return_conditional_losses_64633\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
>__inference_dr2_layer_call_and_return_conditional_losses_64645\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 v
#__inference_dr2_layer_call_fn_64623O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿv
#__inference_dr2_layer_call_fn_64628O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¢
=__inference_f0_layer_call_and_return_conditional_losses_64594a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 z
"__inference_f0_layer_call_fn_64588T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@
ª "ÿÿÿÿÿÿÿÿÿ 7
__inference_loss_fn_0_64674¢

¢ 
ª " 7
__inference_loss_fn_1_64679¢

¢ 
ª " 7
__inference_loss_fn_2_64684¢

¢ 
ª " 7
__inference_loss_fn_3_64689¢

¢ 
ª " »
G__inference_sequential_9_layer_call_and_return_conditional_losses_64455p#$A¢>
7¢4
*'
f0_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 »
G__inference_sequential_9_layer_call_and_return_conditional_losses_64475p#$A¢>
7¢4
*'
f0_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ¹
G__inference_sequential_9_layer_call_and_return_conditional_losses_64536n#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ¹
G__inference_sequential_9_layer_call_and_return_conditional_losses_64568n#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 
,__inference_sequential_9_layer_call_fn_64327c#$A¢>
7¢4
*'
f0_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "ÿÿÿÿÿÿÿÿÿ(
,__inference_sequential_9_layer_call_fn_64435c#$A¢>
7¢4
*'
f0_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª "ÿÿÿÿÿÿÿÿÿ(
,__inference_sequential_9_layer_call_fn_64498a#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "ÿÿÿÿÿÿÿÿÿ(
,__inference_sequential_9_layer_call_fn_64511a#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª "ÿÿÿÿÿÿÿÿÿ(
#__inference_signature_wrapper_64583v#$E¢B
¢ 
;ª8
6
f0_input*'
f0_inputÿÿÿÿÿÿÿÿÿ@@"'ª$
"
d3
d3ÿÿÿÿÿÿÿÿÿ(
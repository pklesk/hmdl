ń
éŗ
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
Į
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
executor_typestring Ø
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ö
p
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d1/kernel
i
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel* 
_output_shapes
:
*
dtype0
g
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d1/bias
`
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes	
:*
dtype0
p
	d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d3/kernel
i
d3/kernel/Read/ReadVariableOpReadVariableOp	d3/kernel* 
_output_shapes
:
*
dtype0
g
d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d3/bias
`
d3/bias/Read/ReadVariableOpReadVariableOpd3/bias*
_output_shapes	
:*
dtype0
o
	d5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_name	d5/kernel
h
d5/kernel/Read/ReadVariableOpReadVariableOp	d5/kernel*
_output_shapes
:	
*
dtype0
f
d5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d5/bias
_
d5/bias/Read/ReadVariableOpReadVariableOpd5/bias*
_output_shapes
:
*
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
~
Adam/d1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d1/kernel/m
w
$Adam/d1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d1/kernel/m* 
_output_shapes
:
*
dtype0
u
Adam/d1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d1/bias/m
n
"Adam/d1/bias/m/Read/ReadVariableOpReadVariableOpAdam/d1/bias/m*
_output_shapes	
:*
dtype0
~
Adam/d3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d3/kernel/m
w
$Adam/d3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d3/kernel/m* 
_output_shapes
:
*
dtype0
u
Adam/d3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d3/bias/m
n
"Adam/d3/bias/m/Read/ReadVariableOpReadVariableOpAdam/d3/bias/m*
_output_shapes	
:*
dtype0
}
Adam/d5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*!
shared_nameAdam/d5/kernel/m
v
$Adam/d5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d5/kernel/m*
_output_shapes
:	
*
dtype0
t
Adam/d5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/d5/bias/m
m
"Adam/d5/bias/m/Read/ReadVariableOpReadVariableOpAdam/d5/bias/m*
_output_shapes
:
*
dtype0
~
Adam/d1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d1/kernel/v
w
$Adam/d1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d1/kernel/v* 
_output_shapes
:
*
dtype0
u
Adam/d1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d1/bias/v
n
"Adam/d1/bias/v/Read/ReadVariableOpReadVariableOpAdam/d1/bias/v*
_output_shapes	
:*
dtype0
~
Adam/d3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d3/kernel/v
w
$Adam/d3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d3/kernel/v* 
_output_shapes
:
*
dtype0
u
Adam/d3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d3/bias/v
n
"Adam/d3/bias/v/Read/ReadVariableOpReadVariableOpAdam/d3/bias/v*
_output_shapes	
:*
dtype0
}
Adam/d5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*!
shared_nameAdam/d5/kernel/v
v
$Adam/d5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d5/kernel/v*
_output_shapes
:	
*
dtype0
t
Adam/d5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/d5/bias/v
m
"Adam/d5/bias/v/Read/ReadVariableOpReadVariableOpAdam/d5/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ę8
value¼8B¹8 B²8
č
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
„
	variables
trainable_variables
 regularization_losses
!	keras_api
"_random_generator
#__call__
*$&call_and_return_all_conditional_losses* 
¦

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
„
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1_random_generator
2__call__
*3&call_and_return_all_conditional_losses* 
¦

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
²
<iter

=beta_1

>beta_2
	?decay
@learning_ratemvmw%mx&my4mz5m{v|v}%v~&v4v5v*
.
0
1
%2
&3
44
55*
.
0
1
%2
&3
44
55*
,
A0
B1
C2
D3
E4
F5* 
°
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Lserving_default* 
* 
* 
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

A0
B1* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
 regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*

C0
D1* 

\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
-	variables
.trainable_variables
/regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*

E0
F1* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
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
* 
* 
.
0
1
2
3
4
5*

k0
l1*
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
A0
B1* 
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
C0
D1* 
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
E0
F1* 
* 
8
	mtotal
	ncount
o	variables
p	keras_api*
H
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

o	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

q0
r1*

t	variables*
|v
VARIABLE_VALUEAdam/d1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_f0_inputPlaceholder*/
_output_shapes
:’’’’’’’’’*
dtype0*$
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_f0_input	d1/kerneld1/bias	d3/kerneld3/bias	d5/kerneld5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *,
f'R%
#__inference_signature_wrapper_19119
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Å	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd3/kernel/Read/ReadVariableOpd3/bias/Read/ReadVariableOpd5/kernel/Read/ReadVariableOpd5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/d1/kernel/m/Read/ReadVariableOp"Adam/d1/bias/m/Read/ReadVariableOp$Adam/d3/kernel/m/Read/ReadVariableOp"Adam/d3/bias/m/Read/ReadVariableOp$Adam/d5/kernel/m/Read/ReadVariableOp"Adam/d5/bias/m/Read/ReadVariableOp$Adam/d1/kernel/v/Read/ReadVariableOp"Adam/d1/bias/v/Read/ReadVariableOp$Adam/d3/kernel/v/Read/ReadVariableOp"Adam/d3/bias/v/Read/ReadVariableOp$Adam/d5/kernel/v/Read/ReadVariableOp"Adam/d5/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
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
__inference__traced_save_19390
¤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d3/kerneld3/bias	d5/kerneld5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/d1/kernel/mAdam/d1/bias/mAdam/d3/kernel/mAdam/d3/bias/mAdam/d5/kernel/mAdam/d5/bias/mAdam/d1/kernel/vAdam/d1/bias/vAdam/d3/kernel/vAdam/d3/bias/vAdam/d5/kernel/vAdam/d5/bias/v*'
Tin 
2*
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
!__inference__traced_restore_19481ø
æ
Y
=__inference_f0_layer_call_and_return_conditional_losses_18658

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õ
\
>__inference_dr2_layer_call_and_return_conditional_losses_18684

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

?
#__inference_dr2_layer_call_fn_19159

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18684a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ī
\
#__inference_dr4_layer_call_fn_19215

inputs
identity¢StatefulPartitionedCallĮ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_18783p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß
ļ
=__inference_d5_layer_call_and_return_conditional_losses_19256

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
`
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
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ā

"__inference_d3_layer_call_fn_19192

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_18699p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ė

E__inference_sequential_layer_call_and_return_conditional_losses_18970
f0_input
d1_18946:

d1_18948:	
d3_18952:

d3_18954:	
d5_18958:	

d5_18960:

identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCall¢d5/StatefulPartitionedCall¢dr2/StatefulPartitionedCall¢dr4/StatefulPartitionedCallµ
f0/PartitionedCallPartitionedCallf0_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_18658ņ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_18946d1_18948*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_18673ā
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18816ū
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_18952d3_18954*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_18699
dr4/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_18783ś
d5/StatefulPartitionedCallStatefulPartitionedCall$dr4/StatefulPartitionedCall:output:0d5_18958d5_18960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_18725`
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
 *    r
IdentityIdentity#d5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
Ł
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr4/StatefulPartitionedCalldr4/StatefulPartitionedCall:Y U
/
_output_shapes
:’’’’’’’’’
"
_user_specified_name
f0_input
Č
+
__inference_loss_fn_0_19261
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
/
Å
E__inference_sequential_layer_call_and_return_conditional_losses_19100

inputs5
!d1_matmul_readvariableop_resource:
1
"d1_biasadd_readvariableop_resource:	5
!d3_matmul_readvariableop_resource:
1
"d3_biasadd_readvariableop_resource:	4
!d5_matmul_readvariableop_resource:	
0
"d5_biasadd_readvariableop_resource:

identity¢d1/BiasAdd/ReadVariableOp¢d1/MatMul/ReadVariableOp¢d3/BiasAdd/ReadVariableOp¢d3/MatMul/ReadVariableOp¢d5/BiasAdd/ReadVariableOp¢d5/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0}
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’y
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’W
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dr2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?|
dr2/dropout/MulMuld1/Relu:activations:0dr2/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dr2/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:¦
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0*
seed2’’’’_
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >³
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
dr2/dropout/CastCastdr2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’v
dr2/dropout/Mul_1Muldr2/dropout/Mul:z:0dr2/dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’|
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
	d3/MatMulMatMuldr2/dropout/Mul_1:z:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’y
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’W
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dr4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?|
dr4/dropout/MulMuld3/Relu:activations:0dr4/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dr4/dropout/ShapeShaped3/Relu:activations:0*
T0*
_output_shapes
:¢
(dr4/dropout/random_uniform/RandomUniformRandomUniformdr4/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0*
seed2_
dr4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >³
dr4/dropout/GreaterEqualGreaterEqual1dr4/dropout/random_uniform/RandomUniform:output:0#dr4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
dr4/dropout/CastCastdr4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’v
dr4/dropout/Mul_1Muldr4/dropout/Mul:z:0dr4/dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’{
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0~
	d5/MatMulMatMuldr4/dropout/Mul_1:z:0 d5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
x
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
\

d5/SoftmaxSoftmaxd5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
`
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
 *    c
IdentityIdentityd5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
ė
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp26
d5/BiasAdd/ReadVariableOpd5/BiasAdd/ReadVariableOp24
d5/MatMul/ReadVariableOpd5/MatMul/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ä

#__inference_signature_wrapper_19119
f0_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *)
f$R"
 __inference__wrapped_model_18645o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:’’’’’’’’’
"
_user_specified_name
f0_input
!
Å
E__inference_sequential_layer_call_and_return_conditional_losses_19051

inputs5
!d1_matmul_readvariableop_resource:
1
"d1_biasadd_readvariableop_resource:	5
!d3_matmul_readvariableop_resource:
1
"d3_biasadd_readvariableop_resource:	4
!d5_matmul_readvariableop_resource:	
0
"d5_biasadd_readvariableop_resource:

identity¢d1/BiasAdd/ReadVariableOp¢d1/MatMul/ReadVariableOp¢d3/BiasAdd/ReadVariableOp¢d3/MatMul/ReadVariableOp¢d5/BiasAdd/ReadVariableOp¢d5/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0}
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’y
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’W
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
dr2/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’|
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
	d3/MatMulMatMuldr2/Identity:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’y
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’W
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
dr4/IdentityIdentityd3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’{
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0~
	d5/MatMulMatMuldr4/Identity:output:0 d5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
x
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
\

d5/SoftmaxSoftmaxd5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
`
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
 *    c
IdentityIdentityd5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
ė
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp26
d5/BiasAdd/ReadVariableOpd5/BiasAdd/ReadVariableOp24
d5/MatMul/ReadVariableOpd5/MatMul/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾

"__inference_d5_layer_call_fn_19243

inputs
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_18725o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ā
ń
=__inference_d1_layer_call_and_return_conditional_losses_19154

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
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
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õ
\
>__inference_dr4_layer_call_and_return_conditional_losses_19220

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	

*__inference_sequential_layer_call_fn_18753
f0_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18738o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:’’’’’’’’’
"
_user_specified_name
f0_input
ß
ļ
=__inference_d5_layer_call_and_return_conditional_losses_18725

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
`
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
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


]
>__inference_dr4_layer_call_and_return_conditional_losses_18783

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0*
seed2’’’’[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ø
>
"__inference_f0_layer_call_fn_19124

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
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_18658a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


]
>__inference_dr4_layer_call_and_return_conditional_losses_19232

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0*
seed2’’’’[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_1_19266
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
	

*__inference_sequential_layer_call_fn_18914
f0_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18882o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:’’’’’’’’’
"
_user_specified_name
f0_input
Õ
\
>__inference_dr2_layer_call_and_return_conditional_losses_19169

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ä"
¦
 __inference__wrapped_model_18645
f0_input@
,sequential_d1_matmul_readvariableop_resource:
<
-sequential_d1_biasadd_readvariableop_resource:	@
,sequential_d3_matmul_readvariableop_resource:
<
-sequential_d3_biasadd_readvariableop_resource:	?
,sequential_d5_matmul_readvariableop_resource:	
;
-sequential_d5_biasadd_readvariableop_resource:

identity¢$sequential/d1/BiasAdd/ReadVariableOp¢#sequential/d1/MatMul/ReadVariableOp¢$sequential/d3/BiasAdd/ReadVariableOp¢#sequential/d3/MatMul/ReadVariableOp¢$sequential/d5/BiasAdd/ReadVariableOp¢#sequential/d5/MatMul/ReadVariableOpd
sequential/f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  {
sequential/f0/ReshapeReshapef0_inputsequential/f0/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
#sequential/d1/MatMul/ReadVariableOpReadVariableOp,sequential_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sequential/d1/MatMulMatMulsequential/f0/Reshape:output:0+sequential/d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
$sequential/d1/BiasAdd/ReadVariableOpReadVariableOp-sequential_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0”
sequential/d1/BiasAddBiasAddsequential/d1/MatMul:product:0,sequential/d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
sequential/d1/ReluRelusequential/d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
sequential/dr2/IdentityIdentity sequential/d1/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
#sequential/d3/MatMul/ReadVariableOpReadVariableOp,sequential_d3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
sequential/d3/MatMulMatMul sequential/dr2/Identity:output:0+sequential/d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
$sequential/d3/BiasAdd/ReadVariableOpReadVariableOp-sequential_d3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0”
sequential/d3/BiasAddBiasAddsequential/d3/MatMul:product:0,sequential/d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
sequential/d3/ReluRelusequential/d3/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
sequential/dr4/IdentityIdentity sequential/d3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
#sequential/d5/MatMul/ReadVariableOpReadVariableOp,sequential_d5_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
sequential/d5/MatMulMatMul sequential/dr4/Identity:output:0+sequential/d5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’

$sequential/d5/BiasAdd/ReadVariableOpReadVariableOp-sequential_d5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0 
sequential/d5/BiasAddBiasAddsequential/d5/MatMul:product:0,sequential/d5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
r
sequential/d5/SoftmaxSoftmaxsequential/d5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
n
IdentityIdentitysequential/d5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
­
NoOpNoOp%^sequential/d1/BiasAdd/ReadVariableOp$^sequential/d1/MatMul/ReadVariableOp%^sequential/d3/BiasAdd/ReadVariableOp$^sequential/d3/MatMul/ReadVariableOp%^sequential/d5/BiasAdd/ReadVariableOp$^sequential/d5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 2L
$sequential/d1/BiasAdd/ReadVariableOp$sequential/d1/BiasAdd/ReadVariableOp2J
#sequential/d1/MatMul/ReadVariableOp#sequential/d1/MatMul/ReadVariableOp2L
$sequential/d3/BiasAdd/ReadVariableOp$sequential/d3/BiasAdd/ReadVariableOp2J
#sequential/d3/MatMul/ReadVariableOp#sequential/d3/MatMul/ReadVariableOp2L
$sequential/d5/BiasAdd/ReadVariableOp$sequential/d5/BiasAdd/ReadVariableOp2J
#sequential/d5/MatMul/ReadVariableOp#sequential/d5/MatMul/ReadVariableOp:Y U
/
_output_shapes
:’’’’’’’’’
"
_user_specified_name
f0_input
æ
Y
=__inference_f0_layer_call_and_return_conditional_losses_19130

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


]
>__inference_dr2_layer_call_and_return_conditional_losses_19181

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0*
seed2’’’’[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Å

E__inference_sequential_layer_call_and_return_conditional_losses_18882

inputs
d1_18858:

d1_18860:	
d3_18864:

d3_18866:	
d5_18870:	

d5_18872:

identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCall¢d5/StatefulPartitionedCall¢dr2/StatefulPartitionedCall¢dr4/StatefulPartitionedCall³
f0/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_18658ņ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_18858d1_18860*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_18673ā
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18816ū
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_18864d3_18866*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_18699
dr4/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_18783ś
d5/StatefulPartitionedCallStatefulPartitionedCall$dr4/StatefulPartitionedCall:output:0d5_18870d5_18872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_18725`
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
 *    r
IdentityIdentity#d5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
Ł
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr4/StatefulPartitionedCalldr4/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

?
#__inference_dr4_layer_call_fn_19210

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_18710a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_3_19276
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
ā
ń
=__inference_d3_layer_call_and_return_conditional_losses_19205

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
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
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
«:


__inference__traced_save_19390
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop(
$savev2_d3_kernel_read_readvariableop&
"savev2_d3_bias_read_readvariableop(
$savev2_d5_kernel_read_readvariableop&
"savev2_d5_bias_read_readvariableop(
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
+savev2_adam_d5_kernel_m_read_readvariableop-
)savev2_adam_d5_bias_m_read_readvariableop/
+savev2_adam_d1_kernel_v_read_readvariableop-
)savev2_adam_d1_bias_v_read_readvariableop/
+savev2_adam_d3_kernel_v_read_readvariableop-
)savev2_adam_d3_bias_v_read_readvariableop/
+savev2_adam_d5_kernel_v_read_readvariableop-
)savev2_adam_d5_bias_v_read_readvariableop
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
: õ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH„
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d3_kernel_read_readvariableop"savev2_d3_bias_read_readvariableop$savev2_d5_kernel_read_readvariableop"savev2_d5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_d1_kernel_m_read_readvariableop)savev2_adam_d1_bias_m_read_readvariableop+savev2_adam_d3_kernel_m_read_readvariableop)savev2_adam_d3_bias_m_read_readvariableop+savev2_adam_d5_kernel_m_read_readvariableop)savev2_adam_d5_bias_m_read_readvariableop+savev2_adam_d1_kernel_v_read_readvariableop)savev2_adam_d1_bias_v_read_readvariableop+savev2_adam_d3_kernel_v_read_readvariableop)savev2_adam_d3_bias_v_read_readvariableop+savev2_adam_d5_kernel_v_read_readvariableop)savev2_adam_d5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	
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

identity_1Identity_1:output:0*Š
_input_shapes¾
»: :
::
::	
:
: : : : : : : : : :
::
::	
:
:
::
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:
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
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: 


]
>__inference_dr2_layer_call_and_return_conditional_losses_18816

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0*
seed2’’’’[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ą
E__inference_sequential_layer_call_and_return_conditional_losses_18942
f0_input
d1_18918:

d1_18920:	
d3_18924:

d3_18926:	
d5_18930:	

d5_18932:

identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCall¢d5/StatefulPartitionedCallµ
f0/PartitionedCallPartitionedCallf0_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_18658ņ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_18918d1_18920*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_18673Ņ
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18684ó
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_18924d3_18926*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_18699Ņ
dr4/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_18710ņ
d5/StatefulPartitionedCallStatefulPartitionedCalldr4/PartitionedCall:output:0d5_18930d5_18932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_18725`
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
 *    r
IdentityIdentity#d5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’

NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall:Y U
/
_output_shapes
:’’’’’’’’’
"
_user_specified_name
f0_input

Ž
E__inference_sequential_layer_call_and_return_conditional_losses_18738

inputs
d1_18674:

d1_18676:	
d3_18700:

d3_18702:	
d5_18726:	

d5_18728:

identity¢d1/StatefulPartitionedCall¢d3/StatefulPartitionedCall¢d5/StatefulPartitionedCall³
f0/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_18658ņ
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_18674d1_18676*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_18673Ņ
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18684ó
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_18700d3_18702*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_18699Ņ
dr4/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_18710ņ
d5/StatefulPartitionedCallStatefulPartitionedCalldr4/PartitionedCall:output:0d5_18726d5_18728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_18725`
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
 *    r
IdentityIdentity#d5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’

NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ā
ń
=__inference_d1_layer_call_and_return_conditional_losses_18673

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
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
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_5_19286
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
Č
+
__inference_loss_fn_4_19281
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
Č
+
__inference_loss_fn_2_19271
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
Ā

"__inference_d1_layer_call_fn_19141

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_18673p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	

*__inference_sequential_layer_call_fn_18999

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18738o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ī
\
#__inference_dr2_layer_call_fn_19164

inputs
identity¢StatefulPartitionedCallĮ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18816p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õ
\
>__inference_dr4_layer_call_and_return_conditional_losses_18710

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
k

!__inference__traced_restore_19481
file_prefix.
assignvariableop_d1_kernel:
)
assignvariableop_1_d1_bias:	0
assignvariableop_2_d3_kernel:
)
assignvariableop_3_d3_bias:	/
assignvariableop_4_d5_kernel:	
(
assignvariableop_5_d5_bias:
&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: 8
$assignvariableop_15_adam_d1_kernel_m:
1
"assignvariableop_16_adam_d1_bias_m:	8
$assignvariableop_17_adam_d3_kernel_m:
1
"assignvariableop_18_adam_d3_bias_m:	7
$assignvariableop_19_adam_d5_kernel_m:	
0
"assignvariableop_20_adam_d5_bias_m:
8
$assignvariableop_21_adam_d1_kernel_v:
1
"assignvariableop_22_adam_d1_bias_v:	8
$assignvariableop_23_adam_d3_kernel_v:
1
"assignvariableop_24_adam_d3_bias_v:	7
$assignvariableop_25_adam_d5_kernel_v:	
0
"assignvariableop_26_adam_d5_bias_v:

identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ų
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
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
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_d5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_d5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_d1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_d1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_d3_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_d3_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_d5_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_d5_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_d1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_d1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_d3_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_d3_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_d5_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_d5_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ”
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
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
ā
ń
=__inference_d3_layer_call_and_return_conditional_losses_18699

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
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
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	

*__inference_sequential_layer_call_fn_19016

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18882o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"ŪL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Æ
serving_default
E
f0_input9
serving_default_f0_input:0’’’’’’’’’6
d50
StatefulPartitionedCall:0’’’’’’’’’
tensorflow/serving/predict:¦

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
„
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
	variables
trainable_variables
 regularization_losses
!	keras_api
"_random_generator
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
»

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1_random_generator
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
»

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
Į
<iter

=beta_1

>beta_2
	?decay
@learning_ratemvmw%mx&my4mz5m{v|v}%v~&v4v5v"
	optimizer
J
0
1
%2
&3
44
55"
trackable_list_wrapper
J
0
1
%2
&3
44
55"
trackable_list_wrapper
J
A0
B1
C2
D3
E4
F5"
trackable_list_wrapper
Ź
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_sequential_layer_call_fn_18753
*__inference_sequential_layer_call_fn_18999
*__inference_sequential_layer_call_fn_19016
*__inference_sequential_layer_call_fn_18914Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ß
E__inference_sequential_layer_call_and_return_conditional_losses_19051
E__inference_sequential_layer_call_and_return_conditional_losses_19100
E__inference_sequential_layer_call_and_return_conditional_losses_18942
E__inference_sequential_layer_call_and_return_conditional_losses_18970Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ĢBÉ
 __inference__wrapped_model_18645f0_input"
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
annotationsŖ *
 
,
Lserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ģ2É
"__inference_f0_layer_call_fn_19124¢
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
annotationsŖ *
 
ē2ä
=__inference_f0_layer_call_and_return_conditional_losses_19130¢
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
annotationsŖ *
 
:
2	d1/kernel
:2d1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ģ2É
"__inference_d1_layer_call_fn_19141¢
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
annotationsŖ *
 
ē2ä
=__inference_d1_layer_call_and_return_conditional_losses_19154¢
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
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
 regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr2_layer_call_fn_19159
#__inference_dr2_layer_call_fn_19164“
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
kwonlydefaultsŖ 
annotationsŖ *
 
ŗ2·
>__inference_dr2_layer_call_and_return_conditional_losses_19169
>__inference_dr2_layer_call_and_return_conditional_losses_19181“
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
kwonlydefaultsŖ 
annotationsŖ *
 
:
2	d3/kernel
:2d3/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
­
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ģ2É
"__inference_d3_layer_call_fn_19192¢
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
annotationsŖ *
 
ē2ä
=__inference_d3_layer_call_and_return_conditional_losses_19205¢
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
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
-	variables
.trainable_variables
/regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr4_layer_call_fn_19210
#__inference_dr4_layer_call_fn_19215“
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
kwonlydefaultsŖ 
annotationsŖ *
 
ŗ2·
>__inference_dr4_layer_call_and_return_conditional_losses_19220
>__inference_dr4_layer_call_and_return_conditional_losses_19232“
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
kwonlydefaultsŖ 
annotationsŖ *
 
:	
2	d5/kernel
:
2d5/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ģ2É
"__inference_d5_layer_call_fn_19243¢
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
annotationsŖ *
 
ē2ä
=__inference_d5_layer_call_and_return_conditional_losses_19256¢
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
annotationsŖ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
²2Æ
__inference_loss_fn_0_19261
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
annotationsŖ *¢ 
²2Æ
__inference_loss_fn_1_19266
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
annotationsŖ *¢ 
²2Æ
__inference_loss_fn_2_19271
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
annotationsŖ *¢ 
²2Æ
__inference_loss_fn_3_19276
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
annotationsŖ *¢ 
²2Æ
__inference_loss_fn_4_19281
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
annotationsŖ *¢ 
²2Æ
__inference_loss_fn_5_19286
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
annotationsŖ *¢ 
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ĖBČ
#__inference_signature_wrapper_19119f0_input"
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
annotationsŖ *
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
A0
B1"
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
C0
D1"
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
E0
F1"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	mtotal
	ncount
o	variables
p	keras_api"
_tf_keras_metric
^
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
": 
2Adam/d1/kernel/m
:2Adam/d1/bias/m
": 
2Adam/d3/kernel/m
:2Adam/d3/bias/m
!:	
2Adam/d5/kernel/m
:
2Adam/d5/bias/m
": 
2Adam/d1/kernel/v
:2Adam/d1/bias/v
": 
2Adam/d3/kernel/v
:2Adam/d3/bias/v
!:	
2Adam/d5/kernel/v
:
2Adam/d5/bias/v
 __inference__wrapped_model_18645l%&459¢6
/¢,
*'
f0_input’’’’’’’’’
Ŗ "'Ŗ$
"
d5
d5’’’’’’’’’

=__inference_d1_layer_call_and_return_conditional_losses_19154^0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 w
"__inference_d1_layer_call_fn_19141Q0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
=__inference_d3_layer_call_and_return_conditional_losses_19205^%&0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 w
"__inference_d3_layer_call_fn_19192Q%&0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
=__inference_d5_layer_call_and_return_conditional_losses_19256]450¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’

 v
"__inference_d5_layer_call_fn_19243P450¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
 
>__inference_dr2_layer_call_and_return_conditional_losses_19169^4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
  
>__inference_dr2_layer_call_and_return_conditional_losses_19181^4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 x
#__inference_dr2_layer_call_fn_19159Q4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’x
#__inference_dr2_layer_call_fn_19164Q4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’ 
>__inference_dr4_layer_call_and_return_conditional_losses_19220^4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
  
>__inference_dr4_layer_call_and_return_conditional_losses_19232^4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 x
#__inference_dr4_layer_call_fn_19210Q4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’x
#__inference_dr4_layer_call_fn_19215Q4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’¢
=__inference_f0_layer_call_and_return_conditional_losses_19130a7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 z
"__inference_f0_layer_call_fn_19124T7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "’’’’’’’’’7
__inference_loss_fn_0_19261¢

¢ 
Ŗ " 7
__inference_loss_fn_1_19266¢

¢ 
Ŗ " 7
__inference_loss_fn_2_19271¢

¢ 
Ŗ " 7
__inference_loss_fn_3_19276¢

¢ 
Ŗ " 7
__inference_loss_fn_4_19281¢

¢ 
Ŗ " 7
__inference_loss_fn_5_19286¢

¢ 
Ŗ " »
E__inference_sequential_layer_call_and_return_conditional_losses_18942r%&45A¢>
7¢4
*'
f0_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 »
E__inference_sequential_layer_call_and_return_conditional_losses_18970r%&45A¢>
7¢4
*'
f0_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 ¹
E__inference_sequential_layer_call_and_return_conditional_losses_19051p%&45?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 ¹
E__inference_sequential_layer_call_and_return_conditional_losses_19100p%&45?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 
*__inference_sequential_layer_call_fn_18753e%&45A¢>
7¢4
*'
f0_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

*__inference_sequential_layer_call_fn_18914e%&45A¢>
7¢4
*'
f0_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

*__inference_sequential_layer_call_fn_18999c%&45?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

*__inference_sequential_layer_call_fn_19016c%&45?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

#__inference_signature_wrapper_19119x%&45E¢B
¢ 
;Ŗ8
6
f0_input*'
f0_input’’’’’’’’’"'Ŗ$
"
d5
d5’’’’’’’’’

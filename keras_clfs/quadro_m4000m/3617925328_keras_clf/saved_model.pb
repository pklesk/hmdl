	
éş
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ú
p
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d1/kernel
i
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel* 
_output_shapes
:
*
dtype0
g
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d1/bias
`
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes	
:*
dtype0
p
	d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d3/kernel
i
d3/kernel/Read/ReadVariableOpReadVariableOp	d3/kernel* 
_output_shapes
:
*
dtype0
g
d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d3/bias
`
d3/bias/Read/ReadVariableOpReadVariableOpd3/bias*
_output_shapes	
:*
dtype0
p
	d5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d5/kernel
i
d5/kernel/Read/ReadVariableOpReadVariableOp	d5/kernel* 
_output_shapes
:
*
dtype0
g
d5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d5/bias
`
d5/bias/Read/ReadVariableOpReadVariableOpd5/bias*
_output_shapes	
:*
dtype0
o
	d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_name	d7/kernel
h
d7/kernel/Read/ReadVariableOpReadVariableOp	d7/kernel*
_output_shapes
:	
*
dtype0
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
*!
shared_nameAdam/d1/kernel/m
w
$Adam/d1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d1/kernel/m* 
_output_shapes
:
*
dtype0
u
Adam/d1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d1/bias/m
n
"Adam/d1/bias/m/Read/ReadVariableOpReadVariableOpAdam/d1/bias/m*
_output_shapes	
:*
dtype0
~
Adam/d3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d3/kernel/m
w
$Adam/d3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d3/kernel/m* 
_output_shapes
:
*
dtype0
u
Adam/d3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d3/bias/m
n
"Adam/d3/bias/m/Read/ReadVariableOpReadVariableOpAdam/d3/bias/m*
_output_shapes	
:*
dtype0
~
Adam/d5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d5/kernel/m
w
$Adam/d5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d5/kernel/m* 
_output_shapes
:
*
dtype0
u
Adam/d5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d5/bias/m
n
"Adam/d5/bias/m/Read/ReadVariableOpReadVariableOpAdam/d5/bias/m*
_output_shapes	
:*
dtype0
}
Adam/d7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*!
shared_nameAdam/d7/kernel/m
v
$Adam/d7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d7/kernel/m*
_output_shapes
:	
*
dtype0
t
Adam/d7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/d7/bias/m
m
"Adam/d7/bias/m/Read/ReadVariableOpReadVariableOpAdam/d7/bias/m*
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
*!
shared_nameAdam/d1/kernel/v
w
$Adam/d1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d1/kernel/v* 
_output_shapes
:
*
dtype0
u
Adam/d1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d1/bias/v
n
"Adam/d1/bias/v/Read/ReadVariableOpReadVariableOpAdam/d1/bias/v*
_output_shapes	
:*
dtype0
~
Adam/d3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d3/kernel/v
w
$Adam/d3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d3/kernel/v* 
_output_shapes
:
*
dtype0
u
Adam/d3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d3/bias/v
n
"Adam/d3/bias/v/Read/ReadVariableOpReadVariableOpAdam/d3/bias/v*
_output_shapes	
:*
dtype0
~
Adam/d5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameAdam/d5/kernel/v
w
$Adam/d5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d5/kernel/v* 
_output_shapes
:
*
dtype0
u
Adam/d5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d5/bias/v
n
"Adam/d5/bias/v/Read/ReadVariableOpReadVariableOpAdam/d5/bias/v*
_output_shapes	
:*
dtype0
}
Adam/d7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*!
shared_nameAdam/d7/kernel/v
v
$Adam/d7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d7/kernel/v*
_output_shapes
:	
*
dtype0
t
Adam/d7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/d7/bias/v
m
"Adam/d7/bias/v/Read/ReadVariableOpReadVariableOpAdam/d7/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
÷G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*˛G
value¨GBĽG BG

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
Ś

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Ľ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$_random_generator
%__call__
*&&call_and_return_all_conditional_losses* 
Ś

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
Ľ
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3_random_generator
4__call__
*5&call_and_return_all_conditional_losses* 
Ś

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
Ľ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B_random_generator
C__call__
*D&call_and_return_all_conditional_losses* 
Ś

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*
ä
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratemm'm(m6m7mEmFmvv'v(v6v7v EvĄFv˘*
<
0
1
'2
(3
64
75
E6
F7*
<
0
1
'2
(3
64
75
E6
F7*
:
R0
S1
T2
U3
V4
W5
X6
Y7* 
°
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

_serving_default* 
* 
* 
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

R0
S1* 

enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
 	variables
!trainable_variables
"regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*

T0
U1* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
/	variables
0trainable_variables
1regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*

V0
W1* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*

X0
Y1* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
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
* 
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
0
1*
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

V0
W1* 
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
X0
Y1* 
* 
<

total

count
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
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
VARIABLE_VALUEAdam/d7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
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
|v
VARIABLE_VALUEAdam/d7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_f0_inputPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
dtype0*$
shape:˙˙˙˙˙˙˙˙˙  

StatefulPartitionedCallStatefulPartitionedCallserving_default_f0_input	d1/kerneld1/bias	d3/kerneld3/bias	d5/kerneld5/bias	d7/kerneld7/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *,
f'R%
#__inference_signature_wrapper_30081
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd3/kernel/Read/ReadVariableOpd3/bias/Read/ReadVariableOpd5/kernel/Read/ReadVariableOpd5/bias/Read/ReadVariableOpd7/kernel/Read/ReadVariableOpd7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/d1/kernel/m/Read/ReadVariableOp"Adam/d1/bias/m/Read/ReadVariableOp$Adam/d3/kernel/m/Read/ReadVariableOp"Adam/d3/bias/m/Read/ReadVariableOp$Adam/d5/kernel/m/Read/ReadVariableOp"Adam/d5/bias/m/Read/ReadVariableOp$Adam/d7/kernel/m/Read/ReadVariableOp"Adam/d7/bias/m/Read/ReadVariableOp$Adam/d1/kernel/v/Read/ReadVariableOp"Adam/d1/bias/v/Read/ReadVariableOp$Adam/d3/kernel/v/Read/ReadVariableOp"Adam/d3/bias/v/Read/ReadVariableOp$Adam/d5/kernel/v/Read/ReadVariableOp"Adam/d5/bias/v/Read/ReadVariableOp$Adam/d7/kernel/v/Read/ReadVariableOp"Adam/d7/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
__inference__traced_save_30431

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d3/kerneld3/bias	d5/kerneld5/bias	d7/kerneld7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/d1/kernel/mAdam/d1/bias/mAdam/d3/kernel/mAdam/d3/bias/mAdam/d5/kernel/mAdam/d5/bias/mAdam/d7/kernel/mAdam/d7/bias/mAdam/d1/kernel/vAdam/d1/bias/vAdam/d3/kernel/vAdam/d3/bias/vAdam/d5/kernel/vAdam/d5/bias/vAdam/d7/kernel/vAdam/d7/bias/v*-
Tin&
$2"*
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
!__inference__traced_restore_30540ŻÂ
Č
+
__inference_loss_fn_0_30274
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
â
ń
=__inference_d3_layer_call_and_return_conditional_losses_30167

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
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
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


]
>__inference_dr4_layer_call_and_return_conditional_losses_29668

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
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2˙˙˙˙[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ő
\
>__inference_dr6_layer_call_and_return_conditional_losses_30233

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_7_30309
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
î
\
#__inference_dr2_layer_call_fn_30126

inputs
identity˘StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_29701p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


]
>__inference_dr2_layer_call_and_return_conditional_losses_30143

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
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2˙˙˙˙[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ő
\
>__inference_dr4_layer_call_and_return_conditional_losses_30182

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
î
\
#__inference_dr4_layer_call_fn_30177

inputs
identity˘StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_29668p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
É%
´
E__inference_sequential_layer_call_and_return_conditional_losses_29855
f0_input
d1_29823:

d1_29825:	
d3_29829:

d3_29831:	
d5_29835:

d5_29837:	
d7_29841:	

d7_29843:

identity˘d1/StatefulPartitionedCall˘d3/StatefulPartitionedCall˘d5/StatefulPartitionedCall˘d7/StatefulPartitionedCallľ
f0/PartitionedCallPartitionedCallf0_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_29478ň
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_29823d1_29825*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_29493Ň
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_29504ó
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_29829d3_29831*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_29519Ň
dr4/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_29530ó
d5/StatefulPartitionedCallStatefulPartitionedCalldr4/PartitionedCall:output:0d5_29835d5_29837*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29545Ň
dr6/PartitionedCallPartitionedCall#d5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29556ň
d7/StatefulPartitionedCallStatefulPartitionedCalldr6/PartitionedCall:output:0d7_29841d7_29843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29571`
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
:˙˙˙˙˙˙˙˙˙
ş
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
"
_user_specified_name
f0_input
Â

"__inference_d3_layer_call_fn_30154

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_29519p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ż)

E__inference_sequential_layer_call_and_return_conditional_losses_29779

inputs
d1_29747:

d1_29749:	
d3_29753:

d3_29755:	
d5_29759:

d5_29761:	
d7_29765:	

d7_29767:

identity˘d1/StatefulPartitionedCall˘d3/StatefulPartitionedCall˘d5/StatefulPartitionedCall˘d7/StatefulPartitionedCall˘dr2/StatefulPartitionedCall˘dr4/StatefulPartitionedCall˘dr6/StatefulPartitionedCallł
f0/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_29478ň
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_29747d1_29749*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_29493â
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_29701ű
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_29753d3_29755*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_29519
dr4/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_29668ű
d5/StatefulPartitionedCallStatefulPartitionedCall$dr4/StatefulPartitionedCall:output:0d5_29759d5_29761*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29545
dr6/StatefulPartitionedCallStatefulPartitionedCall#d5/StatefulPartitionedCall:output:0^dr4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29635ú
d7/StatefulPartitionedCallStatefulPartitionedCall$dr6/StatefulPartitionedCall:output:0d7_29765d7_29767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29571`
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
:˙˙˙˙˙˙˙˙˙

NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr4/StatefulPartitionedCall^dr6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr4/StatefulPartitionedCalldr4/StatefulPartitionedCall2:
dr6/StatefulPartitionedCalldr6/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs
Ő
\
>__inference_dr2_layer_call_and_return_conditional_losses_29504

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

?
#__inference_dr2_layer_call_fn_30121

inputs
identityą
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_29504a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


]
>__inference_dr2_layer_call_and_return_conditional_losses_29701

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
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2˙˙˙˙[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Č
+
__inference_loss_fn_4_30294
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
Ő
\
>__inference_dr6_layer_call_and_return_conditional_losses_29556

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
×*
ć
E__inference_sequential_layer_call_and_return_conditional_losses_29992

inputs5
!d1_matmul_readvariableop_resource:
1
"d1_biasadd_readvariableop_resource:	5
!d3_matmul_readvariableop_resource:
1
"d3_biasadd_readvariableop_resource:	5
!d5_matmul_readvariableop_resource:
1
"d5_biasadd_readvariableop_resource:	4
!d7_matmul_readvariableop_resource:	
0
"d7_biasadd_readvariableop_resource:

identity˘d1/BiasAdd/ReadVariableOp˘d1/MatMul/ReadVariableOp˘d3/BiasAdd/ReadVariableOp˘d3/MatMul/ReadVariableOp˘d5/BiasAdd/ReadVariableOp˘d5/MatMul/ReadVariableOp˘d7/BiasAdd/ReadVariableOp˘d7/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0}
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dr2/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
	d3/MatMulMatMuldr2/Identity:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dr4/IdentityIdentityd3/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
	d5/MatMulMatMuldr4/Identity:output:0 d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
d5/ReluRelud5/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dr6/IdentityIdentityd5/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0~
	d7/MatMulMatMuldr6/Identity:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\

d7/SoftmaxSoftmaxd7/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
˘
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp26
d5/BiasAdd/ReadVariableOpd5/BiasAdd/ReadVariableOp24
d5/MatMul/ReadVariableOpd5/MatMul/ReadVariableOp26
d7/BiasAdd/ReadVariableOpd7/BiasAdd/ReadVariableOp24
d7/MatMul/ReadVariableOpd7/MatMul/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs
â
ń
=__inference_d5_layer_call_and_return_conditional_losses_30218

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
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
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
î
\
#__inference_dr6_layer_call_fn_30228

inputs
identity˘StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29635p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


]
>__inference_dr4_layer_call_and_return_conditional_losses_30194

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
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2˙˙˙˙[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Â

"__inference_d1_layer_call_fn_30103

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_29493p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť	
ž
#__inference_signature_wrapper_30081
f0_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *)
f$R"
 __inference__wrapped_model_29465o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
"
_user_specified_name
f0_input

×
!__inference__traced_restore_30540
file_prefix.
assignvariableop_d1_kernel:
)
assignvariableop_1_d1_bias:	0
assignvariableop_2_d3_kernel:
)
assignvariableop_3_d3_bias:	0
assignvariableop_4_d5_kernel:
)
assignvariableop_5_d5_bias:	/
assignvariableop_6_d7_kernel:	
(
assignvariableop_7_d7_bias:
&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: 8
$assignvariableop_17_adam_d1_kernel_m:
1
"assignvariableop_18_adam_d1_bias_m:	8
$assignvariableop_19_adam_d3_kernel_m:
1
"assignvariableop_20_adam_d3_bias_m:	8
$assignvariableop_21_adam_d5_kernel_m:
1
"assignvariableop_22_adam_d5_bias_m:	7
$assignvariableop_23_adam_d7_kernel_m:	
0
"assignvariableop_24_adam_d7_bias_m:
8
$assignvariableop_25_adam_d1_kernel_v:
1
"assignvariableop_26_adam_d1_bias_v:	8
$assignvariableop_27_adam_d3_kernel_v:
1
"assignvariableop_28_adam_d3_bias_v:	8
$assignvariableop_29_adam_d5_kernel_v:
1
"assignvariableop_30_adam_d5_bias_v:	7
$assignvariableop_31_adam_d7_kernel_v:	
0
"assignvariableop_32_adam_d7_bias_v:

identity_34˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9˛
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ř
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
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
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_d7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_d7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_d1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_d1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_d3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_d3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_d5_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_d5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_d7_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_d7_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_d1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_d1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_d3_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_d3_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp$assignvariableop_29_adam_d5_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_d5_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_d7_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_d7_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ľ
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
ç	
Ĺ
*__inference_sequential_layer_call_fn_29605
f0_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identity˘StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29586o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
"
_user_specified_name
f0_input
â
ń
=__inference_d1_layer_call_and_return_conditional_losses_29493

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
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
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á	
Ă
*__inference_sequential_layer_call_fn_29947

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29779o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs

?
#__inference_dr6_layer_call_fn_30223

inputs
identityą
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29556a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ő
\
>__inference_dr4_layer_call_and_return_conditional_losses_29530

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨
>
"__inference_f0_layer_call_fn_30086

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
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_29478a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙  :W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs
ľ)

E__inference_sequential_layer_call_and_return_conditional_losses_29891
f0_input
d1_29859:

d1_29861:	
d3_29865:

d3_29867:	
d5_29871:

d5_29873:	
d7_29877:	

d7_29879:

identity˘d1/StatefulPartitionedCall˘d3/StatefulPartitionedCall˘d5/StatefulPartitionedCall˘d7/StatefulPartitionedCall˘dr2/StatefulPartitionedCall˘dr4/StatefulPartitionedCall˘dr6/StatefulPartitionedCallľ
f0/PartitionedCallPartitionedCallf0_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_29478ň
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_29859d1_29861*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_29493â
dr2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_29701ű
d3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0d3_29865d3_29867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_29519
dr4/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_29668ű
d5/StatefulPartitionedCallStatefulPartitionedCall$dr4/StatefulPartitionedCall:output:0d5_29871d5_29873*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29545
dr6/StatefulPartitionedCallStatefulPartitionedCall#d5/StatefulPartitionedCall:output:0^dr4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29635ú
d7/StatefulPartitionedCallStatefulPartitionedCall$dr6/StatefulPartitionedCall:output:0d7_29877d7_29879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29571`
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
:˙˙˙˙˙˙˙˙˙

NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr4/StatefulPartitionedCall^dr6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr4/StatefulPartitionedCalldr4/StatefulPartitionedCall2:
dr6/StatefulPartitionedCalldr6/StatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
"
_user_specified_name
f0_input
â
ń
=__inference_d5_layer_call_and_return_conditional_losses_29545

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
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
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ő
\
>__inference_dr2_layer_call_and_return_conditional_losses_30131

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


]
>__inference_dr6_layer_call_and_return_conditional_losses_29635

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
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2˙˙˙˙[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç	
Ĺ
*__inference_sequential_layer_call_fn_29819
f0_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identity˘StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallf0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29779o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
"
_user_specified_name
f0_input
Č
+
__inference_loss_fn_2_30284
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
Ä
+
__inference_loss_fn_3_30289
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
Ă%
˛
E__inference_sequential_layer_call_and_return_conditional_losses_29586

inputs
d1_29494:

d1_29496:	
d3_29520:

d3_29522:	
d5_29546:

d5_29548:	
d7_29572:	

d7_29574:

identity˘d1/StatefulPartitionedCall˘d3/StatefulPartitionedCall˘d5/StatefulPartitionedCall˘d7/StatefulPartitionedCallł
f0/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f0_layer_call_and_return_conditional_losses_29478ň
d1/StatefulPartitionedCallStatefulPartitionedCallf0/PartitionedCall:output:0d1_29494d1_29496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_29493Ň
dr2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_29504ó
d3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0d3_29520d3_29522*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_29519Ň
dr4/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_29530ó
d5/StatefulPartitionedCallStatefulPartitionedCalldr4/PartitionedCall:output:0d5_29546d5_29548*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29545Ň
dr6/PartitionedCallPartitionedCall#d5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29556ň
d7/StatefulPartitionedCallStatefulPartitionedCalldr6/PartitionedCall:output:0d7_29572d7_29574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29571`
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
:˙˙˙˙˙˙˙˙˙
ş
NoOpNoOp^d1/StatefulPartitionedCall^d3/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_5_30299
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
Â

"__inference_d5_layer_call_fn_30205

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29545p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
â
ń
=__inference_d1_layer_call_and_return_conditional_losses_30116

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
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
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ß
ď
=__inference_d7_layer_call_and_return_conditional_losses_30269

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


]
>__inference_dr6_layer_call_and_return_conditional_losses_30245

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
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2˙˙˙˙[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

?
#__inference_dr4_layer_call_fn_30172

inputs
identityą
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr4_layer_call_and_return_conditional_losses_29530a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
D
Ż
__inference__traced_save_30431
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop(
$savev2_d3_kernel_read_readvariableop&
"savev2_d3_bias_read_readvariableop(
$savev2_d5_kernel_read_readvariableop&
"savev2_d5_bias_read_readvariableop(
$savev2_d7_kernel_read_readvariableop&
"savev2_d7_bias_read_readvariableop(
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
+savev2_adam_d7_kernel_m_read_readvariableop-
)savev2_adam_d7_bias_m_read_readvariableop/
+savev2_adam_d1_kernel_v_read_readvariableop-
)savev2_adam_d1_bias_v_read_readvariableop/
+savev2_adam_d3_kernel_v_read_readvariableop-
)savev2_adam_d3_bias_v_read_readvariableop/
+savev2_adam_d5_kernel_v_read_readvariableop-
)savev2_adam_d5_bias_v_read_readvariableop/
+savev2_adam_d7_kernel_v_read_readvariableop-
)savev2_adam_d7_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
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
: Ż
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ř
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHą
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d3_kernel_read_readvariableop"savev2_d3_bias_read_readvariableop$savev2_d5_kernel_read_readvariableop"savev2_d5_bias_read_readvariableop$savev2_d7_kernel_read_readvariableop"savev2_d7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_d1_kernel_m_read_readvariableop)savev2_adam_d1_bias_m_read_readvariableop+savev2_adam_d3_kernel_m_read_readvariableop)savev2_adam_d3_bias_m_read_readvariableop+savev2_adam_d5_kernel_m_read_readvariableop)savev2_adam_d5_bias_m_read_readvariableop+savev2_adam_d7_kernel_m_read_readvariableop)savev2_adam_d7_bias_m_read_readvariableop+savev2_adam_d1_kernel_v_read_readvariableop)savev2_adam_d1_bias_v_read_readvariableop+savev2_adam_d3_kernel_v_read_readvariableop)savev2_adam_d3_bias_v_read_readvariableop+savev2_adam_d5_kernel_v_read_readvariableop)savev2_adam_d5_bias_v_read_readvariableop+savev2_adam_d7_kernel_v_read_readvariableop)savev2_adam_d7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
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

identity_1Identity_1:output:0*
_input_shapes÷
ô: :
::
::
::	
:
: : : : : : : : : :
::
::
::	
:
:
::
::
::	
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
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:	
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
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	
: !

_output_shapes
:
:"

_output_shapes
: 
ż
Y
=__inference_f0_layer_call_and_return_conditional_losses_29478

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙  :W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs
Ä
+
__inference_loss_fn_1_30279
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
Č
+
__inference_loss_fn_6_30304
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
ß
ď
=__inference_d7_layer_call_and_return_conditional_losses_29571

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á	
Ă
*__inference_sequential_layer_call_fn_29926

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29586o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs
ż
Y
=__inference_f0_layer_call_and_return_conditional_losses_30092

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙  :W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs
ó,
ó
 __inference__wrapped_model_29465
f0_input@
,sequential_d1_matmul_readvariableop_resource:
<
-sequential_d1_biasadd_readvariableop_resource:	@
,sequential_d3_matmul_readvariableop_resource:
<
-sequential_d3_biasadd_readvariableop_resource:	@
,sequential_d5_matmul_readvariableop_resource:
<
-sequential_d5_biasadd_readvariableop_resource:	?
,sequential_d7_matmul_readvariableop_resource:	
;
-sequential_d7_biasadd_readvariableop_resource:

identity˘$sequential/d1/BiasAdd/ReadVariableOp˘#sequential/d1/MatMul/ReadVariableOp˘$sequential/d3/BiasAdd/ReadVariableOp˘#sequential/d3/MatMul/ReadVariableOp˘$sequential/d5/BiasAdd/ReadVariableOp˘#sequential/d5/MatMul/ReadVariableOp˘$sequential/d7/BiasAdd/ReadVariableOp˘#sequential/d7/MatMul/ReadVariableOpd
sequential/f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   {
sequential/f0/ReshapeReshapef0_inputsequential/f0/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#sequential/d1/MatMul/ReadVariableOpReadVariableOp,sequential_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sequential/d1/MatMulMatMulsequential/f0/Reshape:output:0+sequential/d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
$sequential/d1/BiasAdd/ReadVariableOpReadVariableOp-sequential_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ą
sequential/d1/BiasAddBiasAddsequential/d1/MatMul:product:0,sequential/d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
sequential/d1/ReluRelusequential/d1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
sequential/dr2/IdentityIdentity sequential/d1/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#sequential/d3/MatMul/ReadVariableOpReadVariableOp,sequential_d3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
sequential/d3/MatMulMatMul sequential/dr2/Identity:output:0+sequential/d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
$sequential/d3/BiasAdd/ReadVariableOpReadVariableOp-sequential_d3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ą
sequential/d3/BiasAddBiasAddsequential/d3/MatMul:product:0,sequential/d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
sequential/d3/ReluRelusequential/d3/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
sequential/dr4/IdentityIdentity sequential/d3/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#sequential/d5/MatMul/ReadVariableOpReadVariableOp,sequential_d5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
sequential/d5/MatMulMatMul sequential/dr4/Identity:output:0+sequential/d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
$sequential/d5/BiasAdd/ReadVariableOpReadVariableOp-sequential_d5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ą
sequential/d5/BiasAddBiasAddsequential/d5/MatMul:product:0,sequential/d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
sequential/d5/ReluRelusequential/d5/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
sequential/dr6/IdentityIdentity sequential/d5/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#sequential/d7/MatMul/ReadVariableOpReadVariableOp,sequential_d7_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
sequential/d7/MatMulMatMul sequential/dr6/Identity:output:0+sequential/d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$sequential/d7/BiasAdd/ReadVariableOpReadVariableOp-sequential_d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0 
sequential/d7/BiasAddBiasAddsequential/d7/MatMul:product:0,sequential/d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
sequential/d7/SoftmaxSoftmaxsequential/d7/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
IdentityIdentitysequential/d7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ú
NoOpNoOp%^sequential/d1/BiasAdd/ReadVariableOp$^sequential/d1/MatMul/ReadVariableOp%^sequential/d3/BiasAdd/ReadVariableOp$^sequential/d3/MatMul/ReadVariableOp%^sequential/d5/BiasAdd/ReadVariableOp$^sequential/d5/MatMul/ReadVariableOp%^sequential/d7/BiasAdd/ReadVariableOp$^sequential/d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 2L
$sequential/d1/BiasAdd/ReadVariableOp$sequential/d1/BiasAdd/ReadVariableOp2J
#sequential/d1/MatMul/ReadVariableOp#sequential/d1/MatMul/ReadVariableOp2L
$sequential/d3/BiasAdd/ReadVariableOp$sequential/d3/BiasAdd/ReadVariableOp2J
#sequential/d3/MatMul/ReadVariableOp#sequential/d3/MatMul/ReadVariableOp2L
$sequential/d5/BiasAdd/ReadVariableOp$sequential/d5/BiasAdd/ReadVariableOp2J
#sequential/d5/MatMul/ReadVariableOp#sequential/d5/MatMul/ReadVariableOp2L
$sequential/d7/BiasAdd/ReadVariableOp$sequential/d7/BiasAdd/ReadVariableOp2J
#sequential/d7/MatMul/ReadVariableOp#sequential/d7/MatMul/ReadVariableOp:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
"
_user_specified_name
f0_input
â
ń
=__inference_d3_layer_call_and_return_conditional_losses_29519

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
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
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ž

"__inference_d7_layer_call_fn_30256

inputs
unknown:	

	unknown_0:

identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29571o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ă?
ć
E__inference_sequential_layer_call_and_return_conditional_losses_30058

inputs5
!d1_matmul_readvariableop_resource:
1
"d1_biasadd_readvariableop_resource:	5
!d3_matmul_readvariableop_resource:
1
"d3_biasadd_readvariableop_resource:	5
!d5_matmul_readvariableop_resource:
1
"d5_biasadd_readvariableop_resource:	4
!d7_matmul_readvariableop_resource:	
0
"d7_biasadd_readvariableop_resource:

identity˘d1/BiasAdd/ReadVariableOp˘d1/MatMul/ReadVariableOp˘d3/BiasAdd/ReadVariableOp˘d3/MatMul/ReadVariableOp˘d5/BiasAdd/ReadVariableOp˘d5/MatMul/ReadVariableOp˘d7/BiasAdd/ReadVariableOp˘d7/MatMul/ReadVariableOpY
f0/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   c

f0/ReshapeReshapeinputsf0/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0}
	d1/MatMulMatMulf0/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dr2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?|
dr2/dropout/MulMuld1/Relu:activations:0dr2/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dr2/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:Ś
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2˙˙˙˙_
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >ł
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
dr2/dropout/CastCastdr2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
dr2/dropout/Mul_1Muldr2/dropout/Mul:z:0dr2/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
	d3/MatMulMatMuldr2/dropout/Mul_1:z:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dr4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?|
dr4/dropout/MulMuld3/Relu:activations:0dr4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dr4/dropout/ShapeShaped3/Relu:activations:0*
T0*
_output_shapes
:˘
(dr4/dropout/random_uniform/RandomUniformRandomUniformdr4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2_
dr4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >ł
dr4/dropout/GreaterEqualGreaterEqual1dr4/dropout/random_uniform/RandomUniform:output:0#dr4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
dr4/dropout/CastCastdr4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
dr4/dropout/Mul_1Muldr4/dropout/Mul:z:0dr4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
	d5/MatMulMatMuldr4/dropout/Mul_1:z:0 d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
d5/ReluRelud5/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dr6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?|
dr6/dropout/MulMuld5/Relu:activations:0dr6/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dr6/dropout/ShapeShaped5/Relu:activations:0*
T0*
_output_shapes
:˘
(dr6/dropout/random_uniform/RandomUniformRandomUniformdr6/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
seed2_
dr6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >ł
dr6/dropout/GreaterEqualGreaterEqual1dr6/dropout/random_uniform/RandomUniform:output:0#dr6/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
dr6/dropout/CastCastdr6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
dr6/dropout/Mul_1Muldr6/dropout/Mul:z:0dr6/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0~
	d7/MatMulMatMuldr6/dropout/Mul_1:z:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\

d7/SoftmaxSoftmaxd7/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
˘
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙  : : : : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp26
d5/BiasAdd/ReadVariableOpd5/BiasAdd/ReadVariableOp24
d5/MatMul/ReadVariableOpd5/MatMul/ReadVariableOp26
d7/BiasAdd/ReadVariableOpd7/BiasAdd/ReadVariableOp24
d7/MatMul/ReadVariableOpd7/MatMul/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
 
_user_specified_nameinputs"ŰL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ż
serving_default
E
f0_input9
serving_default_f0_input:0˙˙˙˙˙˙˙˙˙  6
d70
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict:­Ś
ś
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ľ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ź
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$_random_generator
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
ź
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3_random_generator
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
ź
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B_random_generator
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
ó
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratemm'm(m6m7mEmFmvv'v(v6v7v EvĄFv˘"
	optimizer
X
0
1
'2
(3
64
75
E6
F7"
trackable_list_wrapper
X
0
1
'2
(3
64
75
E6
F7"
trackable_list_wrapper
X
R0
S1
T2
U3
V4
W5
X6
Y7"
trackable_list_wrapper
Ę
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_sequential_layer_call_fn_29605
*__inference_sequential_layer_call_fn_29926
*__inference_sequential_layer_call_fn_29947
*__inference_sequential_layer_call_fn_29819Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_29992
E__inference_sequential_layer_call_and_return_conditional_losses_30058
E__inference_sequential_layer_call_and_return_conditional_losses_29855
E__inference_sequential_layer_call_and_return_conditional_losses_29891Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
ĚBÉ
 __inference__wrapped_model_29465f0_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
,
_serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ě2É
"__inference_f0_layer_call_fn_30086˘
˛
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
annotationsŞ *
 
ç2ä
=__inference_f0_layer_call_and_return_conditional_losses_30092˘
˛
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
annotationsŞ *
 
:
2	d1/kernel
:2d1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
­
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ě2É
"__inference_d1_layer_call_fn_30103˘
˛
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
annotationsŞ *
 
ç2ä
=__inference_d1_layer_call_and_return_conditional_losses_30116˘
˛
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
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
 	variables
!trainable_variables
"regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr2_layer_call_fn_30121
#__inference_dr2_layer_call_fn_30126´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
ş2ˇ
>__inference_dr2_layer_call_and_return_conditional_losses_30131
>__inference_dr2_layer_call_and_return_conditional_losses_30143´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
:
2	d3/kernel
:2d3/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ě2É
"__inference_d3_layer_call_fn_30154˘
˛
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
annotationsŞ *
 
ç2ä
=__inference_d3_layer_call_and_return_conditional_losses_30167˘
˛
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
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
/	variables
0trainable_variables
1regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr4_layer_call_fn_30172
#__inference_dr4_layer_call_fn_30177´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
ş2ˇ
>__inference_dr4_layer_call_and_return_conditional_losses_30182
>__inference_dr4_layer_call_and_return_conditional_losses_30194´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
:
2	d5/kernel
:2d5/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
Ě2É
"__inference_d5_layer_call_fn_30205˘
˛
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
annotationsŞ *
 
ç2ä
=__inference_d5_layer_call_and_return_conditional_losses_30218˘
˛
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
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr6_layer_call_fn_30223
#__inference_dr6_layer_call_fn_30228´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
ş2ˇ
>__inference_dr6_layer_call_and_return_conditional_losses_30233
>__inference_dr6_layer_call_and_return_conditional_losses_30245´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
:	
2	d7/kernel
:
2d7/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
Ě2É
"__inference_d7_layer_call_fn_30256˘
˛
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
annotationsŞ *
 
ç2ä
=__inference_d7_layer_call_and_return_conditional_losses_30269˘
˛
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
annotationsŞ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
˛2Ż
__inference_loss_fn_0_30274
˛
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
annotationsŞ *˘ 
˛2Ż
__inference_loss_fn_1_30279
˛
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
annotationsŞ *˘ 
˛2Ż
__inference_loss_fn_2_30284
˛
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
annotationsŞ *˘ 
˛2Ż
__inference_loss_fn_3_30289
˛
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
annotationsŞ *˘ 
˛2Ż
__inference_loss_fn_4_30294
˛
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
annotationsŞ *˘ 
˛2Ż
__inference_loss_fn_5_30299
˛
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
annotationsŞ *˘ 
˛2Ż
__inference_loss_fn_6_30304
˛
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
annotationsŞ *˘ 
˛2Ż
__inference_loss_fn_7_30309
˛
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
annotationsŞ *˘ 
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBČ
#__inference_signature_wrapper_30081f0_input"
˛
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
annotationsŞ *
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
R0
S1"
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
T0
U1"
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
V0
W1"
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
X0
Y1"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
": 
2Adam/d1/kernel/m
:2Adam/d1/bias/m
": 
2Adam/d3/kernel/m
:2Adam/d3/bias/m
": 
2Adam/d5/kernel/m
:2Adam/d5/bias/m
!:	
2Adam/d7/kernel/m
:
2Adam/d7/bias/m
": 
2Adam/d1/kernel/v
:2Adam/d1/bias/v
": 
2Adam/d3/kernel/v
:2Adam/d3/bias/v
": 
2Adam/d5/kernel/v
:2Adam/d5/bias/v
!:	
2Adam/d7/kernel/v
:
2Adam/d7/bias/v
 __inference__wrapped_model_29465n'(67EF9˘6
/˘,
*'
f0_input˙˙˙˙˙˙˙˙˙  
Ş "'Ş$
"
d7
d7˙˙˙˙˙˙˙˙˙

=__inference_d1_layer_call_and_return_conditional_losses_30116^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 w
"__inference_d1_layer_call_fn_30103Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
=__inference_d3_layer_call_and_return_conditional_losses_30167^'(0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 w
"__inference_d3_layer_call_fn_30154Q'(0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
=__inference_d5_layer_call_and_return_conditional_losses_30218^670˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 w
"__inference_d5_layer_call_fn_30205Q670˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
=__inference_d7_layer_call_and_return_conditional_losses_30269]EF0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 v
"__inference_d7_layer_call_fn_30256PEF0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
 
>__inference_dr2_layer_call_and_return_conditional_losses_30131^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
  
>__inference_dr2_layer_call_and_return_conditional_losses_30143^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 x
#__inference_dr2_layer_call_fn_30121Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙x
#__inference_dr2_layer_call_fn_30126Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙ 
>__inference_dr4_layer_call_and_return_conditional_losses_30182^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
  
>__inference_dr4_layer_call_and_return_conditional_losses_30194^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 x
#__inference_dr4_layer_call_fn_30172Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙x
#__inference_dr4_layer_call_fn_30177Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙ 
>__inference_dr6_layer_call_and_return_conditional_losses_30233^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
  
>__inference_dr6_layer_call_and_return_conditional_losses_30245^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 x
#__inference_dr6_layer_call_fn_30223Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙x
#__inference_dr6_layer_call_fn_30228Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙˘
=__inference_f0_layer_call_and_return_conditional_losses_30092a7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙  
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 z
"__inference_f0_layer_call_fn_30086T7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙  
Ş "˙˙˙˙˙˙˙˙˙7
__inference_loss_fn_0_30274˘

˘ 
Ş " 7
__inference_loss_fn_1_30279˘

˘ 
Ş " 7
__inference_loss_fn_2_30284˘

˘ 
Ş " 7
__inference_loss_fn_3_30289˘

˘ 
Ş " 7
__inference_loss_fn_4_30294˘

˘ 
Ş " 7
__inference_loss_fn_5_30299˘

˘ 
Ş " 7
__inference_loss_fn_6_30304˘

˘ 
Ş " 7
__inference_loss_fn_7_30309˘

˘ 
Ş " ˝
E__inference_sequential_layer_call_and_return_conditional_losses_29855t'(67EFA˘>
7˘4
*'
f0_input˙˙˙˙˙˙˙˙˙  
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 ˝
E__inference_sequential_layer_call_and_return_conditional_losses_29891t'(67EFA˘>
7˘4
*'
f0_input˙˙˙˙˙˙˙˙˙  
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 ť
E__inference_sequential_layer_call_and_return_conditional_losses_29992r'(67EF?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙  
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 ť
E__inference_sequential_layer_call_and_return_conditional_losses_30058r'(67EF?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙  
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 
*__inference_sequential_layer_call_fn_29605g'(67EFA˘>
7˘4
*'
f0_input˙˙˙˙˙˙˙˙˙  
p 

 
Ş "˙˙˙˙˙˙˙˙˙

*__inference_sequential_layer_call_fn_29819g'(67EFA˘>
7˘4
*'
f0_input˙˙˙˙˙˙˙˙˙  
p

 
Ş "˙˙˙˙˙˙˙˙˙

*__inference_sequential_layer_call_fn_29926e'(67EF?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙  
p 

 
Ş "˙˙˙˙˙˙˙˙˙

*__inference_sequential_layer_call_fn_29947e'(67EF?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙  
p

 
Ş "˙˙˙˙˙˙˙˙˙
Ą
#__inference_signature_wrapper_30081z'(67EFE˘B
˘ 
;Ş8
6
f0_input*'
f0_input˙˙˙˙˙˙˙˙˙  "'Ş$
"
d7
d7˙˙˙˙˙˙˙˙˙

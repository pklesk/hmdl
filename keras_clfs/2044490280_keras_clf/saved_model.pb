Ö
Ý
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

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

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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¸®
v
	c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	c0/kernel
o
c0/kernel/Read/ReadVariableOpReadVariableOp	c0/kernel*&
_output_shapes
:@*
dtype0
f
c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	c0/bias
_
c0/bias/Read/ReadVariableOpReadVariableOpc0/bias*
_output_shapes
:@*
dtype0
p
	d4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
b*
shared_name	d4/kernel
i
d4/kernel/Read/ReadVariableOpReadVariableOp	d4/kernel* 
_output_shapes
:
b*
dtype0
g
d4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d4/bias
`
d4/bias/Read/ReadVariableOpReadVariableOpd4/bias*
_output_shapes	
:*
dtype0
o
	d6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_name	d6/kernel
h
d6/kernel/Read/ReadVariableOpReadVariableOp	d6/kernel*
_output_shapes
:	
*
dtype0
f
d6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	d6/bias
_
d6/bias/Read/ReadVariableOpReadVariableOpd6/bias*
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

Adam/c0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/c0/kernel/m
}
$Adam/c0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/m*&
_output_shapes
:@*
dtype0
t
Adam/c0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/c0/bias/m
m
"Adam/c0/bias/m/Read/ReadVariableOpReadVariableOpAdam/c0/bias/m*
_output_shapes
:@*
dtype0
~
Adam/d4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
b*!
shared_nameAdam/d4/kernel/m
w
$Adam/d4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d4/kernel/m* 
_output_shapes
:
b*
dtype0
u
Adam/d4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d4/bias/m
n
"Adam/d4/bias/m/Read/ReadVariableOpReadVariableOpAdam/d4/bias/m*
_output_shapes	
:*
dtype0
}
Adam/d6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*!
shared_nameAdam/d6/kernel/m
v
$Adam/d6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d6/kernel/m*
_output_shapes
:	
*
dtype0
t
Adam/d6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/d6/bias/m
m
"Adam/d6/bias/m/Read/ReadVariableOpReadVariableOpAdam/d6/bias/m*
_output_shapes
:
*
dtype0

Adam/c0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/c0/kernel/v
}
$Adam/c0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/v*&
_output_shapes
:@*
dtype0
t
Adam/c0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/c0/bias/v
m
"Adam/c0/bias/v/Read/ReadVariableOpReadVariableOpAdam/c0/bias/v*
_output_shapes
:@*
dtype0
~
Adam/d4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
b*!
shared_nameAdam/d4/kernel/v
w
$Adam/d4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d4/kernel/v* 
_output_shapes
:
b*
dtype0
u
Adam/d4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/d4/bias/v
n
"Adam/d4/bias/v/Read/ReadVariableOpReadVariableOpAdam/d4/bias/v*
_output_shapes	
:*
dtype0
}
Adam/d6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*!
shared_nameAdam/d6/kernel/v
v
$Adam/d6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d6/kernel/v*
_output_shapes
:	
*
dtype0
t
Adam/d6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/d6/bias/v
m
"Adam/d6/bias/v/Read/ReadVariableOpReadVariableOpAdam/d6/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ù<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*´<
valueª<B§< B <
õ
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¥
	variables
 trainable_variables
!regularization_losses
"	keras_api
#_random_generator
$__call__
*%&call_and_return_all_conditional_losses* 

&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
¦

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
¥
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8_random_generator
9__call__
*:&call_and_return_all_conditional_losses* 
¦

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
¼
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratemm,m-m;m<mvv,v-v;v<v*
.
0
1
,2
-3
;4
<5*
.
0
1
,2
-3
;4
<5*
,
H0
I1
J2
K3
L4
M5* 
°
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Sserving_default* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

H0
I1* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
 trainable_variables
!regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUE	d4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*

J0
K1* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
4	variables
5trainable_variables
6regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*

L0
M1* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
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
5
0
1
2
3
4
5
6*

w0
x1*
* 
* 
* 
* 
* 
* 

H0
I1* 
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
J0
K1* 
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
L0
M1* 
* 
8
	ytotal
	zcount
{	variables
|	keras_api*
J
	}total
	~count

_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

y0
z1*

{	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

}0
~1*

	variables*
|v
VARIABLE_VALUEAdam/c0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_c0_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	d4/kerneld4/bias	d6/kerneld6/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *,
f'R%
#__inference_signature_wrapper_19158
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Å	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamec0/kernel/Read/ReadVariableOpc0/bias/Read/ReadVariableOpd4/kernel/Read/ReadVariableOpd4/bias/Read/ReadVariableOpd6/kernel/Read/ReadVariableOpd6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/c0/kernel/m/Read/ReadVariableOp"Adam/c0/bias/m/Read/ReadVariableOp$Adam/d4/kernel/m/Read/ReadVariableOp"Adam/d4/bias/m/Read/ReadVariableOp$Adam/d6/kernel/m/Read/ReadVariableOp"Adam/d6/bias/m/Read/ReadVariableOp$Adam/c0/kernel/v/Read/ReadVariableOp"Adam/c0/bias/v/Read/ReadVariableOp$Adam/d4/kernel/v/Read/ReadVariableOp"Adam/d4/bias/v/Read/ReadVariableOp$Adam/d6/kernel/v/Read/ReadVariableOp"Adam/d6/bias/v/Read/ReadVariableOpConst*(
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
__inference__traced_save_19439
¤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	d4/kerneld4/bias	d6/kerneld6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/c0/kernel/mAdam/c0/bias/mAdam/d4/kernel/mAdam/d4/bias/mAdam/d6/kernel/mAdam/d6/bias/mAdam/c0/kernel/vAdam/c0/bias/vAdam/d4/kernel/vAdam/d4/bias/vAdam/d6/kernel/vAdam/d6/bias/v*'
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
!__inference__traced_restore_19530ì®
É:


__inference__traced_save_19439
file_prefix(
$savev2_c0_kernel_read_readvariableop&
"savev2_c0_bias_read_readvariableop(
$savev2_d4_kernel_read_readvariableop&
"savev2_d4_bias_read_readvariableop(
$savev2_d6_kernel_read_readvariableop&
"savev2_d6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop/
+savev2_adam_c0_kernel_m_read_readvariableop-
)savev2_adam_c0_bias_m_read_readvariableop/
+savev2_adam_d4_kernel_m_read_readvariableop-
)savev2_adam_d4_bias_m_read_readvariableop/
+savev2_adam_d6_kernel_m_read_readvariableop-
)savev2_adam_d6_bias_m_read_readvariableop/
+savev2_adam_c0_kernel_v_read_readvariableop-
)savev2_adam_c0_bias_v_read_readvariableop/
+savev2_adam_d4_kernel_v_read_readvariableop-
)savev2_adam_d4_bias_v_read_readvariableop/
+savev2_adam_d6_kernel_v_read_readvariableop-
)savev2_adam_d6_bias_v_read_readvariableop
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
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¥
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_c0_kernel_read_readvariableop"savev2_c0_bias_read_readvariableop$savev2_d4_kernel_read_readvariableop"savev2_d4_bias_read_readvariableop$savev2_d6_kernel_read_readvariableop"savev2_d6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_c0_kernel_m_read_readvariableop)savev2_adam_c0_bias_m_read_readvariableop+savev2_adam_d4_kernel_m_read_readvariableop)savev2_adam_d4_bias_m_read_readvariableop+savev2_adam_d6_kernel_m_read_readvariableop)savev2_adam_d6_bias_m_read_readvariableop+savev2_adam_c0_kernel_v_read_readvariableop)savev2_adam_c0_bias_v_read_readvariableop+savev2_adam_d4_kernel_v_read_readvariableop)savev2_adam_d4_bias_v_read_readvariableop+savev2_adam_d6_kernel_v_read_readvariableop)savev2_adam_d6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*ß
_input_shapesÍ
Ê: :@:@:
b::	
:
: : : : : : : : : :@:@:
b::	
:
:@:@:
b::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
b:!
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
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
b:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:,(
&
_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
b:!
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
>__inference_dr5_layer_call_and_return_conditional_losses_18817

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
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed2ÿÿÿÿ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ö
=__inference_c0_layer_call_and_return_conditional_losses_18698

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
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
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
\
>__inference_dr2_layer_call_and_return_conditional_losses_18710

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
ã
E__inference_sequential_layer_call_and_return_conditional_losses_18772

inputs"
c0_18699:@
c0_18701:@
d4_18734:
b
d4_18736:	
d6_18760:	

d6_18762:

identity¢c0/StatefulPartitionedCall¢d4/StatefulPartitionedCall¢d6/StatefulPartitionedCallä
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_18699c0_18701*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_18698×
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_18675Ñ
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18710É
f3/PartitionedCallPartitionedCalldr2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_18718ò
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_18734d4_18736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_18733Ò
dr5/PartitionedCallPartitionedCall#d4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_18744ò
d6/StatefulPartitionedCallStatefulPartitionedCalldr5/PartitionedCall:output:0d6_18760d6_18762*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_18759`
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
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª"

E__inference_sequential_layer_call_and_return_conditional_losses_18917

inputs"
c0_18891:@
c0_18893:@
d4_18899:
b
d4_18901:	
d6_18905:	

d6_18907:

identity¢c0/StatefulPartitionedCall¢d4/StatefulPartitionedCall¢d6/StatefulPartitionedCall¢dr2/StatefulPartitionedCall¢dr5/StatefulPartitionedCallä
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_18891c0_18893*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_18698×
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_18675á
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18856Ñ
f3/PartitionedCallPartitionedCall$dr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_18718ò
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_18899d4_18901*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_18733
dr5/StatefulPartitionedCallStatefulPartitionedCall#d4/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_18817ú
d6/StatefulPartitionedCallStatefulPartitionedCall$dr5/StatefulPartitionedCall:output:0d6_18905d6_18907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_18759`
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
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ù
NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â$
«
 __inference__wrapped_model_18666
c0_inputF
,sequential_c0_conv2d_readvariableop_resource:@;
-sequential_c0_biasadd_readvariableop_resource:@@
,sequential_d4_matmul_readvariableop_resource:
b<
-sequential_d4_biasadd_readvariableop_resource:	?
,sequential_d6_matmul_readvariableop_resource:	
;
-sequential_d6_biasadd_readvariableop_resource:

identity¢$sequential/c0/BiasAdd/ReadVariableOp¢#sequential/c0/Conv2D/ReadVariableOp¢$sequential/d4/BiasAdd/ReadVariableOp¢#sequential/d4/MatMul/ReadVariableOp¢$sequential/d6/BiasAdd/ReadVariableOp¢#sequential/d6/MatMul/ReadVariableOp
#sequential/c0/Conv2D/ReadVariableOpReadVariableOp,sequential_c0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0·
sequential/c0/Conv2DConv2Dc0_input+sequential/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

$sequential/c0/BiasAdd/ReadVariableOpReadVariableOp-sequential_c0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
sequential/c0/BiasAddBiasAddsequential/c0/Conv2D:output:0,sequential/c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
sequential/c0/ReluRelusequential/c0/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
sequential/m1/MaxPoolMaxPool sequential/c0/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
}
sequential/dr2/IdentityIdentitysequential/m1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
sequential/f3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  
sequential/f3/ReshapeReshape sequential/dr2/Identity:output:0sequential/f3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
#sequential/d4/MatMul/ReadVariableOpReadVariableOp,sequential_d4_matmul_readvariableop_resource* 
_output_shapes
:
b*
dtype0
sequential/d4/MatMulMatMulsequential/f3/Reshape:output:0+sequential/d4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential/d4/BiasAdd/ReadVariableOpReadVariableOp-sequential_d4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
sequential/d4/BiasAddBiasAddsequential/d4/MatMul:product:0,sequential/d4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
sequential/d4/ReluRelusequential/d4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
sequential/dr5/IdentityIdentity sequential/d4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#sequential/d6/MatMul/ReadVariableOpReadVariableOp,sequential_d6_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
sequential/d6/MatMulMatMul sequential/dr5/Identity:output:0+sequential/d6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$sequential/d6/BiasAdd/ReadVariableOpReadVariableOp-sequential_d6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0 
sequential/d6/BiasAddBiasAddsequential/d6/MatMul:product:0,sequential/d6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
sequential/d6/SoftmaxSoftmaxsequential/d6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
IdentityIdentitysequential/d6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/d4/BiasAdd/ReadVariableOp$^sequential/d4/MatMul/ReadVariableOp%^sequential/d6/BiasAdd/ReadVariableOp$^sequential/d6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2L
$sequential/c0/BiasAdd/ReadVariableOp$sequential/c0/BiasAdd/ReadVariableOp2J
#sequential/c0/Conv2D/ReadVariableOp#sequential/c0/Conv2D/ReadVariableOp2L
$sequential/d4/BiasAdd/ReadVariableOp$sequential/d4/BiasAdd/ReadVariableOp2J
#sequential/d4/MatMul/ReadVariableOp#sequential/d4/MatMul/ReadVariableOp2L
$sequential/d6/BiasAdd/ReadVariableOp$sequential/d6/BiasAdd/ReadVariableOp2J
#sequential/d6/MatMul/ReadVariableOp#sequential/d6/MatMul/ReadVariableOp:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
c0_input
È
+
__inference_loss_fn_4_19330
identity`
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$d6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
	

*__inference_sequential_layer_call_fn_19036

inputs!
unknown:@
	unknown_0:@
	unknown_1:
b
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
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
ï
=__inference_d6_layer_call_and_return_conditional_losses_19305

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
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú"
Ê
E__inference_sequential_layer_call_and_return_conditional_losses_19089

inputs;
!c0_conv2d_readvariableop_resource:@0
"c0_biasadd_readvariableop_resource:@5
!d4_matmul_readvariableop_resource:
b1
"d4_biasadd_readvariableop_resource:	4
!d6_matmul_readvariableop_resource:	
0
"d6_biasadd_readvariableop_resource:

identity¢c0/BiasAdd/ReadVariableOp¢c0/Conv2D/ReadVariableOp¢d4/BiasAdd/ReadVariableOp¢d4/MatMul/ReadVariableOp¢d6/BiasAdd/ReadVariableOp¢d6/MatMul/ReadVariableOp
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
g
dr2/IdentityIdentitym1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
f3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  r

f3/ReshapeReshapedr2/Identity:output:0f3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb|
d4/MatMul/ReadVariableOpReadVariableOp!d4_matmul_readvariableop_resource* 
_output_shapes
:
b*
dtype0}
	d4/MatMulMatMulf3/Reshape:output:0 d4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
d4/BiasAdd/ReadVariableOpReadVariableOp"d4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d4/BiasAddBiasAddd4/MatMul:product:0!d4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
d4/ReluRelud4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dr5/IdentityIdentityd4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
d6/MatMul/ReadVariableOpReadVariableOp!d6_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0~
	d6/MatMulMatMuldr5/Identity:output:0 d6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
d6/BiasAdd/ReadVariableOpReadVariableOp"d6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d6/BiasAddBiasAddd6/MatMul:product:0!d6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\

d6/SoftmaxSoftmaxd6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^d4/BiasAdd/ReadVariableOp^d4/MatMul/ReadVariableOp^d6/BiasAdd/ReadVariableOp^d6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
d4/BiasAdd/ReadVariableOpd4/BiasAdd/ReadVariableOp24
d4/MatMul/ReadVariableOpd4/MatMul/ReadVariableOp26
d6/BiasAdd/ReadVariableOpd6/BiasAdd/ReadVariableOp24
d6/MatMul/ReadVariableOpd6/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Y
=__inference_m1_layer_call_and_return_conditional_losses_18675

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
?
#__inference_dr2_layer_call_fn_19197

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18710h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½

]
>__inference_dr2_layer_call_and_return_conditional_losses_19219

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:¥
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed2ÿÿÿÿ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	

*__inference_sequential_layer_call_fn_19053

inputs!
unknown:@
	unknown_0:@
	unknown_1:
b
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
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

"__inference_c0_layer_call_fn_19169

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_18698w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
ñ
=__inference_d4_layer_call_and_return_conditional_losses_19254

inputs2
matmul_readvariableop_resource:
b.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
b*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
â
ñ
=__inference_d4_layer_call_and_return_conditional_losses_18733

inputs2
matmul_readvariableop_resource:
b.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
b*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
½

]
>__inference_dr2_layer_call_and_return_conditional_losses_18856

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:¥
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed2ÿÿÿÿ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
\
>__inference_dr5_layer_call_and_return_conditional_losses_19269

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
Ê
E__inference_sequential_layer_call_and_return_conditional_losses_19139

inputs;
!c0_conv2d_readvariableop_resource:@0
"c0_biasadd_readvariableop_resource:@5
!d4_matmul_readvariableop_resource:
b1
"d4_biasadd_readvariableop_resource:	4
!d6_matmul_readvariableop_resource:	
0
"d6_biasadd_readvariableop_resource:

identity¢c0/BiasAdd/ReadVariableOp¢c0/Conv2D/ReadVariableOp¢d4/BiasAdd/ReadVariableOp¢d4/MatMul/ReadVariableOp¢d6/BiasAdd/ReadVariableOp¢d6/MatMul/ReadVariableOp
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
V
dr2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?
dr2/dropout/MulMulm1/MaxPool:output:0dr2/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dr2/dropout/ShapeShapem1/MaxPool:output:0*
T0*
_output_shapes
:­
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed2ÿÿÿÿ_
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >º
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dr2/dropout/CastCastdr2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
dr2/dropout/Mul_1Muldr2/dropout/Mul:z:0dr2/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
f3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  r

f3/ReshapeReshapedr2/dropout/Mul_1:z:0f3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb|
d4/MatMul/ReadVariableOpReadVariableOp!d4_matmul_readvariableop_resource* 
_output_shapes
:
b*
dtype0}
	d4/MatMulMatMulf3/Reshape:output:0 d4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
d4/BiasAdd/ReadVariableOpReadVariableOp"d4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d4/BiasAddBiasAddd4/MatMul:product:0!d4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
d4/ReluRelud4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
dr5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I?|
dr5/dropout/MulMuld4/Relu:activations:0dr5/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
dr5/dropout/ShapeShaped4/Relu:activations:0*
T0*
_output_shapes
:¢
(dr5/dropout/random_uniform/RandomUniformRandomUniformdr5/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed2_
dr5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >³
dr5/dropout/GreaterEqualGreaterEqual1dr5/dropout/random_uniform/RandomUniform:output:0#dr5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dr5/dropout/CastCastdr5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
dr5/dropout/Mul_1Muldr5/dropout/Mul:z:0dr5/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
d6/MatMul/ReadVariableOpReadVariableOp!d6_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0~
	d6/MatMulMatMuldr5/dropout/Mul_1:z:0 d6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
d6/BiasAdd/ReadVariableOpReadVariableOp"d6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d6/BiasAddBiasAddd6/MatMul:product:0!d6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\

d6/SoftmaxSoftmaxd6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^d4/BiasAdd/ReadVariableOp^d4/MatMul/ReadVariableOp^d6/BiasAdd/ReadVariableOp^d6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
d4/BiasAdd/ReadVariableOpd4/BiasAdd/ReadVariableOp24
d4/MatMul/ReadVariableOpd4/MatMul/ReadVariableOp26
d6/BiasAdd/ReadVariableOpd6/BiasAdd/ReadVariableOp24
d6/MatMul/ReadVariableOpd6/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ö
=__inference_c0_layer_call_and_return_conditional_losses_19182

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
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
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
Y
=__inference_f3_layer_call_and_return_conditional_losses_18718

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿbY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


]
>__inference_dr5_layer_call_and_return_conditional_losses_19281

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
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed2ÿÿÿÿ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
>
"__inference_m1_layer_call_fn_19187

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_18675
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
ï
=__inference_d6_layer_call_and_return_conditional_losses_18759

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
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

#__inference_signature_wrapper_19158
c0_input!
unknown:@
	unknown_0:@
	unknown_1:
b
	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *)
f$R"
 __inference__wrapped_model_18666o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
c0_input

Y
=__inference_m1_layer_call_and_return_conditional_losses_19192

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
+
__inference_loss_fn_2_19320
identity`
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$d4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ò
å
E__inference_sequential_layer_call_and_return_conditional_losses_18978
c0_input"
c0_18952:@
c0_18954:@
d4_18960:
b
d4_18962:	
d6_18966:	

d6_18968:

identity¢c0/StatefulPartitionedCall¢d4/StatefulPartitionedCall¢d6/StatefulPartitionedCallæ
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_18952c0_18954*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_18698×
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_18675Ñ
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18710É
f3/PartitionedCallPartitionedCalldr2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_18718ò
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_18960d4_18962*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_18733Ò
dr5/PartitionedCallPartitionedCall#d4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_18744ò
d6/StatefulPartitionedCallStatefulPartitionedCalldr5/PartitionedCall:output:0d6_18966d6_18968*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_18759`
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
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
c0_input
Õ
\
>__inference_dr5_layer_call_and_return_conditional_losses_18744

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
Y
=__inference_f3_layer_call_and_return_conditional_losses_19230

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿbY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

?
#__inference_dr5_layer_call_fn_19259

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_18744a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°"
¡
E__inference_sequential_layer_call_and_return_conditional_losses_19007
c0_input"
c0_18981:@
c0_18983:@
d4_18989:
b
d4_18991:	
d6_18995:	

d6_18997:

identity¢c0/StatefulPartitionedCall¢d4/StatefulPartitionedCall¢d6/StatefulPartitionedCall¢dr2/StatefulPartitionedCall¢dr5/StatefulPartitionedCallæ
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_18981c0_18983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_18698×
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_18675á
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18856Ñ
f3/PartitionedCallPartitionedCall$dr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_18718ò
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_18989d4_18991*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_18733
dr5/StatefulPartitionedCallStatefulPartitionedCall#d4/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_18817ú
d6/StatefulPartitionedCallStatefulPartitionedCall$dr5/StatefulPartitionedCall:output:0d6_18995d6_18997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_18759`
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
d4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
d6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ù
NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
c0_input

\
#__inference_dr2_layer_call_fn_19202

inputs
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_18856w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	

*__inference_sequential_layer_call_fn_18787
c0_input!
unknown:@
	unknown_0:@
	unknown_1:
b
	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
c0_input
ñ
\
>__inference_dr2_layer_call_and_return_conditional_losses_19207

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î
\
#__inference_dr5_layer_call_fn_19264

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_18817p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

"__inference_d4_layer_call_fn_19241

inputs
unknown:
b
	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_18733p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
¾

"__inference_d6_layer_call_fn_19292

inputs
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_18759o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
k
©
!__inference__traced_restore_19530
file_prefix4
assignvariableop_c0_kernel:@(
assignvariableop_1_c0_bias:@0
assignvariableop_2_d4_kernel:
b)
assignvariableop_3_d4_bias:	/
assignvariableop_4_d6_kernel:	
(
assignvariableop_5_d6_bias:
&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: >
$assignvariableop_15_adam_c0_kernel_m:@0
"assignvariableop_16_adam_c0_bias_m:@8
$assignvariableop_17_adam_d4_kernel_m:
b1
"assignvariableop_18_adam_d4_bias_m:	7
$assignvariableop_19_adam_d6_kernel_m:	
0
"assignvariableop_20_adam_d6_bias_m:
>
$assignvariableop_21_adam_c0_kernel_v:@0
"assignvariableop_22_adam_c0_bias_v:@8
$assignvariableop_23_adam_d4_kernel_v:
b1
"assignvariableop_24_adam_d4_bias_v:	7
$assignvariableop_25_adam_d6_kernel_v:	
0
"assignvariableop_26_adam_d6_bias_v:

identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ø
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¨
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
AssignVariableOpAssignVariableOpassignvariableop_c0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_c0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_d4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_d4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_d6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_d6_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_c0_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_c0_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_d4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_d4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_d6_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_d6_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_c0_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_c0_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_d4_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_d4_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_d6_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_d6_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¡
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
Ä
+
__inference_loss_fn_3_19325
identity^
d4/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"d4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
¨
>
"__inference_f3_layer_call_fn_19224

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
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_18718a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	

*__inference_sequential_layer_call_fn_18949
c0_input!
unknown:@
	unknown_0:@
	unknown_1:
b
	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
c0_input
Ä
+
__inference_loss_fn_5_19335
identity^
d6/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"d6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Ä
+
__inference_loss_fn_1_19315
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
È
+
__inference_loss_fn_0_19310
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
_input_shapes "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¯
serving_default
E
c0_input9
serving_default_c0_input:0ÿÿÿÿÿÿÿÿÿ6
d60
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:ì

layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
	variables
 trainable_variables
!regularization_losses
"	keras_api
#_random_generator
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
»

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8_random_generator
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
»

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
Ë
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratemm,m-m;m<mvv,v-v;v<v"
	optimizer
J
0
1
,2
-3
;4
<5"
trackable_list_wrapper
J
0
1
,2
-3
;4
<5"
trackable_list_wrapper
J
H0
I1
J2
K3
L4
M5"
trackable_list_wrapper
Ê
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_sequential_layer_call_fn_18787
*__inference_sequential_layer_call_fn_19036
*__inference_sequential_layer_call_fn_19053
*__inference_sequential_layer_call_fn_18949À
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
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_19089
E__inference_sequential_layer_call_and_return_conditional_losses_19139
E__inference_sequential_layer_call_and_return_conditional_losses_18978
E__inference_sequential_layer_call_and_return_conditional_losses_19007À
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
 __inference__wrapped_model_18666c0_input"
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
Sserving_default"
signature_map
#:!@2	c0/kernel
:@2c0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_c0_layer_call_fn_19169¢
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
=__inference_c0_layer_call_and_return_conditional_losses_19182¢
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
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_m1_layer_call_fn_19187¢
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
=__inference_m1_layer_call_and_return_conditional_losses_19192¢
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
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
 trainable_variables
!regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr2_layer_call_fn_19197
#__inference_dr2_layer_call_fn_19202´
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
>__inference_dr2_layer_call_and_return_conditional_losses_19207
>__inference_dr2_layer_call_and_return_conditional_losses_19219´
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_f3_layer_call_fn_19224¢
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
=__inference_f3_layer_call_and_return_conditional_losses_19230¢
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
:
b2	d4/kernel
:2d4/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_d4_layer_call_fn_19241¢
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
=__inference_d4_layer_call_and_return_conditional_losses_19254¢
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
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
4	variables
5trainable_variables
6regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
#__inference_dr5_layer_call_fn_19259
#__inference_dr5_layer_call_fn_19264´
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
>__inference_dr5_layer_call_and_return_conditional_losses_19269
>__inference_dr5_layer_call_and_return_conditional_losses_19281´
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
:	
2	d6/kernel
:
2d6/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Ì2É
"__inference_d6_layer_call_fn_19292¢
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
=__inference_d6_layer_call_and_return_conditional_losses_19305¢
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
__inference_loss_fn_0_19310
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
__inference_loss_fn_1_19315
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
__inference_loss_fn_2_19320
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
__inference_loss_fn_3_19325
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
__inference_loss_fn_4_19330
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
__inference_loss_fn_5_19335
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
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
#__inference_signature_wrapper_19158c0_input"
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
.
H0
I1"
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
J0
K1"
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
L0
M1"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ytotal
	zcount
{	variables
|	keras_api"
_tf_keras_metric
`
	}total
	~count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
y0
z1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
.
	variables"
_generic_user_object
(:&@2Adam/c0/kernel/m
:@2Adam/c0/bias/m
": 
b2Adam/d4/kernel/m
:2Adam/d4/bias/m
!:	
2Adam/d6/kernel/m
:
2Adam/d6/bias/m
(:&@2Adam/c0/kernel/v
:@2Adam/c0/bias/v
": 
b2Adam/d4/kernel/v
:2Adam/d4/bias/v
!:	
2Adam/d6/kernel/v
:
2Adam/d6/bias/v
 __inference__wrapped_model_18666l,-;<9¢6
/¢,
*'
c0_inputÿÿÿÿÿÿÿÿÿ
ª "'ª$
"
d6
d6ÿÿÿÿÿÿÿÿÿ
­
=__inference_c0_layer_call_and_return_conditional_losses_19182l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
"__inference_c0_layer_call_fn_19169_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@
=__inference_d4_layer_call_and_return_conditional_losses_19254^,-0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿb
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 w
"__inference_d4_layer_call_fn_19241Q,-0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿb
ª "ÿÿÿÿÿÿÿÿÿ
=__inference_d6_layer_call_and_return_conditional_losses_19305];<0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 v
"__inference_d6_layer_call_fn_19292P;<0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
®
>__inference_dr2_layer_call_and_return_conditional_losses_19207l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ®
>__inference_dr2_layer_call_and_return_conditional_losses_19219l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
#__inference_dr2_layer_call_fn_19197_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
#__inference_dr2_layer_call_fn_19202_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@ 
>__inference_dr5_layer_call_and_return_conditional_losses_19269^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
  
>__inference_dr5_layer_call_and_return_conditional_losses_19281^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 x
#__inference_dr5_layer_call_fn_19259Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿx
#__inference_dr5_layer_call_fn_19264Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¢
=__inference_f3_layer_call_and_return_conditional_losses_19230a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿb
 z
"__inference_f3_layer_call_fn_19224T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿb7
__inference_loss_fn_0_19310¢

¢ 
ª " 7
__inference_loss_fn_1_19315¢

¢ 
ª " 7
__inference_loss_fn_2_19320¢

¢ 
ª " 7
__inference_loss_fn_3_19325¢

¢ 
ª " 7
__inference_loss_fn_4_19330¢

¢ 
ª " 7
__inference_loss_fn_5_19335¢

¢ 
ª " à
=__inference_m1_layer_call_and_return_conditional_losses_19192R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
"__inference_m1_layer_call_fn_19187R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
E__inference_sequential_layer_call_and_return_conditional_losses_18978r,-;<A¢>
7¢4
*'
c0_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 »
E__inference_sequential_layer_call_and_return_conditional_losses_19007r,-;<A¢>
7¢4
*'
c0_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¹
E__inference_sequential_layer_call_and_return_conditional_losses_19089p,-;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¹
E__inference_sequential_layer_call_and_return_conditional_losses_19139p,-;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
*__inference_sequential_layer_call_fn_18787e,-;<A¢>
7¢4
*'
c0_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_18949e,-;<A¢>
7¢4
*'
c0_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_19036c,-;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_19053c,-;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

#__inference_signature_wrapper_19158x,-;<E¢B
¢ 
;ª8
6
c0_input*'
c0_inputÿÿÿÿÿÿÿÿÿ"'ª$
"
d6
d6ÿÿÿÿÿÿÿÿÿ

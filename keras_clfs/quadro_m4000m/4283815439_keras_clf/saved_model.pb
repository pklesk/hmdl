й╓
М▌
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ы
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
В
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
delete_old_dirsbool(И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68║о
v
	c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*
shared_name	c0/kernel
o
c0/kernel/Read/ReadVariableOpReadVariableOp	c0/kernel*&
_output_shapes
:		*
dtype0
f
c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	c0/bias
_
c0/bias/Read/ReadVariableOpReadVariableOpc0/bias*
_output_shapes
:*
dtype0
o
	d4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А @*
shared_name	d4/kernel
h
d4/kernel/Read/ReadVariableOpReadVariableOp	d4/kernel*
_output_shapes
:	А @*
dtype0
f
d4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	d4/bias
_
d4/bias/Read/ReadVariableOpReadVariableOpd4/bias*
_output_shapes
:@*
dtype0
n
	d6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*
shared_name	d6/kernel
g
d6/kernel/Read/ReadVariableOpReadVariableOp	d6/kernel*
_output_shapes

:@(*
dtype0
f
d6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name	d6/bias
_
d6/bias/Read/ReadVariableOpReadVariableOpd6/bias*
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
Д
Adam/c0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameAdam/c0/kernel/m
}
$Adam/c0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/m*&
_output_shapes
:		*
dtype0
t
Adam/c0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c0/bias/m
m
"Adam/c0/bias/m/Read/ReadVariableOpReadVariableOpAdam/c0/bias/m*
_output_shapes
:*
dtype0
}
Adam/d4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А @*!
shared_nameAdam/d4/kernel/m
v
$Adam/d4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d4/kernel/m*
_output_shapes
:	А @*
dtype0
t
Adam/d4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/d4/bias/m
m
"Adam/d4/bias/m/Read/ReadVariableOpReadVariableOpAdam/d4/bias/m*
_output_shapes
:@*
dtype0
|
Adam/d6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/d6/kernel/m
u
$Adam/d6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d6/kernel/m*
_output_shapes

:@(*
dtype0
t
Adam/d6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/d6/bias/m
m
"Adam/d6/bias/m/Read/ReadVariableOpReadVariableOpAdam/d6/bias/m*
_output_shapes
:(*
dtype0
Д
Adam/c0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameAdam/c0/kernel/v
}
$Adam/c0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/v*&
_output_shapes
:		*
dtype0
t
Adam/c0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c0/bias/v
m
"Adam/c0/bias/v/Read/ReadVariableOpReadVariableOpAdam/c0/bias/v*
_output_shapes
:*
dtype0
}
Adam/d4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А @*!
shared_nameAdam/d4/kernel/v
v
$Adam/d4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d4/kernel/v*
_output_shapes
:	А @*
dtype0
t
Adam/d4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/d4/bias/v
m
"Adam/d4/bias/v/Read/ReadVariableOpReadVariableOpAdam/d4/bias/v*
_output_shapes
:@*
dtype0
|
Adam/d6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/d6/kernel/v
u
$Adam/d6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d6/kernel/v*
_output_shapes

:@(*
dtype0
t
Adam/d6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/d6/bias/v
m
"Adam/d6/bias/v/Read/ReadVariableOpReadVariableOpAdam/d6/bias/v*
_output_shapes
:(*
dtype0

NoOpNoOp
∙<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┤<
valueк<Bз< Bа<
ї
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
ж

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
е
	variables
 trainable_variables
!regularization_losses
"	keras_api
#_random_generator
$__call__
*%&call_and_return_all_conditional_losses* 
О
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
ж

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
е
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8_random_generator
9__call__
*:&call_and_return_all_conditional_losses* 
ж

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
╝
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratemВmГ,mД-mЕ;mЖ<mЗvИvЙ,vК-vЛ;vМ<vН*
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
░
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
У
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
С
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
С
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
С
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
У
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
С
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
У
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
А	variables
Б	keras_api*
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
А	variables*
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
Л
serving_default_c0_inputPlaceholder*/
_output_shapes
:         @@*
dtype0*$
shape:         @@
Б
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	d4/kerneld4/bias	d6/kerneld6/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *,
f'R%
#__inference_signature_wrapper_69502
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┼	
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
GPU(2*0J 8В *'
f"R 
__inference__traced_save_69783
д
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
GPU(2*0J 8В **
f%R#
!__inference__traced_restore_69874Ап
█
ю
=__inference_d6_layer_call_and_return_conditional_losses_69649

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         (`
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
:         (w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ё"
╔
G__inference_sequential_9_layer_call_and_return_conditional_losses_69433

inputs;
!c0_conv2d_readvariableop_resource:		0
"c0_biasadd_readvariableop_resource:4
!d4_matmul_readvariableop_resource:	А @0
"d4_biasadd_readvariableop_resource:@3
!d6_matmul_readvariableop_resource:@(0
"d6_biasadd_readvariableop_resource:(
identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвd4/BiasAdd/ReadVariableOpвd4/MatMul/ReadVariableOpвd6/BiasAdd/ReadVariableOpвd6/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:         @@Щ

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
g
dr2/IdentityIdentitym1/MaxPool:output:0*
T0*/
_output_shapes
:         Y
f3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f3/ReshapeReshapedr2/Identity:output:0f3/Const:output:0*
T0*(
_output_shapes
:         А {
d4/MatMul/ReadVariableOpReadVariableOp!d4_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0|
	d4/MatMulMatMulf3/Reshape:output:0 d4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
d4/BiasAdd/ReadVariableOpReadVariableOp"d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d4/BiasAddBiasAddd4/MatMul:product:0!d4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @V
d4/ReluRelud4/BiasAdd:output:0*
T0*'
_output_shapes
:         @a
dr5/IdentityIdentityd4/Relu:activations:0*
T0*'
_output_shapes
:         @z
d6/MatMul/ReadVariableOpReadVariableOp!d6_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0~
	d6/MatMulMatMuldr5/Identity:output:0 d6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (x
d6/BiasAdd/ReadVariableOpReadVariableOp"d6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d6/BiasAddBiasAddd6/MatMul:product:0!d6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (\

d6/SoftmaxSoftmaxd6/BiasAdd:output:0*
T0*'
_output_shapes
:         (`
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
:         (ы
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^d4/BiasAdd/ReadVariableOp^d4/MatMul/ReadVariableOp^d6/BiasAdd/ReadVariableOp^d6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
d4/BiasAdd/ReadVariableOpd4/BiasAdd/ReadVariableOp24
d4/MatMul/ReadVariableOpd4/MatMul/ReadVariableOp26
d6/BiasAdd/ReadVariableOpd6/BiasAdd/ReadVariableOp24
d6/MatMul/ReadVariableOpd6/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╖:
Э

__inference__traced_save_69783
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

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ї
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHе
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B С

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_c0_kernel_read_readvariableop"savev2_c0_bias_read_readvariableop$savev2_d4_kernel_read_readvariableop"savev2_d4_bias_read_readvariableop$savev2_d6_kernel_read_readvariableop"savev2_d6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_c0_kernel_m_read_readvariableop)savev2_adam_c0_bias_m_read_readvariableop+savev2_adam_d4_kernel_m_read_readvariableop)savev2_adam_d4_bias_m_read_readvariableop+savev2_adam_d6_kernel_m_read_readvariableop)savev2_adam_d6_bias_m_read_readvariableop+savev2_adam_c0_kernel_v_read_readvariableop)savev2_adam_c0_bias_v_read_readvariableop+savev2_adam_d4_kernel_v_read_readvariableop)savev2_adam_d4_bias_v_read_readvariableop+savev2_adam_d6_kernel_v_read_readvariableop)savev2_adam_d6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*╓
_input_shapes─
┴: :		::	А @:@:@(:(: : : : : : : : : :		::	А @:@:@(:(:		::	А @:@:@(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		: 

_output_shapes
::%!

_output_shapes
:	А @: 

_output_shapes
:@:$ 

_output_shapes

:@(: 

_output_shapes
:(:
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
:		: 

_output_shapes
::%!

_output_shapes
:	А @: 

_output_shapes
:@:$ 

_output_shapes

:@(: 

_output_shapes
:(:,(
&
_output_shapes
:		: 

_output_shapes
::%!

_output_shapes
:	А @: 

_output_shapes
:@:$ 

_output_shapes

:@(: 

_output_shapes
:(:

_output_shapes
: 
з"
Ю
G__inference_sequential_9_layer_call_and_return_conditional_losses_69261

inputs"
c0_69235:		
c0_69237:
d4_69243:	А @
d4_69245:@
d6_69249:@(
d6_69251:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_69235c0_69237*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_69042╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_69019с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_69200╤
f3/PartitionedCallPartitionedCall$dr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_69062ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_69243d4_69245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_69077 
dr5/StatefulPartitionedCallStatefulPartitionedCall#d4/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_69161·
d6/StatefulPartitionedCallStatefulPartitionedCall$dr5/StatefulPartitionedCall:output:0d6_69249d6_69251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_69103`
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
:         (┘
NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
┌
я
=__inference_d4_layer_call_and_return_conditional_losses_69077

inputs1
matmul_readvariableop_resource:	А @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @`
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
 *    a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
ц
З
#__inference_signature_wrapper_69502
c0_input!
unknown:		
	unknown_0:
	unknown_1:	А @
	unknown_2:@
	unknown_3:@(
	unknown_4:(
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *)
f$R"
 __inference__wrapped_model_69010o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
ъ
\
#__inference_dr5_layer_call_fn_69608

inputs
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_69161o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┐
Y
=__inference_f3_layer_call_and_return_conditional_losses_69574

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
г
>
"__inference_m1_layer_call_fn_69531

inputs
identity╥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_69019Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╕
?
#__inference_dr2_layer_call_fn_69541

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_69054h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
я
ф
G__inference_sequential_9_layer_call_and_return_conditional_losses_69322
c0_input"
c0_69296:		
c0_69298:
d4_69304:	А @
d4_69306:@
d6_69310:@(
d6_69312:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_69296c0_69298*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_69042╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_69019╤
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_69054╔
f3/PartitionedCallPartitionedCalldr2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_69062ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_69304d4_69306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_69077╤
dr5/PartitionedCallPartitionedCall#d4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_69088Є
d6/StatefulPartitionedCallStatefulPartitionedCalldr5/PartitionedCall:output:0d6_69310d6_69312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_69103`
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
:         (Э
NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
┐
Y
=__inference_f3_layer_call_and_return_conditional_losses_69062

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
у
Ч
"__inference_c0_layer_call_fn_69513

inputs!
unknown:		
	unknown_0:
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_69042w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Ц	
Р
,__inference_sequential_9_layer_call_fn_69293
c0_input!
unknown:		
	unknown_0:
	unknown_1:	А @
	unknown_2:@
	unknown_3:@(
	unknown_4:(
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_69261o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
К
\
#__inference_dr2_layer_call_fn_69546

inputs
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_69200w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
█
ю
=__inference_d6_layer_call_and_return_conditional_losses_69103

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         (`
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
:         (w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Р	
О
,__inference_sequential_9_layer_call_fn_69397

inputs!
unknown:		
	unknown_0:
	unknown_1:	А @
	unknown_2:@
	unknown_3:@(
	unknown_4:(
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_69261o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_69526

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
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
:         @@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @@`
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
:         @@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╤
\
>__inference_dr5_layer_call_and_return_conditional_losses_69613

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
¤	
]
>__inference_dr5_layer_call_and_return_conditional_losses_69161

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Э
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_69054

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Лk
а
!__inference__traced_restore_69874
file_prefix4
assignvariableop_c0_kernel:		(
assignvariableop_1_c0_bias:/
assignvariableop_2_d4_kernel:	А @(
assignvariableop_3_d4_bias:@.
assignvariableop_4_d6_kernel:@((
assignvariableop_5_d6_bias:(&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: >
$assignvariableop_15_adam_c0_kernel_m:		0
"assignvariableop_16_adam_c0_bias_m:7
$assignvariableop_17_adam_d4_kernel_m:	А @0
"assignvariableop_18_adam_d4_bias_m:@6
$assignvariableop_19_adam_d6_kernel_m:@(0
"assignvariableop_20_adam_d6_bias_m:(>
$assignvariableop_21_adam_c0_kernel_v:		0
"assignvariableop_22_adam_c0_bias_v:7
$assignvariableop_23_adam_d4_kernel_v:	А @0
"assignvariableop_24_adam_d4_bias_v:@6
$assignvariableop_25_adam_d6_kernel_v:@(0
"assignvariableop_26_adam_d6_bias_v:(
identity_28ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9°
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B л
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Д
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOpAssignVariableOpassignvariableop_c0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_1AssignVariableOpassignvariableop_1_c0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_2AssignVariableOpassignvariableop_2_d4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_3AssignVariableOpassignvariableop_3_d4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_4AssignVariableOpassignvariableop_4_d6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_5AssignVariableOpassignvariableop_5_d6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_c0_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_c0_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_d4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_d4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_d6_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_d6_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_c0_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_c0_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_d4_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_d4_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_d6_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_d6_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 б
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: О
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
─
+
__inference_loss_fn_1_69659
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
─
+
__inference_loss_fn_3_69669
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
─
+
__inference_loss_fn_5_69679
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
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_69536

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_69551

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┌
я
=__inference_d4_layer_call_and_return_conditional_losses_69598

inputs1
matmul_readvariableop_resource:	А @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @`
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
 *    a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
╜

]
>__inference_dr2_layer_call_and_return_conditional_losses_69563

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_0_69654
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
╤
\
>__inference_dr5_layer_call_and_return_conditional_losses_69088

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
щ
т
G__inference_sequential_9_layer_call_and_return_conditional_losses_69116

inputs"
c0_69043:		
c0_69045:
d4_69078:	А @
d4_69080:@
d6_69104:@(
d6_69106:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_69043c0_69045*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_69042╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_69019╤
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_69054╔
f3/PartitionedCallPartitionedCalldr2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_69062ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_69078d4_69080*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_69077╤
dr5/PartitionedCallPartitionedCall#d4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_69088Є
d6/StatefulPartitionedCallStatefulPartitionedCalldr5/PartitionedCall:output:0d6_69104d6_69106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_69103`
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
:         (Э
NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╗
П
"__inference_d6_layer_call_fn_69636

inputs
unknown:@(
	unknown_0:(
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_69103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_69019

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
¤	
]
>__inference_dr5_layer_call_and_return_conditional_losses_69625

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Э
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_69042

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
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
:         @@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @@`
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
:         @@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╛
Р
"__inference_d4_layer_call_fn_69585

inputs
unknown:	А @
	unknown_0:@
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_69077o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
Ц	
Р
,__inference_sequential_9_layer_call_fn_69131
c0_input!
unknown:		
	unknown_0:
	unknown_1:	А @
	unknown_2:@
	unknown_3:@(
	unknown_4:(
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_69116o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
№0
╔
G__inference_sequential_9_layer_call_and_return_conditional_losses_69483

inputs;
!c0_conv2d_readvariableop_resource:		0
"c0_biasadd_readvariableop_resource:4
!d4_matmul_readvariableop_resource:	А @0
"d4_biasadd_readvariableop_resource:@3
!d6_matmul_readvariableop_resource:@(0
"d6_biasadd_readvariableop_resource:(
identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвd4/BiasAdd/ReadVariableOpвd4/MatMul/ReadVariableOpвd6/BiasAdd/ReadVariableOpвd6/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:         @@Щ

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
V
dr2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Б
dr2/dropout/MulMulm1/MaxPool:output:0dr2/dropout/Const:output:0*
T0*/
_output_shapes
:         T
dr2/dropout/ShapeShapem1/MaxPool:output:0*
T0*
_output_shapes
:н
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed2    _
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>║
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         
dr2/dropout/CastCastdr2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         }
dr2/dropout/Mul_1Muldr2/dropout/Mul:z:0dr2/dropout/Cast:y:0*
T0*/
_output_shapes
:         Y
f3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f3/ReshapeReshapedr2/dropout/Mul_1:z:0f3/Const:output:0*
T0*(
_output_shapes
:         А {
d4/MatMul/ReadVariableOpReadVariableOp!d4_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0|
	d4/MatMulMatMulf3/Reshape:output:0 d4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
d4/BiasAdd/ReadVariableOpReadVariableOp"d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d4/BiasAddBiasAddd4/MatMul:product:0!d4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @V
d4/ReluRelud4/BiasAdd:output:0*
T0*'
_output_shapes
:         @V
dr5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?{
dr5/dropout/MulMuld4/Relu:activations:0dr5/dropout/Const:output:0*
T0*'
_output_shapes
:         @V
dr5/dropout/ShapeShaped4/Relu:activations:0*
T0*
_output_shapes
:б
(dr5/dropout/random_uniform/RandomUniformRandomUniformdr5/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2_
dr5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>▓
dr5/dropout/GreaterEqualGreaterEqual1dr5/dropout/random_uniform/RandomUniform:output:0#dr5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @w
dr5/dropout/CastCastdr5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @u
dr5/dropout/Mul_1Muldr5/dropout/Mul:z:0dr5/dropout/Cast:y:0*
T0*'
_output_shapes
:         @z
d6/MatMul/ReadVariableOpReadVariableOp!d6_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0~
	d6/MatMulMatMuldr5/dropout/Mul_1:z:0 d6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (x
d6/BiasAdd/ReadVariableOpReadVariableOp"d6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d6/BiasAddBiasAddd6/MatMul:product:0!d6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (\

d6/SoftmaxSoftmaxd6/BiasAdd:output:0*
T0*'
_output_shapes
:         (`
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
:         (ы
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^d4/BiasAdd/ReadVariableOp^d4/MatMul/ReadVariableOp^d6/BiasAdd/ReadVariableOp^d6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
d4/BiasAdd/ReadVariableOpd4/BiasAdd/ReadVariableOp24
d4/MatMul/ReadVariableOpd4/MatMul/ReadVariableOp26
d6/BiasAdd/ReadVariableOpd6/BiasAdd/ReadVariableOp24
d6/MatMul/ReadVariableOpd6/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_2_69664
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
н"
а
G__inference_sequential_9_layer_call_and_return_conditional_losses_69351
c0_input"
c0_69325:		
c0_69327:
d4_69333:	А @
d4_69335:@
d6_69339:@(
d6_69341:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_69325c0_69327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_69042╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_69019с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_69200╤
f3/PartitionedCallPartitionedCall$dr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_69062ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_69333d4_69335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d4_layer_call_and_return_conditional_losses_69077 
dr5/StatefulPartitionedCallStatefulPartitionedCall#d4/StatefulPartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_69161·
d6/StatefulPartitionedCallStatefulPartitionedCall$dr5/StatefulPartitionedCall:output:0d6_69339d6_69341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d6_layer_call_and_return_conditional_losses_69103`
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
:         (┘
NoOpNoOp^c0/StatefulPartitionedCall^d4/StatefulPartitionedCall^d6/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
d4/StatefulPartitionedCalld4/StatefulPartitionedCall28
d6/StatefulPartitionedCalld6/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
╜

]
>__inference_dr2_layer_call_and_return_conditional_losses_69200

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_4_69674
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
Р	
О
,__inference_sequential_9_layer_call_fn_69380

inputs!
unknown:		
	unknown_0:
	unknown_1:	А @
	unknown_2:@
	unknown_3:@(
	unknown_4:(
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (*(
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_69116o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
и
>
"__inference_f3_layer_call_fn_69568

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f3_layer_call_and_return_conditional_losses_69062a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ё%
└
 __inference__wrapped_model_69010
c0_inputH
.sequential_9_c0_conv2d_readvariableop_resource:		=
/sequential_9_c0_biasadd_readvariableop_resource:A
.sequential_9_d4_matmul_readvariableop_resource:	А @=
/sequential_9_d4_biasadd_readvariableop_resource:@@
.sequential_9_d6_matmul_readvariableop_resource:@(=
/sequential_9_d6_biasadd_readvariableop_resource:(
identityИв&sequential_9/c0/BiasAdd/ReadVariableOpв%sequential_9/c0/Conv2D/ReadVariableOpв&sequential_9/d4/BiasAdd/ReadVariableOpв%sequential_9/d4/MatMul/ReadVariableOpв&sequential_9/d6/BiasAdd/ReadVariableOpв%sequential_9/d6/MatMul/ReadVariableOpЬ
%sequential_9/c0/Conv2D/ReadVariableOpReadVariableOp.sequential_9_c0_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0╗
sequential_9/c0/Conv2DConv2Dc0_input-sequential_9/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
Т
&sequential_9/c0/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0н
sequential_9/c0/BiasAddBiasAddsequential_9/c0/Conv2D:output:0.sequential_9/c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@x
sequential_9/c0/ReluRelu sequential_9/c0/BiasAdd:output:0*
T0*/
_output_shapes
:         @@│
sequential_9/m1/MaxPoolMaxPool"sequential_9/c0/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
Б
sequential_9/dr2/IdentityIdentity sequential_9/m1/MaxPool:output:0*
T0*/
_output_shapes
:         f
sequential_9/f3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Щ
sequential_9/f3/ReshapeReshape"sequential_9/dr2/Identity:output:0sequential_9/f3/Const:output:0*
T0*(
_output_shapes
:         А Х
%sequential_9/d4/MatMul/ReadVariableOpReadVariableOp.sequential_9_d4_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0г
sequential_9/d4/MatMulMatMul sequential_9/f3/Reshape:output:0-sequential_9/d4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Т
&sequential_9/d4/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
sequential_9/d4/BiasAddBiasAdd sequential_9/d4/MatMul:product:0.sequential_9/d4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
sequential_9/d4/ReluRelu sequential_9/d4/BiasAdd:output:0*
T0*'
_output_shapes
:         @{
sequential_9/dr5/IdentityIdentity"sequential_9/d4/Relu:activations:0*
T0*'
_output_shapes
:         @Ф
%sequential_9/d6/MatMul/ReadVariableOpReadVariableOp.sequential_9_d6_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0е
sequential_9/d6/MatMulMatMul"sequential_9/dr5/Identity:output:0-sequential_9/d6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (Т
&sequential_9/d6/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d6_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0ж
sequential_9/d6/BiasAddBiasAdd sequential_9/d6/MatMul:product:0.sequential_9/d6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (v
sequential_9/d6/SoftmaxSoftmax sequential_9/d6/BiasAdd:output:0*
T0*'
_output_shapes
:         (p
IdentityIdentity!sequential_9/d6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         (╣
NoOpNoOp'^sequential_9/c0/BiasAdd/ReadVariableOp&^sequential_9/c0/Conv2D/ReadVariableOp'^sequential_9/d4/BiasAdd/ReadVariableOp&^sequential_9/d4/MatMul/ReadVariableOp'^sequential_9/d6/BiasAdd/ReadVariableOp&^sequential_9/d6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @@: : : : : : 2P
&sequential_9/c0/BiasAdd/ReadVariableOp&sequential_9/c0/BiasAdd/ReadVariableOp2N
%sequential_9/c0/Conv2D/ReadVariableOp%sequential_9/c0/Conv2D/ReadVariableOp2P
&sequential_9/d4/BiasAdd/ReadVariableOp&sequential_9/d4/BiasAdd/ReadVariableOp2N
%sequential_9/d4/MatMul/ReadVariableOp%sequential_9/d4/MatMul/ReadVariableOp2P
&sequential_9/d6/BiasAdd/ReadVariableOp&sequential_9/d6/BiasAdd/ReadVariableOp2N
%sequential_9/d6/MatMul/ReadVariableOp%sequential_9/d6/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
Ш
?
#__inference_dr5_layer_call_fn_69603

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_69088`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*п
serving_defaultЫ
E
c0_input9
serving_default_c0_input:0         @@6
d60
StatefulPartitionedCall:0         (tensorflow/serving/predict:ўС
П
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
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
	variables
 trainable_variables
!regularization_losses
"	keras_api
#_random_generator
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
е
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8_random_generator
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
╦
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratemВmГ,mД-mЕ;mЖ<mЗvИvЙ,vК-vЛ;vМ<vН"
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
╩
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
■2√
,__inference_sequential_9_layer_call_fn_69131
,__inference_sequential_9_layer_call_fn_69380
,__inference_sequential_9_layer_call_fn_69397
,__inference_sequential_9_layer_call_fn_69293└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
G__inference_sequential_9_layer_call_and_return_conditional_losses_69433
G__inference_sequential_9_layer_call_and_return_conditional_losses_69483
G__inference_sequential_9_layer_call_and_return_conditional_losses_69322
G__inference_sequential_9_layer_call_and_return_conditional_losses_69351└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠B╔
 __inference__wrapped_model_69010c0_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
Sserving_default"
signature_map
#:!		2	c0/kernel
:2c0/bias
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
н
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
╠2╔
"__inference_c0_layer_call_fn_69513в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
=__inference_c0_layer_call_and_return_conditional_losses_69526в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
╠2╔
"__inference_m1_layer_call_fn_69531в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
=__inference_m1_layer_call_and_return_conditional_losses_69536в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
Д2Б
#__inference_dr2_layer_call_fn_69541
#__inference_dr2_layer_call_fn_69546┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖
>__inference_dr2_layer_call_and_return_conditional_losses_69551
>__inference_dr2_layer_call_and_return_conditional_losses_69563┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
╠2╔
"__inference_f3_layer_call_fn_69568в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
=__inference_f3_layer_call_and_return_conditional_losses_69574в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	А @2	d4/kernel
:@2d4/bias
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
н
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
╠2╔
"__inference_d4_layer_call_fn_69585в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
=__inference_d4_layer_call_and_return_conditional_losses_69598в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
Д2Б
#__inference_dr5_layer_call_fn_69603
#__inference_dr5_layer_call_fn_69608┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖
>__inference_dr5_layer_call_and_return_conditional_losses_69613
>__inference_dr5_layer_call_and_return_conditional_losses_69625┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
:@(2	d6/kernel
:(2d6/bias
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
н
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
╠2╔
"__inference_d6_layer_call_fn_69636в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
=__inference_d6_layer_call_and_return_conditional_losses_69649в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
▓2п
__inference_loss_fn_0_69654П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_1_69659П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_2_69664П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_3_69669П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_4_69674П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_5_69679П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
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
╦B╚
#__inference_signature_wrapper_69502c0_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
А	variables
Б	keras_api"
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
А	variables"
_generic_user_object
(:&		2Adam/c0/kernel/m
:2Adam/c0/bias/m
!:	А @2Adam/d4/kernel/m
:@2Adam/d4/bias/m
 :@(2Adam/d6/kernel/m
:(2Adam/d6/bias/m
(:&		2Adam/c0/kernel/v
:2Adam/c0/bias/v
!:	А @2Adam/d4/kernel/v
:@2Adam/d4/bias/v
 :@(2Adam/d6/kernel/v
:(2Adam/d6/bias/vР
 __inference__wrapped_model_69010l,-;<9в6
/в,
*К'
c0_input         @@
к "'к$
"
d6К
d6         (н
=__inference_c0_layer_call_and_return_conditional_losses_69526l7в4
-в*
(К%
inputs         @@
к "-в*
#К 
0         @@
Ъ Е
"__inference_c0_layer_call_fn_69513_7в4
-в*
(К%
inputs         @@
к " К         @@Ю
=__inference_d4_layer_call_and_return_conditional_losses_69598],-0в-
&в#
!К
inputs         А 
к "%в"
К
0         @
Ъ v
"__inference_d4_layer_call_fn_69585P,-0в-
&в#
!К
inputs         А 
к "К         @Э
=__inference_d6_layer_call_and_return_conditional_losses_69649\;</в,
%в"
 К
inputs         @
к "%в"
К
0         (
Ъ u
"__inference_d6_layer_call_fn_69636O;</в,
%в"
 К
inputs         @
к "К         (о
>__inference_dr2_layer_call_and_return_conditional_losses_69551l;в8
1в.
(К%
inputs         
p 
к "-в*
#К 
0         
Ъ о
>__inference_dr2_layer_call_and_return_conditional_losses_69563l;в8
1в.
(К%
inputs         
p
к "-в*
#К 
0         
Ъ Ж
#__inference_dr2_layer_call_fn_69541_;в8
1в.
(К%
inputs         
p 
к " К         Ж
#__inference_dr2_layer_call_fn_69546_;в8
1в.
(К%
inputs         
p
к " К         Ю
>__inference_dr5_layer_call_and_return_conditional_losses_69613\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ Ю
>__inference_dr5_layer_call_and_return_conditional_losses_69625\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ v
#__inference_dr5_layer_call_fn_69603O3в0
)в&
 К
inputs         @
p 
к "К         @v
#__inference_dr5_layer_call_fn_69608O3в0
)в&
 К
inputs         @
p
к "К         @в
=__inference_f3_layer_call_and_return_conditional_losses_69574a7в4
-в*
(К%
inputs         
к "&в#
К
0         А 
Ъ z
"__inference_f3_layer_call_fn_69568T7в4
-в*
(К%
inputs         
к "К         А 7
__inference_loss_fn_0_69654в

в 
к "К 7
__inference_loss_fn_1_69659в

в 
к "К 7
__inference_loss_fn_2_69664в

в 
к "К 7
__inference_loss_fn_3_69669в

в 
к "К 7
__inference_loss_fn_4_69674в

в 
к "К 7
__inference_loss_fn_5_69679в

в 
к "К р
=__inference_m1_layer_call_and_return_conditional_losses_69536ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_m1_layer_call_fn_69531СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╜
G__inference_sequential_9_layer_call_and_return_conditional_losses_69322r,-;<Aв>
7в4
*К'
c0_input         @@
p 

 
к "%в"
К
0         (
Ъ ╜
G__inference_sequential_9_layer_call_and_return_conditional_losses_69351r,-;<Aв>
7в4
*К'
c0_input         @@
p

 
к "%в"
К
0         (
Ъ ╗
G__inference_sequential_9_layer_call_and_return_conditional_losses_69433p,-;<?в<
5в2
(К%
inputs         @@
p 

 
к "%в"
К
0         (
Ъ ╗
G__inference_sequential_9_layer_call_and_return_conditional_losses_69483p,-;<?в<
5в2
(К%
inputs         @@
p

 
к "%в"
К
0         (
Ъ Х
,__inference_sequential_9_layer_call_fn_69131e,-;<Aв>
7в4
*К'
c0_input         @@
p 

 
к "К         (Х
,__inference_sequential_9_layer_call_fn_69293e,-;<Aв>
7в4
*К'
c0_input         @@
p

 
к "К         (У
,__inference_sequential_9_layer_call_fn_69380c,-;<?в<
5в2
(К%
inputs         @@
p 

 
к "К         (У
,__inference_sequential_9_layer_call_fn_69397c,-;<?в<
5в2
(К%
inputs         @@
p

 
к "К         (Я
#__inference_signature_wrapper_69502x,-;<EвB
в 
;к8
6
c0_input*К'
c0_input         @@"'к$
"
d6К
d6         (
─А	
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
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ь┼
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
p
	d5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А А*
shared_name	d5/kernel
i
d5/kernel/Read/ReadVariableOpReadVariableOp	d5/kernel* 
_output_shapes
:
А А*
dtype0
g
d5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	d5/bias
`
d5/bias/Read/ReadVariableOpReadVariableOpd5/bias*
_output_shapes	
:А*
dtype0
o
	d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*
shared_name	d7/kernel
h
d7/kernel/Read/ReadVariableOpReadVariableOp	d7/kernel*
_output_shapes
:	А
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
Д
Adam/c0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/c0/kernel/m
}
$Adam/c0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/m*&
_output_shapes
: *
dtype0
t
Adam/c0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/c0/bias/m
m
"Adam/c0/bias/m/Read/ReadVariableOpReadVariableOpAdam/c0/bias/m*
_output_shapes
: *
dtype0
Д
Adam/c1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/c1/kernel/m
}
$Adam/c1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c1/kernel/m*&
_output_shapes
: *
dtype0
t
Adam/c1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c1/bias/m
m
"Adam/c1/bias/m/Read/ReadVariableOpReadVariableOpAdam/c1/bias/m*
_output_shapes
:*
dtype0
~
Adam/d5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А А*!
shared_nameAdam/d5/kernel/m
w
$Adam/d5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d5/kernel/m* 
_output_shapes
:
А А*
dtype0
u
Adam/d5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/d5/bias/m
n
"Adam/d5/bias/m/Read/ReadVariableOpReadVariableOpAdam/d5/bias/m*
_output_shapes	
:А*
dtype0
}
Adam/d7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*!
shared_nameAdam/d7/kernel/m
v
$Adam/d7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d7/kernel/m*
_output_shapes
:	А
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
Д
Adam/c0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/c0/kernel/v
}
$Adam/c0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/v*&
_output_shapes
: *
dtype0
t
Adam/c0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/c0/bias/v
m
"Adam/c0/bias/v/Read/ReadVariableOpReadVariableOpAdam/c0/bias/v*
_output_shapes
: *
dtype0
Д
Adam/c1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/c1/kernel/v
}
$Adam/c1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c1/kernel/v*&
_output_shapes
: *
dtype0
t
Adam/c1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c1/bias/v
m
"Adam/c1/bias/v/Read/ReadVariableOpReadVariableOpAdam/c1/bias/v*
_output_shapes
:*
dtype0
~
Adam/d5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А А*!
shared_nameAdam/d5/kernel/v
w
$Adam/d5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d5/kernel/v* 
_output_shapes
:
А А*
dtype0
u
Adam/d5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/d5/bias/v
n
"Adam/d5/bias/v/Read/ReadVariableOpReadVariableOpAdam/d5/bias/v*
_output_shapes	
:А*
dtype0
}
Adam/d7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*!
shared_nameAdam/d7/kernel/v
v
$Adam/d7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d7/kernel/v*
_output_shapes
:	А
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
█G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЦG
valueМGBЙG BВG
Ь
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
ж

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
ж

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
О
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
е
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,_random_generator
-__call__
*.&call_and_return_all_conditional_losses* 
О
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
ж

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*
е
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A_random_generator
B__call__
*C&call_and_return_all_conditional_losses* 
ж

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
ф
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratemТmУmФmХ5mЦ6mЧDmШEmЩvЪvЫvЬvЭ5vЮ6vЯDvаEvб*
<
0
1
2
3
54
65
D6
E7*
<
0
1
2
3
54
65
D6
E7*
:
Q0
R1
S2
T3
U4
V5
W6
X7* 
░
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
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
^serving_default* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

Q0
R1* 
У
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUE	c1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

S0
T1* 
У
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
С
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
С
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
(	variables
)trainable_variables
*regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
С
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUE	d5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*

U0
V1* 
У
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
У
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
=	variables
>trainable_variables
?regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*

W0
X1* 
Ш
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
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
З0
И1*
* 
* 
* 
* 
* 
* 

Q0
R1* 
* 
* 
* 
* 

S0
T1* 
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
U0
V1* 
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
W0
X1* 
* 
<

Йtotal

Кcount
Л	variables
М	keras_api*
M

Нtotal

Оcount
П
_fn_kwargs
Р	variables
С	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Й0
К1*

Л	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Н0
О1*

Р	variables*
|v
VARIABLE_VALUEAdam/c0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Л
serving_default_c0_inputPlaceholder*/
_output_shapes
:           *
dtype0*$
shape:           
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c1/kernelc1/bias	d5/kerneld5/bias	d7/kerneld7/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *,
f'R%
#__inference_signature_wrapper_30053
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamec0/kernel/Read/ReadVariableOpc0/bias/Read/ReadVariableOpc1/kernel/Read/ReadVariableOpc1/bias/Read/ReadVariableOpd5/kernel/Read/ReadVariableOpd5/bias/Read/ReadVariableOpd7/kernel/Read/ReadVariableOpd7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/c0/kernel/m/Read/ReadVariableOp"Adam/c0/bias/m/Read/ReadVariableOp$Adam/c1/kernel/m/Read/ReadVariableOp"Adam/c1/bias/m/Read/ReadVariableOp$Adam/d5/kernel/m/Read/ReadVariableOp"Adam/d5/bias/m/Read/ReadVariableOp$Adam/d7/kernel/m/Read/ReadVariableOp"Adam/d7/bias/m/Read/ReadVariableOp$Adam/c0/kernel/v/Read/ReadVariableOp"Adam/c0/bias/v/Read/ReadVariableOp$Adam/c1/kernel/v/Read/ReadVariableOp"Adam/c1/bias/v/Read/ReadVariableOp$Adam/d5/kernel/v/Read/ReadVariableOp"Adam/d5/bias/v/Read/ReadVariableOp$Adam/d7/kernel/v/Read/ReadVariableOp"Adam/d7/bias/v/Read/ReadVariableOpConst*.
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
GPU(2*0J 8В *'
f"R 
__inference__traced_save_30386
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	d5/kerneld5/bias	d7/kerneld7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/c0/kernel/mAdam/c0/bias/mAdam/c1/kernel/mAdam/c1/bias/mAdam/d5/kernel/mAdam/d5/bias/mAdam/d7/kernel/mAdam/d7/bias/mAdam/c0/kernel/vAdam/c0/bias/vAdam/c1/kernel/vAdam/c1/bias/vAdam/d5/kernel/vAdam/d5/bias/vAdam/d7/kernel/vAdam/d7/bias/v*-
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
GPU(2*0J 8В **
f%R#
!__inference__traced_restore_30495ън
ц%
╝
E__inference_sequential_layer_call_and_return_conditional_losses_29588

inputs"
c0_29494: 
c0_29496: "
c1_29513: 
c1_29515:
d5_29548:
А А
d5_29550:	А
d7_29574:	А

d7_29576:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвd5/StatefulPartitionedCallвd7/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_29494c0_29496*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_29493Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_29513c1_29515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_29512╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
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
=__inference_m2_layer_call_and_return_conditional_losses_29470╤
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
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
>__inference_dr3_layer_call_and_return_conditional_losses_29524╔
f4/PartitionedCallPartitionedCalldr3/PartitionedCall:output:0*
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
=__inference_f4_layer_call_and_return_conditional_losses_29532Є
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_29548d5_29550*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29547╥
dr6/PartitionedCallPartitionedCall#d5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29558Є
d7/StatefulPartitionedCallStatefulPartitionedCalldr6/PartitionedCall:output:0d7_29574d7_29576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29573`
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
:         
║
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
ё	
╧
*__inference_sequential_layer_call_fn_29798
c0_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
А А
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
▀
я
=__inference_d7_layer_call_and_return_conditional_losses_29573

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
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
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
г
>
"__inference_m2_layer_call_fn_30106

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
=__inference_m2_layer_call_and_return_conditional_losses_29470Г
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
д,
Ё
E__inference_sequential_layer_call_and_return_conditional_losses_29971

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource: 0
"c1_biasadd_readvariableop_resource:5
!d5_matmul_readvariableop_resource:
А А1
"d5_biasadd_readvariableop_resource:	А4
!d7_matmul_readvariableop_resource:	А
0
"d7_biasadd_readvariableop_resource:

identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc1/BiasAdd/ReadVariableOpвc1/Conv2D/ReadVariableOpвd5/BiasAdd/ReadVariableOpвd5/MatMul/ReadVariableOpвd7/BiasAdd/ReadVariableOpвd7/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:            В
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0о
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:           Щ

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
g
dr3/IdentityIdentitym2/MaxPool:output:0*
T0*/
_output_shapes
:         Y
f4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f4/ReshapeReshapedr3/Identity:output:0f4/Const:output:0*
T0*(
_output_shapes
:         А |
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0}
	d5/MatMulMatMulf4/Reshape:output:0 d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0А

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АW
d5/ReluRelud5/BiasAdd:output:0*
T0*(
_output_shapes
:         Аb
dr6/IdentityIdentityd5/Relu:activations:0*
T0*(
_output_shapes
:         А{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0~
	d7/MatMulMatMuldr6/Identity:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
\

d7/SoftmaxSoftmaxd7/BiasAdd:output:0*
T0*'
_output_shapes
:         
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
:         
в
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 26
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
:           
 
_user_specified_nameinputs
ё	
╧
*__inference_sequential_layer_call_fn_29607
c0_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
А А
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29588o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
╛
Ў
=__inference_c1_layer_call_and_return_conditional_losses_30101

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           *
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
:           X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:           `
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
:           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:            
 
_user_specified_nameinputs
и
>
"__inference_f4_layer_call_fn_30143

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
=__inference_f4_layer_call_and_return_conditional_losses_29532a
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
╛
Ў
=__inference_c1_layer_call_and_return_conditional_losses_29512

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           *
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
:           X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:           `
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
:           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:            
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_0_30229
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
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_29493

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
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
:            X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:            `
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
:            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╜

]
>__inference_dr3_layer_call_and_return_conditional_losses_30138

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%IТ?l
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
 *   >о
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
__inference_loss_fn_6_30259
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
┐
Y
=__inference_f4_layer_call_and_return_conditional_losses_29532

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
Е
Y
=__inference_m2_layer_call_and_return_conditional_losses_30111

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
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
Е

]
>__inference_dr6_layer_call_and_return_conditional_losses_30200

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ю
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
у
Ч
"__inference_c0_layer_call_fn_30064

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_29493w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
К
\
#__inference_dr3_layer_call_fn_30121

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
>__inference_dr3_layer_call_and_return_conditional_losses_29676w
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
╜.
¤
 __inference__wrapped_model_29461
c0_inputF
,sequential_c0_conv2d_readvariableop_resource: ;
-sequential_c0_biasadd_readvariableop_resource: F
,sequential_c1_conv2d_readvariableop_resource: ;
-sequential_c1_biasadd_readvariableop_resource:@
,sequential_d5_matmul_readvariableop_resource:
А А<
-sequential_d5_biasadd_readvariableop_resource:	А?
,sequential_d7_matmul_readvariableop_resource:	А
;
-sequential_d7_biasadd_readvariableop_resource:

identityИв$sequential/c0/BiasAdd/ReadVariableOpв#sequential/c0/Conv2D/ReadVariableOpв$sequential/c1/BiasAdd/ReadVariableOpв#sequential/c1/Conv2D/ReadVariableOpв$sequential/d5/BiasAdd/ReadVariableOpв#sequential/d5/MatMul/ReadVariableOpв$sequential/d7/BiasAdd/ReadVariableOpв#sequential/d7/MatMul/ReadVariableOpШ
#sequential/c0/Conv2D/ReadVariableOpReadVariableOp,sequential_c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╖
sequential/c0/Conv2DConv2Dc0_input+sequential/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
О
$sequential/c0/BiasAdd/ReadVariableOpReadVariableOp-sequential_c0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0з
sequential/c0/BiasAddBiasAddsequential/c0/Conv2D:output:0,sequential/c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            t
sequential/c0/ReluRelusequential/c0/BiasAdd:output:0*
T0*/
_output_shapes
:            Ш
#sequential/c1/Conv2D/ReadVariableOpReadVariableOp,sequential_c1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╧
sequential/c1/Conv2DConv2D sequential/c0/Relu:activations:0+sequential/c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           *
paddingSAME*
strides
О
$sequential/c1/BiasAdd/ReadVariableOpReadVariableOp-sequential_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0з
sequential/c1/BiasAddBiasAddsequential/c1/Conv2D:output:0,sequential/c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           t
sequential/c1/ReluRelusequential/c1/BiasAdd:output:0*
T0*/
_output_shapes
:           п
sequential/m2/MaxPoolMaxPool sequential/c1/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
}
sequential/dr3/IdentityIdentitysequential/m2/MaxPool:output:0*
T0*/
_output_shapes
:         d
sequential/f4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       У
sequential/f4/ReshapeReshape sequential/dr3/Identity:output:0sequential/f4/Const:output:0*
T0*(
_output_shapes
:         А Т
#sequential/d5/MatMul/ReadVariableOpReadVariableOp,sequential_d5_matmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0Ю
sequential/d5/MatMulMatMulsequential/f4/Reshape:output:0+sequential/d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$sequential/d5/BiasAdd/ReadVariableOpReadVariableOp-sequential_d5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
sequential/d5/BiasAddBiasAddsequential/d5/MatMul:product:0,sequential/d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
sequential/d5/ReluRelusequential/d5/BiasAdd:output:0*
T0*(
_output_shapes
:         Аx
sequential/dr6/IdentityIdentity sequential/d5/Relu:activations:0*
T0*(
_output_shapes
:         АС
#sequential/d7/MatMul/ReadVariableOpReadVariableOp,sequential_d7_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Я
sequential/d7/MatMulMatMul sequential/dr6/Identity:output:0+sequential/d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
О
$sequential/d7/BiasAdd/ReadVariableOpReadVariableOp-sequential_d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0а
sequential/d7/BiasAddBiasAddsequential/d7/MatMul:product:0,sequential/d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
sequential/d7/SoftmaxSoftmaxsequential/d7/BiasAdd:output:0*
T0*'
_output_shapes
:         
n
IdentityIdentitysequential/d7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
·
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/c1/BiasAdd/ReadVariableOp$^sequential/c1/Conv2D/ReadVariableOp%^sequential/d5/BiasAdd/ReadVariableOp$^sequential/d5/MatMul/ReadVariableOp%^sequential/d7/BiasAdd/ReadVariableOp$^sequential/d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 2L
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
:           
"
_user_specified_name
c0_input
Ь
?
#__inference_dr6_layer_call_fn_30178

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29558a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
дВ
ї
!__inference__traced_restore_30495
file_prefix4
assignvariableop_c0_kernel: (
assignvariableop_1_c0_bias: 6
assignvariableop_2_c1_kernel: (
assignvariableop_3_c1_bias:0
assignvariableop_4_d5_kernel:
А А)
assignvariableop_5_d5_bias:	А/
assignvariableop_6_d7_kernel:	А
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
assignvariableop_16_count_1: >
$assignvariableop_17_adam_c0_kernel_m: 0
"assignvariableop_18_adam_c0_bias_m: >
$assignvariableop_19_adam_c1_kernel_m: 0
"assignvariableop_20_adam_c1_bias_m:8
$assignvariableop_21_adam_d5_kernel_m:
А А1
"assignvariableop_22_adam_d5_bias_m:	А7
$assignvariableop_23_adam_d7_kernel_m:	А
0
"assignvariableop_24_adam_d7_bias_m:
>
$assignvariableop_25_adam_c0_kernel_v: 0
"assignvariableop_26_adam_c0_bias_v: >
$assignvariableop_27_adam_c1_kernel_v: 0
"assignvariableop_28_adam_c1_bias_v:8
$assignvariableop_29_adam_d5_kernel_v:
А А1
"assignvariableop_30_adam_d5_bias_v:	А7
$assignvariableop_31_adam_d7_kernel_v:	А
0
"assignvariableop_32_adam_d7_bias_v:

identity_34ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9▓
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*╪
value╬B╦"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH┤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╦
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
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
AssignVariableOp_2AssignVariableOpassignvariableop_2_c1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_3AssignVariableOpassignvariableop_3_c1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_4AssignVariableOpassignvariableop_4_d5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_5AssignVariableOpassignvariableop_5_d5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_d7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_7AssignVariableOpassignvariableop_7_d7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_c0_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_c0_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_c1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_c1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_d5_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_d5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_d7_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_d7_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_c0_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_c0_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_c1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_c1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_29AssignVariableOp$assignvariableop_29_adam_d5_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_d5_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_d7_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_d7_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 е
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: Т
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
─
+
__inference_loss_fn_1_30234
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
ь%
╛
E__inference_sequential_layer_call_and_return_conditional_losses_29834
c0_input"
c0_29801: 
c0_29803: "
c1_29806: 
c1_29808:
d5_29814:
А А
d5_29816:	А
d7_29820:	А

d7_29822:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвd5/StatefulPartitionedCallвd7/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_29801c0_29803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_29493Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_29806c1_29808*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_29512╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
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
=__inference_m2_layer_call_and_return_conditional_losses_29470╤
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
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
>__inference_dr3_layer_call_and_return_conditional_losses_29524╔
f4/PartitionedCallPartitionedCalldr3/PartitionedCall:output:0*
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
=__inference_f4_layer_call_and_return_conditional_losses_29532Є
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_29814d5_29816*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29547╥
dr6/PartitionedCallPartitionedCall#d5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29558Є
d7/StatefulPartitionedCallStatefulPartitionedCalldr6/PartitionedCall:output:0d7_29820d7_29822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29573`
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
:         
║
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
╜

]
>__inference_dr3_layer_call_and_return_conditional_losses_29676

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%IТ?l
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
 *   >о
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
у
Ч
"__inference_c1_layer_call_fn_30088

inputs!
unknown: 
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
:           *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_29512w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:            : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:            
 
_user_specified_nameinputs
ю
\
#__inference_dr6_layer_call_fn_30183

inputs
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29637p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_4_30249
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
т
ё
=__inference_d5_layer_call_and_return_conditional_losses_29547

inputs2
matmul_readvariableop_resource:
А А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А`
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
:         Аw
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
ё
\
>__inference_dr3_layer_call_and_return_conditional_losses_29524

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
д(
°
E__inference_sequential_layer_call_and_return_conditional_losses_29758

inputs"
c0_29725: 
c0_29727: "
c1_29730: 
c1_29732:
d5_29738:
А А
d5_29740:	А
d7_29744:	А

d7_29746:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвd5/StatefulPartitionedCallвd7/StatefulPartitionedCallвdr3/StatefulPartitionedCallвdr6/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_29725c0_29727*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_29493Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_29730c1_29732*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_29512╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
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
=__inference_m2_layer_call_and_return_conditional_losses_29470с
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
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
>__inference_dr3_layer_call_and_return_conditional_losses_29676╤
f4/PartitionedCallPartitionedCall$dr3/StatefulPartitionedCall:output:0*
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
=__inference_f4_layer_call_and_return_conditional_losses_29532Є
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_29738d5_29740*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29547А
dr6/StatefulPartitionedCallStatefulPartitionedCall#d5/StatefulPartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29637·
d7/StatefulPartitionedCallStatefulPartitionedCall$dr6/StatefulPartitionedCall:output:0d7_29744d7_29746*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29573`
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
:         
Ў
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr6/StatefulPartitionedCalldr6/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
┘D
п
__inference__traced_save_30386
file_prefix(
$savev2_c0_kernel_read_readvariableop&
"savev2_c0_bias_read_readvariableop(
$savev2_c1_kernel_read_readvariableop&
"savev2_c1_bias_read_readvariableop(
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
+savev2_adam_c0_kernel_m_read_readvariableop-
)savev2_adam_c0_bias_m_read_readvariableop/
+savev2_adam_c1_kernel_m_read_readvariableop-
)savev2_adam_c1_bias_m_read_readvariableop/
+savev2_adam_d5_kernel_m_read_readvariableop-
)savev2_adam_d5_bias_m_read_readvariableop/
+savev2_adam_d7_kernel_m_read_readvariableop-
)savev2_adam_d7_bias_m_read_readvariableop/
+savev2_adam_c0_kernel_v_read_readvariableop-
)savev2_adam_c0_bias_v_read_readvariableop/
+savev2_adam_c1_kernel_v_read_readvariableop-
)savev2_adam_c1_bias_v_read_readvariableop/
+savev2_adam_d5_kernel_v_read_readvariableop-
)savev2_adam_d5_bias_v_read_readvariableop/
+savev2_adam_d7_kernel_v_read_readvariableop-
)savev2_adam_d7_bias_v_read_readvariableop
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
: п
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*╪
value╬B╦"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH▒
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B С
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_c0_kernel_read_readvariableop"savev2_c0_bias_read_readvariableop$savev2_c1_kernel_read_readvariableop"savev2_c1_bias_read_readvariableop$savev2_d5_kernel_read_readvariableop"savev2_d5_bias_read_readvariableop$savev2_d7_kernel_read_readvariableop"savev2_d7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_c0_kernel_m_read_readvariableop)savev2_adam_c0_bias_m_read_readvariableop+savev2_adam_c1_kernel_m_read_readvariableop)savev2_adam_c1_bias_m_read_readvariableop+savev2_adam_d5_kernel_m_read_readvariableop)savev2_adam_d5_bias_m_read_readvariableop+savev2_adam_d7_kernel_m_read_readvariableop)savev2_adam_d7_bias_m_read_readvariableop+savev2_adam_c0_kernel_v_read_readvariableop)savev2_adam_c0_bias_v_read_readvariableop+savev2_adam_c1_kernel_v_read_readvariableop)savev2_adam_c1_bias_v_read_readvariableop+savev2_adam_d5_kernel_v_read_readvariableop)savev2_adam_d5_bias_v_read_readvariableop+savev2_adam_d7_kernel_v_read_readvariableop)savev2_adam_d7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	Р
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

identity_1Identity_1:output:0*з
_input_shapesХ
Т: : : : ::
А А:А:	А
:
: : : : : : : : : : : : ::
А А:А:	А
:
: : : ::
А А:А:	А
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
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::&"
 
_output_shapes
:
А А:!

_output_shapes	
:А:%!

_output_shapes
:	А
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
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::&"
 
_output_shapes
:
А А:!

_output_shapes	
:А:%!

_output_shapes
:	А
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::&"
 
_output_shapes
:
А А:!

_output_shapes	
:А:% !

_output_shapes
:	А
: !

_output_shapes
:
:"

_output_shapes
: 
╕
?
#__inference_dr3_layer_call_fn_30116

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
>__inference_dr3_layer_call_and_return_conditional_losses_29524h
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
┼	
╚
#__inference_signature_wrapper_30053
c0_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
А А
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *)
f$R"
 __inference__wrapped_model_29461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
▀
я
=__inference_d7_layer_call_and_return_conditional_losses_30224

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
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
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒
\
>__inference_dr6_layer_call_and_return_conditional_losses_30188

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─
+
__inference_loss_fn_3_30244
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
─
+
__inference_loss_fn_7_30264
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
╛
Р
"__inference_d7_layer_call_fn_30211

inputs
unknown:	А

	unknown_0:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29573o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ы	
═
*__inference_sequential_layer_call_fn_29926

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
А А
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
к(
·
E__inference_sequential_layer_call_and_return_conditional_losses_29870
c0_input"
c0_29837: 
c0_29839: "
c1_29842: 
c1_29844:
d5_29850:
А А
d5_29852:	А
d7_29856:	А

d7_29858:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвd5/StatefulPartitionedCallвd7/StatefulPartitionedCallвdr3/StatefulPartitionedCallвdr6/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_29837c0_29839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_29493Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_29842c1_29844*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_29512╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
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
=__inference_m2_layer_call_and_return_conditional_losses_29470с
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
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
>__inference_dr3_layer_call_and_return_conditional_losses_29676╤
f4/PartitionedCallPartitionedCall$dr3/StatefulPartitionedCall:output:0*
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
=__inference_f4_layer_call_and_return_conditional_losses_29532Є
d5/StatefulPartitionedCallStatefulPartitionedCallf4/PartitionedCall:output:0d5_29850d5_29852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29547А
dr6/StatefulPartitionedCallStatefulPartitionedCall#d5/StatefulPartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr6_layer_call_and_return_conditional_losses_29637·
d7/StatefulPartitionedCallStatefulPartitionedCall$dr6/StatefulPartitionedCall:output:0d7_29856d7_29858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_29573`
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
:         
Ў
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^d5/StatefulPartitionedCall^d7/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
d5/StatefulPartitionedCalld5/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr6/StatefulPartitionedCalldr6/StatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
ы	
═
*__inference_sequential_layer_call_fn_29905

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:
	unknown_3:
А А
	unknown_4:	А
	unknown_5:	А

	unknown_6:

identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29588o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_2_30239
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
ё
\
>__inference_dr3_layer_call_and_return_conditional_losses_30126

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
т
ё
=__inference_d5_layer_call_and_return_conditional_losses_30173

inputs2
matmul_readvariableop_resource:
А А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А`
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
:         Аw
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
Е
Y
=__inference_m2_layer_call_and_return_conditional_losses_29470

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
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
╒
\
>__inference_dr6_layer_call_and_return_conditional_losses_29558

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┬
Т
"__inference_d5_layer_call_fn_30160

inputs
unknown:
А А
	unknown_0:	А
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_d5_layer_call_and_return_conditional_losses_29547p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
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
─
+
__inference_loss_fn_5_30254
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
│:
Ё
E__inference_sequential_layer_call_and_return_conditional_losses_30030

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource: 0
"c1_biasadd_readvariableop_resource:5
!d5_matmul_readvariableop_resource:
А А1
"d5_biasadd_readvariableop_resource:	А4
!d7_matmul_readvariableop_resource:	А
0
"d7_biasadd_readvariableop_resource:

identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc1/BiasAdd/ReadVariableOpвc1/Conv2D/ReadVariableOpвd5/BiasAdd/ReadVariableOpвd5/MatMul/ReadVariableOpвd7/BiasAdd/ReadVariableOpвd7/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:            В
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0о
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:           Щ

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:         *
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
 *%IТ?Б
dr3/dropout/MulMulm2/MaxPool:output:0dr3/dropout/Const:output:0*
T0*/
_output_shapes
:         T
dr3/dropout/ShapeShapem2/MaxPool:output:0*
T0*
_output_shapes
:н
(dr3/dropout/random_uniform/RandomUniformRandomUniformdr3/dropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed2    _
dr3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >║
dr3/dropout/GreaterEqualGreaterEqual1dr3/dropout/random_uniform/RandomUniform:output:0#dr3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         
dr3/dropout/CastCastdr3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         }
dr3/dropout/Mul_1Muldr3/dropout/Mul:z:0dr3/dropout/Cast:y:0*
T0*/
_output_shapes
:         Y
f4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f4/ReshapeReshapedr3/dropout/Mul_1:z:0f4/Const:output:0*
T0*(
_output_shapes
:         А |
d5/MatMul/ReadVariableOpReadVariableOp!d5_matmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0}
	d5/MatMulMatMulf4/Reshape:output:0 d5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
d5/BiasAdd/ReadVariableOpReadVariableOp"d5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0А

d5/BiasAddBiasAddd5/MatMul:product:0!d5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АW
d5/ReluRelud5/BiasAdd:output:0*
T0*(
_output_shapes
:         АV
dr6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?|
dr6/dropout/MulMuld5/Relu:activations:0dr6/dropout/Const:output:0*
T0*(
_output_shapes
:         АV
dr6/dropout/ShapeShaped5/Relu:activations:0*
T0*
_output_shapes
:в
(dr6/dropout/random_uniform/RandomUniformRandomUniformdr6/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0*
seed2_
dr6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>│
dr6/dropout/GreaterEqualGreaterEqual1dr6/dropout/random_uniform/RandomUniform:output:0#dr6/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аx
dr6/dropout/CastCastdr6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аv
dr6/dropout/Mul_1Muldr6/dropout/Mul:z:0dr6/dropout/Cast:y:0*
T0*(
_output_shapes
:         А{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0~
	d7/MatMulMatMuldr6/dropout/Mul_1:z:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
\

d7/SoftmaxSoftmaxd7/BiasAdd:output:0*
T0*'
_output_shapes
:         
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
:         
в
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^d5/BiasAdd/ReadVariableOp^d5/MatMul/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:           : : : : : : : : 26
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
:           
 
_user_specified_nameinputs
┐
Y
=__inference_f4_layer_call_and_return_conditional_losses_30149

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
Е

]
>__inference_dr6_layer_call_and_return_conditional_losses_29637

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ю
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_30077

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
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
:            X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:            `
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
:            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           
 
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
serving_default_c0_input:0           6
d70
StatefulPartitionedCall:0         
tensorflow/serving/predict:рд
╢
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
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
е
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,_random_generator
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
е
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A_random_generator
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
є
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratemТmУmФmХ5mЦ6mЧDmШEmЩvЪvЫvЬvЭ5vЮ6vЯDvаEvб"
	optimizer
X
0
1
2
3
54
65
D6
E7"
trackable_list_wrapper
X
0
1
2
3
54
65
D6
E7"
trackable_list_wrapper
X
Q0
R1
S2
T3
U4
V5
W6
X7"
trackable_list_wrapper
╩
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ў2є
*__inference_sequential_layer_call_fn_29607
*__inference_sequential_layer_call_fn_29905
*__inference_sequential_layer_call_fn_29926
*__inference_sequential_layer_call_fn_29798└
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
т2▀
E__inference_sequential_layer_call_and_return_conditional_losses_29971
E__inference_sequential_layer_call_and_return_conditional_losses_30030
E__inference_sequential_layer_call_and_return_conditional_losses_29834
E__inference_sequential_layer_call_and_return_conditional_losses_29870└
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
 __inference__wrapped_model_29461c0_input"Ш
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
^serving_default"
signature_map
#:! 2	c0/kernel
: 2c0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
н
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c0_layer_call_fn_30064в
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
=__inference_c0_layer_call_and_return_conditional_losses_30077в
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
#:! 2	c1/kernel
:2c1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
н
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c1_layer_call_fn_30088в
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
=__inference_c1_layer_call_and_return_conditional_losses_30101в
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
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_m2_layer_call_fn_30106в
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
=__inference_m2_layer_call_and_return_conditional_losses_30111в
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
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
(	variables
)trainable_variables
*regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Д2Б
#__inference_dr3_layer_call_fn_30116
#__inference_dr3_layer_call_fn_30121┤
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
>__inference_dr3_layer_call_and_return_conditional_losses_30126
>__inference_dr3_layer_call_and_return_conditional_losses_30138┤
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
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_f4_layer_call_fn_30143в
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
=__inference_f4_layer_call_and_return_conditional_losses_30149в
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
:
А А2	d5/kernel
:А2d5/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
н
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_d5_layer_call_fn_30160в
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
=__inference_d5_layer_call_and_return_conditional_losses_30173в
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
п
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
=	variables
>trainable_variables
?regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Д2Б
#__inference_dr6_layer_call_fn_30178
#__inference_dr6_layer_call_fn_30183┤
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
>__inference_dr6_layer_call_and_return_conditional_losses_30188
>__inference_dr6_layer_call_and_return_conditional_losses_30200┤
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
:	А
2	d7/kernel
:
2d7/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
▓
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_d7_layer_call_fn_30211в
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
=__inference_d7_layer_call_and_return_conditional_losses_30224в
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
__inference_loss_fn_0_30229П
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
__inference_loss_fn_1_30234П
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
__inference_loss_fn_2_30239П
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
__inference_loss_fn_3_30244П
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
__inference_loss_fn_4_30249П
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
__inference_loss_fn_5_30254П
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
__inference_loss_fn_6_30259П
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
__inference_loss_fn_7_30264П
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
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╦B╚
#__inference_signature_wrapper_30053c0_input"Ф
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
Q0
R1"
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
S0
T1"
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
U0
V1"
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
W0
X1"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Йtotal

Кcount
Л	variables
М	keras_api"
_tf_keras_metric
c

Нtotal

Оcount
П
_fn_kwargs
Р	variables
С	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
Й0
К1"
trackable_list_wrapper
.
Л	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Н0
О1"
trackable_list_wrapper
.
Р	variables"
_generic_user_object
(:& 2Adam/c0/kernel/m
: 2Adam/c0/bias/m
(:& 2Adam/c1/kernel/m
:2Adam/c1/bias/m
": 
А А2Adam/d5/kernel/m
:А2Adam/d5/bias/m
!:	А
2Adam/d7/kernel/m
:
2Adam/d7/bias/m
(:& 2Adam/c0/kernel/v
: 2Adam/c0/bias/v
(:& 2Adam/c1/kernel/v
:2Adam/c1/bias/v
": 
А А2Adam/d5/kernel/v
:А2Adam/d5/bias/v
!:	А
2Adam/d7/kernel/v
:
2Adam/d7/bias/vТ
 __inference__wrapped_model_29461n56DE9в6
/в,
*К'
c0_input           
к "'к$
"
d7К
d7         
н
=__inference_c0_layer_call_and_return_conditional_losses_30077l7в4
-в*
(К%
inputs           
к "-в*
#К 
0            
Ъ Е
"__inference_c0_layer_call_fn_30064_7в4
-в*
(К%
inputs           
к " К            н
=__inference_c1_layer_call_and_return_conditional_losses_30101l7в4
-в*
(К%
inputs            
к "-в*
#К 
0           
Ъ Е
"__inference_c1_layer_call_fn_30088_7в4
-в*
(К%
inputs            
к " К           Я
=__inference_d5_layer_call_and_return_conditional_losses_30173^560в-
&в#
!К
inputs         А 
к "&в#
К
0         А
Ъ w
"__inference_d5_layer_call_fn_30160Q560в-
&в#
!К
inputs         А 
к "К         АЮ
=__inference_d7_layer_call_and_return_conditional_losses_30224]DE0в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ v
"__inference_d7_layer_call_fn_30211PDE0в-
&в#
!К
inputs         А
к "К         
о
>__inference_dr3_layer_call_and_return_conditional_losses_30126l;в8
1в.
(К%
inputs         
p 
к "-в*
#К 
0         
Ъ о
>__inference_dr3_layer_call_and_return_conditional_losses_30138l;в8
1в.
(К%
inputs         
p
к "-в*
#К 
0         
Ъ Ж
#__inference_dr3_layer_call_fn_30116_;в8
1в.
(К%
inputs         
p 
к " К         Ж
#__inference_dr3_layer_call_fn_30121_;в8
1в.
(К%
inputs         
p
к " К         а
>__inference_dr6_layer_call_and_return_conditional_losses_30188^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ а
>__inference_dr6_layer_call_and_return_conditional_losses_30200^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ x
#__inference_dr6_layer_call_fn_30178Q4в1
*в'
!К
inputs         А
p 
к "К         Аx
#__inference_dr6_layer_call_fn_30183Q4в1
*в'
!К
inputs         А
p
к "К         Ав
=__inference_f4_layer_call_and_return_conditional_losses_30149a7в4
-в*
(К%
inputs         
к "&в#
К
0         А 
Ъ z
"__inference_f4_layer_call_fn_30143T7в4
-в*
(К%
inputs         
к "К         А 7
__inference_loss_fn_0_30229в

в 
к "К 7
__inference_loss_fn_1_30234в

в 
к "К 7
__inference_loss_fn_2_30239в

в 
к "К 7
__inference_loss_fn_3_30244в

в 
к "К 7
__inference_loss_fn_4_30249в

в 
к "К 7
__inference_loss_fn_5_30254в

в 
к "К 7
__inference_loss_fn_6_30259в

в 
к "К 7
__inference_loss_fn_7_30264в

в 
к "К р
=__inference_m2_layer_call_and_return_conditional_losses_30111ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_m2_layer_call_fn_30106СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╜
E__inference_sequential_layer_call_and_return_conditional_losses_29834t56DEAв>
7в4
*К'
c0_input           
p 

 
к "%в"
К
0         

Ъ ╜
E__inference_sequential_layer_call_and_return_conditional_losses_29870t56DEAв>
7в4
*К'
c0_input           
p

 
к "%в"
К
0         

Ъ ╗
E__inference_sequential_layer_call_and_return_conditional_losses_29971r56DE?в<
5в2
(К%
inputs           
p 

 
к "%в"
К
0         

Ъ ╗
E__inference_sequential_layer_call_and_return_conditional_losses_30030r56DE?в<
5в2
(К%
inputs           
p

 
к "%в"
К
0         

Ъ Х
*__inference_sequential_layer_call_fn_29607g56DEAв>
7в4
*К'
c0_input           
p 

 
к "К         
Х
*__inference_sequential_layer_call_fn_29798g56DEAв>
7в4
*К'
c0_input           
p

 
к "К         
У
*__inference_sequential_layer_call_fn_29905e56DE?в<
5в2
(К%
inputs           
p 

 
к "К         
У
*__inference_sequential_layer_call_fn_29926e56DE?в<
5в2
(К%
inputs           
p

 
к "К         
б
#__inference_signature_wrapper_30053z56DEEвB
в 
;к8
6
c0_input*К'
c0_input           "'к$
"
d7К
d7         

 Д

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
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68┼н
v
	c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	c0/kernel
o
c0/kernel/Read/ReadVariableOpReadVariableOp	c0/kernel*&
_output_shapes
:*
dtype0
f
c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	c0/bias
_
c0/bias/Read/ReadVariableOpReadVariableOpc0/bias*
_output_shapes
:*
dtype0
v
	c3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	c3/kernel
o
c3/kernel/Read/ReadVariableOpReadVariableOp	c3/kernel*&
_output_shapes
:*
dtype0
f
c3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	c3/bias
_
c3/bias/Read/ReadVariableOpReadVariableOpc3/bias*
_output_shapes
:*
dtype0
o
	d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_name	d7/kernel
h
d7/kernel/Read/ReadVariableOpReadVariableOp	d7/kernel*
_output_shapes
:	А@*
dtype0
f
d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	d7/bias
_
d7/bias/Read/ReadVariableOpReadVariableOpd7/bias*
_output_shapes
:@*
dtype0
n
	d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*
shared_name	d9/kernel
g
d9/kernel/Read/ReadVariableOpReadVariableOp	d9/kernel*
_output_shapes

:@(*
dtype0
f
d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name	d9/bias
_
d9/bias/Read/ReadVariableOpReadVariableOpd9/bias*
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
shape:*!
shared_nameAdam/c0/kernel/m
}
$Adam/c0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/m*&
_output_shapes
:*
dtype0
t
Adam/c0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c0/bias/m
m
"Adam/c0/bias/m/Read/ReadVariableOpReadVariableOpAdam/c0/bias/m*
_output_shapes
:*
dtype0
Д
Adam/c3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/c3/kernel/m
}
$Adam/c3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c3/kernel/m*&
_output_shapes
:*
dtype0
t
Adam/c3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c3/bias/m
m
"Adam/c3/bias/m/Read/ReadVariableOpReadVariableOpAdam/c3/bias/m*
_output_shapes
:*
dtype0
}
Adam/d7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*!
shared_nameAdam/d7/kernel/m
v
$Adam/d7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d7/kernel/m*
_output_shapes
:	А@*
dtype0
t
Adam/d7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/d7/bias/m
m
"Adam/d7/bias/m/Read/ReadVariableOpReadVariableOpAdam/d7/bias/m*
_output_shapes
:@*
dtype0
|
Adam/d9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/d9/kernel/m
u
$Adam/d9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d9/kernel/m*
_output_shapes

:@(*
dtype0
t
Adam/d9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/d9/bias/m
m
"Adam/d9/bias/m/Read/ReadVariableOpReadVariableOpAdam/d9/bias/m*
_output_shapes
:(*
dtype0
Д
Adam/c0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/c0/kernel/v
}
$Adam/c0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/v*&
_output_shapes
:*
dtype0
t
Adam/c0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c0/bias/v
m
"Adam/c0/bias/v/Read/ReadVariableOpReadVariableOpAdam/c0/bias/v*
_output_shapes
:*
dtype0
Д
Adam/c3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/c3/kernel/v
}
$Adam/c3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c3/kernel/v*&
_output_shapes
:*
dtype0
t
Adam/c3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c3/bias/v
m
"Adam/c3/bias/v/Read/ReadVariableOpReadVariableOpAdam/c3/bias/v*
_output_shapes
:*
dtype0
}
Adam/d7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*!
shared_nameAdam/d7/kernel/v
v
$Adam/d7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d7/kernel/v*
_output_shapes
:	А@*
dtype0
t
Adam/d7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/d7/bias/v
m
"Adam/d7/bias/v/Read/ReadVariableOpReadVariableOpAdam/d7/bias/v*
_output_shapes
:@*
dtype0
|
Adam/d9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/d9/kernel/v
u
$Adam/d9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d9/kernel/v*
_output_shapes

:@(*
dtype0
t
Adam/d9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/d9/bias/v
m
"Adam/d9/bias/v/Read/ReadVariableOpReadVariableOpAdam/d9/bias/v*
_output_shapes
:(*
dtype0

NoOpNoOp
╤O
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*МO
valueВOB N B°N
╢
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
ж

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
О
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
е
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&_random_generator
'__call__
*(&call_and_return_all_conditional_losses* 
ж

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
О
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
е
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;_random_generator
<__call__
*=&call_and_return_all_conditional_losses* 
О
>	variables
?trainable_variables
@regularization_losses
A	keras_api
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
е
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P_random_generator
Q__call__
*R&call_and_return_all_conditional_losses* 
ж

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*
ф
[iter

\beta_1

]beta_2
	^decay
_learning_ratemлmм)mн*mоDmпEm░Sm▒Tm▓v│v┤)v╡*v╢Dv╖Ev╕Sv╣Tv║*
<
0
1
)2
*3
D4
E5
S6
T7*
<
0
1
)2
*3
D4
E5
S6
T7*
:
`0
a1
b2
c3
d4
e5
f6
g7* 
░
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

mserving_default* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

`0
a1* 
У
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
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
С
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
С
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
"	variables
#trainable_variables
$regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	c3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*

b0
c1* 
Х
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ц
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
7	variables
8trainable_variables
9regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ц
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUE	d7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*

d0
e1* 
Ш
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	d9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

S0
T1*

S0
T1*

f0
g1* 
Ш
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
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
J
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
9*

а0
б1*
* 
* 
* 
* 
* 
* 

`0
a1* 
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
b0
c1* 
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
d0
e1* 
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
f0
g1* 
* 
<

вtotal

гcount
д	variables
е	keras_api*
M

жtotal

зcount
и
_fn_kwargs
й	variables
к	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

в0
г1*

д	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ж0
з1*

й	variables*
|v
VARIABLE_VALUEAdam/c0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Л
serving_default_c0_inputPlaceholder*/
_output_shapes
:         @@*
dtype0*$
shape:         @@
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c3/kernelc3/bias	d7/kerneld7/bias	d9/kerneld9/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *,
f'R%
#__inference_signature_wrapper_74361
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamec0/kernel/Read/ReadVariableOpc0/bias/Read/ReadVariableOpc3/kernel/Read/ReadVariableOpc3/bias/Read/ReadVariableOpd7/kernel/Read/ReadVariableOpd7/bias/Read/ReadVariableOpd9/kernel/Read/ReadVariableOpd9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/c0/kernel/m/Read/ReadVariableOp"Adam/c0/bias/m/Read/ReadVariableOp$Adam/c3/kernel/m/Read/ReadVariableOp"Adam/c3/bias/m/Read/ReadVariableOp$Adam/d7/kernel/m/Read/ReadVariableOp"Adam/d7/bias/m/Read/ReadVariableOp$Adam/d9/kernel/m/Read/ReadVariableOp"Adam/d9/bias/m/Read/ReadVariableOp$Adam/c0/kernel/v/Read/ReadVariableOp"Adam/c0/bias/v/Read/ReadVariableOp$Adam/c3/kernel/v/Read/ReadVariableOp"Adam/c3/bias/v/Read/ReadVariableOp$Adam/d7/kernel/v/Read/ReadVariableOp"Adam/d7/bias/v/Read/ReadVariableOp$Adam/d9/kernel/v/Read/ReadVariableOp"Adam/d9/bias/v/Read/ReadVariableOpConst*.
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
__inference__traced_save_74731
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c3/kernelc3/bias	d7/kerneld7/bias	d9/kerneld9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/c0/kernel/mAdam/c0/bias/mAdam/c3/kernel/mAdam/c3/bias/mAdam/d7/kernel/mAdam/d7/bias/mAdam/d9/kernel/mAdam/d9/bias/mAdam/c0/kernel/vAdam/c0/bias/vAdam/c3/kernel/vAdam/c3/bias/vAdam/d7/kernel/vAdam/d7/bias/vAdam/d9/kernel/vAdam/d9/bias/v*-
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
!__inference__traced_restore_74840▀Н
╟D
п
__inference__traced_save_74731
file_prefix(
$savev2_c0_kernel_read_readvariableop&
"savev2_c0_bias_read_readvariableop(
$savev2_c3_kernel_read_readvariableop&
"savev2_c3_bias_read_readvariableop(
$savev2_d7_kernel_read_readvariableop&
"savev2_d7_bias_read_readvariableop(
$savev2_d9_kernel_read_readvariableop&
"savev2_d9_bias_read_readvariableop(
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
+savev2_adam_c3_kernel_m_read_readvariableop-
)savev2_adam_c3_bias_m_read_readvariableop/
+savev2_adam_d7_kernel_m_read_readvariableop-
)savev2_adam_d7_bias_m_read_readvariableop/
+savev2_adam_d9_kernel_m_read_readvariableop-
)savev2_adam_d9_bias_m_read_readvariableop/
+savev2_adam_c0_kernel_v_read_readvariableop-
)savev2_adam_c0_bias_v_read_readvariableop/
+savev2_adam_c3_kernel_v_read_readvariableop-
)savev2_adam_c3_bias_v_read_readvariableop/
+savev2_adam_d7_kernel_v_read_readvariableop-
)savev2_adam_d7_bias_v_read_readvariableop/
+savev2_adam_d9_kernel_v_read_readvariableop-
)savev2_adam_d9_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_c0_kernel_read_readvariableop"savev2_c0_bias_read_readvariableop$savev2_c3_kernel_read_readvariableop"savev2_c3_bias_read_readvariableop$savev2_d7_kernel_read_readvariableop"savev2_d7_bias_read_readvariableop$savev2_d9_kernel_read_readvariableop"savev2_d9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_c0_kernel_m_read_readvariableop)savev2_adam_c0_bias_m_read_readvariableop+savev2_adam_c3_kernel_m_read_readvariableop)savev2_adam_c3_bias_m_read_readvariableop+savev2_adam_d7_kernel_m_read_readvariableop)savev2_adam_d7_bias_m_read_readvariableop+savev2_adam_d9_kernel_m_read_readvariableop)savev2_adam_d9_bias_m_read_readvariableop+savev2_adam_c0_kernel_v_read_readvariableop)savev2_adam_c0_bias_v_read_readvariableop+savev2_adam_c3_kernel_v_read_readvariableop)savev2_adam_c3_bias_v_read_readvariableop+savev2_adam_d7_kernel_v_read_readvariableop)savev2_adam_d7_bias_v_read_readvariableop+savev2_adam_d9_kernel_v_read_readvariableop)savev2_adam_d9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*Ю
_input_shapesМ
Й: :::::	А@:@:@(:(: : : : : : : : : :::::	А@:@:@(:(:::::	А@:@:@(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	А@: 

_output_shapes
:@:$ 

_output_shapes

:@(: 

_output_shapes
:(:	
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	А@: 

_output_shapes
:@:$ 

_output_shapes

:@(: 

_output_shapes
:(:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	А@: 

_output_shapes
:@:$  

_output_shapes

:@(: !

_output_shapes
:(:"

_output_shapes
: 
╤
\
>__inference_dr8_layer_call_and_return_conditional_losses_73826

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
ь	
╠
,__inference_sequential_9_layer_call_fn_74223

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	А@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_74051o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_74395

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
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_73753

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @@`
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
:         @@w
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
─
+
__inference_loss_fn_3_74589
identity^
c3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
╛
Ў
=__inference_c3_layer_call_and_return_conditional_losses_73780

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
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
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         `
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▓2
Ъ
 __inference__wrapped_model_73709
c0_inputH
.sequential_9_c0_conv2d_readvariableop_resource:=
/sequential_9_c0_biasadd_readvariableop_resource:H
.sequential_9_c3_conv2d_readvariableop_resource:=
/sequential_9_c3_biasadd_readvariableop_resource:A
.sequential_9_d7_matmul_readvariableop_resource:	А@=
/sequential_9_d7_biasadd_readvariableop_resource:@@
.sequential_9_d9_matmul_readvariableop_resource:@(=
/sequential_9_d9_biasadd_readvariableop_resource:(
identityИв&sequential_9/c0/BiasAdd/ReadVariableOpв%sequential_9/c0/Conv2D/ReadVariableOpв&sequential_9/c3/BiasAdd/ReadVariableOpв%sequential_9/c3/Conv2D/ReadVariableOpв&sequential_9/d7/BiasAdd/ReadVariableOpв%sequential_9/d7/MatMul/ReadVariableOpв&sequential_9/d9/BiasAdd/ReadVariableOpв%sequential_9/d9/MatMul/ReadVariableOpЬ
%sequential_9/c0/Conv2D/ReadVariableOpReadVariableOp.sequential_9_c0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╗
sequential_9/c0/Conv2DConv2Dc0_input-sequential_9/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
Т
&sequential_9/c0/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0н
sequential_9/c0/BiasAddBiasAddsequential_9/c0/Conv2D:output:0.sequential_9/c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@x
sequential_9/c0/ReluRelu sequential_9/c0/BiasAdd:output:0*
T0*/
_output_shapes
:         @@│
sequential_9/m1/MaxPoolMaxPool"sequential_9/c0/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
Б
sequential_9/dr2/IdentityIdentity sequential_9/m1/MaxPool:output:0*
T0*/
_output_shapes
:         Ь
%sequential_9/c3/Conv2D/ReadVariableOpReadVariableOp.sequential_9_c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╒
sequential_9/c3/Conv2DConv2D"sequential_9/dr2/Identity:output:0-sequential_9/c3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
Т
&sequential_9/c3/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0н
sequential_9/c3/BiasAddBiasAddsequential_9/c3/Conv2D:output:0.sequential_9/c3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         x
sequential_9/c3/ReluRelu sequential_9/c3/BiasAdd:output:0*
T0*/
_output_shapes
:         │
sequential_9/m4/MaxPoolMaxPool"sequential_9/c3/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
Б
sequential_9/dr5/IdentityIdentity sequential_9/m4/MaxPool:output:0*
T0*/
_output_shapes
:         f
sequential_9/f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Щ
sequential_9/f6/ReshapeReshape"sequential_9/dr5/Identity:output:0sequential_9/f6/Const:output:0*
T0*(
_output_shapes
:         АХ
%sequential_9/d7/MatMul/ReadVariableOpReadVariableOp.sequential_9_d7_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0г
sequential_9/d7/MatMulMatMul sequential_9/f6/Reshape:output:0-sequential_9/d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Т
&sequential_9/d7/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
sequential_9/d7/BiasAddBiasAdd sequential_9/d7/MatMul:product:0.sequential_9/d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
sequential_9/d7/ReluRelu sequential_9/d7/BiasAdd:output:0*
T0*'
_output_shapes
:         @{
sequential_9/dr8/IdentityIdentity"sequential_9/d7/Relu:activations:0*
T0*'
_output_shapes
:         @Ф
%sequential_9/d9/MatMul/ReadVariableOpReadVariableOp.sequential_9_d9_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0е
sequential_9/d9/MatMulMatMul"sequential_9/dr8/Identity:output:0-sequential_9/d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (Т
&sequential_9/d9/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0ж
sequential_9/d9/BiasAddBiasAdd sequential_9/d9/MatMul:product:0.sequential_9/d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (v
sequential_9/d9/SoftmaxSoftmax sequential_9/d9/BiasAdd:output:0*
T0*'
_output_shapes
:         (p
IdentityIdentity!sequential_9/d9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         (К
NoOpNoOp'^sequential_9/c0/BiasAdd/ReadVariableOp&^sequential_9/c0/Conv2D/ReadVariableOp'^sequential_9/c3/BiasAdd/ReadVariableOp&^sequential_9/c3/Conv2D/ReadVariableOp'^sequential_9/d7/BiasAdd/ReadVariableOp&^sequential_9/d7/MatMul/ReadVariableOp'^sequential_9/d9/BiasAdd/ReadVariableOp&^sequential_9/d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 2P
&sequential_9/c0/BiasAdd/ReadVariableOp&sequential_9/c0/BiasAdd/ReadVariableOp2N
%sequential_9/c0/Conv2D/ReadVariableOp%sequential_9/c0/Conv2D/ReadVariableOp2P
&sequential_9/c3/BiasAdd/ReadVariableOp&sequential_9/c3/BiasAdd/ReadVariableOp2N
%sequential_9/c3/Conv2D/ReadVariableOp%sequential_9/c3/Conv2D/ReadVariableOp2P
&sequential_9/d7/BiasAdd/ReadVariableOp&sequential_9/d7/BiasAdd/ReadVariableOp2N
%sequential_9/d7/MatMul/ReadVariableOp%sequential_9/d7/MatMul/ReadVariableOp2P
&sequential_9/d9/BiasAdd/ReadVariableOp&sequential_9/d9/BiasAdd/ReadVariableOp2N
%sequential_9/d9/MatMul/ReadVariableOp%sequential_9/d9/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
у
Ч
"__inference_c0_layer_call_fn_74372

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_73753w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @@`
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
┬	
┼
#__inference_signature_wrapper_74361
c0_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	А@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *)
f$R"
 __inference__wrapped_model_73709o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
Ш
?
#__inference_dr8_layer_call_fn_74523

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
>__inference_dr8_layer_call_and_return_conditional_losses_73826`
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
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_2_74584
identity`
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
█
ю
=__inference_d9_layer_call_and_return_conditional_losses_74569

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
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
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
╜

]
>__inference_dr2_layer_call_and_return_conditional_losses_74422

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
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
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
:         w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╛
Р
"__inference_d7_layer_call_fn_74505

inputs
unknown:	А@
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
=__inference_d7_layer_call_and_return_conditional_losses_73815o
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
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
г
>
"__inference_m1_layer_call_fn_74390

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
=__inference_m1_layer_call_and_return_conditional_losses_73718Г
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
╜

]
>__inference_dr2_layer_call_and_return_conditional_losses_73977

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
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
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
:         w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
К
\
#__inference_dr5_layer_call_fn_74466

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
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_73944w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_74410

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ь	
╠
,__inference_sequential_9_layer_call_fn_74202

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	А@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_73856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Є	
╬
,__inference_sequential_9_layer_call_fn_73875
c0_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	А@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_73856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
─
+
__inference_loss_fn_1_74579
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
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_73718

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
К+
╗
G__inference_sequential_9_layer_call_and_return_conditional_losses_73856

inputs"
c0_73754:
c0_73756:"
c3_73781:
c3_73783:
d7_73816:	А@
d7_73818:@
d9_73842:@(
d9_73844:(
identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_73754c0_73756*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_73753╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_73718╤
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_73765·
c3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0c3_73781c3_73783*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_73780╫
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_73730╤
dr5/PartitionedCallPartitionedCallm4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_73792╔
f6/PartitionedCallPartitionedCalldr5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_73800ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_73816d7_73818*
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
=__inference_d7_layer_call_and_return_conditional_losses_73815╤
dr8/PartitionedCallPartitionedCall#d7/StatefulPartitionedCall:output:0*
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
>__inference_dr8_layer_call_and_return_conditional_losses_73826Є
d9/StatefulPartitionedCallStatefulPartitionedCalldr8/PartitionedCall:output:0d9_73842d9_73844*
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
=__inference_d9_layer_call_and_return_conditional_losses_73841`
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
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
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
 *    `
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (║
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
К
\
#__inference_dr2_layer_call_fn_74405

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
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_73977w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ў.
Х
G__inference_sequential_9_layer_call_and_return_conditional_losses_74051

inputs"
c0_74016:
c0_74018:"
c3_74023:
c3_74025:
d7_74031:	А@
d7_74033:@
d9_74037:@(
d9_74039:(
identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallвdr8/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_74016c0_74018*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_73753╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_73718с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_73977В
c3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0c3_74023c3_74025*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_73780╫
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_73730 
dr5/StatefulPartitionedCallStatefulPartitionedCallm4/PartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_73944╤
f6/PartitionedCallPartitionedCall$dr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_73800ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_74031d7_74033*
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
=__inference_d7_layer_call_and_return_conditional_losses_73815 
dr8/StatefulPartitionedCallStatefulPartitionedCall#d7/StatefulPartitionedCall:output:0^dr5/StatefulPartitionedCall*
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
>__inference_dr8_layer_call_and_return_conditional_losses_73905·
d9/StatefulPartitionedCallStatefulPartitionedCall$dr8/StatefulPartitionedCall:output:0d9_74037d9_74039*
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
=__inference_d9_layer_call_and_return_conditional_losses_73841`
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
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
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
 *    `
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (Ф
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall^dr8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall2:
dr8/StatefulPartitionedCalldr8/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
а.
я
G__inference_sequential_9_layer_call_and_return_conditional_losses_74270

inputs;
!c0_conv2d_readvariableop_resource:0
"c0_biasadd_readvariableop_resource:;
!c3_conv2d_readvariableop_resource:0
"c3_biasadd_readvariableop_resource:4
!d7_matmul_readvariableop_resource:	А@0
"d7_biasadd_readvariableop_resource:@3
!d9_matmul_readvariableop_resource:@(0
"d9_biasadd_readvariableop_resource:(
identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc3/BiasAdd/ReadVariableOpвc3/Conv2D/ReadVariableOpвd7/BiasAdd/ReadVariableOpвd7/MatMul/ReadVariableOpвd9/BiasAdd/ReadVariableOpвd9/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:         @@Щ

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
g
dr2/IdentityIdentitym1/MaxPool:output:0*
T0*/
_output_shapes
:         В
c3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0о
	c3/Conv2DConv2Ddr2/Identity:output:0 c3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
x
c3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c3/BiasAddBiasAddc3/Conv2D:output:0!c3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^
c3/ReluReluc3/BiasAdd:output:0*
T0*/
_output_shapes
:         Щ

m4/MaxPoolMaxPoolc3/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
g
dr5/IdentityIdentitym4/MaxPool:output:0*
T0*/
_output_shapes
:         Y
f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f6/ReshapeReshapedr5/Identity:output:0f6/Const:output:0*
T0*(
_output_shapes
:         А{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0|
	d7/MatMulMatMulf6/Reshape:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @V
d7/ReluRelud7/BiasAdd:output:0*
T0*'
_output_shapes
:         @a
dr8/IdentityIdentityd7/Relu:activations:0*
T0*'
_output_shapes
:         @z
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0~
	d9/MatMulMatMuldr8/Identity:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (x
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (\

d9/SoftmaxSoftmaxd9/BiasAdd:output:0*
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
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
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
 *    `
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         (в
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c3/BiasAdd/ReadVariableOp^c3/Conv2D/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c3/BiasAdd/ReadVariableOpc3/BiasAdd/ReadVariableOp24
c3/Conv2D/ReadVariableOpc3/Conv2D/ReadVariableOp26
d7/BiasAdd/ReadVariableOpd7/BiasAdd/ReadVariableOp24
d7/MatMul/ReadVariableOpd7/MatMul/ReadVariableOp26
d9/BiasAdd/ReadVariableOpd9/BiasAdd/ReadVariableOp24
d9/MatMul/ReadVariableOpd9/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
¤	
]
>__inference_dr8_layer_call_and_return_conditional_losses_74545

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
╕
?
#__inference_dr5_layer_call_fn_74461

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
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_73792h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
г
>
"__inference_m4_layer_call_fn_74451

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
=__inference_m4_layer_call_and_return_conditional_losses_73730Г
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
┐
Y
=__inference_f6_layer_call_and_return_conditional_losses_73800

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
¤	
]
>__inference_dr8_layer_call_and_return_conditional_losses_73905

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
╚
+
__inference_loss_fn_4_74594
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
╚
+
__inference_loss_fn_6_74604
identity`
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$d9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
─
+
__inference_loss_fn_7_74609
identity^
d9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"d9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_74385

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @@`
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
:         @@w
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
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_73765

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╜

]
>__inference_dr5_layer_call_and_return_conditional_losses_73944

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
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
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
:         w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
█
ю
=__inference_d9_layer_call_and_return_conditional_losses_73841

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
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
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
─
+
__inference_loss_fn_5_74599
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
╛C
я
G__inference_sequential_9_layer_call_and_return_conditional_losses_74338

inputs;
!c0_conv2d_readvariableop_resource:0
"c0_biasadd_readvariableop_resource:;
!c3_conv2d_readvariableop_resource:0
"c3_biasadd_readvariableop_resource:4
!d7_matmul_readvariableop_resource:	А@0
"d7_biasadd_readvariableop_resource:@3
!d9_matmul_readvariableop_resource:@(0
"d9_biasadd_readvariableop_resource:(
identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc3/BiasAdd/ReadVariableOpвc3/Conv2D/ReadVariableOpвd7/BiasAdd/ReadVariableOpвd7/MatMul/ReadVariableOpвd9/BiasAdd/ReadVariableOpвd9/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:         @@Щ

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:         *
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
:         T
dr2/dropout/ShapeShapem1/MaxPool:output:0*
T0*
_output_shapes
:н
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*/
_output_shapes
:         *
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
:         
dr2/dropout/CastCastdr2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         }
dr2/dropout/Mul_1Muldr2/dropout/Mul:z:0dr2/dropout/Cast:y:0*
T0*/
_output_shapes
:         В
c3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0о
	c3/Conv2DConv2Ddr2/dropout/Mul_1:z:0 c3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
x
c3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

c3/BiasAddBiasAddc3/Conv2D:output:0!c3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^
c3/ReluReluc3/BiasAdd:output:0*
T0*/
_output_shapes
:         Щ

m4/MaxPoolMaxPoolc3/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
V
dr5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Б
dr5/dropout/MulMulm4/MaxPool:output:0dr5/dropout/Const:output:0*
T0*/
_output_shapes
:         T
dr5/dropout/ShapeShapem4/MaxPool:output:0*
T0*
_output_shapes
:й
(dr5/dropout/random_uniform/RandomUniformRandomUniformdr5/dropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed2_
dr5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>║
dr5/dropout/GreaterEqualGreaterEqual1dr5/dropout/random_uniform/RandomUniform:output:0#dr5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         
dr5/dropout/CastCastdr5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         }
dr5/dropout/Mul_1Muldr5/dropout/Mul:z:0dr5/dropout/Cast:y:0*
T0*/
_output_shapes
:         Y
f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f6/ReshapeReshapedr5/dropout/Mul_1:z:0f6/Const:output:0*
T0*(
_output_shapes
:         А{
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0|
	d7/MatMulMatMulf6/Reshape:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @V
d7/ReluRelud7/BiasAdd:output:0*
T0*'
_output_shapes
:         @V
dr8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?{
dr8/dropout/MulMuld7/Relu:activations:0dr8/dropout/Const:output:0*
T0*'
_output_shapes
:         @V
dr8/dropout/ShapeShaped7/Relu:activations:0*
T0*
_output_shapes
:б
(dr8/dropout/random_uniform/RandomUniformRandomUniformdr8/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2_
dr8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>▓
dr8/dropout/GreaterEqualGreaterEqual1dr8/dropout/random_uniform/RandomUniform:output:0#dr8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @w
dr8/dropout/CastCastdr8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @u
dr8/dropout/Mul_1Muldr8/dropout/Mul:z:0dr8/dropout/Cast:y:0*
T0*'
_output_shapes
:         @z
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0~
	d9/MatMulMatMuldr8/dropout/Mul_1:z:0 d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (x
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (\

d9/SoftmaxSoftmaxd9/BiasAdd:output:0*
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
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
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
 *    `
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentityd9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         (в
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c3/BiasAdd/ReadVariableOp^c3/Conv2D/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c3/BiasAdd/ReadVariableOpc3/BiasAdd/ReadVariableOp24
c3/Conv2D/ReadVariableOpc3/Conv2D/ReadVariableOp26
d7/BiasAdd/ReadVariableOpd7/BiasAdd/ReadVariableOp24
d7/MatMul/ReadVariableOpd7/MatMul/ReadVariableOp26
d9/BiasAdd/ReadVariableOpd9/BiasAdd/ReadVariableOp24
d9/MatMul/ReadVariableOpd9/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_0_74574
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
ЫВ
ь
!__inference__traced_restore_74840
file_prefix4
assignvariableop_c0_kernel:(
assignvariableop_1_c0_bias:6
assignvariableop_2_c3_kernel:(
assignvariableop_3_c3_bias:/
assignvariableop_4_d7_kernel:	А@(
assignvariableop_5_d7_bias:@.
assignvariableop_6_d9_kernel:@((
assignvariableop_7_d9_bias:(&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: >
$assignvariableop_17_adam_c0_kernel_m:0
"assignvariableop_18_adam_c0_bias_m:>
$assignvariableop_19_adam_c3_kernel_m:0
"assignvariableop_20_adam_c3_bias_m:7
$assignvariableop_21_adam_d7_kernel_m:	А@0
"assignvariableop_22_adam_d7_bias_m:@6
$assignvariableop_23_adam_d9_kernel_m:@(0
"assignvariableop_24_adam_d9_bias_m:(>
$assignvariableop_25_adam_c0_kernel_v:0
"assignvariableop_26_adam_c0_bias_v:>
$assignvariableop_27_adam_c3_kernel_v:0
"assignvariableop_28_adam_c3_bias_v:7
$assignvariableop_29_adam_d7_kernel_v:	А@0
"assignvariableop_30_adam_d7_bias_v:@6
$assignvariableop_31_adam_d9_kernel_v:@(0
"assignvariableop_32_adam_d9_bias_v:(
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
AssignVariableOp_2AssignVariableOpassignvariableop_2_c3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_3AssignVariableOpassignvariableop_3_c3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_4AssignVariableOpassignvariableop_4_d7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_5AssignVariableOpassignvariableop_5_d7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_d9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_7AssignVariableOpassignvariableop_7_d9_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_c3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_c3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_d7_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_d7_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_d9_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_d9_bias_mIdentity_24:output:0"/device:CPU:0*
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
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_c3_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_c3_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_29AssignVariableOp$assignvariableop_29_adam_d7_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_d7_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_d9_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_d9_bias_vIdentity_32:output:0"/device:CPU:0*
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
╜

]
>__inference_dr5_layer_call_and_return_conditional_losses_74483

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
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
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
:         w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Е
Y
=__inference_m4_layer_call_and_return_conditional_losses_73730

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
┌
я
=__inference_d7_layer_call_and_return_conditional_losses_73815

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
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
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Е
Y
=__inference_m4_layer_call_and_return_conditional_losses_74456

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
╤
\
>__inference_dr8_layer_call_and_return_conditional_losses_74533

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
у
Ч
"__inference_c3_layer_call_fn_74433

inputs!
unknown:
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
:         *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_73780w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Є	
╬
,__inference_sequential_9_layer_call_fn_74091
c0_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	А@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         (**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_74051o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
и
>
"__inference_f6_layer_call_fn_74488

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
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_73800a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Р+
╜
G__inference_sequential_9_layer_call_and_return_conditional_losses_74129
c0_input"
c0_74094:
c0_74096:"
c3_74101:
c3_74103:
d7_74109:	А@
d7_74111:@
d9_74115:@(
d9_74117:(
identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_74094c0_74096*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_73753╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_73718╤
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_73765·
c3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0c3_74101c3_74103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_73780╫
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_73730╤
dr5/PartitionedCallPartitionedCallm4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_73792╔
f6/PartitionedCallPartitionedCalldr5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_73800ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_74109d7_74111*
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
=__inference_d7_layer_call_and_return_conditional_losses_73815╤
dr8/PartitionedCallPartitionedCall#d7/StatefulPartitionedCall:output:0*
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
>__inference_dr8_layer_call_and_return_conditional_losses_73826Є
d9/StatefulPartitionedCallStatefulPartitionedCalldr8/PartitionedCall:output:0d9_74115d9_74117*
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
=__inference_d9_layer_call_and_return_conditional_losses_73841`
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
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
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
 *    `
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (║
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input
ё
\
>__inference_dr5_layer_call_and_return_conditional_losses_73792

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┐
Y
=__inference_f6_layer_call_and_return_conditional_losses_74494

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┌
я
=__inference_d7_layer_call_and_return_conditional_losses_74518

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
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
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ъ
\
#__inference_dr8_layer_call_fn_74528

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
>__inference_dr8_layer_call_and_return_conditional_losses_73905o
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
ё
\
>__inference_dr5_layer_call_and_return_conditional_losses_74471

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╛
Ў
=__inference_c3_layer_call_and_return_conditional_losses_74446

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
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
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         `
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╗
П
"__inference_d9_layer_call_fn_74556

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
=__inference_d9_layer_call_and_return_conditional_losses_73841o
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
╕
?
#__inference_dr2_layer_call_fn_74400

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
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_73765h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
№.
Ч
G__inference_sequential_9_layer_call_and_return_conditional_losses_74167
c0_input"
c0_74132:
c0_74134:"
c3_74139:
c3_74141:
d7_74147:	А@
d7_74149:@
d9_74153:@(
d9_74155:(
identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallвdr8/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_74132c0_74134*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_73753╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_73718с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_73977В
c3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0c3_74139c3_74141*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_73780╫
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_73730 
dr5/StatefulPartitionedCallStatefulPartitionedCallm4/PartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_73944╤
f6/PartitionedCallPartitionedCall$dr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_73800ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_74147d7_74149*
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
=__inference_d7_layer_call_and_return_conditional_losses_73815 
dr8/StatefulPartitionedCallStatefulPartitionedCall#d7/StatefulPartitionedCall:output:0^dr5/StatefulPartitionedCall*
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
>__inference_dr8_layer_call_and_return_conditional_losses_73905·
d9/StatefulPartitionedCallStatefulPartitionedCall$dr8/StatefulPartitionedCall:output:0d9_74153d9_74155*
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
=__inference_d9_layer_call_and_return_conditional_losses_73841`
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
c3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c3/bias/Regularizer/ConstConst*
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
 *    `
d9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
d9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
IdentityIdentity#d9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         (Ф
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall^dr8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         @@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall2:
dr8/StatefulPartitionedCalldr8/StatefulPartitionedCall:Y U
/
_output_shapes
:         @@
"
_user_specified_name
c0_input"█L
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
d90
StatefulPartitionedCall:0         (tensorflow/serving/predict:┬┴
╨
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
е
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&_random_generator
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
е
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;_random_generator
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
е
>	variables
?trainable_variables
@regularization_losses
A	keras_api
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
╝
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P_random_generator
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
є
[iter

\beta_1

]beta_2
	^decay
_learning_ratemлmм)mн*mоDmпEm░Sm▒Tm▓v│v┤)v╡*v╢Dv╖Ev╕Sv╣Tv║"
	optimizer
X
0
1
)2
*3
D4
E5
S6
T7"
trackable_list_wrapper
X
0
1
)2
*3
D4
E5
S6
T7"
trackable_list_wrapper
X
`0
a1
b2
c3
d4
e5
f6
g7"
trackable_list_wrapper
╩
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
■2√
,__inference_sequential_9_layer_call_fn_73875
,__inference_sequential_9_layer_call_fn_74202
,__inference_sequential_9_layer_call_fn_74223
,__inference_sequential_9_layer_call_fn_74091└
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_74270
G__inference_sequential_9_layer_call_and_return_conditional_losses_74338
G__inference_sequential_9_layer_call_and_return_conditional_losses_74129
G__inference_sequential_9_layer_call_and_return_conditional_losses_74167└
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
 __inference__wrapped_model_73709c0_input"Ш
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
mserving_default"
signature_map
#:!2	c0/kernel
:2c0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
н
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c0_layer_call_fn_74372в
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
=__inference_c0_layer_call_and_return_conditional_losses_74385в
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
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_m1_layer_call_fn_74390в
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
=__inference_m1_layer_call_and_return_conditional_losses_74395в
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
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
"	variables
#trainable_variables
$regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Д2Б
#__inference_dr2_layer_call_fn_74400
#__inference_dr2_layer_call_fn_74405┤
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
>__inference_dr2_layer_call_and_return_conditional_losses_74410
>__inference_dr2_layer_call_and_return_conditional_losses_74422┤
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
#:!2	c3/kernel
:2c3/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
п
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c3_layer_call_fn_74433в
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
=__inference_c3_layer_call_and_return_conditional_losses_74446в
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
▓
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_m4_layer_call_fn_74451в
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
=__inference_m4_layer_call_and_return_conditional_losses_74456в
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
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
7	variables
8trainable_variables
9regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Д2Б
#__inference_dr5_layer_call_fn_74461
#__inference_dr5_layer_call_fn_74466┤
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
>__inference_dr5_layer_call_and_return_conditional_losses_74471
>__inference_dr5_layer_call_and_return_conditional_losses_74483┤
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
▓
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_f6_layer_call_fn_74488в
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
=__inference_f6_layer_call_and_return_conditional_losses_74494в
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
:	А@2	d7/kernel
:@2d7/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
▓
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_d7_layer_call_fn_74505в
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
=__inference_d7_layer_call_and_return_conditional_losses_74518в
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
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Д2Б
#__inference_dr8_layer_call_fn_74523
#__inference_dr8_layer_call_fn_74528┤
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
>__inference_dr8_layer_call_and_return_conditional_losses_74533
>__inference_dr8_layer_call_and_return_conditional_losses_74545┤
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
:@(2	d9/kernel
:(2d9/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
▓
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_d9_layer_call_fn_74556в
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
=__inference_d9_layer_call_and_return_conditional_losses_74569в
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
__inference_loss_fn_0_74574П
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
__inference_loss_fn_1_74579П
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
__inference_loss_fn_2_74584П
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
__inference_loss_fn_3_74589П
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
__inference_loss_fn_4_74594П
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
__inference_loss_fn_5_74599П
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
__inference_loss_fn_6_74604П
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
__inference_loss_fn_7_74609П
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
f
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
9"
trackable_list_wrapper
0
а0
б1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╦B╚
#__inference_signature_wrapper_74361c0_input"Ф
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
`0
a1"
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
b0
c1"
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
d0
e1"
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
f0
g1"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

вtotal

гcount
д	variables
е	keras_api"
_tf_keras_metric
c

жtotal

зcount
и
_fn_kwargs
й	variables
к	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
в0
г1"
trackable_list_wrapper
.
д	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ж0
з1"
trackable_list_wrapper
.
й	variables"
_generic_user_object
(:&2Adam/c0/kernel/m
:2Adam/c0/bias/m
(:&2Adam/c3/kernel/m
:2Adam/c3/bias/m
!:	А@2Adam/d7/kernel/m
:@2Adam/d7/bias/m
 :@(2Adam/d9/kernel/m
:(2Adam/d9/bias/m
(:&2Adam/c0/kernel/v
:2Adam/c0/bias/v
(:&2Adam/c3/kernel/v
:2Adam/c3/bias/v
!:	А@2Adam/d7/kernel/v
:@2Adam/d7/bias/v
 :@(2Adam/d9/kernel/v
:(2Adam/d9/bias/vТ
 __inference__wrapped_model_73709n)*DEST9в6
/в,
*К'
c0_input         @@
к "'к$
"
d9К
d9         (н
=__inference_c0_layer_call_and_return_conditional_losses_74385l7в4
-в*
(К%
inputs         @@
к "-в*
#К 
0         @@
Ъ Е
"__inference_c0_layer_call_fn_74372_7в4
-в*
(К%
inputs         @@
к " К         @@н
=__inference_c3_layer_call_and_return_conditional_losses_74446l)*7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ Е
"__inference_c3_layer_call_fn_74433_)*7в4
-в*
(К%
inputs         
к " К         Ю
=__inference_d7_layer_call_and_return_conditional_losses_74518]DE0в-
&в#
!К
inputs         А
к "%в"
К
0         @
Ъ v
"__inference_d7_layer_call_fn_74505PDE0в-
&в#
!К
inputs         А
к "К         @Э
=__inference_d9_layer_call_and_return_conditional_losses_74569\ST/в,
%в"
 К
inputs         @
к "%в"
К
0         (
Ъ u
"__inference_d9_layer_call_fn_74556OST/в,
%в"
 К
inputs         @
к "К         (о
>__inference_dr2_layer_call_and_return_conditional_losses_74410l;в8
1в.
(К%
inputs         
p 
к "-в*
#К 
0         
Ъ о
>__inference_dr2_layer_call_and_return_conditional_losses_74422l;в8
1в.
(К%
inputs         
p
к "-в*
#К 
0         
Ъ Ж
#__inference_dr2_layer_call_fn_74400_;в8
1в.
(К%
inputs         
p 
к " К         Ж
#__inference_dr2_layer_call_fn_74405_;в8
1в.
(К%
inputs         
p
к " К         о
>__inference_dr5_layer_call_and_return_conditional_losses_74471l;в8
1в.
(К%
inputs         
p 
к "-в*
#К 
0         
Ъ о
>__inference_dr5_layer_call_and_return_conditional_losses_74483l;в8
1в.
(К%
inputs         
p
к "-в*
#К 
0         
Ъ Ж
#__inference_dr5_layer_call_fn_74461_;в8
1в.
(К%
inputs         
p 
к " К         Ж
#__inference_dr5_layer_call_fn_74466_;в8
1в.
(К%
inputs         
p
к " К         Ю
>__inference_dr8_layer_call_and_return_conditional_losses_74533\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ Ю
>__inference_dr8_layer_call_and_return_conditional_losses_74545\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ v
#__inference_dr8_layer_call_fn_74523O3в0
)в&
 К
inputs         @
p 
к "К         @v
#__inference_dr8_layer_call_fn_74528O3в0
)в&
 К
inputs         @
p
к "К         @в
=__inference_f6_layer_call_and_return_conditional_losses_74494a7в4
-в*
(К%
inputs         
к "&в#
К
0         А
Ъ z
"__inference_f6_layer_call_fn_74488T7в4
-в*
(К%
inputs         
к "К         А7
__inference_loss_fn_0_74574в

в 
к "К 7
__inference_loss_fn_1_74579в

в 
к "К 7
__inference_loss_fn_2_74584в

в 
к "К 7
__inference_loss_fn_3_74589в

в 
к "К 7
__inference_loss_fn_4_74594в

в 
к "К 7
__inference_loss_fn_5_74599в

в 
к "К 7
__inference_loss_fn_6_74604в

в 
к "К 7
__inference_loss_fn_7_74609в

в 
к "К р
=__inference_m1_layer_call_and_return_conditional_losses_74395ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_m1_layer_call_fn_74390СRвO
HвE
CК@
inputs4                                    
к ";К84                                    р
=__inference_m4_layer_call_and_return_conditional_losses_74456ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_m4_layer_call_fn_74451СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ┐
G__inference_sequential_9_layer_call_and_return_conditional_losses_74129t)*DESTAв>
7в4
*К'
c0_input         @@
p 

 
к "%в"
К
0         (
Ъ ┐
G__inference_sequential_9_layer_call_and_return_conditional_losses_74167t)*DESTAв>
7в4
*К'
c0_input         @@
p

 
к "%в"
К
0         (
Ъ ╜
G__inference_sequential_9_layer_call_and_return_conditional_losses_74270r)*DEST?в<
5в2
(К%
inputs         @@
p 

 
к "%в"
К
0         (
Ъ ╜
G__inference_sequential_9_layer_call_and_return_conditional_losses_74338r)*DEST?в<
5в2
(К%
inputs         @@
p

 
к "%в"
К
0         (
Ъ Ч
,__inference_sequential_9_layer_call_fn_73875g)*DESTAв>
7в4
*К'
c0_input         @@
p 

 
к "К         (Ч
,__inference_sequential_9_layer_call_fn_74091g)*DESTAв>
7в4
*К'
c0_input         @@
p

 
к "К         (Х
,__inference_sequential_9_layer_call_fn_74202e)*DEST?в<
5в2
(К%
inputs         @@
p 

 
к "К         (Х
,__inference_sequential_9_layer_call_fn_74223e)*DEST?в<
5в2
(К%
inputs         @@
p

 
к "К         (б
#__inference_signature_wrapper_74361z)*DESTEвB
в 
;к8
6
c0_input*К'
c0_input         @@"'к$
"
d9К
d9         (
н█
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
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68·▌

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
shape:  *
shared_name	c1/kernel
o
c1/kernel/Read/ReadVariableOpReadVariableOp	c1/kernel*&
_output_shapes
:  *
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
p
	d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А А*
shared_name	d9/kernel
i
d9/kernel/Read/ReadVariableOpReadVariableOp	d9/kernel* 
_output_shapes
:
А А*
dtype0
g
d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	d9/bias
`
d9/bias/Read/ReadVariableOpReadVariableOpd9/bias*
_output_shapes	
:А*
dtype0
q

d11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*
shared_name
d11/kernel
j
d11/kernel/Read/ReadVariableOpReadVariableOp
d11/kernel*
_output_shapes
:	А
*
dtype0
h
d11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
d11/bias
a
d11/bias/Read/ReadVariableOpReadVariableOpd11/bias*
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
shape:  *!
shared_nameAdam/c1/kernel/m
}
$Adam/c1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c1/kernel/m*&
_output_shapes
:  *
dtype0
t
Adam/c1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/c1/bias/m
m
"Adam/c1/bias/m/Read/ReadVariableOpReadVariableOpAdam/c1/bias/m*
_output_shapes
: *
dtype0
Д
Adam/c4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameAdam/c4/kernel/m
}
$Adam/c4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c4/kernel/m*&
_output_shapes
: @*
dtype0
t
Adam/c4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/c4/bias/m
m
"Adam/c4/bias/m/Read/ReadVariableOpReadVariableOpAdam/c4/bias/m*
_output_shapes
:@*
dtype0
Д
Adam/c5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameAdam/c5/kernel/m
}
$Adam/c5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c5/kernel/m*&
_output_shapes
:@@*
dtype0
t
Adam/c5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/c5/bias/m
m
"Adam/c5/bias/m/Read/ReadVariableOpReadVariableOpAdam/c5/bias/m*
_output_shapes
:@*
dtype0
~
Adam/d9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А А*!
shared_nameAdam/d9/kernel/m
w
$Adam/d9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d9/kernel/m* 
_output_shapes
:
А А*
dtype0
u
Adam/d9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/d9/bias/m
n
"Adam/d9/bias/m/Read/ReadVariableOpReadVariableOpAdam/d9/bias/m*
_output_shapes	
:А*
dtype0

Adam/d11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*"
shared_nameAdam/d11/kernel/m
x
%Adam/d11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d11/kernel/m*
_output_shapes
:	А
*
dtype0
v
Adam/d11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/d11/bias/m
o
#Adam/d11/bias/m/Read/ReadVariableOpReadVariableOpAdam/d11/bias/m*
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
shape:  *!
shared_nameAdam/c1/kernel/v
}
$Adam/c1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c1/kernel/v*&
_output_shapes
:  *
dtype0
t
Adam/c1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/c1/bias/v
m
"Adam/c1/bias/v/Read/ReadVariableOpReadVariableOpAdam/c1/bias/v*
_output_shapes
: *
dtype0
Д
Adam/c4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameAdam/c4/kernel/v
}
$Adam/c4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c4/kernel/v*&
_output_shapes
: @*
dtype0
t
Adam/c4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/c4/bias/v
m
"Adam/c4/bias/v/Read/ReadVariableOpReadVariableOpAdam/c4/bias/v*
_output_shapes
:@*
dtype0
Д
Adam/c5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameAdam/c5/kernel/v
}
$Adam/c5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c5/kernel/v*&
_output_shapes
:@@*
dtype0
t
Adam/c5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/c5/bias/v
m
"Adam/c5/bias/v/Read/ReadVariableOpReadVariableOpAdam/c5/bias/v*
_output_shapes
:@*
dtype0
~
Adam/d9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А А*!
shared_nameAdam/d9/kernel/v
w
$Adam/d9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d9/kernel/v* 
_output_shapes
:
А А*
dtype0
u
Adam/d9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/d9/bias/v
n
"Adam/d9/bias/v/Read/ReadVariableOpReadVariableOpAdam/d9/bias/v*
_output_shapes	
:А*
dtype0

Adam/d11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*"
shared_nameAdam/d11/kernel/v
x
%Adam/d11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d11/kernel/v*
_output_shapes
:	А
*
dtype0
v
Adam/d11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/d11/bias/v
o
#Adam/d11/bias/v/Read/ReadVariableOpReadVariableOpAdam/d11/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
Ыe
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╓d
value╠dB╔d B┬d
Ж
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
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
ж

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
ж

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
О
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
е
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses* 
ж

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*
ж

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
О
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
е
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M_random_generator
N__call__
*O&call_and_return_all_conditional_losses* 
О
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
ж

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses*
е
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b_random_generator
c__call__
*d&call_and_return_all_conditional_losses* 
ж

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
┤
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem╦m╠m═m╬3m╧4m╨;m╤<m╥Vm╙Wm╘em╒fm╓v╫v╪v┘v┌3v█4v▄;v▌<v▐Vv▀Wvрevсfvт*
Z
0
1
2
3
34
45
;6
<7
V8
W9
e10
f11*
Z
0
1
2
3
34
45
;6
<7
V8
W9
e10
f11*
X
r0
s1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11* 
│
~non_trainable_variables

layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Гserving_default* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

r0
s1* 
Ш
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUE	c1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

t0
u1* 
Ш
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ц
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	c4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*

v0
w1* 
Ш
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUE	c5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*

x0
y1* 
Ш
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ц
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ц
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 
* 
* 
YS
VARIABLE_VALUE	d9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*

z0
{1* 
Ш
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
^	variables
_trainable_variables
`regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 
* 
* 
* 
ZT
VARIABLE_VALUE
d11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEd11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

e0
f1*

|0
}1* 
Ш
╗non_trainable_variables
╝layers
╜metrics
 ╛layer_regularization_losses
┐layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
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
* 
* 
* 
* 
Z
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
11*

└0
┴1*
* 
* 
* 
* 
* 
* 

r0
s1* 
* 
* 
* 
* 

t0
u1* 
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
v0
w1* 
* 
* 
* 
* 

x0
y1* 
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
z0
{1* 
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
|0
}1* 
* 
<

┬total

├count
─	variables
┼	keras_api*
M

╞total

╟count
╚
_fn_kwargs
╔	variables
╩	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

┬0
├1*

─	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

╞0
╟1*

╔	variables*
|v
VARIABLE_VALUEAdam/c0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c5/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c5/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d9/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d9/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/d11/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/d11/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c5/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c5/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/d9/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/d9/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/d11/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/d11/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Л
serving_default_c0_inputPlaceholder*/
_output_shapes
:           *
dtype0*$
shape:           
╦
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	d9/kerneld9/bias
d11/kerneld11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8В *,
f'R%
#__inference_signature_wrapper_30691
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
═
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamec0/kernel/Read/ReadVariableOpc0/bias/Read/ReadVariableOpc1/kernel/Read/ReadVariableOpc1/bias/Read/ReadVariableOpc4/kernel/Read/ReadVariableOpc4/bias/Read/ReadVariableOpc5/kernel/Read/ReadVariableOpc5/bias/Read/ReadVariableOpd9/kernel/Read/ReadVariableOpd9/bias/Read/ReadVariableOpd11/kernel/Read/ReadVariableOpd11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/c0/kernel/m/Read/ReadVariableOp"Adam/c0/bias/m/Read/ReadVariableOp$Adam/c1/kernel/m/Read/ReadVariableOp"Adam/c1/bias/m/Read/ReadVariableOp$Adam/c4/kernel/m/Read/ReadVariableOp"Adam/c4/bias/m/Read/ReadVariableOp$Adam/c5/kernel/m/Read/ReadVariableOp"Adam/c5/bias/m/Read/ReadVariableOp$Adam/d9/kernel/m/Read/ReadVariableOp"Adam/d9/bias/m/Read/ReadVariableOp%Adam/d11/kernel/m/Read/ReadVariableOp#Adam/d11/bias/m/Read/ReadVariableOp$Adam/c0/kernel/v/Read/ReadVariableOp"Adam/c0/bias/v/Read/ReadVariableOp$Adam/c1/kernel/v/Read/ReadVariableOp"Adam/c1/bias/v/Read/ReadVariableOp$Adam/c4/kernel/v/Read/ReadVariableOp"Adam/c4/bias/v/Read/ReadVariableOp$Adam/c5/kernel/v/Read/ReadVariableOp"Adam/c5/bias/v/Read/ReadVariableOp$Adam/d9/kernel/v/Read/ReadVariableOp"Adam/d9/bias/v/Read/ReadVariableOp%Adam/d11/kernel/v/Read/ReadVariableOp#Adam/d11/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
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
__inference__traced_save_31165
─
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	d9/kerneld9/bias
d11/kerneld11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/c0/kernel/mAdam/c0/bias/mAdam/c1/kernel/mAdam/c1/bias/mAdam/c4/kernel/mAdam/c4/bias/mAdam/c5/kernel/mAdam/c5/bias/mAdam/d9/kernel/mAdam/d9/bias/mAdam/d11/kernel/mAdam/d11/bias/mAdam/c0/kernel/vAdam/c0/bias/vAdam/c1/kernel/vAdam/c1/bias/vAdam/c4/kernel/vAdam/c4/bias/vAdam/c5/kernel/vAdam/c5/bias/vAdam/d9/kernel/vAdam/d9/bias/vAdam/d11/kernel/vAdam/d11/bias/v*9
Tin2
02.*
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
!__inference__traced_restore_31310ъМ	
╛
Ў
=__inference_c1_layer_call_and_return_conditional_losses_30739

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
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
:            w
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
__inference_loss_fn_8_30992
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
л
╠
*__inference_sequential_layer_call_fn_30055
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
А А
	unknown_8:	А
	unknown_9:	А


unknown_10:

identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30028o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
─
+
__inference_loss_fn_9_30997
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
Е
Y
=__inference_m6_layer_call_and_return_conditional_losses_29860

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
╕
?
#__inference_dr7_layer_call_fn_30839

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
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_29960h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Е
Y
=__inference_m6_layer_call_and_return_conditional_losses_30834

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
ЩA
└
E__inference_sequential_layer_call_and_return_conditional_losses_30574

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource:  0
"c1_biasadd_readvariableop_resource: ;
!c4_conv2d_readvariableop_resource: @0
"c4_biasadd_readvariableop_resource:@;
!c5_conv2d_readvariableop_resource:@@0
"c5_biasadd_readvariableop_resource:@5
!d9_matmul_readvariableop_resource:
А А1
"d9_biasadd_readvariableop_resource:	А5
"d11_matmul_readvariableop_resource:	А
1
#d11_biasadd_readvariableop_resource:

identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc1/BiasAdd/ReadVariableOpвc1/Conv2D/ReadVariableOpвc4/BiasAdd/ReadVariableOpвc4/Conv2D/ReadVariableOpвc5/BiasAdd/ReadVariableOpвc5/Conv2D/ReadVariableOpвd11/BiasAdd/ReadVariableOpвd11/MatMul/ReadVariableOpвd9/BiasAdd/ReadVariableOpвd9/MatMul/ReadVariableOpВ
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
:  *
dtype0о
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ж

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:            Щ

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
g
dr3/IdentityIdentitym2/MaxPool:output:0*
T0*/
_output_shapes
:          В
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0о
	c4/Conv2DConv2Ddr3/Identity:output:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
x
c4/BiasAdd/ReadVariableOpReadVariableOp"c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж

c4/BiasAddBiasAddc4/Conv2D:output:0!c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:         @В
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
x
c5/BiasAdd/ReadVariableOpReadVariableOp"c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж

c5/BiasAddBiasAddc5/Conv2D:output:0!c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:         @Щ

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
dr7/IdentityIdentitym6/MaxPool:output:0*
T0*/
_output_shapes
:         @Y
f8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f8/ReshapeReshapedr7/Identity:output:0f8/Const:output:0*
T0*(
_output_shapes
:         А |
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0}
	d9/MatMulMatMulf8/Reshape:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0А

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АW
d9/ReluRelud9/BiasAdd:output:0*
T0*(
_output_shapes
:         Аc
dr10/IdentityIdentityd9/Relu:activations:0*
T0*(
_output_shapes
:         А}
d11/MatMul/ReadVariableOpReadVariableOp"d11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Б

d11/MatMulMatMuldr10/Identity:output:0!d11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
z
d11/BiasAdd/ReadVariableOpReadVariableOp#d11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0В
d11/BiasAddBiasAddd11/MatMul:product:0"d11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
^
d11/SoftmaxSoftmaxd11/BiasAdd:output:0*
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
 *    a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentityd11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Т
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^d11/BiasAdd/ReadVariableOp^d11/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
c4/BiasAdd/ReadVariableOpc4/BiasAdd/ReadVariableOp24
c4/Conv2D/ReadVariableOpc4/Conv2D/ReadVariableOp26
c5/BiasAdd/ReadVariableOpc5/BiasAdd/ReadVariableOp24
c5/Conv2D/ReadVariableOpc5/Conv2D/ReadVariableOp28
d11/BiasAdd/ReadVariableOpd11/BiasAdd/ReadVariableOp26
d11/MatMul/ReadVariableOpd11/MatMul/ReadVariableOp26
d9/BiasAdd/ReadVariableOpd9/BiasAdd/ReadVariableOp24
d9/MatMul/ReadVariableOpd9/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
у
Ч
"__inference_c5_layer_call_fn_30811

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_29948w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
е
╩
*__inference_sequential_layer_call_fn_30509

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
А А
	unknown_8:	А
	unknown_9:	А


unknown_10:

identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30273o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
┬
Т
"__inference_d9_layer_call_fn_30883

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
=__inference_d9_layer_call_and_return_conditional_losses_29983p
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
С7
ё
E__inference_sequential_layer_call_and_return_conditional_losses_30028

inputs"
c0_29884: 
c0_29886: "
c1_29903:  
c1_29905: "
c4_29930: @
c4_29932:@"
c5_29949:@@
c5_29951:@
d9_29984:
А А
d9_29986:	А
	d11_30010:	А

	d11_30012:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвc4/StatefulPartitionedCallвc5/StatefulPartitionedCallвd11/StatefulPartitionedCallвd9/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_29884c0_29886*
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
=__inference_c0_layer_call_and_return_conditional_losses_29883Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_29903c1_29905*
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
=__inference_c1_layer_call_and_return_conditional_losses_29902╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_29848╤
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_29914·
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_29930c4_29932*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_29929Б
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_29949c5_29951*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_29948╫
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_29860╤
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_29960╔
f8/PartitionedCallPartitionedCalldr7/PartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_29968Є
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_29984d9_29986*
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
=__inference_d9_layer_call_and_return_conditional_losses_29983╘
dr10/PartitionedCallPartitionedCall#d9/StatefulPartitionedCall:output:0*
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
GPU(2*0J 8В *H
fCRA
?__inference_dr10_layer_call_and_return_conditional_losses_29994ў
d11/StatefulPartitionedCallStatefulPartitionedCalldr10/PartitionedCall:output:0	d11_30010	d11_30012*
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
GPU(2*0J 8В *G
fBR@
>__inference_d11_layer_call_and_return_conditional_losses_30009`
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
 *    a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ї
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall2:
d11/StatefulPartitionedCalld11/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╓
]
?__inference_dr10_layer_call_and_return_conditional_losses_30911

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
ЕY
┘
__inference__traced_save_31165
file_prefix(
$savev2_c0_kernel_read_readvariableop&
"savev2_c0_bias_read_readvariableop(
$savev2_c1_kernel_read_readvariableop&
"savev2_c1_bias_read_readvariableop(
$savev2_c4_kernel_read_readvariableop&
"savev2_c4_bias_read_readvariableop(
$savev2_c5_kernel_read_readvariableop&
"savev2_c5_bias_read_readvariableop(
$savev2_d9_kernel_read_readvariableop&
"savev2_d9_bias_read_readvariableop)
%savev2_d11_kernel_read_readvariableop'
#savev2_d11_bias_read_readvariableop(
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
+savev2_adam_c4_kernel_m_read_readvariableop-
)savev2_adam_c4_bias_m_read_readvariableop/
+savev2_adam_c5_kernel_m_read_readvariableop-
)savev2_adam_c5_bias_m_read_readvariableop/
+savev2_adam_d9_kernel_m_read_readvariableop-
)savev2_adam_d9_bias_m_read_readvariableop0
,savev2_adam_d11_kernel_m_read_readvariableop.
*savev2_adam_d11_bias_m_read_readvariableop/
+savev2_adam_c0_kernel_v_read_readvariableop-
)savev2_adam_c0_bias_v_read_readvariableop/
+savev2_adam_c1_kernel_v_read_readvariableop-
)savev2_adam_c1_bias_v_read_readvariableop/
+savev2_adam_c4_kernel_v_read_readvariableop-
)savev2_adam_c4_bias_v_read_readvariableop/
+savev2_adam_c5_kernel_v_read_readvariableop-
)savev2_adam_c5_bias_v_read_readvariableop/
+savev2_adam_d9_kernel_v_read_readvariableop-
)savev2_adam_d9_bias_v_read_readvariableop0
,savev2_adam_d11_kernel_v_read_readvariableop.
*savev2_adam_d11_bias_v_read_readvariableop
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
: г
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╔
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_c0_kernel_read_readvariableop"savev2_c0_bias_read_readvariableop$savev2_c1_kernel_read_readvariableop"savev2_c1_bias_read_readvariableop$savev2_c4_kernel_read_readvariableop"savev2_c4_bias_read_readvariableop$savev2_c5_kernel_read_readvariableop"savev2_c5_bias_read_readvariableop$savev2_d9_kernel_read_readvariableop"savev2_d9_bias_read_readvariableop%savev2_d11_kernel_read_readvariableop#savev2_d11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_c0_kernel_m_read_readvariableop)savev2_adam_c0_bias_m_read_readvariableop+savev2_adam_c1_kernel_m_read_readvariableop)savev2_adam_c1_bias_m_read_readvariableop+savev2_adam_c4_kernel_m_read_readvariableop)savev2_adam_c4_bias_m_read_readvariableop+savev2_adam_c5_kernel_m_read_readvariableop)savev2_adam_c5_bias_m_read_readvariableop+savev2_adam_d9_kernel_m_read_readvariableop)savev2_adam_d9_bias_m_read_readvariableop,savev2_adam_d11_kernel_m_read_readvariableop*savev2_adam_d11_bias_m_read_readvariableop+savev2_adam_c0_kernel_v_read_readvariableop)savev2_adam_c0_bias_v_read_readvariableop+savev2_adam_c1_kernel_v_read_readvariableop)savev2_adam_c1_bias_v_read_readvariableop+savev2_adam_c4_kernel_v_read_readvariableop)savev2_adam_c4_bias_v_read_readvariableop+savev2_adam_c5_kernel_v_read_readvariableop)savev2_adam_c5_bias_v_read_readvariableop+savev2_adam_d9_kernel_v_read_readvariableop)savev2_adam_d9_bias_v_read_readvariableop,savev2_adam_d11_kernel_v_read_readvariableop*savev2_adam_d11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	Р
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

identity_1Identity_1:output:0*╖
_input_shapesе
в: : : :  : : @:@:@@:@:
А А:А:	А
:
: : : : : : : : : : : :  : : @:@:@@:@:
А А:А:	А
:
: : :  : : @:@:@@:@:
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
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&	"
 
_output_shapes
:
А А:!


_output_shapes	
:А:%!

_output_shapes
:	А
: 

_output_shapes
:
:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
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
:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@:,((
&
_output_shapes
:@@: )

_output_shapes
:@:&*"
 
_output_shapes
:
А А:!+

_output_shapes	
:А:%,!

_output_shapes
:	А
: -

_output_shapes
:
:.

_output_shapes
: 
╛
Ў
=__inference_c4_layer_call_and_return_conditional_losses_29929

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @`
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
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
─
+
__inference_loss_fn_1_30957
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
╠░
У
!__inference__traced_restore_31310
file_prefix4
assignvariableop_c0_kernel: (
assignvariableop_1_c0_bias: 6
assignvariableop_2_c1_kernel:  (
assignvariableop_3_c1_bias: 6
assignvariableop_4_c4_kernel: @(
assignvariableop_5_c4_bias:@6
assignvariableop_6_c5_kernel:@@(
assignvariableop_7_c5_bias:@0
assignvariableop_8_d9_kernel:
А А)
assignvariableop_9_d9_bias:	А1
assignvariableop_10_d11_kernel:	А
*
assignvariableop_11_d11_bias:
'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: >
$assignvariableop_21_adam_c0_kernel_m: 0
"assignvariableop_22_adam_c0_bias_m: >
$assignvariableop_23_adam_c1_kernel_m:  0
"assignvariableop_24_adam_c1_bias_m: >
$assignvariableop_25_adam_c4_kernel_m: @0
"assignvariableop_26_adam_c4_bias_m:@>
$assignvariableop_27_adam_c5_kernel_m:@@0
"assignvariableop_28_adam_c5_bias_m:@8
$assignvariableop_29_adam_d9_kernel_m:
А А1
"assignvariableop_30_adam_d9_bias_m:	А8
%assignvariableop_31_adam_d11_kernel_m:	А
1
#assignvariableop_32_adam_d11_bias_m:
>
$assignvariableop_33_adam_c0_kernel_v: 0
"assignvariableop_34_adam_c0_bias_v: >
$assignvariableop_35_adam_c1_kernel_v:  0
"assignvariableop_36_adam_c1_bias_v: >
$assignvariableop_37_adam_c4_kernel_v: @0
"assignvariableop_38_adam_c4_bias_v:@>
$assignvariableop_39_adam_c5_kernel_v:@@0
"assignvariableop_40_adam_c5_bias_v:@8
$assignvariableop_41_adam_d9_kernel_v:
А А1
"assignvariableop_42_adam_d9_bias_v:	А8
%assignvariableop_43_adam_d11_kernel_v:	А
1
#assignvariableop_44_adam_d11_bias_v:

identity_46ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╠
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B З
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
╕::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
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
AssignVariableOp_4AssignVariableOpassignvariableop_4_c4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_5AssignVariableOpassignvariableop_5_c4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_c5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_7AssignVariableOpassignvariableop_7_c5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_d9_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_9AssignVariableOpassignvariableop_9_d9_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_10AssignVariableOpassignvariableop_10_d11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_11AssignVariableOpassignvariableop_11_d11_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_c0_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_c0_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_c1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_c1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_c4_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_c4_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_c5_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_c5_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_29AssignVariableOp$assignvariableop_29_adam_d9_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_d9_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_d11_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_32AssignVariableOp#assignvariableop_32_adam_d11_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_33AssignVariableOp$assignvariableop_33_adam_c0_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_34AssignVariableOp"assignvariableop_34_adam_c0_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_35AssignVariableOp$assignvariableop_35_adam_c1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_36AssignVariableOp"assignvariableop_36_adam_c1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_37AssignVariableOp$assignvariableop_37_adam_c4_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_38AssignVariableOp"assignvariableop_38_adam_c4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_39AssignVariableOp$assignvariableop_39_adam_c5_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_40AssignVariableOp"assignvariableop_40_adam_c5_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_41AssignVariableOp$assignvariableop_41_adam_d9_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_42AssignVariableOp"assignvariableop_42_adam_d9_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adam_d11_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_44AssignVariableOp#assignvariableop_44_adam_d11_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 н
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: Ъ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
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
т
Ё
>__inference_d11_layer_call_and_return_conditional_losses_30947

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
a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
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
ё
\
>__inference_dr3_layer_call_and_return_conditional_losses_29914

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:          c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╦
,
__inference_loss_fn_10_31002
identitya
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    \
IdentityIdentity%d11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
╜

]
>__inference_dr3_layer_call_and_return_conditional_losses_30167

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
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
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
:          w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
К
\
#__inference_dr7_layer_call_fn_30844

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
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_30124w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╜

]
>__inference_dr7_layer_call_and_return_conditional_losses_30124

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
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
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
:         @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┐
Y
=__inference_f8_layer_call_and_return_conditional_losses_29968

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
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_30715

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
г
>
"__inference_m2_layer_call_fn_30744

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
=__inference_m2_layer_call_and_return_conditional_losses_29848Г
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
е
╩
*__inference_sequential_layer_call_fn_30480

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
А А
	unknown_8:	А
	unknown_9:	А


unknown_10:

identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30028o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_6_30982
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
Ж

^
?__inference_dr10_layer_call_and_return_conditional_losses_30923

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
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
 *   ?з
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
и
>
"__inference_f8_layer_call_fn_30866

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
=__inference_f8_layer_call_and_return_conditional_losses_29968a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_2_30962
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
т
ё
=__inference_d9_layer_call_and_return_conditional_losses_29983

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
Ё
]
$__inference_dr10_layer_call_fn_30906

inputs
identityИвStatefulPartitionedCall┬
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
GPU(2*0J 8В *H
fCRA
?__inference_dr10_layer_call_and_return_conditional_losses_30085p
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
─
+
__inference_loss_fn_7_30987
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
╓
]
?__inference_dr10_layer_call_and_return_conditional_losses_29994

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
╝D
е

 __inference__wrapped_model_29839
c0_inputF
,sequential_c0_conv2d_readvariableop_resource: ;
-sequential_c0_biasadd_readvariableop_resource: F
,sequential_c1_conv2d_readvariableop_resource:  ;
-sequential_c1_biasadd_readvariableop_resource: F
,sequential_c4_conv2d_readvariableop_resource: @;
-sequential_c4_biasadd_readvariableop_resource:@F
,sequential_c5_conv2d_readvariableop_resource:@@;
-sequential_c5_biasadd_readvariableop_resource:@@
,sequential_d9_matmul_readvariableop_resource:
А А<
-sequential_d9_biasadd_readvariableop_resource:	А@
-sequential_d11_matmul_readvariableop_resource:	А
<
.sequential_d11_biasadd_readvariableop_resource:

identityИв$sequential/c0/BiasAdd/ReadVariableOpв#sequential/c0/Conv2D/ReadVariableOpв$sequential/c1/BiasAdd/ReadVariableOpв#sequential/c1/Conv2D/ReadVariableOpв$sequential/c4/BiasAdd/ReadVariableOpв#sequential/c4/Conv2D/ReadVariableOpв$sequential/c5/BiasAdd/ReadVariableOpв#sequential/c5/Conv2D/ReadVariableOpв%sequential/d11/BiasAdd/ReadVariableOpв$sequential/d11/MatMul/ReadVariableOpв$sequential/d9/BiasAdd/ReadVariableOpв#sequential/d9/MatMul/ReadVariableOpШ
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
:  *
dtype0╧
sequential/c1/Conv2DConv2D sequential/c0/Relu:activations:0+sequential/c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
О
$sequential/c1/BiasAdd/ReadVariableOpReadVariableOp-sequential_c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0з
sequential/c1/BiasAddBiasAddsequential/c1/Conv2D:output:0,sequential/c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            t
sequential/c1/ReluRelusequential/c1/BiasAdd:output:0*
T0*/
_output_shapes
:            п
sequential/m2/MaxPoolMaxPool sequential/c1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
}
sequential/dr3/IdentityIdentitysequential/m2/MaxPool:output:0*
T0*/
_output_shapes
:          Ш
#sequential/c4/Conv2D/ReadVariableOpReadVariableOp,sequential_c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╧
sequential/c4/Conv2DConv2D sequential/dr3/Identity:output:0+sequential/c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
О
$sequential/c4/BiasAdd/ReadVariableOpReadVariableOp-sequential_c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0з
sequential/c4/BiasAddBiasAddsequential/c4/Conv2D:output:0,sequential/c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @t
sequential/c4/ReluRelusequential/c4/BiasAdd:output:0*
T0*/
_output_shapes
:         @Ш
#sequential/c5/Conv2D/ReadVariableOpReadVariableOp,sequential_c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╧
sequential/c5/Conv2DConv2D sequential/c4/Relu:activations:0+sequential/c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
О
$sequential/c5/BiasAdd/ReadVariableOpReadVariableOp-sequential_c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0з
sequential/c5/BiasAddBiasAddsequential/c5/Conv2D:output:0,sequential/c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @t
sequential/c5/ReluRelusequential/c5/BiasAdd:output:0*
T0*/
_output_shapes
:         @п
sequential/m6/MaxPoolMaxPool sequential/c5/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
}
sequential/dr7/IdentityIdentitysequential/m6/MaxPool:output:0*
T0*/
_output_shapes
:         @d
sequential/f8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       У
sequential/f8/ReshapeReshape sequential/dr7/Identity:output:0sequential/f8/Const:output:0*
T0*(
_output_shapes
:         А Т
#sequential/d9/MatMul/ReadVariableOpReadVariableOp,sequential_d9_matmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0Ю
sequential/d9/MatMulMatMulsequential/f8/Reshape:output:0+sequential/d9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$sequential/d9/BiasAdd/ReadVariableOpReadVariableOp-sequential_d9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
sequential/d9/BiasAddBiasAddsequential/d9/MatMul:product:0,sequential/d9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
sequential/d9/ReluRelusequential/d9/BiasAdd:output:0*
T0*(
_output_shapes
:         Аy
sequential/dr10/IdentityIdentity sequential/d9/Relu:activations:0*
T0*(
_output_shapes
:         АУ
$sequential/d11/MatMul/ReadVariableOpReadVariableOp-sequential_d11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0в
sequential/d11/MatMulMatMul!sequential/dr10/Identity:output:0,sequential/d11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Р
%sequential/d11/BiasAdd/ReadVariableOpReadVariableOp.sequential_d11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0г
sequential/d11/BiasAddBiasAddsequential/d11/MatMul:product:0-sequential/d11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
t
sequential/d11/SoftmaxSoftmaxsequential/d11/BiasAdd:output:0*
T0*'
_output_shapes
:         
o
IdentityIdentity sequential/d11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Ц
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/c1/BiasAdd/ReadVariableOp$^sequential/c1/Conv2D/ReadVariableOp%^sequential/c4/BiasAdd/ReadVariableOp$^sequential/c4/Conv2D/ReadVariableOp%^sequential/c5/BiasAdd/ReadVariableOp$^sequential/c5/Conv2D/ReadVariableOp&^sequential/d11/BiasAdd/ReadVariableOp%^sequential/d11/MatMul/ReadVariableOp%^sequential/d9/BiasAdd/ReadVariableOp$^sequential/d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 2L
$sequential/c0/BiasAdd/ReadVariableOp$sequential/c0/BiasAdd/ReadVariableOp2J
#sequential/c0/Conv2D/ReadVariableOp#sequential/c0/Conv2D/ReadVariableOp2L
$sequential/c1/BiasAdd/ReadVariableOp$sequential/c1/BiasAdd/ReadVariableOp2J
#sequential/c1/Conv2D/ReadVariableOp#sequential/c1/Conv2D/ReadVariableOp2L
$sequential/c4/BiasAdd/ReadVariableOp$sequential/c4/BiasAdd/ReadVariableOp2J
#sequential/c4/Conv2D/ReadVariableOp#sequential/c4/Conv2D/ReadVariableOp2L
$sequential/c5/BiasAdd/ReadVariableOp$sequential/c5/BiasAdd/ReadVariableOp2J
#sequential/c5/Conv2D/ReadVariableOp#sequential/c5/Conv2D/ReadVariableOp2N
%sequential/d11/BiasAdd/ReadVariableOp%sequential/d11/BiasAdd/ReadVariableOp2L
$sequential/d11/MatMul/ReadVariableOp$sequential/d11/MatMul/ReadVariableOp2L
$sequential/d9/BiasAdd/ReadVariableOp$sequential/d9/BiasAdd/ReadVariableOp2J
#sequential/d9/MatMul/ReadVariableOp#sequential/d9/MatMul/ReadVariableOp:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
╜

]
>__inference_dr7_layer_call_and_return_conditional_losses_30861

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
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
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
:         @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Б;
╠
E__inference_sequential_layer_call_and_return_conditional_losses_30273

inputs"
c0_30224: 
c0_30226: "
c1_30229:  
c1_30231: "
c4_30236: @
c4_30238:@"
c5_30241:@@
c5_30243:@
d9_30249:
А А
d9_30251:	А
	d11_30255:	А

	d11_30257:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвc4/StatefulPartitionedCallвc5/StatefulPartitionedCallвd11/StatefulPartitionedCallвd9/StatefulPartitionedCallвdr10/StatefulPartitionedCallвdr3/StatefulPartitionedCallвdr7/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_30224c0_30226*
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
=__inference_c0_layer_call_and_return_conditional_losses_29883Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_30229c1_30231*
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
=__inference_c1_layer_call_and_return_conditional_losses_29902╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_29848с
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30167В
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_30236c4_30238*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_29929Б
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_30241c5_30243*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_29948╫
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_29860 
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_30124╤
f8/PartitionedCallPartitionedCall$dr7/StatefulPartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_29968Є
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_30249d9_30251*
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
=__inference_d9_layer_call_and_return_conditional_losses_29983В
dr10/StatefulPartitionedCallStatefulPartitionedCall#d9/StatefulPartitionedCall:output:0^dr7/StatefulPartitionedCall*
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
GPU(2*0J 8В *H
fCRA
?__inference_dr10_layer_call_and_return_conditional_losses_30085 
d11/StatefulPartitionedCallStatefulPartitionedCall%dr10/StatefulPartitionedCall:output:0	d11_30255	d11_30257*
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
GPU(2*0J 8В *G
fBR@
>__inference_d11_layer_call_and_return_conditional_losses_30009`
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
 *    a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
╨
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr10/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall2:
d11/StatefulPartitionedCalld11/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2<
dr10/StatefulPartitionedCalldr10/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr7/StatefulPartitionedCalldr7/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╕
?
#__inference_dr3_layer_call_fn_30754

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
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_29914h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ё
\
>__inference_dr7_layer_call_and_return_conditional_losses_30849

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ю
@
$__inference_dr10_layer_call_fn_30901

inputs
identity▓
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
GPU(2*0J 8В *H
fCRA
?__inference_dr10_layer_call_and_return_conditional_losses_29994a
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
└
С
#__inference_d11_layer_call_fn_30934

inputs
unknown:	А

	unknown_0:

identityИвStatefulPartitionedCall┌
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
GPU(2*0J 8В *G
fBR@
>__inference_d11_layer_call_and_return_conditional_losses_30009o
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
╜

]
>__inference_dr3_layer_call_and_return_conditional_losses_30776

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
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
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
:          w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╛
Ў
=__inference_c1_layer_call_and_return_conditional_losses_29902

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
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
:            w
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
у
Ч
"__inference_c1_layer_call_fn_30726

inputs!
unknown:  
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
=__inference_c1_layer_call_and_return_conditional_losses_29902w
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
:            : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:            
 
_user_specified_nameinputs
Е
Y
=__inference_m2_layer_call_and_return_conditional_losses_30749

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
╛
Ў
=__inference_c4_layer_call_and_return_conditional_losses_30800

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @`
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
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
─
+
__inference_loss_fn_3_30967
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
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_29883

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
╚
+
__inference_loss_fn_4_30972
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
╟
,
__inference_loss_fn_11_31007
identity_
d11/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
IdentityIdentity#d11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ё
\
>__inference_dr7_layer_call_and_return_conditional_losses_29960

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ё
\
>__inference_dr3_layer_call_and_return_conditional_losses_30764

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:          c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╛
Ў
=__inference_c5_layer_call_and_return_conditional_losses_30824

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @`
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
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ж

^
?__inference_dr10_layer_call_and_return_conditional_losses_30085

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
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
 *   ?з
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
т
ё
=__inference_d9_layer_call_and_return_conditional_losses_30896

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
╔V
└
E__inference_sequential_layer_call_and_return_conditional_losses_30660

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource:  0
"c1_biasadd_readvariableop_resource: ;
!c4_conv2d_readvariableop_resource: @0
"c4_biasadd_readvariableop_resource:@;
!c5_conv2d_readvariableop_resource:@@0
"c5_biasadd_readvariableop_resource:@5
!d9_matmul_readvariableop_resource:
А А1
"d9_biasadd_readvariableop_resource:	А5
"d11_matmul_readvariableop_resource:	А
1
#d11_biasadd_readvariableop_resource:

identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc1/BiasAdd/ReadVariableOpвc1/Conv2D/ReadVariableOpвc4/BiasAdd/ReadVariableOpвc4/Conv2D/ReadVariableOpвc5/BiasAdd/ReadVariableOpвc5/Conv2D/ReadVariableOpвd11/BiasAdd/ReadVariableOpвd11/MatMul/ReadVariableOpвd9/BiasAdd/ReadVariableOpвd9/MatMul/ReadVariableOpВ
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
:  *
dtype0о
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ж

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:            Щ

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:          *
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
:          T
dr3/dropout/ShapeShapem2/MaxPool:output:0*
T0*
_output_shapes
:н
(dr3/dropout/random_uniform/RandomUniformRandomUniformdr3/dropout/Shape:output:0*
T0*/
_output_shapes
:          *
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
:          
dr3/dropout/CastCastdr3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          }
dr3/dropout/Mul_1Muldr3/dropout/Mul:z:0dr3/dropout/Cast:y:0*
T0*/
_output_shapes
:          В
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0о
	c4/Conv2DConv2Ddr3/dropout/Mul_1:z:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
x
c4/BiasAdd/ReadVariableOpReadVariableOp"c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж

c4/BiasAddBiasAddc4/Conv2D:output:0!c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:         @В
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
x
c5/BiasAdd/ReadVariableOpReadVariableOp"c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж

c5/BiasAddBiasAddc5/Conv2D:output:0!c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:         @Щ

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:         @*
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
 *лкк?Б
dr7/dropout/MulMulm6/MaxPool:output:0dr7/dropout/Const:output:0*
T0*/
_output_shapes
:         @T
dr7/dropout/ShapeShapem6/MaxPool:output:0*
T0*
_output_shapes
:й
(dr7/dropout/random_uniform/RandomUniformRandomUniformdr7/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seed2_
dr7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>║
dr7/dropout/GreaterEqualGreaterEqual1dr7/dropout/random_uniform/RandomUniform:output:0#dr7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @
dr7/dropout/CastCastdr7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @}
dr7/dropout/Mul_1Muldr7/dropout/Mul:z:0dr7/dropout/Cast:y:0*
T0*/
_output_shapes
:         @Y
f8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       r

f8/ReshapeReshapedr7/dropout/Mul_1:z:0f8/Const:output:0*
T0*(
_output_shapes
:         А |
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource* 
_output_shapes
:
А А*
dtype0}
	d9/MatMulMatMulf8/Reshape:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0А

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АW
d9/ReluRelud9/BiasAdd:output:0*
T0*(
_output_shapes
:         АW
dr10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @~
dr10/dropout/MulMuld9/Relu:activations:0dr10/dropout/Const:output:0*
T0*(
_output_shapes
:         АW
dr10/dropout/ShapeShaped9/Relu:activations:0*
T0*
_output_shapes
:д
)dr10/dropout/random_uniform/RandomUniformRandomUniformdr10/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0*
seed2`
dr10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╢
dr10/dropout/GreaterEqualGreaterEqual2dr10/dropout/random_uniform/RandomUniform:output:0$dr10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аz
dr10/dropout/CastCastdr10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аy
dr10/dropout/Mul_1Muldr10/dropout/Mul:z:0dr10/dropout/Cast:y:0*
T0*(
_output_shapes
:         А}
d11/MatMul/ReadVariableOpReadVariableOp"d11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Б

d11/MatMulMatMuldr10/dropout/Mul_1:z:0!d11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
z
d11/BiasAdd/ReadVariableOpReadVariableOp#d11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0В
d11/BiasAddBiasAddd11/MatMul:product:0"d11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
^
d11/SoftmaxSoftmaxd11/BiasAdd:output:0*
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
 *    a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentityd11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Т
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^d11/BiasAdd/ReadVariableOp^d11/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
c4/BiasAdd/ReadVariableOpc4/BiasAdd/ReadVariableOp24
c4/Conv2D/ReadVariableOpc4/Conv2D/ReadVariableOp26
c5/BiasAdd/ReadVariableOpc5/BiasAdd/ReadVariableOp24
c5/Conv2D/ReadVariableOpc5/Conv2D/ReadVariableOp28
d11/BiasAdd/ReadVariableOpd11/BiasAdd/ReadVariableOp26
d11/MatMul/ReadVariableOpd11/MatMul/ReadVariableOp26
d9/BiasAdd/ReadVariableOpd9/BiasAdd/ReadVariableOp24
d9/MatMul/ReadVariableOpd9/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
┐
Y
=__inference_f8_layer_call_and_return_conditional_losses_30872

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
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
у
Ч
"__inference_c4_layer_call_fn_30787

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_29929w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ч7
є
E__inference_sequential_layer_call_and_return_conditional_losses_30381
c0_input"
c0_30332: 
c0_30334: "
c1_30337:  
c1_30339: "
c4_30344: @
c4_30346:@"
c5_30349:@@
c5_30351:@
d9_30357:
А А
d9_30359:	А
	d11_30363:	А

	d11_30365:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвc4/StatefulPartitionedCallвc5/StatefulPartitionedCallвd11/StatefulPartitionedCallвd9/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_30332c0_30334*
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
=__inference_c0_layer_call_and_return_conditional_losses_29883Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_30337c1_30339*
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
=__inference_c1_layer_call_and_return_conditional_losses_29902╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_29848╤
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_29914·
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_30344c4_30346*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_29929Б
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_30349c5_30351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_29948╫
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_29860╤
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_29960╔
f8/PartitionedCallPartitionedCalldr7/PartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_29968Є
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_30357d9_30359*
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
=__inference_d9_layer_call_and_return_conditional_losses_29983╘
dr10/PartitionedCallPartitionedCall#d9/StatefulPartitionedCall:output:0*
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
GPU(2*0J 8В *H
fCRA
?__inference_dr10_layer_call_and_return_conditional_losses_29994ў
d11/StatefulPartitionedCallStatefulPartitionedCalldr10/PartitionedCall:output:0	d11_30363	d11_30365*
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
GPU(2*0J 8В *G
fBR@
>__inference_d11_layer_call_and_return_conditional_losses_30009`
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
 *    a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ї
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall2:
d11/StatefulPartitionedCalld11/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
─
+
__inference_loss_fn_5_30977
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
л
╠
*__inference_sequential_layer_call_fn_30329
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
А А
	unknown_8:	А
	unknown_9:	А


unknown_10:

identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_30273o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
 

┼
#__inference_signature_wrapper_30691
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
А А
	unknown_8:	А
	unknown_9:	А


unknown_10:

identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8В *)
f$R"
 __inference__wrapped_model_29839o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
К
\
#__inference_dr3_layer_call_fn_30759

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
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30167w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
у
Ч
"__inference_c0_layer_call_fn_30702

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
=__inference_c0_layer_call_and_return_conditional_losses_29883w
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
╛
Ў
=__inference_c5_layer_call_and_return_conditional_losses_29948

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @`
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
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
г
>
"__inference_m6_layer_call_fn_30829

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
=__inference_m6_layer_call_and_return_conditional_losses_29860Г
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
З;
╬
E__inference_sequential_layer_call_and_return_conditional_losses_30433
c0_input"
c0_30384: 
c0_30386: "
c1_30389:  
c1_30391: "
c4_30396: @
c4_30398:@"
c5_30401:@@
c5_30403:@
d9_30409:
А А
d9_30411:	А
	d11_30415:	А

	d11_30417:

identityИвc0/StatefulPartitionedCallвc1/StatefulPartitionedCallвc4/StatefulPartitionedCallвc5/StatefulPartitionedCallвd11/StatefulPartitionedCallвd9/StatefulPartitionedCallвdr10/StatefulPartitionedCallвdr3/StatefulPartitionedCallвdr7/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_30384c0_30386*
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
=__inference_c0_layer_call_and_return_conditional_losses_29883Б
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_30389c1_30391*
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
=__inference_c1_layer_call_and_return_conditional_losses_29902╫
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_29848с
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_30167В
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_30396c4_30398*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_29929Б
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_30401c5_30403*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_29948╫
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_29860 
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_30124╤
f8/PartitionedCallPartitionedCall$dr7/StatefulPartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_29968Є
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_30409d9_30411*
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
=__inference_d9_layer_call_and_return_conditional_losses_29983В
dr10/StatefulPartitionedCallStatefulPartitionedCall#d9/StatefulPartitionedCall:output:0^dr7/StatefulPartitionedCall*
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
GPU(2*0J 8В *H
fCRA
?__inference_dr10_layer_call_and_return_conditional_losses_30085 
d11/StatefulPartitionedCallStatefulPartitionedCall%dr10/StatefulPartitionedCall:output:0	d11_30415	d11_30417*
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
GPU(2*0J 8В *G
fBR@
>__inference_d11_layer_call_and_return_conditional_losses_30009`
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
 *    a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
╨
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr10/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:           : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall2:
d11/StatefulPartitionedCalld11/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2<
dr10/StatefulPartitionedCalldr10/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr7/StatefulPartitionedCalldr7/StatefulPartitionedCall:Y U
/
_output_shapes
:           
"
_user_specified_name
c0_input
╚
+
__inference_loss_fn_0_30952
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
т
Ё
>__inference_d11_layer_call_and_return_conditional_losses_30009

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
a
d11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d11/bias/Regularizer/ConstConst*
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
Е
Y
=__inference_m2_layer_call_and_return_conditional_losses_29848

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
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*░
serving_defaultЬ
E
c0_input9
serving_default_c0_input:0           7
d110
StatefulPartitionedCall:0         
tensorflow/serving/predict:║ч
а
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
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
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
╝
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
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
е
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M_random_generator
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
е
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b_random_generator
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
├
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem╦m╠m═m╬3m╧4m╨;m╤<m╥Vm╙Wm╘em╒fm╓v╫v╪v┘v┌3v█4v▄;v▌<v▐Vv▀Wvрevсfvт"
	optimizer
v
0
1
2
3
34
45
;6
<7
V8
W9
e10
f11"
trackable_list_wrapper
v
0
1
2
3
34
45
;6
<7
V8
W9
e10
f11"
trackable_list_wrapper
v
r0
s1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11"
trackable_list_wrapper
═
~non_trainable_variables

layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ў2є
*__inference_sequential_layer_call_fn_30055
*__inference_sequential_layer_call_fn_30480
*__inference_sequential_layer_call_fn_30509
*__inference_sequential_layer_call_fn_30329└
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
E__inference_sequential_layer_call_and_return_conditional_losses_30574
E__inference_sequential_layer_call_and_return_conditional_losses_30660
E__inference_sequential_layer_call_and_return_conditional_losses_30381
E__inference_sequential_layer_call_and_return_conditional_losses_30433└
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
 __inference__wrapped_model_29839c0_input"Ш
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
-
Гserving_default"
signature_map
#:! 2	c0/kernel
: 2c0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
▓
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c0_layer_call_fn_30702в
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
=__inference_c0_layer_call_and_return_conditional_losses_30715в
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
#:!  2	c1/kernel
: 2c1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
▓
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c1_layer_call_fn_30726в
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
=__inference_c1_layer_call_and_return_conditional_losses_30739в
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
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_m2_layer_call_fn_30744в
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
=__inference_m2_layer_call_and_return_conditional_losses_30749в
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
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Д2Б
#__inference_dr3_layer_call_fn_30754
#__inference_dr3_layer_call_fn_30759┤
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
>__inference_dr3_layer_call_and_return_conditional_losses_30764
>__inference_dr3_layer_call_and_return_conditional_losses_30776┤
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
#:! @2	c4/kernel
:@2c4/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
▓
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c4_layer_call_fn_30787в
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
=__inference_c4_layer_call_and_return_conditional_losses_30800в
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
#:!@@2	c5/kernel
:@2c5/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
▓
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_c5_layer_call_fn_30811в
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
=__inference_c5_layer_call_and_return_conditional_losses_30824в
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
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_m6_layer_call_fn_30829в
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
=__inference_m6_layer_call_and_return_conditional_losses_30834в
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
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Д2Б
#__inference_dr7_layer_call_fn_30839
#__inference_dr7_layer_call_fn_30844┤
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
>__inference_dr7_layer_call_and_return_conditional_losses_30849
>__inference_dr7_layer_call_and_return_conditional_losses_30861┤
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
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_f8_layer_call_fn_30866в
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
=__inference_f8_layer_call_and_return_conditional_losses_30872в
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
А А2	d9/kernel
:А2d9/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
▓
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
╠2╔
"__inference_d9_layer_call_fn_30883в
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
=__inference_d9_layer_call_and_return_conditional_losses_30896в
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
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
^	variables
_trainable_variables
`regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ж2Г
$__inference_dr10_layer_call_fn_30901
$__inference_dr10_layer_call_fn_30906┤
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
╝2╣
?__inference_dr10_layer_call_and_return_conditional_losses_30911
?__inference_dr10_layer_call_and_return_conditional_losses_30923┤
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
:	А
2
d11/kernel
:
2d11/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
▓
╗non_trainable_variables
╝layers
╜metrics
 ╛layer_regularization_losses
┐layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
═2╩
#__inference_d11_layer_call_fn_30934в
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
ш2х
>__inference_d11_layer_call_and_return_conditional_losses_30947в
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
__inference_loss_fn_0_30952П
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
__inference_loss_fn_1_30957П
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
__inference_loss_fn_2_30962П
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
__inference_loss_fn_3_30967П
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
__inference_loss_fn_4_30972П
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
__inference_loss_fn_5_30977П
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
__inference_loss_fn_6_30982П
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
__inference_loss_fn_7_30987П
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
__inference_loss_fn_8_30992П
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
__inference_loss_fn_9_30997П
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
│2░
__inference_loss_fn_10_31002П
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
│2░
__inference_loss_fn_11_31007П
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
v
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
11"
trackable_list_wrapper
0
└0
┴1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╦B╚
#__inference_signature_wrapper_30691c0_input"Ф
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
r0
s1"
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
t0
u1"
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
v0
w1"
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
x0
y1"
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
z0
{1"
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
|0
}1"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

┬total

├count
─	variables
┼	keras_api"
_tf_keras_metric
c

╞total

╟count
╚
_fn_kwargs
╔	variables
╩	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
┬0
├1"
trackable_list_wrapper
.
─	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╞0
╟1"
trackable_list_wrapper
.
╔	variables"
_generic_user_object
(:& 2Adam/c0/kernel/m
: 2Adam/c0/bias/m
(:&  2Adam/c1/kernel/m
: 2Adam/c1/bias/m
(:& @2Adam/c4/kernel/m
:@2Adam/c4/bias/m
(:&@@2Adam/c5/kernel/m
:@2Adam/c5/bias/m
": 
А А2Adam/d9/kernel/m
:А2Adam/d9/bias/m
": 	А
2Adam/d11/kernel/m
:
2Adam/d11/bias/m
(:& 2Adam/c0/kernel/v
: 2Adam/c0/bias/v
(:&  2Adam/c1/kernel/v
: 2Adam/c1/bias/v
(:& @2Adam/c4/kernel/v
:@2Adam/c4/bias/v
(:&@@2Adam/c5/kernel/v
:@2Adam/c5/bias/v
": 
А А2Adam/d9/kernel/v
:А2Adam/d9/bias/v
": 	А
2Adam/d11/kernel/v
:
2Adam/d11/bias/vШ
 __inference__wrapped_model_29839t34;<VWef9в6
/в,
*К'
c0_input           
к ")к&
$
d11К
d11         
н
=__inference_c0_layer_call_and_return_conditional_losses_30715l7в4
-в*
(К%
inputs           
к "-в*
#К 
0            
Ъ Е
"__inference_c0_layer_call_fn_30702_7в4
-в*
(К%
inputs           
к " К            н
=__inference_c1_layer_call_and_return_conditional_losses_30739l7в4
-в*
(К%
inputs            
к "-в*
#К 
0            
Ъ Е
"__inference_c1_layer_call_fn_30726_7в4
-в*
(К%
inputs            
к " К            н
=__inference_c4_layer_call_and_return_conditional_losses_30800l347в4
-в*
(К%
inputs          
к "-в*
#К 
0         @
Ъ Е
"__inference_c4_layer_call_fn_30787_347в4
-в*
(К%
inputs          
к " К         @н
=__inference_c5_layer_call_and_return_conditional_losses_30824l;<7в4
-в*
(К%
inputs         @
к "-в*
#К 
0         @
Ъ Е
"__inference_c5_layer_call_fn_30811_;<7в4
-в*
(К%
inputs         @
к " К         @Я
>__inference_d11_layer_call_and_return_conditional_losses_30947]ef0в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ w
#__inference_d11_layer_call_fn_30934Pef0в-
&в#
!К
inputs         А
к "К         
Я
=__inference_d9_layer_call_and_return_conditional_losses_30896^VW0в-
&в#
!К
inputs         А 
к "&в#
К
0         А
Ъ w
"__inference_d9_layer_call_fn_30883QVW0в-
&в#
!К
inputs         А 
к "К         Аб
?__inference_dr10_layer_call_and_return_conditional_losses_30911^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ б
?__inference_dr10_layer_call_and_return_conditional_losses_30923^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ y
$__inference_dr10_layer_call_fn_30901Q4в1
*в'
!К
inputs         А
p 
к "К         Аy
$__inference_dr10_layer_call_fn_30906Q4в1
*в'
!К
inputs         А
p
к "К         Ао
>__inference_dr3_layer_call_and_return_conditional_losses_30764l;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ о
>__inference_dr3_layer_call_and_return_conditional_losses_30776l;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ Ж
#__inference_dr3_layer_call_fn_30754_;в8
1в.
(К%
inputs          
p 
к " К          Ж
#__inference_dr3_layer_call_fn_30759_;в8
1в.
(К%
inputs          
p
к " К          о
>__inference_dr7_layer_call_and_return_conditional_losses_30849l;в8
1в.
(К%
inputs         @
p 
к "-в*
#К 
0         @
Ъ о
>__inference_dr7_layer_call_and_return_conditional_losses_30861l;в8
1в.
(К%
inputs         @
p
к "-в*
#К 
0         @
Ъ Ж
#__inference_dr7_layer_call_fn_30839_;в8
1в.
(К%
inputs         @
p 
к " К         @Ж
#__inference_dr7_layer_call_fn_30844_;в8
1в.
(К%
inputs         @
p
к " К         @в
=__inference_f8_layer_call_and_return_conditional_losses_30872a7в4
-в*
(К%
inputs         @
к "&в#
К
0         А 
Ъ z
"__inference_f8_layer_call_fn_30866T7в4
-в*
(К%
inputs         @
к "К         А 7
__inference_loss_fn_0_30952в

в 
к "К 8
__inference_loss_fn_10_31002в

в 
к "К 8
__inference_loss_fn_11_31007в

в 
к "К 7
__inference_loss_fn_1_30957в

в 
к "К 7
__inference_loss_fn_2_30962в

в 
к "К 7
__inference_loss_fn_3_30967в

в 
к "К 7
__inference_loss_fn_4_30972в

в 
к "К 7
__inference_loss_fn_5_30977в

в 
к "К 7
__inference_loss_fn_6_30982в

в 
к "К 7
__inference_loss_fn_7_30987в

в 
к "К 7
__inference_loss_fn_8_30992в

в 
к "К 7
__inference_loss_fn_9_30997в

в 
к "К р
=__inference_m2_layer_call_and_return_conditional_losses_30749ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_m2_layer_call_fn_30744СRвO
HвE
CК@
inputs4                                    
к ";К84                                    р
=__inference_m6_layer_call_and_return_conditional_losses_30834ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_m6_layer_call_fn_30829СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ┴
E__inference_sequential_layer_call_and_return_conditional_losses_30381x34;<VWefAв>
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
Ъ ┴
E__inference_sequential_layer_call_and_return_conditional_losses_30433x34;<VWefAв>
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
Ъ ┐
E__inference_sequential_layer_call_and_return_conditional_losses_30574v34;<VWef?в<
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
Ъ ┐
E__inference_sequential_layer_call_and_return_conditional_losses_30660v34;<VWef?в<
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
Ъ Щ
*__inference_sequential_layer_call_fn_30055k34;<VWefAв>
7в4
*К'
c0_input           
p 

 
к "К         
Щ
*__inference_sequential_layer_call_fn_30329k34;<VWefAв>
7в4
*К'
c0_input           
p

 
к "К         
Ч
*__inference_sequential_layer_call_fn_30480i34;<VWef?в<
5в2
(К%
inputs           
p 

 
к "К         
Ч
*__inference_sequential_layer_call_fn_30509i34;<VWef?в<
5в2
(К%
inputs           
p

 
к "К         
и
#__inference_signature_wrapper_30691А34;<VWefEвB
в 
;к8
6
c0_input*К'
c0_input           ")к&
$
d11К
d11         

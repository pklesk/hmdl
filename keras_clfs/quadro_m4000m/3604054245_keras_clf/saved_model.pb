ֺ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
v
	c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	c0/kernel
o
c0/kernel/Read/ReadVariableOpReadVariableOp	c0/kernel*&
_output_shapes
: *
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
shape:  *
shared_name	c1/kernel
o
c1/kernel/Read/ReadVariableOpReadVariableOp	c1/kernel*&
_output_shapes
:  *
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
w
	c8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_name	c8/kernel
p
c8/kernel/Read/ReadVariableOpReadVariableOp	c8/kernel*'
_output_shapes
:@�*
dtype0
g
c8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	c8/bias
`
c8/bias/Read/ReadVariableOpReadVariableOpc8/bias*
_output_shapes	
:�*
dtype0
x
	c9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_name	c9/kernel
q
c9/kernel/Read/ReadVariableOpReadVariableOp	c9/kernel*(
_output_shapes
:��*
dtype0
g
c9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	c9/bias
`
c9/bias/Read/ReadVariableOpReadVariableOpc9/bias*
_output_shapes	
:�*
dtype0
r

d13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*
shared_name
d13/kernel
k
d13/kernel/Read/ReadVariableOpReadVariableOp
d13/kernel* 
_output_shapes
:
�	�*
dtype0
i
d13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
d13/bias
b
d13/bias/Read/ReadVariableOpReadVariableOpd13/bias*
_output_shapes	
:�*
dtype0
q

d15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_name
d15/kernel
j
d15/kernel/Read/ReadVariableOpReadVariableOp
d15/kernel*
_output_shapes
:	�
*
dtype0
h
d15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
d15/bias
a
d15/bias/Read/ReadVariableOpReadVariableOpd15/bias*
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
�
Adam/c0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/c0/kernel/m
}
$Adam/c0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/m*&
_output_shapes
: *
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
�
Adam/c1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameAdam/c1/kernel/m
}
$Adam/c1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c1/kernel/m*&
_output_shapes
:  *
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
�
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
�
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
�
Adam/c8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameAdam/c8/kernel/m
~
$Adam/c8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c8/kernel/m*'
_output_shapes
:@�*
dtype0
u
Adam/c8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/c8/bias/m
n
"Adam/c8/bias/m/Read/ReadVariableOpReadVariableOpAdam/c8/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/c9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameAdam/c9/kernel/m

$Adam/c9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c9/kernel/m*(
_output_shapes
:��*
dtype0
u
Adam/c9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/c9/bias/m
n
"Adam/c9/bias/m/Read/ReadVariableOpReadVariableOpAdam/c9/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/d13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*"
shared_nameAdam/d13/kernel/m
y
%Adam/d13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d13/kernel/m* 
_output_shapes
:
�	�*
dtype0
w
Adam/d13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameAdam/d13/bias/m
p
#Adam/d13/bias/m/Read/ReadVariableOpReadVariableOpAdam/d13/bias/m*
_output_shapes	
:�*
dtype0

Adam/d15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*"
shared_nameAdam/d15/kernel/m
x
%Adam/d15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d15/kernel/m*
_output_shapes
:	�
*
dtype0
v
Adam/d15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/d15/bias/m
o
#Adam/d15/bias/m/Read/ReadVariableOpReadVariableOpAdam/d15/bias/m*
_output_shapes
:
*
dtype0
�
Adam/c0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/c0/kernel/v
}
$Adam/c0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c0/kernel/v*&
_output_shapes
: *
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
�
Adam/c1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameAdam/c1/kernel/v
}
$Adam/c1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c1/kernel/v*&
_output_shapes
:  *
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
�
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
�
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
�
Adam/c8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameAdam/c8/kernel/v
~
$Adam/c8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c8/kernel/v*'
_output_shapes
:@�*
dtype0
u
Adam/c8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/c8/bias/v
n
"Adam/c8/bias/v/Read/ReadVariableOpReadVariableOpAdam/c8/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/c9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameAdam/c9/kernel/v

$Adam/c9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c9/kernel/v*(
_output_shapes
:��*
dtype0
u
Adam/c9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/c9/bias/v
n
"Adam/c9/bias/v/Read/ReadVariableOpReadVariableOpAdam/c9/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/d13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*"
shared_nameAdam/d13/kernel/v
y
%Adam/d13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d13/kernel/v* 
_output_shapes
:
�	�*
dtype0
w
Adam/d13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameAdam/d13/bias/v
p
#Adam/d13/bias/v/Read/ReadVariableOpReadVariableOpAdam/d13/bias/v*
_output_shapes	
:�*
dtype0

Adam/d15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*"
shared_nameAdam/d15/kernel/v
x
%Adam/d15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d15/kernel/v*
_output_shapes
:	�
*
dtype0
v
Adam/d15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/d15/bias/v
o
#Adam/d15/bias/v/Read/ReadVariableOpReadVariableOpAdam/d15/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ۂ
valueЂB̂ BĂ
�
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
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4_random_generator
5__call__
*6&call_and_return_all_conditional_losses* 
�

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
�

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses* 
�

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses*
�

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n_random_generator
o__call__
*p&call_and_return_all_conditional_losses* 
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
�

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�"m�#m�7m�8m�?m�@m�Tm�Um�\m�]m�wm�xm�	�m�	�m�v�v�"v�#v�7v�8v�?v�@v�Tv�Uv�\v�]v�wv�xv�	�v�	�v�*
|
0
1
"2
#3
74
85
?6
@7
T8
U9
\10
]11
w12
x13
�14
�15*
|
0
1
"2
#3
74
85
?6
@7
T8
U9
\10
]11
w12
x13
�14
�15*
�
�0
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
�15* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

�serving_default* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUE	c1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
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
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 
* 
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
0	variables
1trainable_variables
2regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	c4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUE	c5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
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
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 
* 
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
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 
* 
* 
* 
YS
VARIABLE_VALUE	c8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

T0
U1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUE	c9/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc9/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

\0
]1*

\0
]1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*
* 
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
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 
* 
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
j	variables
ktrainable_variables
lregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 
* 
* 
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
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 
* 
* 
ZT
VARIABLE_VALUE
d13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEd13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

w0
x1*

w0
x1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
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
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
ZT
VARIABLE_VALUE
d15/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEd15/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
* 
* 
* 
* 
z
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
11
12
13
14
15*

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 

�0
�1* 
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

�0
�1* 
* 
* 
* 
* 

�0
�1* 
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

�0
�1* 
* 
* 
* 
* 

�0
�1* 
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

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
<

�total

�count
�	variables
�	keras_api*
M

�total

�count
�
_fn_kwargs
�	variables
�	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
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
VARIABLE_VALUEAdam/c8/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c8/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c9/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c9/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/d13/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/d13/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/d15/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/d15/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/c8/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c8/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/c9/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/c9/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/d13/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/d13/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/d15/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/d15/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_c0_inputPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	c8/kernelc8/bias	c9/kernelc9/bias
d13/kerneld13/bias
d15/kerneld15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *,
f'R%
#__inference_signature_wrapper_20753
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamec0/kernel/Read/ReadVariableOpc0/bias/Read/ReadVariableOpc1/kernel/Read/ReadVariableOpc1/bias/Read/ReadVariableOpc4/kernel/Read/ReadVariableOpc4/bias/Read/ReadVariableOpc5/kernel/Read/ReadVariableOpc5/bias/Read/ReadVariableOpc8/kernel/Read/ReadVariableOpc8/bias/Read/ReadVariableOpc9/kernel/Read/ReadVariableOpc9/bias/Read/ReadVariableOpd13/kernel/Read/ReadVariableOpd13/bias/Read/ReadVariableOpd15/kernel/Read/ReadVariableOpd15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$Adam/c0/kernel/m/Read/ReadVariableOp"Adam/c0/bias/m/Read/ReadVariableOp$Adam/c1/kernel/m/Read/ReadVariableOp"Adam/c1/bias/m/Read/ReadVariableOp$Adam/c4/kernel/m/Read/ReadVariableOp"Adam/c4/bias/m/Read/ReadVariableOp$Adam/c5/kernel/m/Read/ReadVariableOp"Adam/c5/bias/m/Read/ReadVariableOp$Adam/c8/kernel/m/Read/ReadVariableOp"Adam/c8/bias/m/Read/ReadVariableOp$Adam/c9/kernel/m/Read/ReadVariableOp"Adam/c9/bias/m/Read/ReadVariableOp%Adam/d13/kernel/m/Read/ReadVariableOp#Adam/d13/bias/m/Read/ReadVariableOp%Adam/d15/kernel/m/Read/ReadVariableOp#Adam/d15/bias/m/Read/ReadVariableOp$Adam/c0/kernel/v/Read/ReadVariableOp"Adam/c0/bias/v/Read/ReadVariableOp$Adam/c1/kernel/v/Read/ReadVariableOp"Adam/c1/bias/v/Read/ReadVariableOp$Adam/c4/kernel/v/Read/ReadVariableOp"Adam/c4/bias/v/Read/ReadVariableOp$Adam/c5/kernel/v/Read/ReadVariableOp"Adam/c5/bias/v/Read/ReadVariableOp$Adam/c8/kernel/v/Read/ReadVariableOp"Adam/c8/bias/v/Read/ReadVariableOp$Adam/c9/kernel/v/Read/ReadVariableOp"Adam/c9/bias/v/Read/ReadVariableOp%Adam/d13/kernel/v/Read/ReadVariableOp#Adam/d13/bias/v/Read/ReadVariableOp%Adam/d15/kernel/v/Read/ReadVariableOp#Adam/d15/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
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
__inference__traced_save_21368
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	c8/kernelc8/bias	c9/kernelc9/bias
d13/kerneld13/bias
d15/kerneld15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/c0/kernel/mAdam/c0/bias/mAdam/c1/kernel/mAdam/c1/bias/mAdam/c4/kernel/mAdam/c4/bias/mAdam/c5/kernel/mAdam/c5/bias/mAdam/c8/kernel/mAdam/c8/bias/mAdam/c9/kernel/mAdam/c9/bias/mAdam/d13/kernel/mAdam/d13/bias/mAdam/d15/kernel/mAdam/d15/bias/mAdam/c0/kernel/vAdam/c0/bias/vAdam/c1/kernel/vAdam/c1/bias/vAdam/c4/kernel/vAdam/c4/bias/vAdam/c5/kernel/vAdam/c5/bias/vAdam/c8/kernel/vAdam/c8/bias/vAdam/c9/kernel/vAdam/c9/bias/vAdam/d13/kernel/vAdam/d13/bias/vAdam/d15/kernel/vAdam/d15/bias/v*E
Tin>
<2:*
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
!__inference__traced_restore_21549��
�

^
?__inference_dr11_layer_call_and_return_conditional_losses_21008

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_c1_layer_call_fn_20788

inputs!
unknown:  
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
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_19716w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
>
"__inference_m6_layer_call_fn_20891

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
=__inference_m6_layer_call_and_return_conditional_losses_19662�
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
�

^
?__inference_dr11_layer_call_and_return_conditional_losses_19996

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0*
seed2����[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
=__inference_c9_layer_call_and_return_conditional_losses_20971

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
?
#__inference_f12_layer_call_fn_21013

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
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_19828a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
+
__inference_loss_fn_5_21124
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
�
+
__inference_loss_fn_8_21139
identity`
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
,
__inference_loss_fn_13_21164
identity_
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
IdentityIdentity#d13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
>__inference_d15_layer_call_and_return_conditional_losses_21094

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
a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
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
�
\
#__inference_dr7_layer_call_fn_20906

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
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_20039w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_c5_layer_call_and_return_conditional_losses_19762

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
"__inference_c8_layer_call_fn_20934

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_19789x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
+
__inference_loss_fn_4_21119
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
�
�
"__inference_c4_layer_call_fn_20849

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_19743w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_19650

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
�
�
>__inference_d13_layer_call_and_return_conditional_losses_21043

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
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
:����������a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
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
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_20479

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�	�

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_19892o
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
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
$__inference_dr14_layer_call_fn_21053

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
GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_19957p
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
�
�
=__inference_c4_layer_call_and_return_conditional_losses_19743

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
+
__inference_loss_fn_1_21104
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
�
\
>__inference_dr7_layer_call_and_return_conditional_losses_20911

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
"__inference_c9_layer_call_fn_20958

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_19808x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_19927
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�	�

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_19892o
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
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
c0_input
�
�
=__inference_c1_layer_call_and_return_conditional_losses_19716

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� `
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
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_20826

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
\
#__inference_dr3_layer_call_fn_20821

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
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_20082w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
=__inference_c8_layer_call_and_return_conditional_losses_20947

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
+
__inference_loss_fn_0_21099
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
�M
�
E__inference_sequential_layer_call_and_return_conditional_losses_20212

inputs"
c0_20147: 
c0_20149: "
c1_20152:  
c1_20154: "
c4_20159: @
c4_20161:@"
c5_20164:@@
c5_20166:@#
c8_20171:@�
c8_20173:	�$
c9_20176:��
c9_20178:	�
	d13_20184:
�	�
	d13_20186:	�
	d15_20190:	�

	d15_20192:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�dr11/StatefulPartitionedCall�dr14/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_20147c0_20149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_19697�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_20152c1_20154*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_19716�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_19650�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_20082�
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_20159c4_20161*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_19743�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_20164c5_20166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_19762�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_19662�
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_20039�
c8/StatefulPartitionedCallStatefulPartitionedCall$dr7/StatefulPartitionedCall:output:0c8_20171c8_20173*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_19789�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_20176c9_20178*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_19808�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_19674�
dr11/StatefulPartitionedCallStatefulPartitionedCallm10/PartitionedCall:output:0^dr7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_19996�
f12/PartitionedCallPartitionedCall%dr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_19828�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_20184	d13_20186*
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
GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_19843�
dr14/StatefulPartitionedCallStatefulPartitionedCall$d13/StatefulPartitionedCall:output:0^dr11/StatefulPartitionedCall*
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
GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_19957�
d15/StatefulPartitionedCallStatefulPartitionedCall%dr14/StatefulPartitionedCall:output:0	d15_20190	d15_20192*
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
GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_19869`
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
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall^dr11/StatefulPartitionedCall^dr14/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall2<
dr11/StatefulPartitionedCalldr11/StatefulPartitionedCall2<
dr14/StatefulPartitionedCalldr14/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr7/StatefulPartitionedCalldr7/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
=__inference_c0_layer_call_and_return_conditional_losses_20777

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� `
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
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�!
!__inference__traced_restore_21549
file_prefix4
assignvariableop_c0_kernel: (
assignvariableop_1_c0_bias: 6
assignvariableop_2_c1_kernel:  (
assignvariableop_3_c1_bias: 6
assignvariableop_4_c4_kernel: @(
assignvariableop_5_c4_bias:@6
assignvariableop_6_c5_kernel:@@(
assignvariableop_7_c5_bias:@7
assignvariableop_8_c8_kernel:@�)
assignvariableop_9_c8_bias:	�9
assignvariableop_10_c9_kernel:��*
assignvariableop_11_c9_bias:	�2
assignvariableop_12_d13_kernel:
�	�+
assignvariableop_13_d13_bias:	�1
assignvariableop_14_d15_kernel:	�
*
assignvariableop_15_d15_bias:
'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: >
$assignvariableop_25_adam_c0_kernel_m: 0
"assignvariableop_26_adam_c0_bias_m: >
$assignvariableop_27_adam_c1_kernel_m:  0
"assignvariableop_28_adam_c1_bias_m: >
$assignvariableop_29_adam_c4_kernel_m: @0
"assignvariableop_30_adam_c4_bias_m:@>
$assignvariableop_31_adam_c5_kernel_m:@@0
"assignvariableop_32_adam_c5_bias_m:@?
$assignvariableop_33_adam_c8_kernel_m:@�1
"assignvariableop_34_adam_c8_bias_m:	�@
$assignvariableop_35_adam_c9_kernel_m:��1
"assignvariableop_36_adam_c9_bias_m:	�9
%assignvariableop_37_adam_d13_kernel_m:
�	�2
#assignvariableop_38_adam_d13_bias_m:	�8
%assignvariableop_39_adam_d15_kernel_m:	�
1
#assignvariableop_40_adam_d15_bias_m:
>
$assignvariableop_41_adam_c0_kernel_v: 0
"assignvariableop_42_adam_c0_bias_v: >
$assignvariableop_43_adam_c1_kernel_v:  0
"assignvariableop_44_adam_c1_bias_v: >
$assignvariableop_45_adam_c4_kernel_v: @0
"assignvariableop_46_adam_c4_bias_v:@>
$assignvariableop_47_adam_c5_kernel_v:@@0
"assignvariableop_48_adam_c5_bias_v:@?
$assignvariableop_49_adam_c8_kernel_v:@�1
"assignvariableop_50_adam_c8_bias_v:	�@
$assignvariableop_51_adam_c9_kernel_v:��1
"assignvariableop_52_adam_c9_bias_v:	�9
%assignvariableop_53_adam_d13_kernel_v:
�	�2
#assignvariableop_54_adam_d13_bias_v:	�8
%assignvariableop_55_adam_d15_kernel_v:	�
1
#assignvariableop_56_adam_d15_bias_v:

identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9� 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_c0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_c0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_c1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_c1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_c4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_c4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_c5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_c5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_c8_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_c8_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_c9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_c9_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_d13_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_d13_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_d15_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_d15_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_c0_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_c0_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_c1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_c1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp$assignvariableop_29_adam_c4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_c4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_c5_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_c5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp$assignvariableop_33_adam_c8_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp"assignvariableop_34_adam_c8_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp$assignvariableop_35_adam_c9_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp"assignvariableop_36_adam_c9_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_d13_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_d13_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_d15_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_adam_d15_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp$assignvariableop_41_adam_c0_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp"assignvariableop_42_adam_c0_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp$assignvariableop_43_adam_c1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp"assignvariableop_44_adam_c1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp$assignvariableop_45_adam_c4_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp"assignvariableop_46_adam_c4_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp$assignvariableop_47_adam_c5_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp"assignvariableop_48_adam_c5_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp$assignvariableop_49_adam_c8_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp"assignvariableop_50_adam_c8_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp$assignvariableop_51_adam_c9_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp"assignvariableop_52_adam_c9_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp%assignvariableop_53_adam_d13_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp#assignvariableop_54_adam_d13_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp%assignvariableop_55_adam_d15_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp#assignvariableop_56_adam_d15_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
*__inference_sequential_layer_call_fn_20516

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�	�

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_20212o
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
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
Y
=__inference_m6_layer_call_and_return_conditional_losses_20896

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
�
\
>__inference_dr7_layer_call_and_return_conditional_losses_19774

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
]
$__inference_dr11_layer_call_fn_20991

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_19996x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
+
__inference_loss_fn_6_21129
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
�
@
$__inference_dr14_layer_call_fn_21048

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
GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_19854a
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
�

^
?__inference_dr14_layer_call_and_return_conditional_losses_21070

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
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
 *   >�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
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
,
__inference_loss_fn_12_21159
identitya
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    \
IdentityIdentity%d13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
=__inference_c8_layer_call_and_return_conditional_losses_19789

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
,
__inference_loss_fn_14_21169
identitya
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    \
IdentityIdentity%d15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
]
?__inference_dr11_layer_call_and_return_conditional_losses_20996

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
,
__inference_loss_fn_15_21174
identity_
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
IdentityIdentity#d15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�m
�
__inference__traced_save_21368
file_prefix(
$savev2_c0_kernel_read_readvariableop&
"savev2_c0_bias_read_readvariableop(
$savev2_c1_kernel_read_readvariableop&
"savev2_c1_bias_read_readvariableop(
$savev2_c4_kernel_read_readvariableop&
"savev2_c4_bias_read_readvariableop(
$savev2_c5_kernel_read_readvariableop&
"savev2_c5_bias_read_readvariableop(
$savev2_c8_kernel_read_readvariableop&
"savev2_c8_bias_read_readvariableop(
$savev2_c9_kernel_read_readvariableop&
"savev2_c9_bias_read_readvariableop)
%savev2_d13_kernel_read_readvariableop'
#savev2_d13_bias_read_readvariableop)
%savev2_d15_kernel_read_readvariableop'
#savev2_d15_bias_read_readvariableop(
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
+savev2_adam_c8_kernel_m_read_readvariableop-
)savev2_adam_c8_bias_m_read_readvariableop/
+savev2_adam_c9_kernel_m_read_readvariableop-
)savev2_adam_c9_bias_m_read_readvariableop0
,savev2_adam_d13_kernel_m_read_readvariableop.
*savev2_adam_d13_bias_m_read_readvariableop0
,savev2_adam_d15_kernel_m_read_readvariableop.
*savev2_adam_d15_bias_m_read_readvariableop/
+savev2_adam_c0_kernel_v_read_readvariableop-
)savev2_adam_c0_bias_v_read_readvariableop/
+savev2_adam_c1_kernel_v_read_readvariableop-
)savev2_adam_c1_bias_v_read_readvariableop/
+savev2_adam_c4_kernel_v_read_readvariableop-
)savev2_adam_c4_bias_v_read_readvariableop/
+savev2_adam_c5_kernel_v_read_readvariableop-
)savev2_adam_c5_bias_v_read_readvariableop/
+savev2_adam_c8_kernel_v_read_readvariableop-
)savev2_adam_c8_bias_v_read_readvariableop/
+savev2_adam_c9_kernel_v_read_readvariableop-
)savev2_adam_c9_bias_v_read_readvariableop0
,savev2_adam_d13_kernel_v_read_readvariableop.
*savev2_adam_d13_bias_v_read_readvariableop0
,savev2_adam_d15_kernel_v_read_readvariableop.
*savev2_adam_d15_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
: � 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_c0_kernel_read_readvariableop"savev2_c0_bias_read_readvariableop$savev2_c1_kernel_read_readvariableop"savev2_c1_bias_read_readvariableop$savev2_c4_kernel_read_readvariableop"savev2_c4_bias_read_readvariableop$savev2_c5_kernel_read_readvariableop"savev2_c5_bias_read_readvariableop$savev2_c8_kernel_read_readvariableop"savev2_c8_bias_read_readvariableop$savev2_c9_kernel_read_readvariableop"savev2_c9_bias_read_readvariableop%savev2_d13_kernel_read_readvariableop#savev2_d13_bias_read_readvariableop%savev2_d15_kernel_read_readvariableop#savev2_d15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_adam_c0_kernel_m_read_readvariableop)savev2_adam_c0_bias_m_read_readvariableop+savev2_adam_c1_kernel_m_read_readvariableop)savev2_adam_c1_bias_m_read_readvariableop+savev2_adam_c4_kernel_m_read_readvariableop)savev2_adam_c4_bias_m_read_readvariableop+savev2_adam_c5_kernel_m_read_readvariableop)savev2_adam_c5_bias_m_read_readvariableop+savev2_adam_c8_kernel_m_read_readvariableop)savev2_adam_c8_bias_m_read_readvariableop+savev2_adam_c9_kernel_m_read_readvariableop)savev2_adam_c9_bias_m_read_readvariableop,savev2_adam_d13_kernel_m_read_readvariableop*savev2_adam_d13_bias_m_read_readvariableop,savev2_adam_d15_kernel_m_read_readvariableop*savev2_adam_d15_bias_m_read_readvariableop+savev2_adam_c0_kernel_v_read_readvariableop)savev2_adam_c0_bias_v_read_readvariableop+savev2_adam_c1_kernel_v_read_readvariableop)savev2_adam_c1_bias_v_read_readvariableop+savev2_adam_c4_kernel_v_read_readvariableop)savev2_adam_c4_bias_v_read_readvariableop+savev2_adam_c5_kernel_v_read_readvariableop)savev2_adam_c5_bias_v_read_readvariableop+savev2_adam_c8_kernel_v_read_readvariableop)savev2_adam_c8_bias_v_read_readvariableop+savev2_adam_c9_kernel_v_read_readvariableop)savev2_adam_c9_bias_v_read_readvariableop,savev2_adam_d13_kernel_v_read_readvariableop*savev2_adam_d13_bias_v_read_readvariableop,savev2_adam_d15_kernel_v_read_readvariableop*savev2_adam_d15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : @:@:@@:@:@�:�:��:�:
�	�:�:	�
:
: : : : : : : : : : : :  : : @:@:@@:@:@�:�:��:�:
�	�:�:	�
:
: : :  : : @:@:@@:@:@�:�:��:�:
�	�:�:	�
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
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 
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
:@:-	)
'
_output_shapes
:@�:!


_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:%!

_output_shapes
:	�
: 

_output_shapes
:
:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:-")
'
_output_shapes
:@�:!#

_output_shapes	
:�:.$*
(
_output_shapes
:��:!%

_output_shapes	
:�:&&"
 
_output_shapes
:
�	�:!'

_output_shapes	
:�:%(!

_output_shapes
:	�
: )

_output_shapes
:
:,*(
&
_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
:  : -

_output_shapes
: :,.(
&
_output_shapes
: @: /

_output_shapes
:@:,0(
&
_output_shapes
:@@: 1

_output_shapes
:@:-2)
'
_output_shapes
:@�:!3

_output_shapes	
:�:.4*
(
_output_shapes
:��:!5

_output_shapes	
:�:&6"
 
_output_shapes
:
�	�:!7

_output_shapes	
:�:%8!

_output_shapes
:	�
: 9

_output_shapes
:
::

_output_shapes
: 
�
�
=__inference_c1_layer_call_and_return_conditional_losses_20801

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� `
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
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
]
?__inference_dr14_layer_call_and_return_conditional_losses_21058

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
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_20082

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
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
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
:��������� w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�V
�
E__inference_sequential_layer_call_and_return_conditional_losses_20601

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource:  0
"c1_biasadd_readvariableop_resource: ;
!c4_conv2d_readvariableop_resource: @0
"c4_biasadd_readvariableop_resource:@;
!c5_conv2d_readvariableop_resource:@@0
"c5_biasadd_readvariableop_resource:@<
!c8_conv2d_readvariableop_resource:@�1
"c8_biasadd_readvariableop_resource:	�=
!c9_conv2d_readvariableop_resource:��1
"c9_biasadd_readvariableop_resource:	�6
"d13_matmul_readvariableop_resource:
�	�2
#d13_biasadd_readvariableop_resource:	�5
"d15_matmul_readvariableop_resource:	�
1
#d15_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�c4/BiasAdd/ReadVariableOp�c4/Conv2D/ReadVariableOp�c5/BiasAdd/ReadVariableOp�c5/Conv2D/ReadVariableOp�c8/BiasAdd/ReadVariableOp�c8/Conv2D/ReadVariableOp�c9/BiasAdd/ReadVariableOp�c9/Conv2D/ReadVariableOp�d13/BiasAdd/ReadVariableOp�d13/MatMul/ReadVariableOp�d15/BiasAdd/ReadVariableOp�d15/MatMul/ReadVariableOp�
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
g
dr3/IdentityIdentitym2/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
	c4/Conv2DConv2Ddr3/Identity:output:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c4/BiasAdd/ReadVariableOpReadVariableOp"c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c4/BiasAddBiasAddc4/Conv2D:output:0!c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c5/BiasAdd/ReadVariableOpReadVariableOp"c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c5/BiasAddBiasAddc5/Conv2D:output:0!c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
g
dr7/IdentityIdentitym6/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
c8/Conv2D/ReadVariableOpReadVariableOp!c8_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
	c8/Conv2DConv2Ddr7/Identity:output:0 c8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c8/BiasAdd/ReadVariableOpReadVariableOp"c8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c8/BiasAddBiasAddc8/Conv2D:output:0!c8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c8/ReluReluc8/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
c9/Conv2D/ReadVariableOpReadVariableOp!c9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
	c9/Conv2DConv2Dc8/Relu:activations:0 c9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c9/BiasAdd/ReadVariableOpReadVariableOp"c9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c9/BiasAddBiasAddc9/Conv2D:output:0!c9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c9/ReluReluc9/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
m10/MaxPoolMaxPoolc9/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
j
dr11/IdentityIdentitym10/MaxPool:output:0*
T0*0
_output_shapes
:����������Z
	f12/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  u
f12/ReshapeReshapedr11/Identity:output:0f12/Const:output:0*
T0*(
_output_shapes
:����������	~
d13/MatMul/ReadVariableOpReadVariableOp"d13_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0�

d13/MatMulMatMulf12/Reshape:output:0!d13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
d13/BiasAdd/ReadVariableOpReadVariableOp#d13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
d13/BiasAddBiasAddd13/MatMul:product:0"d13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Y
d13/ReluRelud13/BiasAdd:output:0*
T0*(
_output_shapes
:����������d
dr14/IdentityIdentityd13/Relu:activations:0*
T0*(
_output_shapes
:����������}
d15/MatMul/ReadVariableOpReadVariableOp"d15_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�

d15/MatMulMatMuldr14/Identity:output:0!d15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
d15/BiasAdd/ReadVariableOpReadVariableOp#d15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
d15/BiasAddBiasAddd15/MatMul:product:0"d15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
^
d15/SoftmaxSoftmaxd15/BiasAdd:output:0*
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
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentityd15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^c8/BiasAdd/ReadVariableOp^c8/Conv2D/ReadVariableOp^c9/BiasAdd/ReadVariableOp^c9/Conv2D/ReadVariableOp^d13/BiasAdd/ReadVariableOp^d13/MatMul/ReadVariableOp^d15/BiasAdd/ReadVariableOp^d15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
c4/BiasAdd/ReadVariableOpc4/BiasAdd/ReadVariableOp24
c4/Conv2D/ReadVariableOpc4/Conv2D/ReadVariableOp26
c5/BiasAdd/ReadVariableOpc5/BiasAdd/ReadVariableOp24
c5/Conv2D/ReadVariableOpc5/Conv2D/ReadVariableOp26
c8/BiasAdd/ReadVariableOpc8/BiasAdd/ReadVariableOp24
c8/Conv2D/ReadVariableOpc8/Conv2D/ReadVariableOp26
c9/BiasAdd/ReadVariableOpc9/BiasAdd/ReadVariableOp24
c9/Conv2D/ReadVariableOpc9/Conv2D/ReadVariableOp28
d13/BiasAdd/ReadVariableOpd13/BiasAdd/ReadVariableOp26
d13/MatMul/ReadVariableOpd13/MatMul/ReadVariableOp28
d15/BiasAdd/ReadVariableOpd15/BiasAdd/ReadVariableOp26
d15/MatMul/ReadVariableOpd15/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_19728

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�H
�
E__inference_sequential_layer_call_and_return_conditional_losses_20352
c0_input"
c0_20287: 
c0_20289: "
c1_20292:  
c1_20294: "
c4_20299: @
c4_20301:@"
c5_20304:@@
c5_20306:@#
c8_20311:@�
c8_20313:	�$
c9_20316:��
c9_20318:	�
	d13_20324:
�	�
	d13_20326:	�
	d15_20330:	�

	d15_20332:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_20287c0_20289*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_19697�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_20292c1_20294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_19716�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_19650�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_19728�
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_20299c4_20301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_19743�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_20304c5_20306*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_19762�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_19662�
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_19774�
c8/StatefulPartitionedCallStatefulPartitionedCalldr7/PartitionedCall:output:0c8_20311c8_20313*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_19789�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_20316c9_20318*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_19808�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_19674�
dr11/PartitionedCallPartitionedCallm10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_19820�
f12/PartitionedCallPartitionedCalldr11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_19828�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_20324	d13_20326*
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
GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_19843�
dr14/PartitionedCallPartitionedCall$d13/StatefulPartitionedCall:output:0*
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
GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_19854�
d15/StatefulPartitionedCallStatefulPartitionedCalldr14/PartitionedCall:output:0	d15_20330	d15_20332*
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
GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_19869`
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
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
c0_input
�
>
"__inference_m2_layer_call_fn_20806

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
=__inference_m2_layer_call_and_return_conditional_losses_19650�
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
�
?
#__inference_m10_layer_call_fn_20976

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
GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_19674�
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
�
�
>__inference_d13_layer_call_and_return_conditional_losses_19843

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
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
:����������a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
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
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
�
=__inference_c5_layer_call_and_return_conditional_losses_20886

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
?
#__inference_dr7_layer_call_fn_20901

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
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_19774h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
"__inference_c0_layer_call_fn_20764

inputs!
unknown: 
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
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_19697w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_20838

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
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
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
:��������� w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

^
?__inference_dr14_layer_call_and_return_conditional_losses_19957

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
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
 *   >�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
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
#__inference_d13_layer_call_fn_21030

inputs
unknown:
�	�
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
GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_19843p
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
:����������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
Z
>__inference_m10_layer_call_and_return_conditional_losses_20981

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
�

]
>__inference_dr7_layer_call_and_return_conditional_losses_20039

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
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
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
:���������@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
Z
>__inference_f12_layer_call_and_return_conditional_losses_19828

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_c5_layer_call_fn_20873

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_19762w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
,
__inference_loss_fn_11_21154
identity^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
#__inference_signature_wrapper_20753
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�	�

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *)
f$R"
 __inference__wrapped_model_19641o
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
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
c0_input
�
Z
>__inference_f12_layer_call_and_return_conditional_losses_21019

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
=__inference_c0_layer_call_and_return_conditional_losses_19697

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� `
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
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
Y
=__inference_m6_layer_call_and_return_conditional_losses_19662

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
+
__inference_loss_fn_2_21109
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
�
+
__inference_loss_fn_3_21114
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
>__inference_d15_layer_call_and_return_conditional_losses_19869

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
a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
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
�

]
>__inference_dr7_layer_call_and_return_conditional_losses_20923

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
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
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
:���������@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
,
__inference_loss_fn_10_21149
identity`
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    [
IdentityIdentity$c9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�Z
�
 __inference__wrapped_model_19641
c0_inputF
,sequential_c0_conv2d_readvariableop_resource: ;
-sequential_c0_biasadd_readvariableop_resource: F
,sequential_c1_conv2d_readvariableop_resource:  ;
-sequential_c1_biasadd_readvariableop_resource: F
,sequential_c4_conv2d_readvariableop_resource: @;
-sequential_c4_biasadd_readvariableop_resource:@F
,sequential_c5_conv2d_readvariableop_resource:@@;
-sequential_c5_biasadd_readvariableop_resource:@G
,sequential_c8_conv2d_readvariableop_resource:@�<
-sequential_c8_biasadd_readvariableop_resource:	�H
,sequential_c9_conv2d_readvariableop_resource:��<
-sequential_c9_biasadd_readvariableop_resource:	�A
-sequential_d13_matmul_readvariableop_resource:
�	�=
.sequential_d13_biasadd_readvariableop_resource:	�@
-sequential_d15_matmul_readvariableop_resource:	�
<
.sequential_d15_biasadd_readvariableop_resource:

identity��$sequential/c0/BiasAdd/ReadVariableOp�#sequential/c0/Conv2D/ReadVariableOp�$sequential/c1/BiasAdd/ReadVariableOp�#sequential/c1/Conv2D/ReadVariableOp�$sequential/c4/BiasAdd/ReadVariableOp�#sequential/c4/Conv2D/ReadVariableOp�$sequential/c5/BiasAdd/ReadVariableOp�#sequential/c5/Conv2D/ReadVariableOp�$sequential/c8/BiasAdd/ReadVariableOp�#sequential/c8/Conv2D/ReadVariableOp�$sequential/c9/BiasAdd/ReadVariableOp�#sequential/c9/Conv2D/ReadVariableOp�%sequential/d13/BiasAdd/ReadVariableOp�$sequential/d13/MatMul/ReadVariableOp�%sequential/d15/BiasAdd/ReadVariableOp�$sequential/d15/MatMul/ReadVariableOp�
#sequential/c0/Conv2D/ReadVariableOpReadVariableOp,sequential_c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential/c0/Conv2DConv2Dc0_input+sequential/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� t
sequential/c0/ReluRelusequential/c0/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
#sequential/c1/Conv2D/ReadVariableOpReadVariableOp,sequential_c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
sequential/c1/Conv2DConv2D sequential/c0/Relu:activations:0+sequential/c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
$sequential/c1/BiasAdd/ReadVariableOpReadVariableOp-sequential_c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential/c1/BiasAddBiasAddsequential/c1/Conv2D:output:0,sequential/c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� t
sequential/c1/ReluRelusequential/c1/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
sequential/m2/MaxPoolMaxPool sequential/c1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
}
sequential/dr3/IdentityIdentitysequential/m2/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
#sequential/c4/Conv2D/ReadVariableOpReadVariableOp,sequential_c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
sequential/c4/Conv2DConv2D sequential/dr3/Identity:output:0+sequential/c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
$sequential/c4/BiasAdd/ReadVariableOpReadVariableOp-sequential_c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/c4/BiasAddBiasAddsequential/c4/Conv2D:output:0,sequential/c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@t
sequential/c4/ReluRelusequential/c4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
#sequential/c5/Conv2D/ReadVariableOpReadVariableOp,sequential_c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
sequential/c5/Conv2DConv2D sequential/c4/Relu:activations:0+sequential/c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
$sequential/c5/BiasAdd/ReadVariableOpReadVariableOp-sequential_c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/c5/BiasAddBiasAddsequential/c5/Conv2D:output:0,sequential/c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@t
sequential/c5/ReluRelusequential/c5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
sequential/m6/MaxPoolMaxPool sequential/c5/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
}
sequential/dr7/IdentityIdentitysequential/m6/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
#sequential/c8/Conv2D/ReadVariableOpReadVariableOp,sequential_c8_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
sequential/c8/Conv2DConv2D sequential/dr7/Identity:output:0+sequential/c8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$sequential/c8/BiasAdd/ReadVariableOpReadVariableOp-sequential_c8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/c8/BiasAddBiasAddsequential/c8/Conv2D:output:0,sequential/c8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������u
sequential/c8/ReluRelusequential/c8/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
#sequential/c9/Conv2D/ReadVariableOpReadVariableOp,sequential_c9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
sequential/c9/Conv2DConv2D sequential/c8/Relu:activations:0+sequential/c9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$sequential/c9/BiasAdd/ReadVariableOpReadVariableOp-sequential_c9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/c9/BiasAddBiasAddsequential/c9/Conv2D:output:0,sequential/c9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������u
sequential/c9/ReluRelusequential/c9/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
sequential/m10/MaxPoolMaxPool sequential/c9/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
sequential/dr11/IdentityIdentitysequential/m10/MaxPool:output:0*
T0*0
_output_shapes
:����������e
sequential/f12/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
sequential/f12/ReshapeReshape!sequential/dr11/Identity:output:0sequential/f12/Const:output:0*
T0*(
_output_shapes
:����������	�
$sequential/d13/MatMul/ReadVariableOpReadVariableOp-sequential_d13_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0�
sequential/d13/MatMulMatMulsequential/f12/Reshape:output:0,sequential/d13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%sequential/d13/BiasAdd/ReadVariableOpReadVariableOp.sequential_d13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/d13/BiasAddBiasAddsequential/d13/MatMul:product:0-sequential/d13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
sequential/d13/ReluRelusequential/d13/BiasAdd:output:0*
T0*(
_output_shapes
:����������z
sequential/dr14/IdentityIdentity!sequential/d13/Relu:activations:0*
T0*(
_output_shapes
:�����������
$sequential/d15/MatMul/ReadVariableOpReadVariableOp-sequential_d15_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
sequential/d15/MatMulMatMul!sequential/dr14/Identity:output:0,sequential/d15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
%sequential/d15/BiasAdd/ReadVariableOpReadVariableOp.sequential_d15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential/d15/BiasAddBiasAddsequential/d15/MatMul:product:0-sequential/d15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
sequential/d15/SoftmaxSoftmaxsequential/d15/BiasAdd:output:0*
T0*'
_output_shapes
:���������
o
IdentityIdentity sequential/d15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/c1/BiasAdd/ReadVariableOp$^sequential/c1/Conv2D/ReadVariableOp%^sequential/c4/BiasAdd/ReadVariableOp$^sequential/c4/Conv2D/ReadVariableOp%^sequential/c5/BiasAdd/ReadVariableOp$^sequential/c5/Conv2D/ReadVariableOp%^sequential/c8/BiasAdd/ReadVariableOp$^sequential/c8/Conv2D/ReadVariableOp%^sequential/c9/BiasAdd/ReadVariableOp$^sequential/c9/Conv2D/ReadVariableOp&^sequential/d13/BiasAdd/ReadVariableOp%^sequential/d13/MatMul/ReadVariableOp&^sequential/d15/BiasAdd/ReadVariableOp%^sequential/d15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2L
$sequential/c0/BiasAdd/ReadVariableOp$sequential/c0/BiasAdd/ReadVariableOp2J
#sequential/c0/Conv2D/ReadVariableOp#sequential/c0/Conv2D/ReadVariableOp2L
$sequential/c1/BiasAdd/ReadVariableOp$sequential/c1/BiasAdd/ReadVariableOp2J
#sequential/c1/Conv2D/ReadVariableOp#sequential/c1/Conv2D/ReadVariableOp2L
$sequential/c4/BiasAdd/ReadVariableOp$sequential/c4/BiasAdd/ReadVariableOp2J
#sequential/c4/Conv2D/ReadVariableOp#sequential/c4/Conv2D/ReadVariableOp2L
$sequential/c5/BiasAdd/ReadVariableOp$sequential/c5/BiasAdd/ReadVariableOp2J
#sequential/c5/Conv2D/ReadVariableOp#sequential/c5/Conv2D/ReadVariableOp2L
$sequential/c8/BiasAdd/ReadVariableOp$sequential/c8/BiasAdd/ReadVariableOp2J
#sequential/c8/Conv2D/ReadVariableOp#sequential/c8/Conv2D/ReadVariableOp2L
$sequential/c9/BiasAdd/ReadVariableOp$sequential/c9/BiasAdd/ReadVariableOp2J
#sequential/c9/Conv2D/ReadVariableOp#sequential/c9/Conv2D/ReadVariableOp2N
%sequential/d13/BiasAdd/ReadVariableOp%sequential/d13/BiasAdd/ReadVariableOp2L
$sequential/d13/MatMul/ReadVariableOp$sequential/d13/MatMul/ReadVariableOp2N
%sequential/d15/BiasAdd/ReadVariableOp%sequential/d15/BiasAdd/ReadVariableOp2L
$sequential/d15/MatMul/ReadVariableOp$sequential/d15/MatMul/ReadVariableOp:Y U
/
_output_shapes
:���������
"
_user_specified_name
c0_input
�
+
__inference_loss_fn_9_21144
identity^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentity"c8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
#__inference_d15_layer_call_fn_21081

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
GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_19869o
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
�M
�
E__inference_sequential_layer_call_and_return_conditional_losses_20420
c0_input"
c0_20355: 
c0_20357: "
c1_20360:  
c1_20362: "
c4_20367: @
c4_20369:@"
c5_20372:@@
c5_20374:@#
c8_20379:@�
c8_20381:	�$
c9_20384:��
c9_20386:	�
	d13_20392:
�	�
	d13_20394:	�
	d15_20398:	�

	d15_20400:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�dr11/StatefulPartitionedCall�dr14/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_20355c0_20357*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_19697�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_20360c1_20362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_19716�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_19650�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_20082�
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_20367c4_20369*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_19743�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_20372c5_20374*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_19762�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_19662�
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_20039�
c8/StatefulPartitionedCallStatefulPartitionedCall$dr7/StatefulPartitionedCall:output:0c8_20379c8_20381*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_19789�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_20384c9_20386*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_19808�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_19674�
dr11/StatefulPartitionedCallStatefulPartitionedCallm10/PartitionedCall:output:0^dr7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_19996�
f12/PartitionedCallPartitionedCall%dr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_19828�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_20392	d13_20394*
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
GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_19843�
dr14/StatefulPartitionedCallStatefulPartitionedCall$d13/StatefulPartitionedCall:output:0^dr11/StatefulPartitionedCall*
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
GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_19957�
d15/StatefulPartitionedCallStatefulPartitionedCall%dr14/StatefulPartitionedCall:output:0	d15_20398	d15_20400*
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
GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_19869`
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
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall^dr11/StatefulPartitionedCall^dr14/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall2<
dr11/StatefulPartitionedCalldr11/StatefulPartitionedCall2<
dr14/StatefulPartitionedCalldr14/StatefulPartitionedCall2:
dr3/StatefulPartitionedCalldr3/StatefulPartitionedCall2:
dr7/StatefulPartitionedCalldr7/StatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
c0_input
�
�
=__inference_c4_layer_call_and_return_conditional_losses_20862

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�H
�
E__inference_sequential_layer_call_and_return_conditional_losses_19892

inputs"
c0_19698: 
c0_19700: "
c1_19717:  
c1_19719: "
c4_19744: @
c4_19746:@"
c5_19763:@@
c5_19765:@#
c8_19790:@�
c8_19792:	�$
c9_19809:��
c9_19811:	�
	d13_19844:
�	�
	d13_19846:	�
	d15_19870:	�

	d15_19872:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�c8/StatefulPartitionedCall�c9/StatefulPartitionedCall�d13/StatefulPartitionedCall�d15/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_19698c0_19700*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_19697�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_19717c1_19719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_19716�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_19650�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_19728�
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_19744c4_19746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_19743�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_19763c5_19765*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_19762�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_19662�
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_19774�
c8/StatefulPartitionedCallStatefulPartitionedCalldr7/PartitionedCall:output:0c8_19790c8_19792*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c8_layer_call_and_return_conditional_losses_19789�
c9/StatefulPartitionedCallStatefulPartitionedCall#c8/StatefulPartitionedCall:output:0c9_19809c9_19811*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c9_layer_call_and_return_conditional_losses_19808�
m10/PartitionedCallPartitionedCall#c9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_m10_layer_call_and_return_conditional_losses_19674�
dr11/PartitionedCallPartitionedCallm10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_19820�
f12/PartitionedCallPartitionedCalldr11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_f12_layer_call_and_return_conditional_losses_19828�
d13/StatefulPartitionedCallStatefulPartitionedCallf12/PartitionedCall:output:0	d13_19844	d13_19846*
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
GPU(2*0J 8� *G
fBR@
>__inference_d13_layer_call_and_return_conditional_losses_19843�
dr14/PartitionedCallPartitionedCall$d13/StatefulPartitionedCall:output:0*
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
GPU(2*0J 8� *H
fCRA
?__inference_dr14_layer_call_and_return_conditional_losses_19854�
d15/StatefulPartitionedCallStatefulPartitionedCalldr14/PartitionedCall:output:0	d15_19870	d15_19872*
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
GPU(2*0J 8� *G
fBR@
>__inference_d15_layer_call_and_return_conditional_losses_19869`
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
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
IdentityIdentity$d15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^c8/StatefulPartitionedCall^c9/StatefulPartitionedCall^d13/StatefulPartitionedCall^d15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall28
c8/StatefulPartitionedCallc8/StatefulPartitionedCall28
c9/StatefulPartitionedCallc9/StatefulPartitionedCall2:
d13/StatefulPartitionedCalld13/StatefulPartitionedCall2:
d15/StatefulPartitionedCalld15/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
+
__inference_loss_fn_7_21134
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
�
?
#__inference_dr3_layer_call_fn_20816

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
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_19728h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
]
?__inference_dr14_layer_call_and_return_conditional_losses_19854

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
�
�
=__inference_c9_layer_call_and_return_conditional_losses_19808

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������`
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
Z
>__inference_m10_layer_call_and_return_conditional_losses_19674

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
@
$__inference_dr11_layer_call_fn_20986

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *H
fCRA
?__inference_dr11_layer_call_and_return_conditional_losses_19820i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_20284
c0_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�	�

unknown_12:	�

unknown_13:	�


unknown_14:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*2
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_20212o
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
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
c0_input
�r
�
E__inference_sequential_layer_call_and_return_conditional_losses_20714

inputs;
!c0_conv2d_readvariableop_resource: 0
"c0_biasadd_readvariableop_resource: ;
!c1_conv2d_readvariableop_resource:  0
"c1_biasadd_readvariableop_resource: ;
!c4_conv2d_readvariableop_resource: @0
"c4_biasadd_readvariableop_resource:@;
!c5_conv2d_readvariableop_resource:@@0
"c5_biasadd_readvariableop_resource:@<
!c8_conv2d_readvariableop_resource:@�1
"c8_biasadd_readvariableop_resource:	�=
!c9_conv2d_readvariableop_resource:��1
"c9_biasadd_readvariableop_resource:	�6
"d13_matmul_readvariableop_resource:
�	�2
#d13_biasadd_readvariableop_resource:	�5
"d15_matmul_readvariableop_resource:	�
1
#d15_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�c4/BiasAdd/ReadVariableOp�c4/Conv2D/ReadVariableOp�c5/BiasAdd/ReadVariableOp�c5/Conv2D/ReadVariableOp�c8/BiasAdd/ReadVariableOp�c8/Conv2D/ReadVariableOp�c9/BiasAdd/ReadVariableOp�c9/Conv2D/ReadVariableOp�d13/BiasAdd/ReadVariableOp�d13/MatMul/ReadVariableOp�d15/BiasAdd/ReadVariableOp�d15/MatMul/ReadVariableOp�
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
x
c1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�

c1/BiasAddBiasAddc1/Conv2D:output:0!c1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:��������� *
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
:��������� T
dr3/dropout/ShapeShapem2/MaxPool:output:0*
T0*
_output_shapes
:�
(dr3/dropout/random_uniform/RandomUniformRandomUniformdr3/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
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
:��������� 
dr3/dropout/CastCastdr3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� }
dr3/dropout/Mul_1Muldr3/dropout/Mul:z:0dr3/dropout/Cast:y:0*
T0*/
_output_shapes
:��������� �
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
	c4/Conv2DConv2Ddr3/dropout/Mul_1:z:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c4/BiasAdd/ReadVariableOpReadVariableOp"c4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c4/BiasAddBiasAddc4/Conv2D:output:0!c4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
x
c5/BiasAdd/ReadVariableOpReadVariableOp"c5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�

c5/BiasAddBiasAddc5/Conv2D:output:0!c5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:���������@*
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
 *%I�?�
dr7/dropout/MulMulm6/MaxPool:output:0dr7/dropout/Const:output:0*
T0*/
_output_shapes
:���������@T
dr7/dropout/ShapeShapem6/MaxPool:output:0*
T0*
_output_shapes
:�
(dr7/dropout/random_uniform/RandomUniformRandomUniformdr7/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype0*
seed2_
dr7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dr7/dropout/GreaterEqualGreaterEqual1dr7/dropout/random_uniform/RandomUniform:output:0#dr7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@
dr7/dropout/CastCastdr7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@}
dr7/dropout/Mul_1Muldr7/dropout/Mul:z:0dr7/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@�
c8/Conv2D/ReadVariableOpReadVariableOp!c8_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
	c8/Conv2DConv2Ddr7/dropout/Mul_1:z:0 c8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c8/BiasAdd/ReadVariableOpReadVariableOp"c8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c8/BiasAddBiasAddc8/Conv2D:output:0!c8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c8/ReluReluc8/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
c9/Conv2D/ReadVariableOpReadVariableOp!c9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
	c9/Conv2DConv2Dc8/Relu:activations:0 c9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
y
c9/BiasAdd/ReadVariableOpReadVariableOp"c9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

c9/BiasAddBiasAddc9/Conv2D:output:0!c9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������_
c9/ReluReluc9/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
m10/MaxPoolMaxPoolc9/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
W
dr11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?�
dr11/dropout/MulMulm10/MaxPool:output:0dr11/dropout/Const:output:0*
T0*0
_output_shapes
:����������V
dr11/dropout/ShapeShapem10/MaxPool:output:0*
T0*
_output_shapes
:�
)dr11/dropout/random_uniform/RandomUniformRandomUniformdr11/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0*
seed2`
dr11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dr11/dropout/GreaterEqualGreaterEqual2dr11/dropout/random_uniform/RandomUniform:output:0$dr11/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:�����������
dr11/dropout/CastCastdr11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:�����������
dr11/dropout/Mul_1Muldr11/dropout/Mul:z:0dr11/dropout/Cast:y:0*
T0*0
_output_shapes
:����������Z
	f12/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  u
f12/ReshapeReshapedr11/dropout/Mul_1:z:0f12/Const:output:0*
T0*(
_output_shapes
:����������	~
d13/MatMul/ReadVariableOpReadVariableOp"d13_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0�

d13/MatMulMatMulf12/Reshape:output:0!d13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
d13/BiasAdd/ReadVariableOpReadVariableOp#d13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
d13/BiasAddBiasAddd13/MatMul:product:0"d13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Y
d13/ReluRelud13/BiasAdd:output:0*
T0*(
_output_shapes
:����������W
dr14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?
dr14/dropout/MulMuld13/Relu:activations:0dr14/dropout/Const:output:0*
T0*(
_output_shapes
:����������X
dr14/dropout/ShapeShaped13/Relu:activations:0*
T0*
_output_shapes
:�
)dr14/dropout/random_uniform/RandomUniformRandomUniformdr14/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2`
dr14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dr14/dropout/GreaterEqualGreaterEqual2dr14/dropout/random_uniform/RandomUniform:output:0$dr14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������z
dr14/dropout/CastCastdr14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������y
dr14/dropout/Mul_1Muldr14/dropout/Mul:z:0dr14/dropout/Cast:y:0*
T0*(
_output_shapes
:����������}
d15/MatMul/ReadVariableOpReadVariableOp"d15_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�

d15/MatMulMatMuldr14/dropout/Mul_1:z:0!d15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
d15/BiasAdd/ReadVariableOpReadVariableOp#d15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
d15/BiasAddBiasAddd15/MatMul:product:0"d15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
^
d15/SoftmaxSoftmaxd15/BiasAdd:output:0*
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
c8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c8/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    `
c9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
c9/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d13/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
d15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
d15/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    d
IdentityIdentityd15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^c8/BiasAdd/ReadVariableOp^c8/Conv2D/ReadVariableOp^c9/BiasAdd/ReadVariableOp^c9/Conv2D/ReadVariableOp^d13/BiasAdd/ReadVariableOp^d13/MatMul/ReadVariableOp^d15/BiasAdd/ReadVariableOp^d15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 26
c0/BiasAdd/ReadVariableOpc0/BiasAdd/ReadVariableOp24
c0/Conv2D/ReadVariableOpc0/Conv2D/ReadVariableOp26
c1/BiasAdd/ReadVariableOpc1/BiasAdd/ReadVariableOp24
c1/Conv2D/ReadVariableOpc1/Conv2D/ReadVariableOp26
c4/BiasAdd/ReadVariableOpc4/BiasAdd/ReadVariableOp24
c4/Conv2D/ReadVariableOpc4/Conv2D/ReadVariableOp26
c5/BiasAdd/ReadVariableOpc5/BiasAdd/ReadVariableOp24
c5/Conv2D/ReadVariableOpc5/Conv2D/ReadVariableOp26
c8/BiasAdd/ReadVariableOpc8/BiasAdd/ReadVariableOp24
c8/Conv2D/ReadVariableOpc8/Conv2D/ReadVariableOp26
c9/BiasAdd/ReadVariableOpc9/BiasAdd/ReadVariableOp24
c9/Conv2D/ReadVariableOpc9/Conv2D/ReadVariableOp28
d13/BiasAdd/ReadVariableOpd13/BiasAdd/ReadVariableOp26
d13/MatMul/ReadVariableOpd13/MatMul/ReadVariableOp28
d15/BiasAdd/ReadVariableOpd15/BiasAdd/ReadVariableOp26
d15/MatMul/ReadVariableOpd15/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_20811

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
�
]
?__inference_dr11_layer_call_and_return_conditional_losses_19820

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs"�L
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
serving_default_c0_input:0���������7
d150
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
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
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
�

"kernel
#bias
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
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4_random_generator
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
�

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
�

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n_random_generator
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
�

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�"m�#m�7m�8m�?m�@m�Tm�Um�\m�]m�wm�xm�	�m�	�m�v�v�"v�#v�7v�8v�?v�@v�Tv�Uv�\v�]v�wv�xv�	�v�	�v�"
	optimizer
�
0
1
"2
#3
74
85
?6
@7
T8
U9
\10
]11
w12
x13
�14
�15"
trackable_list_wrapper
�
0
1
"2
#3
74
85
?6
@7
T8
U9
\10
]11
w12
x13
�14
�15"
trackable_list_wrapper
�
�0
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
�15"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_sequential_layer_call_fn_19927
*__inference_sequential_layer_call_fn_20479
*__inference_sequential_layer_call_fn_20516
*__inference_sequential_layer_call_fn_20284�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_sequential_layer_call_and_return_conditional_losses_20601
E__inference_sequential_layer_call_and_return_conditional_losses_20714
E__inference_sequential_layer_call_and_return_conditional_losses_20352
E__inference_sequential_layer_call_and_return_conditional_losses_20420�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_19641c0_input"�
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
-
�serving_default"
signature_map
#:! 2	c0/kernel
: 2c0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�2�
"__inference_c0_layer_call_fn_20764�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_c0_layer_call_and_return_conditional_losses_20777�
���
FullArgSpec
args�
jself
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
annotations� *
 
#:!  2	c1/kernel
: 2c1/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�2�
"__inference_c1_layer_call_fn_20788�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_c1_layer_call_and_return_conditional_losses_20801�
���
FullArgSpec
args�
jself
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
annotations� *
 
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
�2�
"__inference_m2_layer_call_fn_20806�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_m2_layer_call_and_return_conditional_losses_20811�
���
FullArgSpec
args�
jself
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
annotations� *
 
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
0	variables
1trainable_variables
2regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
#__inference_dr3_layer_call_fn_20816
#__inference_dr3_layer_call_fn_20821�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_dr3_layer_call_and_return_conditional_losses_20826
>__inference_dr3_layer_call_and_return_conditional_losses_20838�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
#:! @2	c4/kernel
:@2c4/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�2�
"__inference_c4_layer_call_fn_20849�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_c4_layer_call_and_return_conditional_losses_20862�
���
FullArgSpec
args�
jself
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
annotations� *
 
#:!@@2	c5/kernel
:@2c5/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�2�
"__inference_c5_layer_call_fn_20873�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_c5_layer_call_and_return_conditional_losses_20886�
���
FullArgSpec
args�
jself
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
annotations� *
 
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
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�2�
"__inference_m6_layer_call_fn_20891�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_m6_layer_call_and_return_conditional_losses_20896�
���
FullArgSpec
args�
jself
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
annotations� *
 
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
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
#__inference_dr7_layer_call_fn_20901
#__inference_dr7_layer_call_fn_20906�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_dr7_layer_call_and_return_conditional_losses_20911
>__inference_dr7_layer_call_and_return_conditional_losses_20923�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
$:"@�2	c8/kernel
:�2c8/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�2�
"__inference_c8_layer_call_fn_20934�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_c8_layer_call_and_return_conditional_losses_20947�
���
FullArgSpec
args�
jself
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
annotations� *
 
%:#��2	c9/kernel
:�2c9/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�2�
"__inference_c9_layer_call_fn_20958�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
=__inference_c9_layer_call_and_return_conditional_losses_20971�
���
FullArgSpec
args�
jself
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
annotations� *
 
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
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�2�
#__inference_m10_layer_call_fn_20976�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
>__inference_m10_layer_call_and_return_conditional_losses_20981�
���
FullArgSpec
args�
jself
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
annotations� *
 
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
j	variables
ktrainable_variables
lregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
$__inference_dr11_layer_call_fn_20986
$__inference_dr11_layer_call_fn_20991�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_dr11_layer_call_and_return_conditional_losses_20996
?__inference_dr11_layer_call_and_return_conditional_losses_21008�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�2�
#__inference_f12_layer_call_fn_21013�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
>__inference_f12_layer_call_and_return_conditional_losses_21019�
���
FullArgSpec
args�
jself
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
annotations� *
 
:
�	�2
d13/kernel
:�2d13/bias
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�2�
#__inference_d13_layer_call_fn_21030�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
>__inference_d13_layer_call_and_return_conditional_losses_21043�
���
FullArgSpec
args�
jself
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
annotations� *
 
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
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
$__inference_dr14_layer_call_fn_21048
$__inference_dr14_layer_call_fn_21053�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_dr14_layer_call_and_return_conditional_losses_21058
?__inference_dr14_layer_call_and_return_conditional_losses_21070�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	�
2
d15/kernel
:
2d15/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
#__inference_d15_layer_call_fn_21081�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
>__inference_d15_layer_call_and_return_conditional_losses_21094�
���
FullArgSpec
args�
jself
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
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�2�
__inference_loss_fn_0_21099�
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
�2�
__inference_loss_fn_1_21104�
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
�2�
__inference_loss_fn_2_21109�
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
�2�
__inference_loss_fn_3_21114�
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
�2�
__inference_loss_fn_4_21119�
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
�2�
__inference_loss_fn_5_21124�
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
�2�
__inference_loss_fn_6_21129�
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
�2�
__inference_loss_fn_7_21134�
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
�2�
__inference_loss_fn_8_21139�
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
�2�
__inference_loss_fn_9_21144�
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
�2�
__inference_loss_fn_10_21149�
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
�2�
__inference_loss_fn_11_21154�
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
�2�
__inference_loss_fn_12_21159�
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
�2�
__inference_loss_fn_13_21164�
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
�2�
__inference_loss_fn_14_21169�
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
�2�
__inference_loss_fn_15_21174�
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
 "
trackable_list_wrapper
�
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
11
12
13
14
15"
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
#__inference_signature_wrapper_20753c0_input"�
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
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
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
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
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
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
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
0
�0
�1"
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
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
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
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
(:& 2Adam/c0/kernel/m
: 2Adam/c0/bias/m
(:&  2Adam/c1/kernel/m
: 2Adam/c1/bias/m
(:& @2Adam/c4/kernel/m
:@2Adam/c4/bias/m
(:&@@2Adam/c5/kernel/m
:@2Adam/c5/bias/m
):'@�2Adam/c8/kernel/m
:�2Adam/c8/bias/m
*:(��2Adam/c9/kernel/m
:�2Adam/c9/bias/m
#:!
�	�2Adam/d13/kernel/m
:�2Adam/d13/bias/m
": 	�
2Adam/d15/kernel/m
:
2Adam/d15/bias/m
(:& 2Adam/c0/kernel/v
: 2Adam/c0/bias/v
(:&  2Adam/c1/kernel/v
: 2Adam/c1/bias/v
(:& @2Adam/c4/kernel/v
:@2Adam/c4/bias/v
(:&@@2Adam/c5/kernel/v
:@2Adam/c5/bias/v
):'@�2Adam/c8/kernel/v
:�2Adam/c8/bias/v
*:(��2Adam/c9/kernel/v
:�2Adam/c9/bias/v
#:!
�	�2Adam/d13/kernel/v
:�2Adam/d13/bias/v
": 	�
2Adam/d15/kernel/v
:
2Adam/d15/bias/v�
 __inference__wrapped_model_19641z"#78?@TU\]wx��9�6
/�,
*�'
c0_input���������
� ")�&
$
d15�
d15���������
�
=__inference_c0_layer_call_and_return_conditional_losses_20777l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
"__inference_c0_layer_call_fn_20764_7�4
-�*
(�%
inputs���������
� " ���������� �
=__inference_c1_layer_call_and_return_conditional_losses_20801l"#7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
"__inference_c1_layer_call_fn_20788_"#7�4
-�*
(�%
inputs��������� 
� " ���������� �
=__inference_c4_layer_call_and_return_conditional_losses_20862l787�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
"__inference_c4_layer_call_fn_20849_787�4
-�*
(�%
inputs��������� 
� " ����������@�
=__inference_c5_layer_call_and_return_conditional_losses_20886l?@7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
"__inference_c5_layer_call_fn_20873_?@7�4
-�*
(�%
inputs���������@
� " ����������@�
=__inference_c8_layer_call_and_return_conditional_losses_20947mTU7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
"__inference_c8_layer_call_fn_20934`TU7�4
-�*
(�%
inputs���������@
� "!������������
=__inference_c9_layer_call_and_return_conditional_losses_20971n\]8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
"__inference_c9_layer_call_fn_20958a\]8�5
.�+
)�&
inputs����������
� "!������������
>__inference_d13_layer_call_and_return_conditional_losses_21043^wx0�-
&�#
!�
inputs����������	
� "&�#
�
0����������
� x
#__inference_d13_layer_call_fn_21030Qwx0�-
&�#
!�
inputs����������	
� "������������
>__inference_d15_layer_call_and_return_conditional_losses_21094_��0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� y
#__inference_d15_layer_call_fn_21081R��0�-
&�#
!�
inputs����������
� "����������
�
?__inference_dr11_layer_call_and_return_conditional_losses_20996n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
?__inference_dr11_layer_call_and_return_conditional_losses_21008n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
$__inference_dr11_layer_call_fn_20986a<�9
2�/
)�&
inputs����������
p 
� "!������������
$__inference_dr11_layer_call_fn_20991a<�9
2�/
)�&
inputs����������
p
� "!������������
?__inference_dr14_layer_call_and_return_conditional_losses_21058^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
?__inference_dr14_layer_call_and_return_conditional_losses_21070^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� y
$__inference_dr14_layer_call_fn_21048Q4�1
*�'
!�
inputs����������
p 
� "�����������y
$__inference_dr14_layer_call_fn_21053Q4�1
*�'
!�
inputs����������
p
� "������������
>__inference_dr3_layer_call_and_return_conditional_losses_20826l;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
>__inference_dr3_layer_call_and_return_conditional_losses_20838l;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
#__inference_dr3_layer_call_fn_20816_;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
#__inference_dr3_layer_call_fn_20821_;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
>__inference_dr7_layer_call_and_return_conditional_losses_20911l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
>__inference_dr7_layer_call_and_return_conditional_losses_20923l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
#__inference_dr7_layer_call_fn_20901_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
#__inference_dr7_layer_call_fn_20906_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
>__inference_f12_layer_call_and_return_conditional_losses_21019b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������	
� |
#__inference_f12_layer_call_fn_21013U8�5
.�+
)�&
inputs����������
� "�����������	7
__inference_loss_fn_0_21099�

� 
� "� 8
__inference_loss_fn_10_21149�

� 
� "� 8
__inference_loss_fn_11_21154�

� 
� "� 8
__inference_loss_fn_12_21159�

� 
� "� 8
__inference_loss_fn_13_21164�

� 
� "� 8
__inference_loss_fn_14_21169�

� 
� "� 8
__inference_loss_fn_15_21174�

� 
� "� 7
__inference_loss_fn_1_21104�

� 
� "� 7
__inference_loss_fn_2_21109�

� 
� "� 7
__inference_loss_fn_3_21114�

� 
� "� 7
__inference_loss_fn_4_21119�

� 
� "� 7
__inference_loss_fn_5_21124�

� 
� "� 7
__inference_loss_fn_6_21129�

� 
� "� 7
__inference_loss_fn_7_21134�

� 
� "� 7
__inference_loss_fn_8_21139�

� 
� "� 7
__inference_loss_fn_9_21144�

� 
� "� �
>__inference_m10_layer_call_and_return_conditional_losses_20981�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
#__inference_m10_layer_call_fn_20976�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
=__inference_m2_layer_call_and_return_conditional_losses_20811�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
"__inference_m2_layer_call_fn_20806�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
=__inference_m6_layer_call_and_return_conditional_losses_20896�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
"__inference_m6_layer_call_fn_20891�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_sequential_layer_call_and_return_conditional_losses_20352~"#78?@TU\]wx��A�>
7�4
*�'
c0_input���������
p 

 
� "%�"
�
0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_20420~"#78?@TU\]wx��A�>
7�4
*�'
c0_input���������
p

 
� "%�"
�
0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_20601|"#78?@TU\]wx��?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_20714|"#78?@TU\]wx��?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������

� �
*__inference_sequential_layer_call_fn_19927q"#78?@TU\]wx��A�>
7�4
*�'
c0_input���������
p 

 
� "����������
�
*__inference_sequential_layer_call_fn_20284q"#78?@TU\]wx��A�>
7�4
*�'
c0_input���������
p

 
� "����������
�
*__inference_sequential_layer_call_fn_20479o"#78?@TU\]wx��?�<
5�2
(�%
inputs���������
p 

 
� "����������
�
*__inference_sequential_layer_call_fn_20516o"#78?@TU\]wx��?�<
5�2
(�%
inputs���������
p

 
� "����������
�
#__inference_signature_wrapper_20753�"#78?@TU\]wx��E�B
� 
;�8
6
c0_input*�'
c0_input���������")�&
$
d15�
d15���������

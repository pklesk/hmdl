��
��
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
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
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
v
Adam/v/d11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/v/d11/bias
o
#Adam/v/d11/bias/Read/ReadVariableOpReadVariableOpAdam/v/d11/bias*
_output_shapes
:
*
dtype0
v
Adam/m/d11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/m/d11/bias
o
#Adam/m/d11/bias/Read/ReadVariableOpReadVariableOpAdam/m/d11/bias*
_output_shapes
:
*
dtype0

Adam/v/d11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*"
shared_nameAdam/v/d11/kernel
x
%Adam/v/d11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d11/kernel*
_output_shapes
:	�
*
dtype0

Adam/m/d11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*"
shared_nameAdam/m/d11/kernel
x
%Adam/m/d11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d11/kernel*
_output_shapes
:	�
*
dtype0
u
Adam/v/d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/v/d9/bias
n
"Adam/v/d9/bias/Read/ReadVariableOpReadVariableOpAdam/v/d9/bias*
_output_shapes	
:�*
dtype0
u
Adam/m/d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/m/d9/bias
n
"Adam/m/d9/bias/Read/ReadVariableOpReadVariableOpAdam/m/d9/bias*
_output_shapes	
:�*
dtype0
~
Adam/v/d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
� �*!
shared_nameAdam/v/d9/kernel
w
$Adam/v/d9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d9/kernel* 
_output_shapes
:
� �*
dtype0
~
Adam/m/d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
� �*!
shared_nameAdam/m/d9/kernel
w
$Adam/m/d9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d9/kernel* 
_output_shapes
:
� �*
dtype0
t
Adam/v/c5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/c5/bias
m
"Adam/v/c5/bias/Read/ReadVariableOpReadVariableOpAdam/v/c5/bias*
_output_shapes
:@*
dtype0
t
Adam/m/c5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/c5/bias
m
"Adam/m/c5/bias/Read/ReadVariableOpReadVariableOpAdam/m/c5/bias*
_output_shapes
:@*
dtype0
�
Adam/v/c5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameAdam/v/c5/kernel
}
$Adam/v/c5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c5/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/m/c5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameAdam/m/c5/kernel
}
$Adam/m/c5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c5/kernel*&
_output_shapes
:@@*
dtype0
t
Adam/v/c4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/c4/bias
m
"Adam/v/c4/bias/Read/ReadVariableOpReadVariableOpAdam/v/c4/bias*
_output_shapes
:@*
dtype0
t
Adam/m/c4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/c4/bias
m
"Adam/m/c4/bias/Read/ReadVariableOpReadVariableOpAdam/m/c4/bias*
_output_shapes
:@*
dtype0
�
Adam/v/c4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameAdam/v/c4/kernel
}
$Adam/v/c4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c4/kernel*&
_output_shapes
: @*
dtype0
�
Adam/m/c4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameAdam/m/c4/kernel
}
$Adam/m/c4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c4/kernel*&
_output_shapes
: @*
dtype0
t
Adam/v/c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/v/c1/bias
m
"Adam/v/c1/bias/Read/ReadVariableOpReadVariableOpAdam/v/c1/bias*
_output_shapes
: *
dtype0
t
Adam/m/c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/m/c1/bias
m
"Adam/m/c1/bias/Read/ReadVariableOpReadVariableOpAdam/m/c1/bias*
_output_shapes
: *
dtype0
�
Adam/v/c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameAdam/v/c1/kernel
}
$Adam/v/c1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c1/kernel*&
_output_shapes
:  *
dtype0
�
Adam/m/c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameAdam/m/c1/kernel
}
$Adam/m/c1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c1/kernel*&
_output_shapes
:  *
dtype0
t
Adam/v/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/v/c0/bias
m
"Adam/v/c0/bias/Read/ReadVariableOpReadVariableOpAdam/v/c0/bias*
_output_shapes
: *
dtype0
t
Adam/m/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/m/c0/bias
m
"Adam/m/c0/bias/Read/ReadVariableOpReadVariableOpAdam/m/c0/bias*
_output_shapes
: *
dtype0
�
Adam/v/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/v/c0/kernel
}
$Adam/v/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c0/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/m/c0/kernel
}
$Adam/m/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c0/kernel*&
_output_shapes
: *
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
q

d11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_name
d11/kernel
j
d11/kernel/Read/ReadVariableOpReadVariableOp
d11/kernel*
_output_shapes
:	�
*
dtype0
g
d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d9/bias
`
d9/bias/Read/ReadVariableOpReadVariableOpd9/bias*
_output_shapes	
:�*
dtype0
p
	d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
� �*
shared_name	d9/kernel
i
d9/kernel/Read/ReadVariableOpReadVariableOp	d9/kernel* 
_output_shapes
:
� �*
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
�
serving_default_c0_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	d9/kerneld9/bias
d11/kerneld11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *,
f'R%
#__inference_signature_wrapper_31904

NoOpNoOp
�i
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�h
value�hB�h B�h
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
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator* 
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op*
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
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator* 
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
�
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias*
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator* 
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias*
Z
0
1
%2
&3
;4
<5
D6
E7
`8
a9
o10
p11*
Z
0
1
%2
&3
;4
<5
D6
E7
`8
a9
o10
p11*
X
q0
r1
s2
t3
u4
v5
w6
x7
y8
z9
{10
|11* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

0
1*

0
1*

q0
r1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

%0
&1*

%0
&1*

s0
t1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

;0
<1*

;0
<1*

u0
v1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

D0
E1*

D0
E1*

w0
x1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	c5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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

�trace_0* 

�trace_0* 
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
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

`0
a1*

`0
a1*

y0
z1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	d9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

o0
p1*

o0
p1*

{0
|1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ZT
VARIABLE_VALUE
d11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEd11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
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
�0
�1*
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
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11* 
* 
* 
* 
* 

q0
r1* 
* 
* 
* 
* 
* 
* 

s0
t1* 
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
* 
* 
* 

u0
v1* 
* 
* 
* 
* 
* 
* 

w0
x1* 
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
y0
z1* 
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
{0
|1* 
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
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
[U
VARIABLE_VALUEAdam/m/c0/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/c0/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/c0/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/c0/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/c1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/c1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/c1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/c1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/c4/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/c4/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/c4/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/c4/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/c5/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/c5/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/c5/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/c5/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/d9/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/d9/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/d9/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/d9/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/d11/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/d11/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/d11/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/d11/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
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

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	d9/kerneld9/bias
d11/kerneld11/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c1/kernelAdam/v/c1/kernelAdam/m/c1/biasAdam/v/c1/biasAdam/m/c4/kernelAdam/v/c4/kernelAdam/m/c4/biasAdam/v/c4/biasAdam/m/c5/kernelAdam/v/c5/kernelAdam/m/c5/biasAdam/v/c5/biasAdam/m/d9/kernelAdam/v/d9/kernelAdam/m/d9/biasAdam/v/d9/biasAdam/m/d11/kernelAdam/v/d11/kernelAdam/m/d11/biasAdam/v/d11/biastotal_1count_1totalcountConst*7
Tin0
.2,*
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
__inference__traced_save_32764
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c1/kernelc1/bias	c4/kernelc4/bias	c5/kernelc5/bias	d9/kerneld9/bias
d11/kerneld11/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c1/kernelAdam/v/c1/kernelAdam/m/c1/biasAdam/v/c1/biasAdam/m/c4/kernelAdam/v/c4/kernelAdam/m/c4/biasAdam/v/c4/biasAdam/m/c5/kernelAdam/v/c5/kernelAdam/m/c5/biasAdam/v/c5/biasAdam/m/d9/kernelAdam/v/d9/kernelAdam/m/d9/biasAdam/v/d9/biasAdam/m/d11/kernelAdam/v/d11/kernelAdam/m/d11/biasAdam/v/d11/biastotal_1count_1totalcount*6
Tin/
-2+*
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
!__inference__traced_restore_32900��

�
\
>__inference_dr7_layer_call_and_return_conditional_losses_32347

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_c1_layer_call_and_return_conditional_losses_31325

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
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
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
+
__inference_loss_fn_0_32434
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
�
+
__inference_loss_fn_4_32454
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
�D
�

 __inference__wrapped_model_31265
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
� �<
-sequential_d9_biasadd_readvariableop_resource:	�@
-sequential_d11_matmul_readvariableop_resource:	�
<
.sequential_d11_biasadd_readvariableop_resource:

identity��$sequential/c0/BiasAdd/ReadVariableOp�#sequential/c0/Conv2D/ReadVariableOp�$sequential/c1/BiasAdd/ReadVariableOp�#sequential/c1/Conv2D/ReadVariableOp�$sequential/c4/BiasAdd/ReadVariableOp�#sequential/c4/Conv2D/ReadVariableOp�$sequential/c5/BiasAdd/ReadVariableOp�#sequential/c5/Conv2D/ReadVariableOp�%sequential/d11/BiasAdd/ReadVariableOp�$sequential/d11/MatMul/ReadVariableOp�$sequential/d9/BiasAdd/ReadVariableOp�#sequential/d9/MatMul/ReadVariableOp�
#sequential/c0/Conv2D/ReadVariableOpReadVariableOp,sequential_c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential/c0/Conv2DConv2Dc0_input+sequential/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   t
sequential/c0/ReluRelusequential/c0/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
#sequential/c1/Conv2D/ReadVariableOpReadVariableOp,sequential_c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
sequential/c1/Conv2DConv2D sequential/c0/Relu:activations:0+sequential/c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   t
sequential/c1/ReluRelusequential/c1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
sequential/m2/MaxPoolMaxPool sequential/c1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
}
sequential/dr3/IdentityIdentitysequential/m2/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
#sequential/c4/Conv2D/ReadVariableOpReadVariableOp,sequential_c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
sequential/c4/Conv2DConv2D sequential/dr3/Identity:output:0+sequential/c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@t
sequential/c4/ReluRelusequential/c4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
#sequential/c5/Conv2D/ReadVariableOpReadVariableOp,sequential_c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
sequential/c5/Conv2DConv2D sequential/c4/Relu:activations:0+sequential/c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@t
sequential/c5/ReluRelusequential/c5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
sequential/m6/MaxPoolMaxPool sequential/c5/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
}
sequential/dr7/IdentityIdentitysequential/m6/MaxPool:output:0*
T0*/
_output_shapes
:���������@d
sequential/f8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
sequential/f8/ReshapeReshape sequential/dr7/Identity:output:0sequential/f8/Const:output:0*
T0*(
_output_shapes
:���������� �
#sequential/d9/MatMul/ReadVariableOpReadVariableOp,sequential_d9_matmul_readvariableop_resource* 
_output_shapes
:
� �*
dtype0�
sequential/d9/MatMulMatMulsequential/f8/Reshape:output:0+sequential/d9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$sequential/d9/BiasAdd/ReadVariableOpReadVariableOp-sequential_d9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/d9/BiasAddBiasAddsequential/d9/MatMul:product:0,sequential/d9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
sequential/d9/ReluRelusequential/d9/BiasAdd:output:0*
T0*(
_output_shapes
:����������y
sequential/dr10/IdentityIdentity sequential/d9/Relu:activations:0*
T0*(
_output_shapes
:�����������
$sequential/d11/MatMul/ReadVariableOpReadVariableOp-sequential_d11_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
sequential/d11/MatMulMatMul!sequential/dr10/Identity:output:0,sequential/d11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
%sequential/d11/BiasAdd/ReadVariableOpReadVariableOp.sequential_d11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential/d11/BiasAddBiasAddsequential/d11/MatMul:product:0-sequential/d11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
sequential/d11/SoftmaxSoftmaxsequential/d11/BiasAdd:output:0*
T0*'
_output_shapes
:���������
o
IdentityIdentity sequential/d11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/c1/BiasAdd/ReadVariableOp$^sequential/c1/Conv2D/ReadVariableOp%^sequential/c4/BiasAdd/ReadVariableOp$^sequential/c4/Conv2D/ReadVariableOp%^sequential/c5/BiasAdd/ReadVariableOp$^sequential/c5/Conv2D/ReadVariableOp&^sequential/d11/BiasAdd/ReadVariableOp%^sequential/d11/MatMul/ReadVariableOp%^sequential/d9/BiasAdd/ReadVariableOp$^sequential/d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2L
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
:���������  
"
_user_specified_name
c0_input
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_32266

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
+
__inference_loss_fn_2_32444
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
�

�
#__inference_signature_wrapper_31904
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
� �
	unknown_8:	�
	unknown_9:	�


unknown_10:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *)
f$R"
 __inference__wrapped_model_31265o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
�
=__inference_d9_layer_call_and_return_conditional_losses_31420

inputs2
matmul_readvariableop_resource:
� �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
� �*
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
:����������`
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
:����������w
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
=__inference_c5_layer_call_and_return_conditional_losses_32310

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
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
>
"__inference_f8_layer_call_fn_32352

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
=__inference_f8_layer_call_and_return_conditional_losses_31405a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
O
"__inference__update_step_xla_32180
gradient
variable:	�
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�
: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�

"
_user_specified_name
gradient
�
K
"__inference__update_step_xla_32175
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
V
"__inference__update_step_xla_32130
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
: 
"
_user_specified_name
gradient
�
�
>__inference_d11_layer_call_and_return_conditional_losses_31453

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
�
,
__inference_loss_fn_10_32484
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
�
J
"__inference__update_step_xla_32185
gradient
variable:
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:
: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:

"
_user_specified_name
gradient
�
]
$__inference_dr10_layer_call_fn_32385

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
?__inference_dr10_layer_call_and_return_conditional_losses_31438p
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
�
J
"__inference__update_step_xla_32155
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
�
J
"__inference__update_step_xla_32145
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
: 
"
_user_specified_name
gradient
�
�
*__inference_sequential_layer_call_fn_31974

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
� �
	unknown_8:	�
	unknown_9:	�


unknown_10:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_31675o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
V
"__inference__update_step_xla_32140
gradient"
variable:  *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:  : *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:  
"
_user_specified_name
gradient
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_32239

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
�A
�
E__inference_sequential_layer_call_and_return_conditional_losses_32125

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
� �1
"d9_biasadd_readvariableop_resource:	�5
"d11_matmul_readvariableop_resource:	�
1
#d11_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�c4/BiasAdd/ReadVariableOp�c4/Conv2D/ReadVariableOp�c5/BiasAdd/ReadVariableOp�c5/Conv2D/ReadVariableOp�d11/BiasAdd/ReadVariableOp�d11/MatMul/ReadVariableOp�d9/BiasAdd/ReadVariableOp�d9/MatMul/ReadVariableOp�
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
g
dr3/IdentityIdentitym2/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
	c4/Conv2DConv2Ddr3/Identity:output:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
g
dr7/IdentityIdentitym6/MaxPool:output:0*
T0*/
_output_shapes
:���������@Y
f8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   r

f8/ReshapeReshapedr7/Identity:output:0f8/Const:output:0*
T0*(
_output_shapes
:���������� |
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource* 
_output_shapes
:
� �*
dtype0}
	d9/MatMulMatMulf8/Reshape:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d9/ReluRelud9/BiasAdd:output:0*
T0*(
_output_shapes
:����������c
dr10/IdentityIdentityd9/Relu:activations:0*
T0*(
_output_shapes
:����������}
d11/MatMul/ReadVariableOpReadVariableOp"d11_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�

d11/MatMulMatMuldr10/Identity:output:0!d11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
d11/BiasAdd/ReadVariableOpReadVariableOp#d11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
d11/BiasAddBiasAddd11/MatMul:product:0"d11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
^
d11/SoftmaxSoftmaxd11/BiasAdd:output:0*
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
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^d11/BiasAdd/ReadVariableOp^d11/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 26
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
:���������  
 
_user_specified_nameinputs
�
@
$__inference_dr10_layer_call_fn_32390

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
?__inference_dr10_layer_call_and_return_conditional_losses_31519a
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
�
>__inference_d11_layer_call_and_return_conditional_losses_32429

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

^
?__inference_dr10_layer_call_and_return_conditional_losses_32402

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
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
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
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
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
�
Y
=__inference_f8_layer_call_and_return_conditional_losses_31405

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
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
+
__inference_loss_fn_1_32439
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
�
+
__inference_loss_fn_9_32479
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
�
>
"__inference_m2_layer_call_fn_32234

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
=__inference_m2_layer_call_and_return_conditional_losses_31271�
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
=__inference_d9_layer_call_and_return_conditional_losses_32380

inputs2
matmul_readvariableop_resource:
� �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
� �*
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
:����������`
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
:����������w
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
�
V
"__inference__update_step_xla_32150
gradient"
variable: @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: @: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
: @
"
_user_specified_name
gradient
�7
�
E__inference_sequential_layer_call_and_return_conditional_losses_31675

inputs"
c0_31626: 
c0_31628: "
c1_31631:  
c1_31633: "
c4_31638: @
c4_31640:@"
c5_31643:@@
c5_31645:@
d9_31651:
� �
d9_31653:	�
	d11_31657:	�

	d11_31659:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�d11/StatefulPartitionedCall�d9/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_31626c0_31628*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31306�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_31631c1_31633*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31325�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31271�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31490�
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_31638c4_31640*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31359�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_31643c5_31645*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31378�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31283�
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_31507�
f8/PartitionedCallPartitionedCalldr7/PartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_31405�
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_31651d9_31653*
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
GPU(2*0J 8� *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_31420�
dr10/PartitionedCallPartitionedCall#d9/StatefulPartitionedCall:output:0*
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
?__inference_dr10_layer_call_and_return_conditional_losses_31519�
d11/StatefulPartitionedCallStatefulPartitionedCalldr10/PartitionedCall:output:0	d11_31657	d11_31659*
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
>__inference_d11_layer_call_and_return_conditional_losses_31453`
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
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall2:
d11/StatefulPartitionedCalld11/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_31594

inputs"
c0_31545: 
c0_31547: "
c1_31550:  
c1_31552: "
c4_31557: @
c4_31559:@"
c5_31562:@@
c5_31564:@
d9_31570:
� �
d9_31572:	�
	d11_31576:	�

	d11_31578:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�d11/StatefulPartitionedCall�d9/StatefulPartitionedCall�dr10/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_31545c0_31547*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31306�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_31550c1_31552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31325�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31271�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31344�
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_31557c4_31559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31359�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_31562c5_31564*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31378�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31283�
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_31397�
f8/PartitionedCallPartitionedCall$dr7/StatefulPartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_31405�
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_31570d9_31572*
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
GPU(2*0J 8� *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_31420�
dr10/StatefulPartitionedCallStatefulPartitionedCall#d9/StatefulPartitionedCall:output:0^dr7/StatefulPartitionedCall*
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
?__inference_dr10_layer_call_and_return_conditional_losses_31438�
d11/StatefulPartitionedCallStatefulPartitionedCall%dr10/StatefulPartitionedCall:output:0	d11_31576	d11_31578*
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
>__inference_d11_layer_call_and_return_conditional_losses_31453`
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
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr10/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 28
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
:���������  
 
_user_specified_nameinputs
�
J
"__inference__update_step_xla_32135
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
: 
"
_user_specified_name
gradient
�
+
__inference_loss_fn_6_32464
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
�
\
>__inference_dr7_layer_call_and_return_conditional_losses_31507

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
]
?__inference_dr10_layer_call_and_return_conditional_losses_31519

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
�
P
"__inference__update_step_xla_32170
gradient
variable:
� �*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
� �: *
	_noinline(:($
"
_user_specified_name
variable:J F
 
_output_shapes
:
� �
"
_user_specified_name
gradient
�

]
>__inference_dr7_layer_call_and_return_conditional_losses_32342

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
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
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
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

^
?__inference_dr10_layer_call_and_return_conditional_losses_31438

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
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
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
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
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
+
__inference_loss_fn_5_32459
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
�
\
>__inference_dr3_layer_call_and_return_conditional_losses_31490

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
=__inference_c5_layer_call_and_return_conditional_losses_31378

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
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
Y
=__inference_m6_layer_call_and_return_conditional_losses_31283

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
�
\
#__inference_dr7_layer_call_fn_32325

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
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_31397w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

]
>__inference_dr7_layer_call_and_return_conditional_losses_31397

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
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
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
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
=__inference_c4_layer_call_and_return_conditional_losses_31359

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
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
#__inference_d11_layer_call_fn_32416

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
>__inference_d11_layer_call_and_return_conditional_losses_31453o
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
�
�
"__inference_c5_layer_call_fn_32297

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
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31378w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
Y
=__inference_f8_layer_call_and_return_conditional_losses_32358

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
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_32261

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
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
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
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
+
__inference_loss_fn_8_32474
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
�W
�
E__inference_sequential_layer_call_and_return_conditional_losses_32060

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
� �1
"d9_biasadd_readvariableop_resource:	�5
"d11_matmul_readvariableop_resource:	�
1
#d11_biasadd_readvariableop_resource:

identity��c0/BiasAdd/ReadVariableOp�c0/Conv2D/ReadVariableOp�c1/BiasAdd/ReadVariableOp�c1/Conv2D/ReadVariableOp�c4/BiasAdd/ReadVariableOp�c4/Conv2D/ReadVariableOp�c5/BiasAdd/ReadVariableOp�c5/Conv2D/ReadVariableOp�d11/BiasAdd/ReadVariableOp�d11/MatMul/ReadVariableOp�d9/BiasAdd/ReadVariableOp�d9/MatMul/ReadVariableOp�
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   ^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
c1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
	c1/Conv2DConv2Dc0/Relu:activations:0 c1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   ^
c1/ReluReluc1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �

m2/MaxPoolMaxPoolc1/Relu:activations:0*/
_output_shapes
:��������� *
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
:��������� b
dr3/dropout/ShapeShapem2/MaxPool:output:0*
T0*
_output_shapes
::���
(dr3/dropout/random_uniform/RandomUniformRandomUniformdr3/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
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
:��������� X
dr3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr3/dropout/SelectV2SelectV2dr3/dropout/GreaterEqual:z:0dr3/dropout/Mul:z:0dr3/dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� �
c4/Conv2D/ReadVariableOpReadVariableOp!c4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
	c4/Conv2DConv2Ddr3/dropout/SelectV2:output:0 c4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@^
c4/ReluReluc4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
c5/Conv2D/ReadVariableOpReadVariableOp!c5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
	c5/Conv2DConv2Dc4/Relu:activations:0 c5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@^
c5/ReluReluc5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�

m6/MaxPoolMaxPoolc5/Relu:activations:0*/
_output_shapes
:���������@*
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
:���������@b
dr7/dropout/ShapeShapem6/MaxPool:output:0*
T0*
_output_shapes
::���
(dr7/dropout/random_uniform/RandomUniformRandomUniformdr7/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
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
:���������@X
dr7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr7/dropout/SelectV2SelectV2dr7/dropout/GreaterEqual:z:0dr7/dropout/Mul:z:0dr7/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@Y
f8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z

f8/ReshapeReshapedr7/dropout/SelectV2:output:0f8/Const:output:0*
T0*(
_output_shapes
:���������� |
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource* 
_output_shapes
:
� �*
dtype0}
	d9/MatMulMatMulf8/Reshape:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d9/ReluRelud9/BiasAdd:output:0*
T0*(
_output_shapes
:����������W
dr10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%I�?~
dr10/dropout/MulMuld9/Relu:activations:0dr10/dropout/Const:output:0*
T0*(
_output_shapes
:����������e
dr10/dropout/ShapeShaped9/Relu:activations:0*
T0*
_output_shapes
::���
)dr10/dropout/random_uniform/RandomUniformRandomUniformdr10/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2`
dr10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
dr10/dropout/GreaterEqualGreaterEqual2dr10/dropout/random_uniform/RandomUniform:output:0$dr10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������Y
dr10/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dr10/dropout/SelectV2SelectV2dr10/dropout/GreaterEqual:z:0dr10/dropout/Mul:z:0dr10/dropout/Const_1:output:0*
T0*(
_output_shapes
:����������}
d11/MatMul/ReadVariableOpReadVariableOp"d11_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�

d11/MatMulMatMuldr10/dropout/SelectV2:output:0!d11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
d11/BiasAdd/ReadVariableOpReadVariableOp#d11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
d11/BiasAddBiasAddd11/MatMul:product:0"d11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
^
d11/SoftmaxSoftmaxd11/BiasAdd:output:0*
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
:���������
�
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c1/BiasAdd/ReadVariableOp^c1/Conv2D/ReadVariableOp^c4/BiasAdd/ReadVariableOp^c4/Conv2D/ReadVariableOp^c5/BiasAdd/ReadVariableOp^c5/Conv2D/ReadVariableOp^d11/BiasAdd/ReadVariableOp^d11/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 26
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
:���������  
 
_user_specified_nameinputs
�
]
?__inference_dr10_layer_call_and_return_conditional_losses_32407

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
=__inference_c4_layer_call_and_return_conditional_losses_32288

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
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@`
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_31621
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
� �
	unknown_8:	�
	unknown_9:	�


unknown_10:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_31594o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
�
"__inference_d9_layer_call_fn_32367

inputs
unknown:
� �
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
GPU(2*0J 8� *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_31420p
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
:���������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
>
"__inference_m6_layer_call_fn_32315

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
=__inference_m6_layer_call_and_return_conditional_losses_31283�
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
�
?
#__inference_dr3_layer_call_fn_32249

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
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31490h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
Y
=__inference_m2_layer_call_and_return_conditional_losses_31271

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
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_31472
c0_input"
c0_31307: 
c0_31309: "
c1_31326:  
c1_31328: "
c4_31360: @
c4_31362:@"
c5_31379:@@
c5_31381:@
d9_31421:
� �
d9_31423:	�
	d11_31454:	�

	d11_31456:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�d11/StatefulPartitionedCall�d9/StatefulPartitionedCall�dr10/StatefulPartitionedCall�dr3/StatefulPartitionedCall�dr7/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_31307c0_31309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31306�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_31326c1_31328*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31325�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31271�
dr3/StatefulPartitionedCallStatefulPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31344�
c4/StatefulPartitionedCallStatefulPartitionedCall$dr3/StatefulPartitionedCall:output:0c4_31360c4_31362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31359�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_31379c5_31381*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31378�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31283�
dr7/StatefulPartitionedCallStatefulPartitionedCallm6/PartitionedCall:output:0^dr3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_31397�
f8/PartitionedCallPartitionedCall$dr7/StatefulPartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_31405�
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_31421d9_31423*
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
GPU(2*0J 8� *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_31420�
dr10/StatefulPartitionedCallStatefulPartitionedCall#d9/StatefulPartitionedCall:output:0^dr7/StatefulPartitionedCall*
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
?__inference_dr10_layer_call_and_return_conditional_losses_31438�
d11/StatefulPartitionedCallStatefulPartitionedCall%dr10/StatefulPartitionedCall:output:0	d11_31454	d11_31456*
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
>__inference_d11_layer_call_and_return_conditional_losses_31453`
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
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr10/StatefulPartitionedCall^dr3/StatefulPartitionedCall^dr7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 28
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
:���������  
"
_user_specified_name
c0_input
�
�
"__inference_c0_layer_call_fn_32194

inputs!
unknown: 
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
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31306w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
J
"__inference__update_step_xla_32165
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
�
+
__inference_loss_fn_7_32469
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
�
+
__inference_loss_fn_3_32449
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
�
\
#__inference_dr3_layer_call_fn_32244

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
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31344w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
=__inference_c0_layer_call_and_return_conditional_losses_32207

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
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
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�$
__inference__traced_save_32764
file_prefix:
 read_disablecopyonread_c0_kernel: .
 read_1_disablecopyonread_c0_bias: <
"read_2_disablecopyonread_c1_kernel:  .
 read_3_disablecopyonread_c1_bias: <
"read_4_disablecopyonread_c4_kernel: @.
 read_5_disablecopyonread_c4_bias:@<
"read_6_disablecopyonread_c5_kernel:@@.
 read_7_disablecopyonread_c5_bias:@6
"read_8_disablecopyonread_d9_kernel:
� �/
 read_9_disablecopyonread_d9_bias:	�7
$read_10_disablecopyonread_d11_kernel:	�
0
"read_11_disablecopyonread_d11_bias:
-
#read_12_disablecopyonread_iteration:	 1
'read_13_disablecopyonread_learning_rate: D
*read_14_disablecopyonread_adam_m_c0_kernel: D
*read_15_disablecopyonread_adam_v_c0_kernel: 6
(read_16_disablecopyonread_adam_m_c0_bias: 6
(read_17_disablecopyonread_adam_v_c0_bias: D
*read_18_disablecopyonread_adam_m_c1_kernel:  D
*read_19_disablecopyonread_adam_v_c1_kernel:  6
(read_20_disablecopyonread_adam_m_c1_bias: 6
(read_21_disablecopyonread_adam_v_c1_bias: D
*read_22_disablecopyonread_adam_m_c4_kernel: @D
*read_23_disablecopyonread_adam_v_c4_kernel: @6
(read_24_disablecopyonread_adam_m_c4_bias:@6
(read_25_disablecopyonread_adam_v_c4_bias:@D
*read_26_disablecopyonread_adam_m_c5_kernel:@@D
*read_27_disablecopyonread_adam_v_c5_kernel:@@6
(read_28_disablecopyonread_adam_m_c5_bias:@6
(read_29_disablecopyonread_adam_v_c5_bias:@>
*read_30_disablecopyonread_adam_m_d9_kernel:
� �>
*read_31_disablecopyonread_adam_v_d9_kernel:
� �7
(read_32_disablecopyonread_adam_m_d9_bias:	�7
(read_33_disablecopyonread_adam_v_d9_bias:	�>
+read_34_disablecopyonread_adam_m_d11_kernel:	�
>
+read_35_disablecopyonread_adam_v_d11_kernel:	�
7
)read_36_disablecopyonread_adam_m_d11_bias:
7
)read_37_disablecopyonread_adam_v_d11_bias:
+
!read_38_disablecopyonread_total_1: +
!read_39_disablecopyonread_count_1: )
read_40_disablecopyonread_total: )
read_41_disablecopyonread_count: 
savev2_const
identity_85��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnRead read_disablecopyonread_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp read_disablecopyonread_c0_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: t
Read_1/DisableCopyOnReadDisableCopyOnRead read_1_disablecopyonread_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp read_1_disablecopyonread_c0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_c1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:  t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_c1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_c4_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_c4_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: @t
Read_5/DisableCopyOnReadDisableCopyOnRead read_5_disablecopyonread_c4_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp read_5_disablecopyonread_c4_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_c5_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_c5_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@t
Read_7/DisableCopyOnReadDisableCopyOnRead read_7_disablecopyonread_c5_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp read_7_disablecopyonread_c5_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_d9_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_d9_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
� �*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
� �g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
� �t
Read_9/DisableCopyOnReadDisableCopyOnRead read_9_disablecopyonread_d9_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp read_9_disablecopyonread_d9_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�y
Read_10/DisableCopyOnReadDisableCopyOnRead$read_10_disablecopyonread_d11_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp$read_10_disablecopyonread_d11_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
w
Read_11/DisableCopyOnReadDisableCopyOnRead"read_11_disablecopyonread_d11_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp"read_11_disablecopyonread_d11_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:
x
Read_12/DisableCopyOnReadDisableCopyOnRead#read_12_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp#read_12_disablecopyonread_iteration^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_learning_rate^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_adam_m_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_adam_m_c0_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_adam_v_c0_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_adam_v_c0_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
: }
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_adam_m_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_adam_m_c0_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_17/DisableCopyOnReadDisableCopyOnRead(read_17_disablecopyonread_adam_v_c0_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp(read_17_disablecopyonread_adam_v_c0_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_18/DisableCopyOnReadDisableCopyOnRead*read_18_disablecopyonread_adam_m_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp*read_18_disablecopyonread_adam_m_c1_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
:  
Read_19/DisableCopyOnReadDisableCopyOnRead*read_19_disablecopyonread_adam_v_c1_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp*read_19_disablecopyonread_adam_v_c1_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*&
_output_shapes
:  }
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_adam_m_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_adam_m_c1_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_21/DisableCopyOnReadDisableCopyOnRead(read_21_disablecopyonread_adam_v_c1_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp(read_21_disablecopyonread_adam_v_c1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_22/DisableCopyOnReadDisableCopyOnRead*read_22_disablecopyonread_adam_m_c4_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp*read_22_disablecopyonread_adam_m_c4_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_23/DisableCopyOnReadDisableCopyOnRead*read_23_disablecopyonread_adam_v_c4_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp*read_23_disablecopyonread_adam_v_c4_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*&
_output_shapes
: @}
Read_24/DisableCopyOnReadDisableCopyOnRead(read_24_disablecopyonread_adam_m_c4_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp(read_24_disablecopyonread_adam_m_c4_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_adam_v_c4_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_adam_v_c4_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_26/DisableCopyOnReadDisableCopyOnRead*read_26_disablecopyonread_adam_m_c5_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp*read_26_disablecopyonread_adam_m_c5_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@
Read_27/DisableCopyOnReadDisableCopyOnRead*read_27_disablecopyonread_adam_v_c5_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp*read_27_disablecopyonread_adam_v_c5_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@}
Read_28/DisableCopyOnReadDisableCopyOnRead(read_28_disablecopyonread_adam_m_c5_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp(read_28_disablecopyonread_adam_m_c5_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_29/DisableCopyOnReadDisableCopyOnRead(read_29_disablecopyonread_adam_v_c5_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp(read_29_disablecopyonread_adam_v_c5_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_30/DisableCopyOnReadDisableCopyOnRead*read_30_disablecopyonread_adam_m_d9_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp*read_30_disablecopyonread_adam_m_d9_kernel^Read_30/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
� �*
dtype0q
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
� �g
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0* 
_output_shapes
:
� �
Read_31/DisableCopyOnReadDisableCopyOnRead*read_31_disablecopyonread_adam_v_d9_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp*read_31_disablecopyonread_adam_v_d9_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
� �*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
� �g
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
� �}
Read_32/DisableCopyOnReadDisableCopyOnRead(read_32_disablecopyonread_adam_m_d9_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp(read_32_disablecopyonread_adam_m_d9_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_33/DisableCopyOnReadDisableCopyOnRead(read_33_disablecopyonread_adam_v_d9_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp(read_33_disablecopyonread_adam_v_d9_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead+read_34_disablecopyonread_adam_m_d11_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp+read_34_disablecopyonread_adam_m_d11_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_35/DisableCopyOnReadDisableCopyOnRead+read_35_disablecopyonread_adam_v_d11_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp+read_35_disablecopyonread_adam_v_d11_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
~
Read_36/DisableCopyOnReadDisableCopyOnRead)read_36_disablecopyonread_adam_m_d11_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp)read_36_disablecopyonread_adam_m_d11_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:
~
Read_37/DisableCopyOnReadDisableCopyOnRead)read_37_disablecopyonread_adam_v_d11_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp)read_37_disablecopyonread_adam_v_d11_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_38/DisableCopyOnReadDisableCopyOnRead!read_38_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp!read_38_disablecopyonread_total_1^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_39/DisableCopyOnReadDisableCopyOnRead!read_39_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp!read_39_disablecopyonread_count_1^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_40/DisableCopyOnReadDisableCopyOnReadread_40_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpread_40_disablecopyonread_total^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_41/DisableCopyOnReadDisableCopyOnReadread_41_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpread_41_disablecopyonread_count^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *9
dtypes/
-2+	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_84Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_85IdentityIdentity_84:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_85Identity_85:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:+

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
=__inference_c0_layer_call_and_return_conditional_losses_31306

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
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
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
?
#__inference_dr7_layer_call_fn_32330

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
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_31507h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

]
>__inference_dr3_layer_call_and_return_conditional_losses_31344

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
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
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
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_31945

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
� �
	unknown_8:	�
	unknown_9:	�


unknown_10:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_31594o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�7
�
E__inference_sequential_layer_call_and_return_conditional_losses_31539
c0_input"
c0_31475: 
c0_31477: "
c1_31480:  
c1_31482: "
c4_31492: @
c4_31494:@"
c5_31497:@@
c5_31499:@
d9_31510:
� �
d9_31512:	�
	d11_31521:	�

	d11_31523:

identity��c0/StatefulPartitionedCall�c1/StatefulPartitionedCall�c4/StatefulPartitionedCall�c5/StatefulPartitionedCall�d11/StatefulPartitionedCall�d9/StatefulPartitionedCall�
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_31475c0_31477*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_31306�
c1/StatefulPartitionedCallStatefulPartitionedCall#c0/StatefulPartitionedCall:output:0c1_31480c1_31482*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31325�
m2/PartitionedCallPartitionedCall#c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m2_layer_call_and_return_conditional_losses_31271�
dr3/PartitionedCallPartitionedCallm2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr3_layer_call_and_return_conditional_losses_31490�
c4/StatefulPartitionedCallStatefulPartitionedCalldr3/PartitionedCall:output:0c4_31492c4_31494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31359�
c5/StatefulPartitionedCallStatefulPartitionedCall#c4/StatefulPartitionedCall:output:0c5_31497c5_31499*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c5_layer_call_and_return_conditional_losses_31378�
m6/PartitionedCallPartitionedCall#c5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_m6_layer_call_and_return_conditional_losses_31283�
dr7/PartitionedCallPartitionedCallm6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *G
fBR@
>__inference_dr7_layer_call_and_return_conditional_losses_31507�
f8/PartitionedCallPartitionedCalldr7/PartitionedCall:output:0*
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
=__inference_f8_layer_call_and_return_conditional_losses_31405�
d9/StatefulPartitionedCallStatefulPartitionedCallf8/PartitionedCall:output:0d9_31510d9_31512*
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
GPU(2*0J 8� *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_31420�
dr10/PartitionedCallPartitionedCall#d9/StatefulPartitionedCall:output:0*
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
?__inference_dr10_layer_call_and_return_conditional_losses_31519�
d11/StatefulPartitionedCallStatefulPartitionedCalldr10/PartitionedCall:output:0	d11_31521	d11_31523*
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
>__inference_d11_layer_call_and_return_conditional_losses_31453`
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
:���������
�
NoOpNoOp^c0/StatefulPartitionedCall^c1/StatefulPartitionedCall^c4/StatefulPartitionedCall^c5/StatefulPartitionedCall^d11/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c1/StatefulPartitionedCallc1/StatefulPartitionedCall28
c4/StatefulPartitionedCallc4/StatefulPartitionedCall28
c5/StatefulPartitionedCallc5/StatefulPartitionedCall2:
d11/StatefulPartitionedCalld11/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input
�
�
=__inference_c1_layer_call_and_return_conditional_losses_32229

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
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
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   `
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
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
"__inference_c4_layer_call_fn_32275

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
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c4_layer_call_and_return_conditional_losses_31359w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
V
"__inference__update_step_xla_32160
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient
˭
�
!__inference__traced_restore_32900
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
� �)
assignvariableop_9_d9_bias:	�1
assignvariableop_10_d11_kernel:	�
*
assignvariableop_11_d11_bias:
'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: >
$assignvariableop_14_adam_m_c0_kernel: >
$assignvariableop_15_adam_v_c0_kernel: 0
"assignvariableop_16_adam_m_c0_bias: 0
"assignvariableop_17_adam_v_c0_bias: >
$assignvariableop_18_adam_m_c1_kernel:  >
$assignvariableop_19_adam_v_c1_kernel:  0
"assignvariableop_20_adam_m_c1_bias: 0
"assignvariableop_21_adam_v_c1_bias: >
$assignvariableop_22_adam_m_c4_kernel: @>
$assignvariableop_23_adam_v_c4_kernel: @0
"assignvariableop_24_adam_m_c4_bias:@0
"assignvariableop_25_adam_v_c4_bias:@>
$assignvariableop_26_adam_m_c5_kernel:@@>
$assignvariableop_27_adam_v_c5_kernel:@@0
"assignvariableop_28_adam_m_c5_bias:@0
"assignvariableop_29_adam_v_c5_bias:@8
$assignvariableop_30_adam_m_d9_kernel:
� �8
$assignvariableop_31_adam_v_d9_kernel:
� �1
"assignvariableop_32_adam_m_d9_bias:	�1
"assignvariableop_33_adam_v_d9_bias:	�8
%assignvariableop_34_adam_m_d11_kernel:	�
8
%assignvariableop_35_adam_v_d11_kernel:	�
1
#assignvariableop_36_adam_m_d11_bias:
1
#assignvariableop_37_adam_v_d11_bias:
%
assignvariableop_38_total_1: %
assignvariableop_39_count_1: #
assignvariableop_40_total: #
assignvariableop_41_count: 
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_c0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_c0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_c1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_c1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_c4_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_c4_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_c5_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_c5_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_d9_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_d9_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_d11_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_d11_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_adam_m_c0_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_v_c0_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_m_c0_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_adam_v_c0_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_adam_m_c1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_v_c1_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_m_c1_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_adam_v_c1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_adam_m_c4_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_v_c4_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_m_c4_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_adam_v_c4_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_adam_m_c5_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_v_c5_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_m_c5_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp"assignvariableop_29_adam_v_c5_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_adam_m_d9_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_v_d9_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_m_d9_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp"assignvariableop_33_adam_v_d9_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_m_d11_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp%assignvariableop_35_adam_v_d11_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_adam_m_d11_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp#assignvariableop_37_adam_v_d11_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
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
�
,
__inference_loss_fn_11_32489
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
�
�
"__inference_c1_layer_call_fn_32216

inputs!
unknown:  
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
:���������   *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *F
fAR?
=__inference_c1_layer_call_and_return_conditional_losses_31325w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
Y
=__inference_m6_layer_call_and_return_conditional_losses_32320

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
�
*__inference_sequential_layer_call_fn_31702
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
� �
	unknown_8:	�
	unknown_9:	�


unknown_10:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*.
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_31675o
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
c0_input"�
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
c0_input9
serving_default_c0_input:0���������  7
d110
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
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
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op"
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
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator"
_tf_keras_layer
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias"
_tf_keras_layer
v
0
1
%2
&3
;4
<5
D6
E7
`8
a9
o10
p11"
trackable_list_wrapper
v
0
1
%2
&3
;4
<5
D6
E7
`8
a9
o10
p11"
trackable_list_wrapper
v
q0
r1
s2
t3
u4
v5
w6
x7
y8
z9
{10
|11"
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
*__inference_sequential_layer_call_fn_31621
*__inference_sequential_layer_call_fn_31702
*__inference_sequential_layer_call_fn_31945
*__inference_sequential_layer_call_fn_31974�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_31472
E__inference_sequential_layer_call_and_return_conditional_losses_31539
E__inference_sequential_layer_call_and_return_conditional_losses_32060
E__inference_sequential_layer_call_and_return_conditional_losses_32125�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_31265c0_input"�
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
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c0_layer_call_fn_32194�
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
 z�trace_0
�
�trace_02�
=__inference_c0_layer_call_and_return_conditional_losses_32207�
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
 z�trace_0
#:! 2	c0/kernel
: 2c0/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
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
 0
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c1_layer_call_fn_32216�
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
 z�trace_0
�
�trace_02�
=__inference_c1_layer_call_and_return_conditional_losses_32229�
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
 z�trace_0
#:!  2	c1/kernel
: 2c1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
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
 0
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
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_m2_layer_call_fn_32234�
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
 z�trace_0
�
�trace_02�
=__inference_m2_layer_call_and_return_conditional_losses_32239�
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
 z�trace_0
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
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_dr3_layer_call_fn_32244
#__inference_dr3_layer_call_fn_32249�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_dr3_layer_call_and_return_conditional_losses_32261
>__inference_dr3_layer_call_and_return_conditional_losses_32266�
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
 z�trace_0z�trace_1
"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c4_layer_call_fn_32275�
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
 z�trace_0
�
�trace_02�
=__inference_c4_layer_call_and_return_conditional_losses_32288�
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
 z�trace_0
#:! @2	c4/kernel
:@2c4/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
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
 0
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_c5_layer_call_fn_32297�
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
 z�trace_0
�
�trace_02�
=__inference_c5_layer_call_and_return_conditional_losses_32310�
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
 z�trace_0
#:!@@2	c5/kernel
:@2c5/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
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
 0
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
�
�trace_02�
"__inference_m6_layer_call_fn_32315�
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
 z�trace_0
�
�trace_02�
=__inference_m6_layer_call_and_return_conditional_losses_32320�
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
 z�trace_0
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
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_dr7_layer_call_fn_32325
#__inference_dr7_layer_call_fn_32330�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_dr7_layer_call_and_return_conditional_losses_32342
>__inference_dr7_layer_call_and_return_conditional_losses_32347�
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
 z�trace_0z�trace_1
"
_generic_user_object
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
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_f8_layer_call_fn_32352�
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
 z�trace_0
�
�trace_02�
=__inference_f8_layer_call_and_return_conditional_losses_32358�
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
 z�trace_0
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_d9_layer_call_fn_32367�
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
 z�trace_0
�
�trace_02�
=__inference_d9_layer_call_and_return_conditional_losses_32380�
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
 z�trace_0
:
� �2	d9/kernel
:�2d9/bias
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
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_dr10_layer_call_fn_32385
$__inference_dr10_layer_call_fn_32390�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_dr10_layer_call_and_return_conditional_losses_32402
?__inference_dr10_layer_call_and_return_conditional_losses_32407�
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
 z�trace_0z�trace_1
"
_generic_user_object
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_d11_layer_call_fn_32416�
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
 z�trace_0
�
�trace_02�
>__inference_d11_layer_call_and_return_conditional_losses_32429�
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
 z�trace_0
:	�
2
d11/kernel
:
2d11/bias
�
�trace_02�
__inference_loss_fn_0_32434�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_32439�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_32444�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_32449�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_32454�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_32459�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_32464�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_32469�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_8_32474�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_9_32479�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_10_32484�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_11_32489�
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
annotations� *� z�trace_0
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
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_sequential_layer_call_fn_31621c0_input"�
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
*__inference_sequential_layer_call_fn_31702c0_input"�
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
*__inference_sequential_layer_call_fn_31945inputs"�
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
*__inference_sequential_layer_call_fn_31974inputs"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_31472c0_input"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_31539c0_input"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_32060inputs"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_32125inputs"�
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
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_112�
"__inference__update_step_xla_32130
"__inference__update_step_xla_32135
"__inference__update_step_xla_32140
"__inference__update_step_xla_32145
"__inference__update_step_xla_32150
"__inference__update_step_xla_32155
"__inference__update_step_xla_32160
"__inference__update_step_xla_32165
"__inference__update_step_xla_32170
"__inference__update_step_xla_32175
"__inference__update_step_xla_32180
"__inference__update_step_xla_32185�
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
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11
�B�
#__inference_signature_wrapper_31904c0_input"�
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
.
q0
r1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c0_layer_call_fn_32194inputs"�
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
=__inference_c0_layer_call_and_return_conditional_losses_32207inputs"�
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
s0
t1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c1_layer_call_fn_32216inputs"�
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
=__inference_c1_layer_call_and_return_conditional_losses_32229inputs"�
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
"__inference_m2_layer_call_fn_32234inputs"�
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
=__inference_m2_layer_call_and_return_conditional_losses_32239inputs"�
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
#__inference_dr3_layer_call_fn_32244inputs"�
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
#__inference_dr3_layer_call_fn_32249inputs"�
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
>__inference_dr3_layer_call_and_return_conditional_losses_32261inputs"�
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
>__inference_dr3_layer_call_and_return_conditional_losses_32266inputs"�
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
u0
v1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c4_layer_call_fn_32275inputs"�
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
=__inference_c4_layer_call_and_return_conditional_losses_32288inputs"�
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
w0
x1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_c5_layer_call_fn_32297inputs"�
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
=__inference_c5_layer_call_and_return_conditional_losses_32310inputs"�
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
"__inference_m6_layer_call_fn_32315inputs"�
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
=__inference_m6_layer_call_and_return_conditional_losses_32320inputs"�
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
#__inference_dr7_layer_call_fn_32325inputs"�
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
#__inference_dr7_layer_call_fn_32330inputs"�
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
>__inference_dr7_layer_call_and_return_conditional_losses_32342inputs"�
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
>__inference_dr7_layer_call_and_return_conditional_losses_32347inputs"�
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_f8_layer_call_fn_32352inputs"�
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
=__inference_f8_layer_call_and_return_conditional_losses_32358inputs"�
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
y0
z1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_d9_layer_call_fn_32367inputs"�
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
=__inference_d9_layer_call_and_return_conditional_losses_32380inputs"�
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
$__inference_dr10_layer_call_fn_32385inputs"�
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
$__inference_dr10_layer_call_fn_32390inputs"�
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
?__inference_dr10_layer_call_and_return_conditional_losses_32402inputs"�
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
?__inference_dr10_layer_call_and_return_conditional_losses_32407inputs"�
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
{0
|1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_d11_layer_call_fn_32416inputs"�
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
>__inference_d11_layer_call_and_return_conditional_losses_32429inputs"�
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
__inference_loss_fn_0_32434"�
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
__inference_loss_fn_1_32439"�
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
__inference_loss_fn_2_32444"�
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
__inference_loss_fn_3_32449"�
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
__inference_loss_fn_4_32454"�
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
__inference_loss_fn_5_32459"�
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
__inference_loss_fn_6_32464"�
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
__inference_loss_fn_7_32469"�
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
__inference_loss_fn_8_32474"�
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
__inference_loss_fn_9_32479"�
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
__inference_loss_fn_10_32484"�
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
__inference_loss_fn_11_32489"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
(:& 2Adam/m/c0/kernel
(:& 2Adam/v/c0/kernel
: 2Adam/m/c0/bias
: 2Adam/v/c0/bias
(:&  2Adam/m/c1/kernel
(:&  2Adam/v/c1/kernel
: 2Adam/m/c1/bias
: 2Adam/v/c1/bias
(:& @2Adam/m/c4/kernel
(:& @2Adam/v/c4/kernel
:@2Adam/m/c4/bias
:@2Adam/v/c4/bias
(:&@@2Adam/m/c5/kernel
(:&@@2Adam/v/c5/kernel
:@2Adam/m/c5/bias
:@2Adam/v/c5/bias
": 
� �2Adam/m/d9/kernel
": 
� �2Adam/v/d9/kernel
:�2Adam/m/d9/bias
:�2Adam/v/d9/bias
": 	�
2Adam/m/d11/kernel
": 	�
2Adam/v/d11/kernel
:
2Adam/m/d11/bias
:
2Adam/v/d11/bias
�B�
"__inference__update_step_xla_32130gradientvariable"�
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
"__inference__update_step_xla_32135gradientvariable"�
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
"__inference__update_step_xla_32140gradientvariable"�
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
"__inference__update_step_xla_32145gradientvariable"�
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
"__inference__update_step_xla_32150gradientvariable"�
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
"__inference__update_step_xla_32155gradientvariable"�
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
"__inference__update_step_xla_32160gradientvariable"�
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
"__inference__update_step_xla_32165gradientvariable"�
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
"__inference__update_step_xla_32170gradientvariable"�
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
"__inference__update_step_xla_32175gradientvariable"�
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
"__inference__update_step_xla_32180gradientvariable"�
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
"__inference__update_step_xla_32185gradientvariable"�
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__update_step_xla_32130~x�u
n�k
!�
gradient 
<�9	%�"
� 
�
p
` VariableSpec 
`�ӧ��?
� "
 �
"__inference__update_step_xla_32135f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`�␩��?
� "
 �
"__inference__update_step_xla_32140~x�u
n�k
!�
gradient  
<�9	%�"
�  
�
p
` VariableSpec 
`ࢷ���?
� "
 �
"__inference__update_step_xla_32145f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`ࡷ���?
� "
 �
"__inference__update_step_xla_32150~x�u
n�k
!�
gradient @
<�9	%�"
� @
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_32155f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`ࢼ���?
� "
 �
"__inference__update_step_xla_32160~x�u
n�k
!�
gradient@@
<�9	%�"
�@@
�
p
` VariableSpec 
`��˧��?
� "
 �
"__inference__update_step_xla_32165f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`�˧��?
� "
 �
"__inference__update_step_xla_32170rl�i
b�_
�
gradient
� �
6�3	�
�
� �
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_32175hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
"__inference__update_step_xla_32180pj�g
`�]
�
gradient	�

5�2	�
�	�

�
p
` VariableSpec 
`૷���?
� "
 �
"__inference__update_step_xla_32185f`�]
V�S
�
gradient

0�-	�
�

�
p
` VariableSpec 
`�궧��?
� "
 �
 __inference__wrapped_model_31265t%&;<DE`aop9�6
/�,
*�'
c0_input���������  
� ")�&
$
d11�
d11���������
�
=__inference_c0_layer_call_and_return_conditional_losses_32207s7�4
-�*
(�%
inputs���������  
� "4�1
*�'
tensor_0���������   
� �
"__inference_c0_layer_call_fn_32194h7�4
-�*
(�%
inputs���������  
� ")�&
unknown���������   �
=__inference_c1_layer_call_and_return_conditional_losses_32229s%&7�4
-�*
(�%
inputs���������   
� "4�1
*�'
tensor_0���������   
� �
"__inference_c1_layer_call_fn_32216h%&7�4
-�*
(�%
inputs���������   
� ")�&
unknown���������   �
=__inference_c4_layer_call_and_return_conditional_losses_32288s;<7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������@
� �
"__inference_c4_layer_call_fn_32275h;<7�4
-�*
(�%
inputs��������� 
� ")�&
unknown���������@�
=__inference_c5_layer_call_and_return_conditional_losses_32310sDE7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
"__inference_c5_layer_call_fn_32297hDE7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
>__inference_d11_layer_call_and_return_conditional_losses_32429dop0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������

� �
#__inference_d11_layer_call_fn_32416Yop0�-
&�#
!�
inputs����������
� "!�
unknown���������
�
=__inference_d9_layer_call_and_return_conditional_losses_32380e`a0�-
&�#
!�
inputs���������� 
� "-�*
#� 
tensor_0����������
� �
"__inference_d9_layer_call_fn_32367Z`a0�-
&�#
!�
inputs���������� 
� ""�
unknown�����������
?__inference_dr10_layer_call_and_return_conditional_losses_32402e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
?__inference_dr10_layer_call_and_return_conditional_losses_32407e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
$__inference_dr10_layer_call_fn_32385Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
$__inference_dr10_layer_call_fn_32390Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
>__inference_dr3_layer_call_and_return_conditional_losses_32261s;�8
1�.
(�%
inputs��������� 
p
� "4�1
*�'
tensor_0��������� 
� �
>__inference_dr3_layer_call_and_return_conditional_losses_32266s;�8
1�.
(�%
inputs��������� 
p 
� "4�1
*�'
tensor_0��������� 
� �
#__inference_dr3_layer_call_fn_32244h;�8
1�.
(�%
inputs��������� 
p
� ")�&
unknown��������� �
#__inference_dr3_layer_call_fn_32249h;�8
1�.
(�%
inputs��������� 
p 
� ")�&
unknown��������� �
>__inference_dr7_layer_call_and_return_conditional_losses_32342s;�8
1�.
(�%
inputs���������@
p
� "4�1
*�'
tensor_0���������@
� �
>__inference_dr7_layer_call_and_return_conditional_losses_32347s;�8
1�.
(�%
inputs���������@
p 
� "4�1
*�'
tensor_0���������@
� �
#__inference_dr7_layer_call_fn_32325h;�8
1�.
(�%
inputs���������@
p
� ")�&
unknown���������@�
#__inference_dr7_layer_call_fn_32330h;�8
1�.
(�%
inputs���������@
p 
� ")�&
unknown���������@�
=__inference_f8_layer_call_and_return_conditional_losses_32358h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
tensor_0���������� 
� �
"__inference_f8_layer_call_fn_32352]7�4
-�*
(�%
inputs���������@
� ""�
unknown���������� @
__inference_loss_fn_0_32434!�

� 
� "�
unknown A
__inference_loss_fn_10_32484!�

� 
� "�
unknown A
__inference_loss_fn_11_32489!�

� 
� "�
unknown @
__inference_loss_fn_1_32439!�

� 
� "�
unknown @
__inference_loss_fn_2_32444!�

� 
� "�
unknown @
__inference_loss_fn_3_32449!�

� 
� "�
unknown @
__inference_loss_fn_4_32454!�

� 
� "�
unknown @
__inference_loss_fn_5_32459!�

� 
� "�
unknown @
__inference_loss_fn_6_32464!�

� 
� "�
unknown @
__inference_loss_fn_7_32469!�

� 
� "�
unknown @
__inference_loss_fn_8_32474!�

� 
� "�
unknown @
__inference_loss_fn_9_32479!�

� 
� "�
unknown �
=__inference_m2_layer_call_and_return_conditional_losses_32239�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
"__inference_m2_layer_call_fn_32234�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
=__inference_m6_layer_call_and_return_conditional_losses_32320�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
"__inference_m6_layer_call_fn_32315�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
E__inference_sequential_layer_call_and_return_conditional_losses_31472%&;<DE`aopA�>
7�4
*�'
c0_input���������  
p

 
� ",�)
"�
tensor_0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_31539%&;<DE`aopA�>
7�4
*�'
c0_input���������  
p 

 
� ",�)
"�
tensor_0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_32060}%&;<DE`aop?�<
5�2
(�%
inputs���������  
p

 
� ",�)
"�
tensor_0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_32125}%&;<DE`aop?�<
5�2
(�%
inputs���������  
p 

 
� ",�)
"�
tensor_0���������

� �
*__inference_sequential_layer_call_fn_31621t%&;<DE`aopA�>
7�4
*�'
c0_input���������  
p

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_31702t%&;<DE`aopA�>
7�4
*�'
c0_input���������  
p 

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_31945r%&;<DE`aop?�<
5�2
(�%
inputs���������  
p

 
� "!�
unknown���������
�
*__inference_sequential_layer_call_fn_31974r%&;<DE`aop?�<
5�2
(�%
inputs���������  
p 

 
� "!�
unknown���������
�
#__inference_signature_wrapper_31904�%&;<DE`aopE�B
� 
;�8
6
c0_input*�'
c0_input���������  ")�&
$
d11�
d11���������

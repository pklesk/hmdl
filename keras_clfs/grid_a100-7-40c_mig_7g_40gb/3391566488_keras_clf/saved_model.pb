Д
нЌ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

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
$
DisableCopyOnRead
resource
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
С
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
executor_typestring Ј
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
 "serve*2.12.02v2.12.0-rc1-12-g0db597d0d758Ы	
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
t
Adam/v/d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/v/d9/bias
m
"Adam/v/d9/bias/Read/ReadVariableOpReadVariableOpAdam/v/d9/bias*
_output_shapes
:(*
dtype0
t
Adam/m/d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/m/d9/bias
m
"Adam/m/d9/bias/Read/ReadVariableOpReadVariableOpAdam/m/d9/bias*
_output_shapes
:(*
dtype0
|
Adam/v/d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/v/d9/kernel
u
$Adam/v/d9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d9/kernel*
_output_shapes

:@(*
dtype0
|
Adam/m/d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/m/d9/kernel
u
$Adam/m/d9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d9/kernel*
_output_shapes

:@(*
dtype0
t
Adam/v/d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/d7/bias
m
"Adam/v/d7/bias/Read/ReadVariableOpReadVariableOpAdam/v/d7/bias*
_output_shapes
:@*
dtype0
t
Adam/m/d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/d7/bias
m
"Adam/m/d7/bias/Read/ReadVariableOpReadVariableOpAdam/m/d7/bias*
_output_shapes
:@*
dtype0
}
Adam/v/d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 @*!
shared_nameAdam/v/d7/kernel
v
$Adam/v/d7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d7/kernel*
_output_shapes
:	 @*
dtype0
}
Adam/m/d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 @*!
shared_nameAdam/m/d7/kernel
v
$Adam/m/d7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d7/kernel*
_output_shapes
:	 @*
dtype0
t
Adam/v/c3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/c3/bias
m
"Adam/v/c3/bias/Read/ReadVariableOpReadVariableOpAdam/v/c3/bias*
_output_shapes
:*
dtype0
t
Adam/m/c3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/c3/bias
m
"Adam/m/c3/bias/Read/ReadVariableOpReadVariableOpAdam/m/c3/bias*
_output_shapes
:*
dtype0

Adam/v/c3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/v/c3/kernel
}
$Adam/v/c3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c3/kernel*&
_output_shapes
:*
dtype0

Adam/m/c3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/m/c3/kernel
}
$Adam/m/c3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c3/kernel*&
_output_shapes
:*
dtype0
t
Adam/v/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/c0/bias
m
"Adam/v/c0/bias/Read/ReadVariableOpReadVariableOpAdam/v/c0/bias*
_output_shapes
:*
dtype0
t
Adam/m/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/c0/bias
m
"Adam/m/c0/bias/Read/ReadVariableOpReadVariableOpAdam/m/c0/bias*
_output_shapes
:*
dtype0

Adam/v/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/v/c0/kernel
}
$Adam/v/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c0/kernel*&
_output_shapes
:*
dtype0

Adam/m/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/m/c0/kernel
}
$Adam/m/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c0/kernel*&
_output_shapes
:*
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
o
	d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 @*
shared_name	d7/kernel
h
d7/kernel/Read/ReadVariableOpReadVariableOp	d7/kernel*
_output_shapes
:	 @*
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

serving_default_c0_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ@@*
dtype0*$
shape:џџџџџџџџџ@@

StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c3/kernelc3/bias	d7/kerneld7/bias	d9/kerneld9/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *,
f'R%
#__inference_signature_wrapper_87432

NoOpNoOp
S
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*МR
valueВRBЏR BЈR
Ж
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 
Ѕ
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator* 
Ш
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias
 2_jit_compiled_convolution_op*

3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
Ѕ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator* 

@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
І
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias*
Ѕ
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator* 
І
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias*
<
0
1
02
13
L4
M5
[6
\7*
<
0
1
02
13
L4
M5
[6
\7*
:
]0
^1
_2
`3
a4
b5
c6
d7* 
А
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
6
ntrace_0
otrace_1
ptrace_2
qtrace_3* 
* 

r
_variables
s_iterations
t_learning_rate
u_index_dict
v
_momentums
w_velocities
x_update_step_xla*

yserving_default* 

0
1*

0
1*

]0
^1* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

00
11*

00
11*

_0
`1* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

trace_0* 

trace_0* 
YS
VARIABLE_VALUE	c3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

Єtrace_0
Ѕtrace_1* 

Іtrace_0
Їtrace_1* 
* 
* 
* 
* 

Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

­trace_0* 

Ўtrace_0* 

L0
M1*

L0
M1*

a0
b1* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
YS
VARIABLE_VALUE	d7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

Лtrace_0
Мtrace_1* 

Нtrace_0
Оtrace_1* 
* 

[0
\1*

[0
\1*

c0
d1* 

Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

Фtrace_0* 

Хtrace_0* 
YS
VARIABLE_VALUE	d9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

Цtrace_0* 

Чtrace_0* 

Шtrace_0* 

Щtrace_0* 

Ъtrace_0* 

Ыtrace_0* 

Ьtrace_0* 

Эtrace_0* 
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
Ю0
Я1*
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

s0
а1
б2
в3
г4
д5
е6
ж7
з8
и9
й10
к11
л12
м13
н14
о15
п16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
а0
в1
д2
ж3
и4
к5
м6
о7*
D
б0
г1
е2
з3
й4
л5
н6
п7*
r
рtrace_0
сtrace_1
тtrace_2
уtrace_3
фtrace_4
хtrace_5
цtrace_6
чtrace_7* 
* 
* 
* 
* 

]0
^1* 
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
_0
`1* 
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
a0
b1* 
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
c0
d1* 
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
ш	variables
щ	keras_api

ъtotal

ыcount*
M
ь	variables
э	keras_api

юtotal

яcount
№
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
VARIABLE_VALUEAdam/m/c3/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/c3/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/c3/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/c3/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/d7/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/d7/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/d7/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/d7/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/d9/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/d9/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/d9/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/d9/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

ъ0
ы1*

ш	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ю0
я1*

ь	variables*
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
й
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c3/kernelc3/bias	d7/kerneld7/bias	d9/kerneld9/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c3/kernelAdam/v/c3/kernelAdam/m/c3/biasAdam/v/c3/biasAdam/m/d7/kernelAdam/v/d7/kernelAdam/m/d7/biasAdam/v/d7/biasAdam/m/d9/kernelAdam/v/d9/kernelAdam/m/d9/biasAdam/v/d9/biastotal_1count_1totalcountConst*+
Tin$
"2 *
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
__inference__traced_save_88080
д
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	c3/kernelc3/bias	d7/kerneld7/bias	d9/kerneld9/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/c3/kernelAdam/v/c3/kernelAdam/m/c3/biasAdam/v/c3/biasAdam/m/d7/kernelAdam/v/d7/kernelAdam/m/d7/biasAdam/v/d7/biasAdam/m/d9/kernelAdam/v/d9/kernelAdam/m/d9/biasAdam/v/d9/biastotal_1count_1totalcount**
Tin#
!2*
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
!__inference__traced_restore_88180ђВ
З
N
"__inference__update_step_xla_87632
gradient
variable:@(*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@(: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:@(
"
_user_specified_name
gradient
і.

G__inference_sequential_9_layer_call_and_return_conditional_losses_87200

inputs"
c0_87165:
c0_87167:"
c3_87172:
c3_87174:
d7_87180:	 @
d7_87182:@
d9_87186:@(
d9_87188:(
identityЂc0/StatefulPartitionedCallЂc3/StatefulPartitionedCallЂd7/StatefulPartitionedCallЂd9/StatefulPartitionedCallЂdr2/StatefulPartitionedCallЂdr5/StatefulPartitionedCallЂdr8/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_87165c0_87167*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_86982з
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_86947с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_87001
c3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0c3_87172c3_87174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_87016з
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_86959џ
dr5/StatefulPartitionedCallStatefulPartitionedCallm4/PartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_87035б
f6/PartitionedCallPartitionedCall$dr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_87043ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_87180d7_87182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_87058џ
dr8/StatefulPartitionedCallStatefulPartitionedCall#d7/StatefulPartitionedCall:output:0^dr5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr8_layer_call_and_return_conditional_losses_87076њ
d9/StatefulPartitionedCallStatefulPartitionedCall$dr8/StatefulPartitionedCall:output:0d9_87186d9_87188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_87091`
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
:џџџџџџџџџ(
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall^dr8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall2:
dr8/StatefulPartitionedCalldr8/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
О
і
=__inference_c0_layer_call_and_return_conditional_losses_86982

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
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
:џџџџџџџџџ@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@`
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
:џџџџџџџџџ@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
л
ю
=__inference_d9_layer_call_and_return_conditional_losses_87091

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(`
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
:џџџџџџџџџ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ѓ

]
>__inference_dr8_layer_call_and_return_conditional_losses_87076

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0*
seed2џџџџ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
И
?
#__inference_dr2_layer_call_fn_87679

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_87119h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  :W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
П
Y
=__inference_f6_layer_call_and_return_conditional_losses_87766

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_87612
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:
"
_user_specified_name
gradient
ќ.

G__inference_sequential_9_layer_call_and_return_conditional_losses_87106
c0_input"
c0_86983:
c0_86985:"
c3_87017:
c3_87019:
d7_87059:	 @
d7_87061:@
d9_87092:@(
d9_87094:(
identityЂc0/StatefulPartitionedCallЂc3/StatefulPartitionedCallЂd7/StatefulPartitionedCallЂd9/StatefulPartitionedCallЂdr2/StatefulPartitionedCallЂdr5/StatefulPartitionedCallЂdr8/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_86983c0_86985*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_86982з
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_86947с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_87001
c3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0c3_87017c3_87019*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_87016з
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_86959џ
dr5/StatefulPartitionedCallStatefulPartitionedCallm4/PartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_87035б
f6/PartitionedCallPartitionedCall$dr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_87043ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_87059d7_87061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_87058џ
dr8/StatefulPartitionedCallStatefulPartitionedCall#d7/StatefulPartitionedCall:output:0^dr5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr8_layer_call_and_return_conditional_losses_87076њ
d9/StatefulPartitionedCallStatefulPartitionedCall$dr8/StatefulPartitionedCall:output:0d9_87092d9_87094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_87091`
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
:џџџџџџџџџ(
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall^dr8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall2:
dr8/StatefulPartitionedCalldr8/StatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ@@
"
_user_specified_name
c0_input
у

"__inference_c0_layer_call_fn_87646

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_86982w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ј
>
"__inference_f6_layer_call_fn_87760

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_87043a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Y
=__inference_m1_layer_call_and_return_conditional_losses_86947

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Y
=__inference_m4_layer_call_and_return_conditional_losses_87728

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_87627
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
Ф
+
__inference_loss_fn_5_87867
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
Ћ
J
"__inference__update_step_xla_87617
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
л

]
>__inference_dr5_layer_call_and_return_conditional_losses_87750

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯЅ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2џџџџ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л

]
>__inference_dr5_layer_call_and_return_conditional_losses_87035

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯЅ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2џџџџ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В2

 __inference__wrapped_model_86941
c0_inputH
.sequential_9_c0_conv2d_readvariableop_resource:=
/sequential_9_c0_biasadd_readvariableop_resource:H
.sequential_9_c3_conv2d_readvariableop_resource:=
/sequential_9_c3_biasadd_readvariableop_resource:A
.sequential_9_d7_matmul_readvariableop_resource:	 @=
/sequential_9_d7_biasadd_readvariableop_resource:@@
.sequential_9_d9_matmul_readvariableop_resource:@(=
/sequential_9_d9_biasadd_readvariableop_resource:(
identityЂ&sequential_9/c0/BiasAdd/ReadVariableOpЂ%sequential_9/c0/Conv2D/ReadVariableOpЂ&sequential_9/c3/BiasAdd/ReadVariableOpЂ%sequential_9/c3/Conv2D/ReadVariableOpЂ&sequential_9/d7/BiasAdd/ReadVariableOpЂ%sequential_9/d7/MatMul/ReadVariableOpЂ&sequential_9/d9/BiasAdd/ReadVariableOpЂ%sequential_9/d9/MatMul/ReadVariableOp
%sequential_9/c0/Conv2D/ReadVariableOpReadVariableOp.sequential_9_c0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Л
sequential_9/c0/Conv2DConv2Dc0_input-sequential_9/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides

&sequential_9/c0/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0­
sequential_9/c0/BiasAddBiasAddsequential_9/c0/Conv2D:output:0.sequential_9/c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@x
sequential_9/c0/ReluRelu sequential_9/c0/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@Г
sequential_9/m1/MaxPoolMaxPool"sequential_9/c0/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides

sequential_9/dr2/IdentityIdentity sequential_9/m1/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  
%sequential_9/c3/Conv2D/ReadVariableOpReadVariableOp.sequential_9_c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0е
sequential_9/c3/Conv2DConv2D"sequential_9/dr2/Identity:output:0-sequential_9/c3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides

&sequential_9/c3/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0­
sequential_9/c3/BiasAddBiasAddsequential_9/c3/Conv2D:output:0.sequential_9/c3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  x
sequential_9/c3/ReluRelu sequential_9/c3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  Г
sequential_9/m4/MaxPoolMaxPool"sequential_9/c3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

sequential_9/dr5/IdentityIdentity sequential_9/m4/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџf
sequential_9/f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
sequential_9/f6/ReshapeReshape"sequential_9/dr5/Identity:output:0sequential_9/f6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
%sequential_9/d7/MatMul/ReadVariableOpReadVariableOp.sequential_9_d7_matmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0Ѓ
sequential_9/d7/MatMulMatMul sequential_9/f6/Reshape:output:0-sequential_9/d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&sequential_9/d7/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
sequential_9/d7/BiasAddBiasAdd sequential_9/d7/MatMul:product:0.sequential_9/d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@p
sequential_9/d7/ReluRelu sequential_9/d7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
sequential_9/dr8/IdentityIdentity"sequential_9/d7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@
%sequential_9/d9/MatMul/ReadVariableOpReadVariableOp.sequential_9_d9_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0Ѕ
sequential_9/d9/MatMulMatMul"sequential_9/dr8/Identity:output:0-sequential_9/d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
&sequential_9/d9/BiasAdd/ReadVariableOpReadVariableOp/sequential_9_d9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0І
sequential_9/d9/BiasAddBiasAdd sequential_9/d9/MatMul:product:0.sequential_9/d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(v
sequential_9/d9/SoftmaxSoftmax sequential_9/d9/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(p
IdentityIdentity!sequential_9/d9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ(
NoOpNoOp'^sequential_9/c0/BiasAdd/ReadVariableOp&^sequential_9/c0/Conv2D/ReadVariableOp'^sequential_9/c3/BiasAdd/ReadVariableOp&^sequential_9/c3/Conv2D/ReadVariableOp'^sequential_9/d7/BiasAdd/ReadVariableOp&^sequential_9/d7/MatMul/ReadVariableOp'^sequential_9/d9/BiasAdd/ReadVariableOp&^sequential_9/d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 2P
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
:џџџџџџџџџ@@
"
_user_specified_name
c0_input
к
я
=__inference_d7_layer_call_and_return_conditional_losses_87058

inputs1
matmul_readvariableop_resource:	 @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@`
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
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ш
+
__inference_loss_fn_4_87862
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
И
?
#__inference_dr5_layer_call_fn_87738

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_87131h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

\
#__inference_dr2_layer_call_fn_87674

inputs
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_87001w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
+
Н
G__inference_sequential_9_layer_call_and_return_conditional_losses_87159
c0_input"
c0_87109:
c0_87111:"
c3_87121:
c3_87123:
d7_87134:	 @
d7_87136:@
d9_87145:@(
d9_87147:(
identityЂc0/StatefulPartitionedCallЂc3/StatefulPartitionedCallЂd7/StatefulPartitionedCallЂd9/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_87109c0_87111*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_86982з
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_86947б
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_87119њ
c3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0c3_87121c3_87123*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_87016з
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_86959б
dr5/PartitionedCallPartitionedCallm4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_87131Щ
f6/PartitionedCallPartitionedCalldr5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_87043ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_87134d7_87136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_87058б
dr8/PartitionedCallPartitionedCall#d7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr8_layer_call_and_return_conditional_losses_87143ђ
d9/StatefulPartitionedCallStatefulPartitionedCalldr8/PartitionedCall:output:0d9_87145d9_87147*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_87091`
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
:џџџџџџџџџ(К
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ@@
"
_user_specified_name
c0_input
О
і
=__inference_c3_layer_call_and_return_conditional_losses_87016

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
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
:џџџџџџџџџ  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  `
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
:џџџџџџџџџ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_87602
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:
"
_user_specified_name
gradient
А}
Ѕ
!__inference__traced_restore_88180
file_prefix4
assignvariableop_c0_kernel:(
assignvariableop_1_c0_bias:6
assignvariableop_2_c3_kernel:(
assignvariableop_3_c3_bias:/
assignvariableop_4_d7_kernel:	 @(
assignvariableop_5_d7_bias:@.
assignvariableop_6_d9_kernel:@((
assignvariableop_7_d9_bias:(&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: >
$assignvariableop_10_adam_m_c0_kernel:>
$assignvariableop_11_adam_v_c0_kernel:0
"assignvariableop_12_adam_m_c0_bias:0
"assignvariableop_13_adam_v_c0_bias:>
$assignvariableop_14_adam_m_c3_kernel:>
$assignvariableop_15_adam_v_c3_kernel:0
"assignvariableop_16_adam_m_c3_bias:0
"assignvariableop_17_adam_v_c3_bias:7
$assignvariableop_18_adam_m_d7_kernel:	 @7
$assignvariableop_19_adam_v_d7_kernel:	 @0
"assignvariableop_20_adam_m_d7_bias:@0
"assignvariableop_21_adam_v_d7_bias:@6
$assignvariableop_22_adam_m_d9_kernel:@(6
$assignvariableop_23_adam_v_d9_kernel:@(0
"assignvariableop_24_adam_m_d9_bias:(0
"assignvariableop_25_adam_v_d9_bias:(%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
valueзBдB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B К
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOpAssignVariableOpassignvariableop_c0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_1AssignVariableOpassignvariableop_1_c0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_2AssignVariableOpassignvariableop_2_c3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_3AssignVariableOpassignvariableop_3_c3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_4AssignVariableOpassignvariableop_4_d7_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_5AssignVariableOpassignvariableop_5_d7_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_6AssignVariableOpassignvariableop_6_d9_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_7AssignVariableOpassignvariableop_7_d9_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_10AssignVariableOp$assignvariableop_10_adam_m_c0_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_11AssignVariableOp$assignvariableop_11_adam_v_c0_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_12AssignVariableOp"assignvariableop_12_adam_m_c0_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_13AssignVariableOp"assignvariableop_13_adam_v_c0_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_14AssignVariableOp$assignvariableop_14_adam_m_c3_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_v_c3_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_m_c3_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_17AssignVariableOp"assignvariableop_17_adam_v_c3_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_18AssignVariableOp$assignvariableop_18_adam_m_d7_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_v_d7_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_m_d7_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_21AssignVariableOp"assignvariableop_21_adam_v_d7_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_22AssignVariableOp$assignvariableop_22_adam_m_d9_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_v_d9_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_m_d9_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_25AssignVariableOp"assignvariableop_25_adam_v_d9_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 у
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: а
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
Ш
+
__inference_loss_fn_6_87872
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
Ф
+
__inference_loss_fn_3_87857
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
л

]
>__inference_dr2_layer_call_and_return_conditional_losses_87001

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯЅ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
dtype0*
seed2џџџџ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  :W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Т	
Х
#__inference_signature_wrapper_87432
c0_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	 @
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *)
f$R"
 __inference__wrapped_model_86941o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ@@
"
_user_specified_name
c0_input
Ф
+
__inference_loss_fn_1_87847
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
ђ	
Ю
,__inference_sequential_9_layer_call_fn_87278
c0_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	 @
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_87259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ@@
"
_user_specified_name
c0_input
П
Y
=__inference_f6_layer_call_and_return_conditional_losses_87043

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
>
"__inference_m1_layer_call_fn_87664

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_86947
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б
\
>__inference_dr8_layer_call_and_return_conditional_losses_87815

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О
і
=__inference_c0_layer_call_and_return_conditional_losses_87659

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
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
:џџџџџџџџџ@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@`
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
:џџџџџџџџџ@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

\
#__inference_dr5_layer_call_fn_87733

inputs
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_87035w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
\
>__inference_dr5_layer_call_and_return_conditional_losses_87131

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь	
Ь
,__inference_sequential_9_layer_call_fn_87482

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	 @
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_87259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_87119

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ  c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  :W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
 .
я
G__inference_sequential_9_layer_call_and_return_conditional_losses_87597

inputs;
!c0_conv2d_readvariableop_resource:0
"c0_biasadd_readvariableop_resource:;
!c3_conv2d_readvariableop_resource:0
"c3_biasadd_readvariableop_resource:4
!d7_matmul_readvariableop_resource:	 @0
"d7_biasadd_readvariableop_resource:@3
!d9_matmul_readvariableop_resource:@(0
"d9_biasadd_readvariableop_resource:(
identityЂc0/BiasAdd/ReadVariableOpЂc0/Conv2D/ReadVariableOpЂc3/BiasAdd/ReadVariableOpЂc3/Conv2D/ReadVariableOpЂd7/BiasAdd/ReadVariableOpЂd7/MatMul/ReadVariableOpЂd9/BiasAdd/ReadVariableOpЂd9/MatMul/ReadVariableOp
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
g
dr2/IdentityIdentitym1/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  
c3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
	c3/Conv2DConv2Ddr2/Identity:output:0 c3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides
x
c3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0

c3/BiasAddBiasAddc3/Conv2D:output:0!c3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  ^
c3/ReluReluc3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  

m4/MaxPoolMaxPoolc3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
g
dr5/IdentityIdentitym4/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџY
f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   r

f6/ReshapeReshapedr5/Identity:output:0f6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ {
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0|
	d7/MatMulMatMulf6/Reshape:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
d7/ReluRelud7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
dr8/IdentityIdentityd7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@z
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0~
	d9/MatMulMatMuldr8/Identity:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(\

d9/SoftmaxSoftmaxd9/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(`
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
:џџџџџџџџџ(Ђ
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c3/BiasAdd/ReadVariableOp^c3/Conv2D/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 26
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
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ђ	
Ю
,__inference_sequential_9_layer_call_fn_87219
c0_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	 @
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallc0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_87200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ@@
"
_user_specified_name
c0_input
ё
\
>__inference_dr5_layer_call_and_return_conditional_losses_87755

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
+
Л
G__inference_sequential_9_layer_call_and_return_conditional_losses_87259

inputs"
c0_87224:
c0_87226:"
c3_87231:
c3_87233:
d7_87239:	 @
d7_87241:@
d9_87245:@(
d9_87247:(
identityЂc0/StatefulPartitionedCallЂc3/StatefulPartitionedCallЂd7/StatefulPartitionedCallЂd9/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_87224c0_87226*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_86982з
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_86947б
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_87119њ
c3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0c3_87231c3_87233*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_87016з
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_86959б
dr5/PartitionedCallPartitionedCallm4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_87131Щ
f6/PartitionedCallPartitionedCalldr5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_87043ё
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_87239d7_87241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_87058б
dr8/PartitionedCallPartitionedCall#d7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr8_layer_call_and_return_conditional_losses_87143ђ
d9/StatefulPartitionedCallStatefulPartitionedCalldr8/PartitionedCall:output:0d9_87245d9_87247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_87091`
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
:џџџџџџџџџ(К
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
л
ю
=__inference_d9_layer_call_and_return_conditional_losses_87837

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(`
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
:џџџџџџџџџ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
К
O
"__inference__update_step_xla_87622
gradient
variable:	 @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	 @: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	 @
"
_user_specified_name
gradient
Ш
+
__inference_loss_fn_2_87852
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
ь	
Ь
,__inference_sequential_9_layer_call_fn_87461

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	 @
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_87200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

Y
=__inference_m4_layer_call_and_return_conditional_losses_86959

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
>
"__inference_m4_layer_call_fn_87723

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_86959
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л

]
>__inference_dr2_layer_call_and_return_conditional_losses_87691

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯЅ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
dtype0*
seed2џџџџ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  :W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Ѓ

]
>__inference_dr8_layer_call_and_return_conditional_losses_87810

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0*
seed2џџџџ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_87637
gradient
variable:(*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:(: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:(
"
_user_specified_name
gradient
к
я
=__inference_d7_layer_call_and_return_conditional_losses_87788

inputs1
matmul_readvariableop_resource:	 @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@`
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
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

?
#__inference_dr8_layer_call_fn_87798

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr8_layer_call_and_return_conditional_losses_87143`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
у

"__inference_c3_layer_call_fn_87705

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_87016w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Ыи

__inference__traced_save_88080
file_prefix:
 read_disablecopyonread_c0_kernel:.
 read_1_disablecopyonread_c0_bias:<
"read_2_disablecopyonread_c3_kernel:.
 read_3_disablecopyonread_c3_bias:5
"read_4_disablecopyonread_d7_kernel:	 @.
 read_5_disablecopyonread_d7_bias:@4
"read_6_disablecopyonread_d9_kernel:@(.
 read_7_disablecopyonread_d9_bias:(,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: D
*read_10_disablecopyonread_adam_m_c0_kernel:D
*read_11_disablecopyonread_adam_v_c0_kernel:6
(read_12_disablecopyonread_adam_m_c0_bias:6
(read_13_disablecopyonread_adam_v_c0_bias:D
*read_14_disablecopyonread_adam_m_c3_kernel:D
*read_15_disablecopyonread_adam_v_c3_kernel:6
(read_16_disablecopyonread_adam_m_c3_bias:6
(read_17_disablecopyonread_adam_v_c3_bias:=
*read_18_disablecopyonread_adam_m_d7_kernel:	 @=
*read_19_disablecopyonread_adam_v_d7_kernel:	 @6
(read_20_disablecopyonread_adam_m_d7_bias:@6
(read_21_disablecopyonread_adam_v_d7_bias:@<
*read_22_disablecopyonread_adam_m_d9_kernel:@(<
*read_23_disablecopyonread_adam_v_d9_kernel:@(6
(read_24_disablecopyonread_adam_m_d9_bias:(6
(read_25_disablecopyonread_adam_v_d9_bias:(+
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
: r
Read/DisableCopyOnReadDisableCopyOnRead read_disablecopyonread_c0_kernel"/device:CPU:0*
_output_shapes
 Є
Read/ReadVariableOpReadVariableOp read_disablecopyonread_c0_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:t
Read_1/DisableCopyOnReadDisableCopyOnRead read_1_disablecopyonread_c0_bias"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOp read_1_disablecopyonread_c0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_c3_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_c3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_c3_bias"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_c3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_d7_kernel"/device:CPU:0*
_output_shapes
 Ѓ
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_d7_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 @*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 @d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	 @t
Read_5/DisableCopyOnReadDisableCopyOnRead read_5_disablecopyonread_d7_bias"/device:CPU:0*
_output_shapes
 
Read_5/ReadVariableOpReadVariableOp read_5_disablecopyonread_d7_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_d9_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_d9_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@(t
Read_7/DisableCopyOnReadDisableCopyOnRead read_7_disablecopyonread_d9_bias"/device:CPU:0*
_output_shapes
 
Read_7/ReadVariableOpReadVariableOp read_7_disablecopyonread_d9_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:(v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_adam_m_c0_kernel"/device:CPU:0*
_output_shapes
 Д
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_adam_m_c0_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_11/DisableCopyOnReadDisableCopyOnRead*read_11_disablecopyonread_adam_v_c0_kernel"/device:CPU:0*
_output_shapes
 Д
Read_11/ReadVariableOpReadVariableOp*read_11_disablecopyonread_adam_v_c0_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
:}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_adam_m_c0_bias"/device:CPU:0*
_output_shapes
 І
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_adam_m_c0_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_adam_v_c0_bias"/device:CPU:0*
_output_shapes
 І
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_adam_v_c0_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_adam_m_c3_kernel"/device:CPU:0*
_output_shapes
 Д
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_adam_m_c3_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_adam_v_c3_kernel"/device:CPU:0*
_output_shapes
 Д
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_adam_v_c3_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
:}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_adam_m_c3_bias"/device:CPU:0*
_output_shapes
 І
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_adam_m_c3_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_17/DisableCopyOnReadDisableCopyOnRead(read_17_disablecopyonread_adam_v_c3_bias"/device:CPU:0*
_output_shapes
 І
Read_17/ReadVariableOpReadVariableOp(read_17_disablecopyonread_adam_v_c3_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_18/DisableCopyOnReadDisableCopyOnRead*read_18_disablecopyonread_adam_m_d7_kernel"/device:CPU:0*
_output_shapes
 ­
Read_18/ReadVariableOpReadVariableOp*read_18_disablecopyonread_adam_m_d7_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 @*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 @f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	 @
Read_19/DisableCopyOnReadDisableCopyOnRead*read_19_disablecopyonread_adam_v_d7_kernel"/device:CPU:0*
_output_shapes
 ­
Read_19/ReadVariableOpReadVariableOp*read_19_disablecopyonread_adam_v_d7_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 @*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 @f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	 @}
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_adam_m_d7_bias"/device:CPU:0*
_output_shapes
 І
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_adam_m_d7_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_21/DisableCopyOnReadDisableCopyOnRead(read_21_disablecopyonread_adam_v_d7_bias"/device:CPU:0*
_output_shapes
 І
Read_21/ReadVariableOpReadVariableOp(read_21_disablecopyonread_adam_v_d7_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_22/DisableCopyOnReadDisableCopyOnRead*read_22_disablecopyonread_adam_m_d9_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_22/ReadVariableOpReadVariableOp*read_22_disablecopyonread_adam_m_d9_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:@(
Read_23/DisableCopyOnReadDisableCopyOnRead*read_23_disablecopyonread_adam_v_d9_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_23/ReadVariableOpReadVariableOp*read_23_disablecopyonread_adam_v_d9_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:@(}
Read_24/DisableCopyOnReadDisableCopyOnRead(read_24_disablecopyonread_adam_m_d9_bias"/device:CPU:0*
_output_shapes
 І
Read_24/ReadVariableOpReadVariableOp(read_24_disablecopyonread_adam_m_d9_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:(}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_adam_v_d9_bias"/device:CPU:0*
_output_shapes
 І
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_adam_v_d9_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:(v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: И
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
valueзBдB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЋ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: љ
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ЁD
я
G__inference_sequential_9_layer_call_and_return_conditional_losses_87550

inputs;
!c0_conv2d_readvariableop_resource:0
"c0_biasadd_readvariableop_resource:;
!c3_conv2d_readvariableop_resource:0
"c3_biasadd_readvariableop_resource:4
!d7_matmul_readvariableop_resource:	 @0
"d7_biasadd_readvariableop_resource:@3
!d9_matmul_readvariableop_resource:@(0
"d9_biasadd_readvariableop_resource:(
identityЂc0/BiasAdd/ReadVariableOpЂc0/Conv2D/ReadVariableOpЂc3/BiasAdd/ReadVariableOpЂc3/Conv2D/ReadVariableOpЂd7/BiasAdd/ReadVariableOpЂd7/MatMul/ReadVariableOpЂd9/BiasAdd/ReadVariableOpЂd9/MatMul/ReadVariableOp
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
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
 *ЋЊЊ?
dr2/dropout/MulMulm1/MaxPool:output:0dr2/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  b
dr2/dropout/ShapeShapem1/MaxPool:output:0*
T0*
_output_shapes
::эЯ­
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
dtype0*
seed2џџџџ_
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >К
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  X
dr2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ћ
dr2/dropout/SelectV2SelectV2dr2/dropout/GreaterEqual:z:0dr2/dropout/Mul:z:0dr2/dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  
c3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ж
	c3/Conv2DConv2Ddr2/dropout/SelectV2:output:0 c3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides
x
c3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0

c3/BiasAddBiasAddc3/Conv2D:output:0!c3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  ^
c3/ReluReluc3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  

m4/MaxPoolMaxPoolc3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
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
 *ЋЊЊ?
dr5/dropout/MulMulm4/MaxPool:output:0dr5/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџb
dr5/dropout/ShapeShapem4/MaxPool:output:0*
T0*
_output_shapes
::эЯЉ
(dr5/dropout/random_uniform/RandomUniformRandomUniformdr5/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2_
dr5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >К
dr5/dropout/GreaterEqualGreaterEqual1dr5/dropout/random_uniform/RandomUniform:output:0#dr5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџX
dr5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ћ
dr5/dropout/SelectV2SelectV2dr5/dropout/GreaterEqual:z:0dr5/dropout/Mul:z:0dr5/dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџY
f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   z

f6/ReshapeReshapedr5/dropout/SelectV2:output:0f6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ {
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0|
	d7/MatMulMatMulf6/Reshape:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
d7/ReluRelud7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
dr8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?{
dr8/dropout/MulMuld7/Relu:activations:0dr8/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@d
dr8/dropout/ShapeShaped7/Relu:activations:0*
T0*
_output_shapes
::эЯЁ
(dr8/dropout/random_uniform/RandomUniformRandomUniformdr8/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0*
seed2_
dr8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >В
dr8/dropout/GreaterEqualGreaterEqual1dr8/dropout/random_uniform/RandomUniform:output:0#dr8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
dr8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ѓ
dr8/dropout/SelectV2SelectV2dr8/dropout/GreaterEqual:z:0dr8/dropout/Mul:z:0dr8/dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@z
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0
	d9/MatMulMatMuldr8/dropout/SelectV2:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(\

d9/SoftmaxSoftmaxd9/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(`
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
:џџџџџџџџџ(Ђ
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c3/BiasAdd/ReadVariableOp^c3/Conv2D/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ@@: : : : : : : : 26
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
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_87607
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Ф
+
__inference_loss_fn_7_87877
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
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_87696

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ  c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  :W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Л

"__inference_d9_layer_call_fn_87824

inputs
unknown:@(
	unknown_0:(
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d9_layer_call_and_return_conditional_losses_87091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ш
+
__inference_loss_fn_0_87842
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

Y
=__inference_m1_layer_call_and_return_conditional_losses_87669

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б
\
>__inference_dr8_layer_call_and_return_conditional_losses_87143

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О

"__inference_d7_layer_call_fn_87775

inputs
unknown:	 @
	unknown_0:@
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *F
fAR?
=__inference_d7_layer_call_and_return_conditional_losses_87058o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ъ
\
#__inference_dr8_layer_call_fn_87793

inputs
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *G
fBR@
>__inference_dr8_layer_call_and_return_conditional_losses_87076o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О
і
=__inference_c3_layer_call_and_return_conditional_losses_87718

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
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
:џџџџџџџџџ  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  `
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
:џџџџџџџџџ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Џ
serving_default
E
c0_input9
serving_default_c0_input:0џџџџџџџџџ@@6
d90
StatefulPartitionedCall:0џџџџџџџџџ(tensorflow/serving/predict:жИ
а
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
М
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator"
_tf_keras_layer
н
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias
 2_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
М
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
Ѕ
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
М
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator"
_tf_keras_layer
Л
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias"
_tf_keras_layer
X
0
1
02
13
L4
M5
[6
\7"
trackable_list_wrapper
X
0
1
02
13
L4
M5
[6
\7"
trackable_list_wrapper
X
]0
^1
_2
`3
a4
b5
c6
d7"
trackable_list_wrapper
Ъ
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
л
jtrace_0
ktrace_1
ltrace_2
mtrace_32№
,__inference_sequential_9_layer_call_fn_87219
,__inference_sequential_9_layer_call_fn_87278
,__inference_sequential_9_layer_call_fn_87461
,__inference_sequential_9_layer_call_fn_87482Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zjtrace_0zktrace_1zltrace_2zmtrace_3
Ч
ntrace_0
otrace_1
ptrace_2
qtrace_32м
G__inference_sequential_9_layer_call_and_return_conditional_losses_87106
G__inference_sequential_9_layer_call_and_return_conditional_losses_87159
G__inference_sequential_9_layer_call_and_return_conditional_losses_87550
G__inference_sequential_9_layer_call_and_return_conditional_losses_87597Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zntrace_0zotrace_1zptrace_2zqtrace_3
ЬBЩ
 __inference__wrapped_model_86941c0_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

r
_variables
s_iterations
t_learning_rate
u_index_dict
v
_momentums
w_velocities
x_update_step_xla"
experimentalOptimizer
,
yserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
м
trace_02П
"__inference_c0_layer_call_fn_87646
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
љ
trace_02к
=__inference_c0_layer_call_and_return_conditional_losses_87659
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
#:!2	c0/kernel
:2c0/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
о
trace_02П
"__inference_m1_layer_call_fn_87664
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
љ
trace_02к
=__inference_m1_layer_call_and_return_conditional_losses_87669
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
Б
trace_0
trace_12і
#__inference_dr2_layer_call_fn_87674
#__inference_dr2_layer_call_fn_87679Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ч
trace_0
trace_12Ќ
>__inference_dr2_layer_call_and_return_conditional_losses_87691
>__inference_dr2_layer_call_and_return_conditional_losses_87696Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
о
trace_02П
"__inference_c3_layer_call_fn_87705
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
љ
trace_02к
=__inference_c3_layer_call_and_return_conditional_losses_87718
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
#:!2	c3/kernel
:2c3/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
о
trace_02П
"__inference_m4_layer_call_fn_87723
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
љ
trace_02к
=__inference_m4_layer_call_and_return_conditional_losses_87728
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
Б
Єtrace_0
Ѕtrace_12і
#__inference_dr5_layer_call_fn_87733
#__inference_dr5_layer_call_fn_87738Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЄtrace_0zЅtrace_1
ч
Іtrace_0
Їtrace_12Ќ
>__inference_dr5_layer_call_and_return_conditional_losses_87750
>__inference_dr5_layer_call_and_return_conditional_losses_87755Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0zЇtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
о
­trace_02П
"__inference_f6_layer_call_fn_87760
В
FullArgSpec
args

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
annotationsЊ *
 z­trace_0
љ
Ўtrace_02к
=__inference_f6_layer_call_and_return_conditional_losses_87766
В
FullArgSpec
args

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
annotationsЊ *
 zЎtrace_0
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
В
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
о
Дtrace_02П
"__inference_d7_layer_call_fn_87775
В
FullArgSpec
args

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
annotationsЊ *
 zДtrace_0
љ
Еtrace_02к
=__inference_d7_layer_call_and_return_conditional_losses_87788
В
FullArgSpec
args

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
annotationsЊ *
 zЕtrace_0
:	 @2	d7/kernel
:@2d7/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
Б
Лtrace_0
Мtrace_12і
#__inference_dr8_layer_call_fn_87793
#__inference_dr8_layer_call_fn_87798Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0zМtrace_1
ч
Нtrace_0
Оtrace_12Ќ
>__inference_dr8_layer_call_and_return_conditional_losses_87810
>__inference_dr8_layer_call_and_return_conditional_losses_87815Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0zОtrace_1
"
_generic_user_object
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
В
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
о
Фtrace_02П
"__inference_d9_layer_call_fn_87824
В
FullArgSpec
args

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
annotationsЊ *
 zФtrace_0
љ
Хtrace_02к
=__inference_d9_layer_call_and_return_conditional_losses_87837
В
FullArgSpec
args

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
annotationsЊ *
 zХtrace_0
:@(2	d9/kernel
:(2d9/bias
Ю
Цtrace_02Џ
__inference_loss_fn_0_87842
В
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
annotationsЊ *Ђ zЦtrace_0
Ю
Чtrace_02Џ
__inference_loss_fn_1_87847
В
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
annotationsЊ *Ђ zЧtrace_0
Ю
Шtrace_02Џ
__inference_loss_fn_2_87852
В
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
annotationsЊ *Ђ zШtrace_0
Ю
Щtrace_02Џ
__inference_loss_fn_3_87857
В
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
annotationsЊ *Ђ zЩtrace_0
Ю
Ъtrace_02Џ
__inference_loss_fn_4_87862
В
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
annotationsЊ *Ђ zЪtrace_0
Ю
Ыtrace_02Џ
__inference_loss_fn_5_87867
В
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
annotationsЊ *Ђ zЫtrace_0
Ю
Ьtrace_02Џ
__inference_loss_fn_6_87872
В
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
annotationsЊ *Ђ zЬtrace_0
Ю
Эtrace_02Џ
__inference_loss_fn_7_87877
В
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
annotationsЊ *Ђ zЭtrace_0
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
Ю0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѕBђ
,__inference_sequential_9_layer_call_fn_87219c0_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
,__inference_sequential_9_layer_call_fn_87278c0_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
,__inference_sequential_9_layer_call_fn_87461inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
,__inference_sequential_9_layer_call_fn_87482inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_9_layer_call_and_return_conditional_losses_87106c0_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_9_layer_call_and_return_conditional_losses_87159c0_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_9_layer_call_and_return_conditional_losses_87550inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_9_layer_call_and_return_conditional_losses_87597inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ў
s0
а1
б2
в3
г4
д5
е6
ж7
з8
и9
й10
к11
л12
м13
н14
о15
п16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
а0
в1
д2
ж3
и4
к5
м6
о7"
trackable_list_wrapper
`
б0
г1
е2
з3
й4
л5
н6
п7"
trackable_list_wrapper
Е
рtrace_0
сtrace_1
тtrace_2
уtrace_3
фtrace_4
хtrace_5
цtrace_6
чtrace_72в
"__inference__update_step_xla_87602
"__inference__update_step_xla_87607
"__inference__update_step_xla_87612
"__inference__update_step_xla_87617
"__inference__update_step_xla_87622
"__inference__update_step_xla_87627
"__inference__update_step_xla_87632
"__inference__update_step_xla_87637Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zрtrace_0zсtrace_1zтtrace_2zуtrace_3zфtrace_4zхtrace_5zцtrace_6zчtrace_7
ЫBШ
#__inference_signature_wrapper_87432c0_input"
В
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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЬBЩ
"__inference_c0_layer_call_fn_87646inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
чBф
=__inference_c0_layer_call_and_return_conditional_losses_87659inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЬBЩ
"__inference_m1_layer_call_fn_87664inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
чBф
=__inference_m1_layer_call_and_return_conditional_losses_87669inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
оBл
#__inference_dr2_layer_call_fn_87674inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
оBл
#__inference_dr2_layer_call_fn_87679inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
>__inference_dr2_layer_call_and_return_conditional_losses_87691inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
>__inference_dr2_layer_call_and_return_conditional_losses_87696inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЬBЩ
"__inference_c3_layer_call_fn_87705inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
чBф
=__inference_c3_layer_call_and_return_conditional_losses_87718inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЬBЩ
"__inference_m4_layer_call_fn_87723inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
чBф
=__inference_m4_layer_call_and_return_conditional_losses_87728inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
оBл
#__inference_dr5_layer_call_fn_87733inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
оBл
#__inference_dr5_layer_call_fn_87738inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
>__inference_dr5_layer_call_and_return_conditional_losses_87750inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
>__inference_dr5_layer_call_and_return_conditional_losses_87755inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
ЬBЩ
"__inference_f6_layer_call_fn_87760inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
чBф
=__inference_f6_layer_call_and_return_conditional_losses_87766inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЬBЩ
"__inference_d7_layer_call_fn_87775inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
чBф
=__inference_d7_layer_call_and_return_conditional_losses_87788inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
оBл
#__inference_dr8_layer_call_fn_87793inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
оBл
#__inference_dr8_layer_call_fn_87798inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
>__inference_dr8_layer_call_and_return_conditional_losses_87810inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
>__inference_dr8_layer_call_and_return_conditional_losses_87815inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЬBЩ
"__inference_d9_layer_call_fn_87824inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
чBф
=__inference_d9_layer_call_and_return_conditional_losses_87837inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ВBЏ
__inference_loss_fn_0_87842"
В
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
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_1_87847"
В
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
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_2_87852"
В
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
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_3_87857"
В
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
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_4_87862"
В
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
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_5_87867"
В
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
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_6_87872"
В
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
annotationsЊ *Ђ 
ВBЏ
__inference_loss_fn_7_87877"
В
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
annotationsЊ *Ђ 
R
ш	variables
щ	keras_api

ъtotal

ыcount"
_tf_keras_metric
c
ь	variables
э	keras_api

юtotal

яcount
№
_fn_kwargs"
_tf_keras_metric
(:&2Adam/m/c0/kernel
(:&2Adam/v/c0/kernel
:2Adam/m/c0/bias
:2Adam/v/c0/bias
(:&2Adam/m/c3/kernel
(:&2Adam/v/c3/kernel
:2Adam/m/c3/bias
:2Adam/v/c3/bias
!:	 @2Adam/m/d7/kernel
!:	 @2Adam/v/d7/kernel
:@2Adam/m/d7/bias
:@2Adam/v/d7/bias
 :@(2Adam/m/d9/kernel
 :@(2Adam/v/d9/kernel
:(2Adam/m/d9/bias
:(2Adam/v/d9/bias
эBъ
"__inference__update_step_xla_87602gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_87607gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_87612gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_87617gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_87622gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_87627gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_87632gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_87637gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
ъ0
ы1"
trackable_list_wrapper
.
ш	variables"
_generic_user_object
:  (2total
:  (2count
0
ю0
я1"
trackable_list_wrapper
.
ь	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЄ
"__inference__update_step_xla_87602~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`ройВћ?
Њ "
 
"__inference__update_step_xla_87607f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рёПЂВћ?
Њ "
 Є
"__inference__update_step_xla_87612~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`рйВћ?
Њ "
 
"__inference__update_step_xla_87617f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рйВћ?
Њ "
 
"__inference__update_step_xla_87622pjЂg
`Ђ]

gradient	 @
52	Ђ
њ	 @

p
` VariableSpec 
`рЧЙћ?
Њ "
 
"__inference__update_step_xla_87627f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рэРЙћ?
Њ "
 
"__inference__update_step_xla_87632nhЂe
^Ђ[

gradient@(
41	Ђ
њ@(

p
` VariableSpec 
`рєЙћ?
Њ "
 
"__inference__update_step_xla_87637f`Ђ]
VЂS

gradient(
0-	Ђ
њ(

p
` VariableSpec 
`рЦєЙћ?
Њ "
 
 __inference__wrapped_model_86941n01LM[\9Ђ6
/Ђ,
*'
c0_inputџџџџџџџџџ@@
Њ "'Њ$
"
d9
d9џџџџџџџџџ(Д
=__inference_c0_layer_call_and_return_conditional_losses_87659s7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@@
 
"__inference_c0_layer_call_fn_87646h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ ")&
unknownџџџџџџџџџ@@Д
=__inference_c3_layer_call_and_return_conditional_losses_87718s017Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ  
 
"__inference_c3_layer_call_fn_87705h017Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  
Њ ")&
unknownџџџџџџџџџ  Ѕ
=__inference_d7_layer_call_and_return_conditional_losses_87788dLM0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
"__inference_d7_layer_call_fn_87775YLM0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџ@Є
=__inference_d9_layer_call_and_return_conditional_losses_87837c[\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ(
 ~
"__inference_d9_layer_call_fn_87824X[\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ(Е
>__inference_dr2_layer_call_and_return_conditional_losses_87691s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ  
 Е
>__inference_dr2_layer_call_and_return_conditional_losses_87696s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ  
 
#__inference_dr2_layer_call_fn_87674h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  
p
Њ ")&
unknownџџџџџџџџџ  
#__inference_dr2_layer_call_fn_87679h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  
p 
Њ ")&
unknownџџџџџџџџџ  Е
>__inference_dr5_layer_call_and_return_conditional_losses_87750s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 Е
>__inference_dr5_layer_call_and_return_conditional_losses_87755s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
#__inference_dr5_layer_call_fn_87733h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ ")&
unknownџџџџџџџџџ
#__inference_dr5_layer_call_fn_87738h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ ")&
unknownџџџџџџџџџЅ
>__inference_dr8_layer_call_and_return_conditional_losses_87810c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 Ѕ
>__inference_dr8_layer_call_and_return_conditional_losses_87815c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
#__inference_dr8_layer_call_fn_87793X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "!
unknownџџџџџџџџџ@
#__inference_dr8_layer_call_fn_87798X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "!
unknownџџџџџџџџџ@Љ
=__inference_f6_layer_call_and_return_conditional_losses_87766h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
"__inference_f6_layer_call_fn_87760]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџ @
__inference_loss_fn_0_87842!Ђ

Ђ 
Њ "
unknown @
__inference_loss_fn_1_87847!Ђ

Ђ 
Њ "
unknown @
__inference_loss_fn_2_87852!Ђ

Ђ 
Њ "
unknown @
__inference_loss_fn_3_87857!Ђ

Ђ 
Њ "
unknown @
__inference_loss_fn_4_87862!Ђ

Ђ 
Њ "
unknown @
__inference_loss_fn_5_87867!Ђ

Ђ 
Њ "
unknown @
__inference_loss_fn_6_87872!Ђ

Ђ 
Њ "
unknown @
__inference_loss_fn_7_87877!Ђ

Ђ 
Њ "
unknown ч
=__inference_m1_layer_call_and_return_conditional_losses_87669ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 С
"__inference_m1_layer_call_fn_87664RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџч
=__inference_m4_layer_call_and_return_conditional_losses_87728ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 С
"__inference_m4_layer_call_fn_87723RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЦ
G__inference_sequential_9_layer_call_and_return_conditional_losses_87106{01LM[\AЂ>
7Ђ4
*'
c0_inputџџџџџџџџџ@@
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ(
 Ц
G__inference_sequential_9_layer_call_and_return_conditional_losses_87159{01LM[\AЂ>
7Ђ4
*'
c0_inputџџџџџџџџџ@@
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ(
 Ф
G__inference_sequential_9_layer_call_and_return_conditional_losses_87550y01LM[\?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ(
 Ф
G__inference_sequential_9_layer_call_and_return_conditional_losses_87597y01LM[\?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ(
  
,__inference_sequential_9_layer_call_fn_87219p01LM[\AЂ>
7Ђ4
*'
c0_inputџџџџџџџџџ@@
p

 
Њ "!
unknownџџџџџџџџџ( 
,__inference_sequential_9_layer_call_fn_87278p01LM[\AЂ>
7Ђ4
*'
c0_inputџџџџџџџџџ@@
p 

 
Њ "!
unknownџџџџџџџџџ(
,__inference_sequential_9_layer_call_fn_87461n01LM[\?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p

 
Њ "!
unknownџџџџџџџџџ(
,__inference_sequential_9_layer_call_fn_87482n01LM[\?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p 

 
Њ "!
unknownџџџџџџџџџ(Ё
#__inference_signature_wrapper_87432z01LM[\EЂB
Ђ 
;Њ8
6
c0_input*'
c0_inputџџџџџџџџџ@@"'Њ$
"
d9
d9џџџџџџџџџ(
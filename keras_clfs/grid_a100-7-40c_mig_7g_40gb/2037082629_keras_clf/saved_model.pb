╛Ы
▌м
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
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
$
DisableCopyOnRead
resourceИ
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
 И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758Ъ╠	
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
shape:
*
shared_nameAdam/v/d9/bias
m
"Adam/v/d9/bias/Read/ReadVariableOpReadVariableOpAdam/v/d9/bias*
_output_shapes
:
*
dtype0
t
Adam/m/d9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameAdam/m/d9/bias
m
"Adam/m/d9/bias/Read/ReadVariableOpReadVariableOpAdam/m/d9/bias*
_output_shapes
:
*
dtype0
}
Adam/v/d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*!
shared_nameAdam/v/d9/kernel
v
$Adam/v/d9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d9/kernel*
_output_shapes
:	А
*
dtype0
}
Adam/m/d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*!
shared_nameAdam/m/d9/kernel
v
$Adam/m/d9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d9/kernel*
_output_shapes
:	А
*
dtype0
u
Adam/v/d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/v/d7/bias
n
"Adam/v/d7/bias/Read/ReadVariableOpReadVariableOpAdam/v/d7/bias*
_output_shapes	
:А*
dtype0
u
Adam/m/d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/m/d7/bias
n
"Adam/m/d7/bias/Read/ReadVariableOpReadVariableOpAdam/m/d7/bias*
_output_shapes	
:А*
dtype0
~
Adam/v/d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А1А*!
shared_nameAdam/v/d7/kernel
w
$Adam/v/d7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d7/kernel* 
_output_shapes
:
А1А*
dtype0
~
Adam/m/d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А1А*!
shared_nameAdam/m/d7/kernel
w
$Adam/m/d7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d7/kernel* 
_output_shapes
:
А1А*
dtype0
u
Adam/v/c3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/v/c3/bias
n
"Adam/v/c3/bias/Read/ReadVariableOpReadVariableOpAdam/v/c3/bias*
_output_shapes	
:А*
dtype0
u
Adam/m/c3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameAdam/m/c3/bias
n
"Adam/m/c3/bias/Read/ReadVariableOpReadVariableOpAdam/m/c3/bias*
_output_shapes	
:А*
dtype0
Е
Adam/v/c3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*!
shared_nameAdam/v/c3/kernel
~
$Adam/v/c3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c3/kernel*'
_output_shapes
:@А*
dtype0
Е
Adam/m/c3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*!
shared_nameAdam/m/c3/kernel
~
$Adam/m/c3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c3/kernel*'
_output_shapes
:@А*
dtype0
t
Adam/v/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/c0/bias
m
"Adam/v/c0/bias/Read/ReadVariableOpReadVariableOpAdam/v/c0/bias*
_output_shapes
:@*
dtype0
t
Adam/m/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/c0/bias
m
"Adam/m/c0/bias/Read/ReadVariableOpReadVariableOpAdam/m/c0/bias*
_output_shapes
:@*
dtype0
Д
Adam/v/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/v/c0/kernel
}
$Adam/v/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c0/kernel*&
_output_shapes
:@*
dtype0
Д
Adam/m/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/m/c0/kernel
}
$Adam/m/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c0/kernel*&
_output_shapes
:@*
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
shape:
*
shared_name	d9/bias
_
d9/bias/Read/ReadVariableOpReadVariableOpd9/bias*
_output_shapes
:
*
dtype0
o
	d9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*
shared_name	d9/kernel
h
d9/kernel/Read/ReadVariableOpReadVariableOp	d9/kernel*
_output_shapes
:	А
*
dtype0
g
d7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	d7/bias
`
d7/bias/Read/ReadVariableOpReadVariableOpd7/bias*
_output_shapes	
:А*
dtype0
p
	d7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А1А*
shared_name	d7/kernel
i
d7/kernel/Read/ReadVariableOpReadVariableOp	d7/kernel* 
_output_shapes
:
А1А*
dtype0
g
c3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	c3/bias
`
c3/bias/Read/ReadVariableOpReadVariableOpc3/bias*
_output_shapes	
:А*
dtype0
w
	c3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*
shared_name	c3/kernel
p
c3/kernel/Read/ReadVariableOpReadVariableOp	c3/kernel*'
_output_shapes
:@А*
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
Л
serving_default_c0_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_c0_input	c0/kernelc0/bias	c3/kernelc3/bias	d7/kerneld7/bias	d9/kerneld9/bias*
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
#__inference_signature_wrapper_20602

NoOpNoOp
БS
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╝R
value▓RBпR BиR
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
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
О
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 
е
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator* 
╚
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias
 2_jit_compiled_convolution_op*
О
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
е
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator* 
О
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
ж
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias*
е
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator* 
ж
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
░
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
Б
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
У
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
Аtrace_0* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

Жtrace_0* 

Зtrace_0* 
* 
* 
* 
Ц
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

Нtrace_0
Оtrace_1* 

Пtrace_0
Рtrace_1* 
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
Ш
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Цtrace_0* 

Чtrace_0* 
YS
VARIABLE_VALUE	c3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

Эtrace_0* 

Юtrace_0* 
* 
* 
* 
Ц
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

дtrace_0
еtrace_1* 

жtrace_0
зtrace_1* 
* 
* 
* 
* 
Ц
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 

L0
M1*

L0
M1*

a0
b1* 
Ш
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

┤trace_0* 

╡trace_0* 
YS
VARIABLE_VALUE	d7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

╗trace_0
╝trace_1* 

╜trace_0
╛trace_1* 
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
Ш
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

─trace_0* 

┼trace_0* 
YS
VARIABLE_VALUE	d9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

╞trace_0* 

╟trace_0* 

╚trace_0* 

╔trace_0* 

╩trace_0* 

╦trace_0* 

╠trace_0* 

═trace_0* 
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
╬0
╧1*
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
Т
s0
╨1
╤2
╥3
╙4
╘5
╒6
╓7
╫8
╪9
┘10
┌11
█12
▄13
▌14
▐15
▀16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
╨0
╥1
╘2
╓3
╪4
┌5
▄6
▐7*
D
╤0
╙1
╒2
╫3
┘4
█5
▌6
▀7*
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
Ё
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
┘
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
GPU(2*0J 8В *'
f"R 
__inference__traced_save_21250
╘
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
GPU(2*0J 8В **
f%R#
!__inference__traced_restore_21350┘│
Т+
╛
E__inference_sequential_layer_call_and_return_conditional_losses_20429

inputs"
c0_20394:@
c0_20396:@#
c3_20401:@А
c3_20403:	А
d7_20409:
А1А
d7_20411:	А
d9_20415:	А

d9_20417:

identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_20394c0_20396*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_20152╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_20117╤
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_20289√
c3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0c3_20401c3_20403*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_20186╪
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_20129╥
dr5/PartitionedCallPartitionedCallm4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_20301╔
f6/PartitionedCallPartitionedCalldr5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_20213Є
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_20409d7_20411*
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
=__inference_d7_layer_call_and_return_conditional_losses_20228╥
dr8/PartitionedCallPartitionedCall#d7/StatefulPartitionedCall:output:0*
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
>__inference_dr8_layer_call_and_return_conditional_losses_20313Є
d9/StatefulPartitionedCallStatefulPartitionedCalldr8/PartitionedCall:output:0d9_20415d9_20417*
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
=__inference_d9_layer_call_and_return_conditional_losses_20261`
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
:         
║
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╒
\
>__inference_dr8_layer_call_and_return_conditional_losses_20985

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
З┘
в
__inference__traced_save_21250
file_prefix:
 read_disablecopyonread_c0_kernel:@.
 read_1_disablecopyonread_c0_bias:@=
"read_2_disablecopyonread_c3_kernel:@А/
 read_3_disablecopyonread_c3_bias:	А6
"read_4_disablecopyonread_d7_kernel:
А1А/
 read_5_disablecopyonread_d7_bias:	А5
"read_6_disablecopyonread_d9_kernel:	А
.
 read_7_disablecopyonread_d9_bias:
,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: D
*read_10_disablecopyonread_adam_m_c0_kernel:@D
*read_11_disablecopyonread_adam_v_c0_kernel:@6
(read_12_disablecopyonread_adam_m_c0_bias:@6
(read_13_disablecopyonread_adam_v_c0_bias:@E
*read_14_disablecopyonread_adam_m_c3_kernel:@АE
*read_15_disablecopyonread_adam_v_c3_kernel:@А7
(read_16_disablecopyonread_adam_m_c3_bias:	А7
(read_17_disablecopyonread_adam_v_c3_bias:	А>
*read_18_disablecopyonread_adam_m_d7_kernel:
А1А>
*read_19_disablecopyonread_adam_v_d7_kernel:
А1А7
(read_20_disablecopyonread_adam_m_d7_bias:	А7
(read_21_disablecopyonread_adam_v_d7_bias:	А=
*read_22_disablecopyonread_adam_m_d9_kernel:	А
=
*read_23_disablecopyonread_adam_v_d9_kernel:	А
6
(read_24_disablecopyonread_adam_m_d9_bias:
6
(read_25_disablecopyonread_adam_v_d9_bias:
+
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
: r
Read/DisableCopyOnReadDisableCopyOnRead read_disablecopyonread_c0_kernel"/device:CPU:0*
_output_shapes
 д
Read/ReadVariableOpReadVariableOp read_disablecopyonread_c0_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@t
Read_1/DisableCopyOnReadDisableCopyOnRead read_1_disablecopyonread_c0_bias"/device:CPU:0*
_output_shapes
 Ь
Read_1/ReadVariableOpReadVariableOp read_1_disablecopyonread_c0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_c3_kernel"/device:CPU:0*
_output_shapes
 л
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_c3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0v

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аl

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*'
_output_shapes
:@Аt
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_c3_bias"/device:CPU:0*
_output_shapes
 Э
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_c3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аv
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_d7_kernel"/device:CPU:0*
_output_shapes
 д
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_d7_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
А1А*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
А1Аe

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
А1Аt
Read_5/DisableCopyOnReadDisableCopyOnRead read_5_disablecopyonread_d7_bias"/device:CPU:0*
_output_shapes
 Э
Read_5/ReadVariableOpReadVariableOp read_5_disablecopyonread_d7_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аv
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_d9_kernel"/device:CPU:0*
_output_shapes
 г
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_d9_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А
*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А
f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	А
t
Read_7/DisableCopyOnReadDisableCopyOnRead read_7_disablecopyonread_d9_bias"/device:CPU:0*
_output_shapes
 Ь
Read_7/ReadVariableOpReadVariableOp read_7_disablecopyonread_d9_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Ъ
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
 Ю
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
 ┤
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_adam_m_c0_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:@
Read_11/DisableCopyOnReadDisableCopyOnRead*read_11_disablecopyonread_adam_v_c0_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_11/ReadVariableOpReadVariableOp*read_11_disablecopyonread_adam_v_c0_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
:@}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_adam_m_c0_bias"/device:CPU:0*
_output_shapes
 ж
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_adam_m_c0_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_adam_v_c0_bias"/device:CPU:0*
_output_shapes
 ж
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_adam_v_c0_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_adam_m_c3_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_adam_m_c3_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0x
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*'
_output_shapes
:@А
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_adam_v_c3_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_adam_v_c3_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0x
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*'
_output_shapes
:@А}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_adam_m_c3_bias"/device:CPU:0*
_output_shapes
 з
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_adam_m_c3_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_17/DisableCopyOnReadDisableCopyOnRead(read_17_disablecopyonread_adam_v_c3_bias"/device:CPU:0*
_output_shapes
 з
Read_17/ReadVariableOpReadVariableOp(read_17_disablecopyonread_adam_v_c3_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:А
Read_18/DisableCopyOnReadDisableCopyOnRead*read_18_disablecopyonread_adam_m_d7_kernel"/device:CPU:0*
_output_shapes
 о
Read_18/ReadVariableOpReadVariableOp*read_18_disablecopyonread_adam_m_d7_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
А1А*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
А1Аg
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
А1А
Read_19/DisableCopyOnReadDisableCopyOnRead*read_19_disablecopyonread_adam_v_d7_kernel"/device:CPU:0*
_output_shapes
 о
Read_19/ReadVariableOpReadVariableOp*read_19_disablecopyonread_adam_v_d7_kernel^Read_19/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
А1А*
dtype0q
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
А1Аg
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
А1А}
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_adam_m_d7_bias"/device:CPU:0*
_output_shapes
 з
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_adam_m_d7_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_21/DisableCopyOnReadDisableCopyOnRead(read_21_disablecopyonread_adam_v_d7_bias"/device:CPU:0*
_output_shapes
 з
Read_21/ReadVariableOpReadVariableOp(read_21_disablecopyonread_adam_v_d7_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:А
Read_22/DisableCopyOnReadDisableCopyOnRead*read_22_disablecopyonread_adam_m_d9_kernel"/device:CPU:0*
_output_shapes
 н
Read_22/ReadVariableOpReadVariableOp*read_22_disablecopyonread_adam_m_d9_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А
*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А
f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	А

Read_23/DisableCopyOnReadDisableCopyOnRead*read_23_disablecopyonread_adam_v_d9_kernel"/device:CPU:0*
_output_shapes
 н
Read_23/ReadVariableOpReadVariableOp*read_23_disablecopyonread_adam_v_d9_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А
*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А
f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	А
}
Read_24/DisableCopyOnReadDisableCopyOnRead(read_24_disablecopyonread_adam_m_d9_bias"/device:CPU:0*
_output_shapes
 ж
Read_24/ReadVariableOpReadVariableOp(read_24_disablecopyonread_adam_m_d9_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:
}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_adam_v_d9_bias"/device:CPU:0*
_output_shapes
 ж
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_adam_v_d9_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
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
 Ы
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
 Щ
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
 Щ
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
: ╕
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
value╫B╘B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHл
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Е
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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
: ∙
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
─
+
__inference_loss_fn_1_21017
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
╧
V
"__inference__update_step_xla_20772
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:@
"
_user_specified_name
gradient
О
\
#__inference_dr5_layer_call_fn_20903

inputs
identityИвStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_20205x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╞
°
=__inference_c3_layer_call_and_return_conditional_losses_20186

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А`
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
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Д/
Ъ
E__inference_sequential_layer_call_and_return_conditional_losses_20276
c0_input"
c0_20153:@
c0_20155:@#
c3_20187:@А
c3_20189:	А
d7_20229:
А1А
d7_20231:	А
d9_20262:	А

d9_20264:

identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallвdr8/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_20153c0_20155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_20152╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_20117с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_20171Г
c3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0c3_20187c3_20189*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_20186╪
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_20129А
dr5/StatefulPartitionedCallStatefulPartitionedCallm4/PartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_20205╤
f6/PartitionedCallPartitionedCall$dr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_20213Є
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_20229d7_20231*
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
=__inference_d7_layer_call_and_return_conditional_losses_20228А
dr8/StatefulPartitionedCallStatefulPartitionedCall#d7/StatefulPartitionedCall:output:0^dr5/StatefulPartitionedCall*
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
>__inference_dr8_layer_call_and_return_conditional_losses_20246·
d9/StatefulPartitionedCallStatefulPartitionedCall$dr8/StatefulPartitionedCall:output:0d9_20262d9_20264*
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
=__inference_d9_layer_call_and_return_conditional_losses_20261`
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
:         
Ф
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall^dr8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall2:
dr8/StatefulPartitionedCalldr8/StatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
c0_input
т

]
>__inference_dr5_layer_call_and_return_conditional_losses_20920

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%IТ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧ж
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ь
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
г
>
"__inference_m1_layer_call_fn_20834

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
=__inference_m1_layer_call_and_return_conditional_losses_20117Г
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
╚
+
__inference_loss_fn_4_21032
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
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_20289

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ї
\
>__inference_dr5_layer_call_and_return_conditional_losses_20925

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         Аd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
э	
╧
*__inference_sequential_layer_call_fn_20652

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А
	unknown_3:
А1А
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
E__inference_sequential_layer_call_and_return_conditional_losses_20429o
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
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_0_21012
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
╝
?
#__inference_dr5_layer_call_fn_20908

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_20301i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
к
>
"__inference_f6_layer_call_fn_20930

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
:         А1* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_20213a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
║
O
"__inference__update_step_xla_20802
gradient
variable:	А
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	А
: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	А

"
_user_specified_name
gradient
▓.
Є
E__inference_sequential_layer_call_and_return_conditional_losses_20767

inputs;
!c0_conv2d_readvariableop_resource:@0
"c0_biasadd_readvariableop_resource:@<
!c3_conv2d_readvariableop_resource:@А1
"c3_biasadd_readvariableop_resource:	А5
!d7_matmul_readvariableop_resource:
А1А1
"d7_biasadd_readvariableop_resource:	А4
!d9_matmul_readvariableop_resource:	А
0
"d9_biasadd_readvariableop_resource:

identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc3/BiasAdd/ReadVariableOpвc3/Conv2D/ReadVariableOpвd7/BiasAdd/ReadVariableOpвd7/MatMul/ReadVariableOpвd9/BiasAdd/ReadVariableOpвd9/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:         @Щ

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
dr2/IdentityIdentitym1/MaxPool:output:0*
T0*/
_output_shapes
:         @Г
c3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0п
	c3/Conv2DConv2Ddr2/Identity:output:0 c3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
y
c3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0З

c3/BiasAddBiasAddc3/Conv2D:output:0!c3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А_
c3/ReluReluc3/BiasAdd:output:0*
T0*0
_output_shapes
:         АЪ

m4/MaxPoolMaxPoolc3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
h
dr5/IdentityIdentitym4/MaxPool:output:0*
T0*0
_output_shapes
:         АY
f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  r

f6/ReshapeReshapedr5/Identity:output:0f6/Const:output:0*
T0*(
_output_shapes
:         А1|
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource* 
_output_shapes
:
А1А*
dtype0}
	d7/MatMulMatMulf6/Reshape:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0А

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АW
d7/ReluRelud7/BiasAdd:output:0*
T0*(
_output_shapes
:         Аb
dr8/IdentityIdentityd7/Relu:activations:0*
T0*(
_output_shapes
:         А{
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0~
	d9/MatMulMatMuldr8/Identity:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
x
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
\

d9/SoftmaxSoftmaxd9/BiasAdd:output:0*
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
:         
в
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c3/BiasAdd/ReadVariableOp^c3/Conv2D/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 26
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
:         
 
_user_specified_nameinputs
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_20866

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╚
+
__inference_loss_fn_6_21042
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
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_20829

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @`
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
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
т
ё
=__inference_d7_layer_call_and_return_conditional_losses_20228

inputs2
matmul_readvariableop_resource:
А1А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А1А*
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
:         А1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
є	
╤
*__inference_sequential_layer_call_fn_20448
c0_input!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А
	unknown_3:
А1А
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
E__inference_sequential_layer_call_and_return_conditional_losses_20429o
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
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
c0_input
■.
Ш
E__inference_sequential_layer_call_and_return_conditional_losses_20370

inputs"
c0_20335:@
c0_20337:@#
c3_20342:@А
c3_20344:	А
d7_20350:
А1А
d7_20352:	А
d9_20356:	А

d9_20358:

identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallвdr8/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_20335c0_20337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_20152╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_20117с
dr2/StatefulPartitionedCallStatefulPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_20171Г
c3/StatefulPartitionedCallStatefulPartitionedCall$dr2/StatefulPartitionedCall:output:0c3_20342c3_20344*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_20186╪
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_20129А
dr5/StatefulPartitionedCallStatefulPartitionedCallm4/PartitionedCall:output:0^dr2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_20205╤
f6/PartitionedCallPartitionedCall$dr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_20213Є
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_20350d7_20352*
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
=__inference_d7_layer_call_and_return_conditional_losses_20228А
dr8/StatefulPartitionedCallStatefulPartitionedCall#d7/StatefulPartitionedCall:output:0^dr5/StatefulPartitionedCall*
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
>__inference_dr8_layer_call_and_return_conditional_losses_20246·
d9/StatefulPartitionedCallStatefulPartitionedCall$dr8/StatefulPartitionedCall:output:0d9_20356d9_20358*
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
=__inference_d9_layer_call_and_return_conditional_losses_20261`
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
:         
Ф
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall^dr2/StatefulPartitionedCall^dr5/StatefulPartitionedCall^dr8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall2:
dr2/StatefulPartitionedCalldr2/StatefulPartitionedCall2:
dr5/StatefulPartitionedCalldr5/StatefulPartitionedCall2:
dr8/StatefulPartitionedCalldr8/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┴
Y
=__inference_f6_layer_call_and_return_conditional_losses_20936

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А1Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ю
\
#__inference_dr8_layer_call_fn_20963

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
>__inference_dr8_layer_call_and_return_conditional_losses_20246p
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
__inference_loss_fn_5_21037
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
т

]
>__inference_dr5_layer_call_and_return_conditional_losses_20205

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%IТ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧ж
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ь
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Е
Y
=__inference_m4_layer_call_and_return_conditional_losses_20898

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
э	
╧
*__inference_sequential_layer_call_fn_20631

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А
	unknown_3:
А1А
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
E__inference_sequential_layer_call_and_return_conditional_losses_20370o
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
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╜
P
"__inference__update_step_xla_20792
gradient
variable:
А1А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
А1А: *
	_noinline(:($
"
_user_specified_name
variable:J F
 
_output_shapes
:
А1А
"
_user_specified_name
gradient
Ь
?
#__inference_dr8_layer_call_fn_20968

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
>__inference_dr8_layer_call_and_return_conditional_losses_20313a
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
К
\
#__inference_dr2_layer_call_fn_20844

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
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_20171w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┐}
┤
!__inference__traced_restore_21350
file_prefix4
assignvariableop_c0_kernel:@(
assignvariableop_1_c0_bias:@7
assignvariableop_2_c3_kernel:@А)
assignvariableop_3_c3_bias:	А0
assignvariableop_4_d7_kernel:
А1А)
assignvariableop_5_d7_bias:	А/
assignvariableop_6_d9_kernel:	А
(
assignvariableop_7_d9_bias:
&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: >
$assignvariableop_10_adam_m_c0_kernel:@>
$assignvariableop_11_adam_v_c0_kernel:@0
"assignvariableop_12_adam_m_c0_bias:@0
"assignvariableop_13_adam_v_c0_bias:@?
$assignvariableop_14_adam_m_c3_kernel:@А?
$assignvariableop_15_adam_v_c3_kernel:@А1
"assignvariableop_16_adam_m_c3_bias:	А1
"assignvariableop_17_adam_v_c3_bias:	А8
$assignvariableop_18_adam_m_d7_kernel:
А1А8
$assignvariableop_19_adam_v_d7_kernel:
А1А1
"assignvariableop_20_adam_m_d7_bias:	А1
"assignvariableop_21_adam_v_d7_bias:	А7
$assignvariableop_22_adam_m_d9_kernel:	А
7
$assignvariableop_23_adam_v_d9_kernel:	А
0
"assignvariableop_24_adam_m_d9_bias:
0
"assignvariableop_25_adam_v_d9_bias:
%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╗
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
value╫B╘B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHо
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ║
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Р
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOpAssignVariableOpassignvariableop_c0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_1AssignVariableOpassignvariableop_1_c0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_2AssignVariableOpassignvariableop_2_c3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_3AssignVariableOpassignvariableop_3_c3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_4AssignVariableOpassignvariableop_4_d7_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_5AssignVariableOpassignvariableop_5_d7_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_6AssignVariableOpassignvariableop_6_d9_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_7AssignVariableOpassignvariableop_7_d9_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_10AssignVariableOp$assignvariableop_10_adam_m_c0_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_11AssignVariableOp$assignvariableop_11_adam_v_c0_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_12AssignVariableOp"assignvariableop_12_adam_m_c0_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_13AssignVariableOp"assignvariableop_13_adam_v_c0_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_14AssignVariableOp$assignvariableop_14_adam_m_c3_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_v_c3_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_m_c3_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_17AssignVariableOp"assignvariableop_17_adam_v_c3_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_18AssignVariableOp$assignvariableop_18_adam_m_d7_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_v_d7_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_m_d7_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_21AssignVariableOp"assignvariableop_21_adam_v_d7_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_22AssignVariableOp$assignvariableop_22_adam_m_d9_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_v_d9_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_m_d9_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_25AssignVariableOp"assignvariableop_25_adam_v_d9_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:▓
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
: ╨
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
█

]
>__inference_dr2_layer_call_and_return_conditional_losses_20861

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
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
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
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ш+
└
E__inference_sequential_layer_call_and_return_conditional_losses_20329
c0_input"
c0_20279:@
c0_20281:@#
c3_20291:@А
c3_20293:	А
d7_20304:
А1А
d7_20306:	А
d9_20315:	А

d9_20317:

identityИвc0/StatefulPartitionedCallвc3/StatefulPartitionedCallвd7/StatefulPartitionedCallвd9/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_20279c0_20281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_20152╫
m1/PartitionedCallPartitionedCall#c0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m1_layer_call_and_return_conditional_losses_20117╤
dr2/PartitionedCallPartitionedCallm1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_20289√
c3/StatefulPartitionedCallStatefulPartitionedCalldr2/PartitionedCall:output:0c3_20291c3_20293*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_20186╪
m4/PartitionedCallPartitionedCall#c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_m4_layer_call_and_return_conditional_losses_20129╥
dr5/PartitionedCallPartitionedCallm4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr5_layer_call_and_return_conditional_losses_20301╔
f6/PartitionedCallPartitionedCalldr5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_f6_layer_call_and_return_conditional_losses_20213Є
d7/StatefulPartitionedCallStatefulPartitionedCallf6/PartitionedCall:output:0d7_20304d7_20306*
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
=__inference_d7_layer_call_and_return_conditional_losses_20228╥
dr8/PartitionedCallPartitionedCall#d7/StatefulPartitionedCall:output:0*
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
>__inference_dr8_layer_call_and_return_conditional_losses_20313Є
d9/StatefulPartitionedCallStatefulPartitionedCalldr8/PartitionedCall:output:0d9_20315d9_20317*
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
=__inference_d9_layer_call_and_return_conditional_losses_20261`
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
:         
║
NoOpNoOp^c0/StatefulPartitionedCall^c3/StatefulPartitionedCall^d7/StatefulPartitionedCall^d9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 28
c0/StatefulPartitionedCallc0/StatefulPartitionedCall28
c3/StatefulPartitionedCallc3/StatefulPartitionedCall28
d7/StatefulPartitionedCalld7/StatefulPartitionedCall28
d9/StatefulPartitionedCalld9/StatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
c0_input
г
>
"__inference_m4_layer_call_fn_20893

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
=__inference_m4_layer_call_and_return_conditional_losses_20129Г
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
╚
+
__inference_loss_fn_2_21022
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
╥
W
"__inference__update_step_xla_20782
gradient#
variable:@А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:@А: *
	_noinline(:($
"
_user_specified_name
variable:Q M
'
_output_shapes
:@А
"
_user_specified_name
gradient
є	
╤
*__inference_sequential_layer_call_fn_20389
c0_input!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А
	unknown_3:
А1А
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
E__inference_sequential_layer_call_and_return_conditional_losses_20370o
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
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
c0_input
─
+
__inference_loss_fn_3_21027
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
л
J
"__inference__update_step_xla_20807
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
о
K
"__inference__update_step_xla_20787
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
─
+
__inference_loss_fn_7_21047
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
▀
я
=__inference_d9_layer_call_and_return_conditional_losses_21007

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
т
ё
=__inference_d7_layer_call_and_return_conditional_losses_20958

inputs2
matmul_readvariableop_resource:
А1А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А1А*
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
:         А1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
к

]
>__inference_dr8_layer_call_and_return_conditional_losses_20980

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%IТ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Ю
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
 *   >з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
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
╒
\
>__inference_dr8_layer_call_and_return_conditional_losses_20313

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
┴
Y
=__inference_f6_layer_call_and_return_conditional_losses_20213

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А1Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╟	
╩
#__inference_signature_wrapper_20602
c0_input!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А
	unknown_3:
А1А
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
 __inference__wrapped_model_20111o
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
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
c0_input
▀
я
=__inference_d9_layer_call_and_return_conditional_losses_20261

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
=__inference_m4_layer_call_and_return_conditional_losses_20129

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
█

]
>__inference_dr2_layer_call_and_return_conditional_losses_20171

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
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧е
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
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
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╛
Р
"__inference_d9_layer_call_fn_20994

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
=__inference_d9_layer_call_and_return_conditional_losses_20261o
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
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_20839

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
л
J
"__inference__update_step_xla_20777
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
ч
Щ
"__inference_c3_layer_call_fn_20875

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c3_layer_call_and_return_conditional_losses_20186x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
к

]
>__inference_dr8_layer_call_and_return_conditional_losses_20246

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%IТ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Ю
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
 *   >з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
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
┬
Т
"__inference_d7_layer_call_fn_20945

inputs
unknown:
А1А
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
=__inference_d7_layer_call_and_return_conditional_losses_20228p
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
:         А1: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
о
K
"__inference__update_step_xla_20797
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_20117

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
у
Ч
"__inference_c0_layer_call_fn_20816

inputs!
unknown:@
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
:         @*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8В *F
fAR?
=__inference_c0_layer_call_and_return_conditional_losses_20152w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╞
°
=__inference_c3_layer_call_and_return_conditional_losses_20888

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А`
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
 *    j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ї
\
>__inference_dr5_layer_call_and_return_conditional_losses_20301

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         Аd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ў0
 
 __inference__wrapped_model_20111
c0_inputF
,sequential_c0_conv2d_readvariableop_resource:@;
-sequential_c0_biasadd_readvariableop_resource:@G
,sequential_c3_conv2d_readvariableop_resource:@А<
-sequential_c3_biasadd_readvariableop_resource:	А@
,sequential_d7_matmul_readvariableop_resource:
А1А<
-sequential_d7_biasadd_readvariableop_resource:	А?
,sequential_d9_matmul_readvariableop_resource:	А
;
-sequential_d9_biasadd_readvariableop_resource:

identityИв$sequential/c0/BiasAdd/ReadVariableOpв#sequential/c0/Conv2D/ReadVariableOpв$sequential/c3/BiasAdd/ReadVariableOpв#sequential/c3/Conv2D/ReadVariableOpв$sequential/d7/BiasAdd/ReadVariableOpв#sequential/d7/MatMul/ReadVariableOpв$sequential/d9/BiasAdd/ReadVariableOpв#sequential/d9/MatMul/ReadVariableOpШ
#sequential/c0/Conv2D/ReadVariableOpReadVariableOp,sequential_c0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╖
sequential/c0/Conv2DConv2Dc0_input+sequential/c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
О
$sequential/c0/BiasAdd/ReadVariableOpReadVariableOp-sequential_c0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0з
sequential/c0/BiasAddBiasAddsequential/c0/Conv2D:output:0,sequential/c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @t
sequential/c0/ReluRelusequential/c0/BiasAdd:output:0*
T0*/
_output_shapes
:         @п
sequential/m1/MaxPoolMaxPool sequential/c0/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
}
sequential/dr2/IdentityIdentitysequential/m1/MaxPool:output:0*
T0*/
_output_shapes
:         @Щ
#sequential/c3/Conv2D/ReadVariableOpReadVariableOp,sequential_c3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╨
sequential/c3/Conv2DConv2D sequential/dr2/Identity:output:0+sequential/c3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
П
$sequential/c3/BiasAdd/ReadVariableOpReadVariableOp-sequential_c3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0и
sequential/c3/BiasAddBiasAddsequential/c3/Conv2D:output:0,sequential/c3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аu
sequential/c3/ReluRelusequential/c3/BiasAdd:output:0*
T0*0
_output_shapes
:         А░
sequential/m4/MaxPoolMaxPool sequential/c3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
~
sequential/dr5/IdentityIdentitysequential/m4/MaxPool:output:0*
T0*0
_output_shapes
:         Аd
sequential/f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  У
sequential/f6/ReshapeReshape sequential/dr5/Identity:output:0sequential/f6/Const:output:0*
T0*(
_output_shapes
:         А1Т
#sequential/d7/MatMul/ReadVariableOpReadVariableOp,sequential_d7_matmul_readvariableop_resource* 
_output_shapes
:
А1А*
dtype0Ю
sequential/d7/MatMulMatMulsequential/f6/Reshape:output:0+sequential/d7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$sequential/d7/BiasAdd/ReadVariableOpReadVariableOp-sequential_d7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
sequential/d7/BiasAddBiasAddsequential/d7/MatMul:product:0,sequential/d7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
sequential/d7/ReluRelusequential/d7/BiasAdd:output:0*
T0*(
_output_shapes
:         Аx
sequential/dr8/IdentityIdentity sequential/d7/Relu:activations:0*
T0*(
_output_shapes
:         АС
#sequential/d9/MatMul/ReadVariableOpReadVariableOp,sequential_d9_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Я
sequential/d9/MatMulMatMul sequential/dr8/Identity:output:0+sequential/d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
О
$sequential/d9/BiasAdd/ReadVariableOpReadVariableOp-sequential_d9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0а
sequential/d9/BiasAddBiasAddsequential/d9/MatMul:product:0,sequential/d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
sequential/d9/SoftmaxSoftmaxsequential/d9/BiasAdd:output:0*
T0*'
_output_shapes
:         
n
IdentityIdentitysequential/d9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
·
NoOpNoOp%^sequential/c0/BiasAdd/ReadVariableOp$^sequential/c0/Conv2D/ReadVariableOp%^sequential/c3/BiasAdd/ReadVariableOp$^sequential/c3/Conv2D/ReadVariableOp%^sequential/d7/BiasAdd/ReadVariableOp$^sequential/d7/MatMul/ReadVariableOp%^sequential/d9/BiasAdd/ReadVariableOp$^sequential/d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2L
$sequential/c0/BiasAdd/ReadVariableOp$sequential/c0/BiasAdd/ReadVariableOp2J
#sequential/c0/Conv2D/ReadVariableOp#sequential/c0/Conv2D/ReadVariableOp2L
$sequential/c3/BiasAdd/ReadVariableOp$sequential/c3/BiasAdd/ReadVariableOp2J
#sequential/c3/Conv2D/ReadVariableOp#sequential/c3/Conv2D/ReadVariableOp2L
$sequential/d7/BiasAdd/ReadVariableOp$sequential/d7/BiasAdd/ReadVariableOp2J
#sequential/d7/MatMul/ReadVariableOp#sequential/d7/MatMul/ReadVariableOp2L
$sequential/d9/BiasAdd/ReadVariableOp$sequential/d9/BiasAdd/ReadVariableOp2J
#sequential/d9/MatMul/ReadVariableOp#sequential/d9/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         
"
_user_specified_name
c0_input
╕
?
#__inference_dr2_layer_call_fn_20849

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
:         @* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8В *G
fBR@
>__inference_dr2_layer_call_and_return_conditional_losses_20289h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╣D
Є
E__inference_sequential_layer_call_and_return_conditional_losses_20720

inputs;
!c0_conv2d_readvariableop_resource:@0
"c0_biasadd_readvariableop_resource:@<
!c3_conv2d_readvariableop_resource:@А1
"c3_biasadd_readvariableop_resource:	А5
!d7_matmul_readvariableop_resource:
А1А1
"d7_biasadd_readvariableop_resource:	А4
!d9_matmul_readvariableop_resource:	А
0
"d9_biasadd_readvariableop_resource:

identityИвc0/BiasAdd/ReadVariableOpвc0/Conv2D/ReadVariableOpвc3/BiasAdd/ReadVariableOpвc3/Conv2D/ReadVariableOpвd7/BiasAdd/ReadVariableOpвd7/MatMul/ReadVariableOpвd9/BiasAdd/ReadVariableOpвd9/MatMul/ReadVariableOpВ
c0/Conv2D/ReadVariableOpReadVariableOp!c0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Я
	c0/Conv2DConv2Dinputs c0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
x
c0/BiasAdd/ReadVariableOpReadVariableOp"c0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж

c0/BiasAddBiasAddc0/Conv2D:output:0!c0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @^
c0/ReluReluc0/BiasAdd:output:0*
T0*/
_output_shapes
:         @Щ

m1/MaxPoolMaxPoolc0/Relu:activations:0*/
_output_shapes
:         @*
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
 *%IТ?Б
dr2/dropout/MulMulm1/MaxPool:output:0dr2/dropout/Const:output:0*
T0*/
_output_shapes
:         @b
dr2/dropout/ShapeShapem1/MaxPool:output:0*
T0*
_output_shapes
::э╧н
(dr2/dropout/random_uniform/RandomUniformRandomUniformdr2/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seed2    _
dr2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >║
dr2/dropout/GreaterEqualGreaterEqual1dr2/dropout/random_uniform/RandomUniform:output:0#dr2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @X
dr2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    л
dr2/dropout/SelectV2SelectV2dr2/dropout/GreaterEqual:z:0dr2/dropout/Mul:z:0dr2/dropout/Const_1:output:0*
T0*/
_output_shapes
:         @Г
c3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╖
	c3/Conv2DConv2Ddr2/dropout/SelectV2:output:0 c3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
y
c3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0З

c3/BiasAddBiasAddc3/Conv2D:output:0!c3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А_
c3/ReluReluc3/BiasAdd:output:0*
T0*0
_output_shapes
:         АЪ

m4/MaxPoolMaxPoolc3/Relu:activations:0*0
_output_shapes
:         А*
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
 *%IТ?В
dr5/dropout/MulMulm4/MaxPool:output:0dr5/dropout/Const:output:0*
T0*0
_output_shapes
:         Аb
dr5/dropout/ShapeShapem4/MaxPool:output:0*
T0*
_output_shapes
::э╧к
(dr5/dropout/random_uniform/RandomUniformRandomUniformdr5/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype0*
seed2_
dr5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >╗
dr5/dropout/GreaterEqualGreaterEqual1dr5/dropout/random_uniform/RandomUniform:output:0#dr5/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         АX
dr5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    м
dr5/dropout/SelectV2SelectV2dr5/dropout/GreaterEqual:z:0dr5/dropout/Mul:z:0dr5/dropout/Const_1:output:0*
T0*0
_output_shapes
:         АY
f6/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  z

f6/ReshapeReshapedr5/dropout/SelectV2:output:0f6/Const:output:0*
T0*(
_output_shapes
:         А1|
d7/MatMul/ReadVariableOpReadVariableOp!d7_matmul_readvariableop_resource* 
_output_shapes
:
А1А*
dtype0}
	d7/MatMulMatMulf6/Reshape:output:0 d7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
d7/BiasAdd/ReadVariableOpReadVariableOp"d7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0А

d7/BiasAddBiasAddd7/MatMul:product:0!d7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АW
d7/ReluRelud7/BiasAdd:output:0*
T0*(
_output_shapes
:         АV
dr8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *%IТ?|
dr8/dropout/MulMuld7/Relu:activations:0dr8/dropout/Const:output:0*
T0*(
_output_shapes
:         Аd
dr8/dropout/ShapeShaped7/Relu:activations:0*
T0*
_output_shapes
::э╧в
(dr8/dropout/random_uniform/RandomUniformRandomUniformdr8/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0*
seed2_
dr8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >│
dr8/dropout/GreaterEqualGreaterEqual1dr8/dropout/random_uniform/RandomUniform:output:0#dr8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АX
dr8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    д
dr8/dropout/SelectV2SelectV2dr8/dropout/GreaterEqual:z:0dr8/dropout/Mul:z:0dr8/dropout/Const_1:output:0*
T0*(
_output_shapes
:         А{
d9/MatMul/ReadVariableOpReadVariableOp!d9_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ж
	d9/MatMulMatMuldr8/dropout/SelectV2:output:0 d9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
x
d9/BiasAdd/ReadVariableOpReadVariableOp"d9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0

d9/BiasAddBiasAddd9/MatMul:product:0!d9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
\

d9/SoftmaxSoftmaxd9/BiasAdd:output:0*
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
:         
в
NoOpNoOp^c0/BiasAdd/ReadVariableOp^c0/Conv2D/ReadVariableOp^c3/BiasAdd/ReadVariableOp^c3/Conv2D/ReadVariableOp^d7/BiasAdd/ReadVariableOp^d7/MatMul/ReadVariableOp^d9/BiasAdd/ReadVariableOp^d9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 26
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
:         
 
_user_specified_nameinputs
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_20152

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @`
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
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs"є
L
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
serving_default_c0_input:0         6
d90
StatefulPartitionedCall:0         
tensorflow/serving/predict:█╕
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
▌
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
е
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator"
_tf_keras_layer
▌
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
е
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
е
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
╝
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator"
_tf_keras_layer
╗
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
╩
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
╙
jtrace_0
ktrace_1
ltrace_2
mtrace_32ш
*__inference_sequential_layer_call_fn_20389
*__inference_sequential_layer_call_fn_20448
*__inference_sequential_layer_call_fn_20631
*__inference_sequential_layer_call_fn_20652╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zjtrace_0zktrace_1zltrace_2zmtrace_3
┐
ntrace_0
otrace_1
ptrace_2
qtrace_32╘
E__inference_sequential_layer_call_and_return_conditional_losses_20276
E__inference_sequential_layer_call_and_return_conditional_losses_20329
E__inference_sequential_layer_call_and_return_conditional_losses_20720
E__inference_sequential_layer_call_and_return_conditional_losses_20767╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zntrace_0zotrace_1zptrace_2zqtrace_3
╠B╔
 __inference__wrapped_model_20111c0_input"Ш
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
Ь
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
н
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
▄
trace_02┐
"__inference_c0_layer_call_fn_20816Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 ztrace_0
∙
Аtrace_02┌
=__inference_c0_layer_call_and_return_conditional_losses_20829Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zАtrace_0
#:!@2	c0/kernel
:@2c0/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
▐
Жtrace_02┐
"__inference_m1_layer_call_fn_20834Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЖtrace_0
∙
Зtrace_02┌
=__inference_m1_layer_call_and_return_conditional_losses_20839Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЗtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
▒
Нtrace_0
Оtrace_12Ў
#__inference_dr2_layer_call_fn_20844
#__inference_dr2_layer_call_fn_20849й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0zОtrace_1
ч
Пtrace_0
Рtrace_12м
>__inference_dr2_layer_call_and_return_conditional_losses_20861
>__inference_dr2_layer_call_and_return_conditional_losses_20866й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zПtrace_0zРtrace_1
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
▓
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
▐
Цtrace_02┐
"__inference_c3_layer_call_fn_20875Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЦtrace_0
∙
Чtrace_02┌
=__inference_c3_layer_call_and_return_conditional_losses_20888Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЧtrace_0
$:"@А2	c3/kernel
:А2c3/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
▐
Эtrace_02┐
"__inference_m4_layer_call_fn_20893Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЭtrace_0
∙
Юtrace_02┌
=__inference_m4_layer_call_and_return_conditional_losses_20898Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЮtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
▒
дtrace_0
еtrace_12Ў
#__inference_dr5_layer_call_fn_20903
#__inference_dr5_layer_call_fn_20908й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zдtrace_0zеtrace_1
ч
жtrace_0
зtrace_12м
>__inference_dr5_layer_call_and_return_conditional_losses_20920
>__inference_dr5_layer_call_and_return_conditional_losses_20925й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zжtrace_0zзtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
▐
нtrace_02┐
"__inference_f6_layer_call_fn_20930Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zнtrace_0
∙
оtrace_02┌
=__inference_f6_layer_call_and_return_conditional_losses_20936Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zоtrace_0
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
▓
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
▐
┤trace_02┐
"__inference_d7_layer_call_fn_20945Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z┤trace_0
∙
╡trace_02┌
=__inference_d7_layer_call_and_return_conditional_losses_20958Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╡trace_0
:
А1А2	d7/kernel
:А2d7/bias
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
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
▒
╗trace_0
╝trace_12Ў
#__inference_dr8_layer_call_fn_20963
#__inference_dr8_layer_call_fn_20968й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0z╝trace_1
ч
╜trace_0
╛trace_12м
>__inference_dr8_layer_call_and_return_conditional_losses_20980
>__inference_dr8_layer_call_and_return_conditional_losses_20985й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0z╛trace_1
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
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
▐
─trace_02┐
"__inference_d9_layer_call_fn_20994Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z─trace_0
∙
┼trace_02┌
=__inference_d9_layer_call_and_return_conditional_losses_21007Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z┼trace_0
:	А
2	d9/kernel
:
2d9/bias
╬
╞trace_02п
__inference_loss_fn_0_21012П
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
annotationsк *в z╞trace_0
╬
╟trace_02п
__inference_loss_fn_1_21017П
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
annotationsк *в z╟trace_0
╬
╚trace_02п
__inference_loss_fn_2_21022П
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
annotationsк *в z╚trace_0
╬
╔trace_02п
__inference_loss_fn_3_21027П
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
annotationsк *в z╔trace_0
╬
╩trace_02п
__inference_loss_fn_4_21032П
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
annotationsк *в z╩trace_0
╬
╦trace_02п
__inference_loss_fn_5_21037П
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
annotationsк *в z╦trace_0
╬
╠trace_02п
__inference_loss_fn_6_21042П
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
annotationsк *в z╠trace_0
╬
═trace_02п
__inference_loss_fn_7_21047П
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
annotationsк *в z═trace_0
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
╬0
╧1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
єBЁ
*__inference_sequential_layer_call_fn_20389c0_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
*__inference_sequential_layer_call_fn_20448c0_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
*__inference_sequential_layer_call_fn_20631inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
*__inference_sequential_layer_call_fn_20652inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
E__inference_sequential_layer_call_and_return_conditional_losses_20276c0_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
E__inference_sequential_layer_call_and_return_conditional_losses_20329c0_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
E__inference_sequential_layer_call_and_return_conditional_losses_20720inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
E__inference_sequential_layer_call_and_return_conditional_losses_20767inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
о
s0
╨1
╤2
╥3
╙4
╘5
╒6
╓7
╫8
╪9
┘10
┌11
█12
▄13
▌14
▐15
▀16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
╨0
╥1
╘2
╓3
╪4
┌5
▄6
▐7"
trackable_list_wrapper
`
╤0
╙1
╒2
╫3
┘4
█5
▌6
▀7"
trackable_list_wrapper
╡
рtrace_0
сtrace_1
тtrace_2
уtrace_3
фtrace_4
хtrace_5
цtrace_6
чtrace_72╥
"__inference__update_step_xla_20772
"__inference__update_step_xla_20777
"__inference__update_step_xla_20782
"__inference__update_step_xla_20787
"__inference__update_step_xla_20792
"__inference__update_step_xla_20797
"__inference__update_step_xla_20802
"__inference__update_step_xla_20807п
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0zрtrace_0zсtrace_1zтtrace_2zуtrace_3zфtrace_4zхtrace_5zцtrace_6zчtrace_7
╦B╚
#__inference_signature_wrapper_20602c0_input"Ф
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
]0
^1"
trackable_list_wrapper
 "
trackable_dict_wrapper
╠B╔
"__inference_c0_layer_call_fn_20816inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
чBф
=__inference_c0_layer_call_and_return_conditional_losses_20829inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╠B╔
"__inference_m1_layer_call_fn_20834inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
чBф
=__inference_m1_layer_call_and_return_conditional_losses_20839inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
▐B█
#__inference_dr2_layer_call_fn_20844inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐B█
#__inference_dr2_layer_call_fn_20849inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
>__inference_dr2_layer_call_and_return_conditional_losses_20861inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
>__inference_dr2_layer_call_and_return_conditional_losses_20866inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

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
_0
`1"
trackable_list_wrapper
 "
trackable_dict_wrapper
╠B╔
"__inference_c3_layer_call_fn_20875inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
чBф
=__inference_c3_layer_call_and_return_conditional_losses_20888inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╠B╔
"__inference_m4_layer_call_fn_20893inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
чBф
=__inference_m4_layer_call_and_return_conditional_losses_20898inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
▐B█
#__inference_dr5_layer_call_fn_20903inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐B█
#__inference_dr5_layer_call_fn_20908inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
>__inference_dr5_layer_call_and_return_conditional_losses_20920inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
>__inference_dr5_layer_call_and_return_conditional_losses_20925inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╠B╔
"__inference_f6_layer_call_fn_20930inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
чBф
=__inference_f6_layer_call_and_return_conditional_losses_20936inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╠B╔
"__inference_d7_layer_call_fn_20945inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
чBф
=__inference_d7_layer_call_and_return_conditional_losses_20958inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
▐B█
#__inference_dr8_layer_call_fn_20963inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐B█
#__inference_dr8_layer_call_fn_20968inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
>__inference_dr8_layer_call_and_return_conditional_losses_20980inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
>__inference_dr8_layer_call_and_return_conditional_losses_20985inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

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
c0
d1"
trackable_list_wrapper
 "
trackable_dict_wrapper
╠B╔
"__inference_d9_layer_call_fn_20994inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
чBф
=__inference_d9_layer_call_and_return_conditional_losses_21007inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
▓Bп
__inference_loss_fn_0_21012"П
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
▓Bп
__inference_loss_fn_1_21017"П
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
▓Bп
__inference_loss_fn_2_21022"П
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
▓Bп
__inference_loss_fn_3_21027"П
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
▓Bп
__inference_loss_fn_4_21032"П
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
▓Bп
__inference_loss_fn_5_21037"П
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
▓Bп
__inference_loss_fn_6_21042"П
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
▓Bп
__inference_loss_fn_7_21047"П
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
Ё
_fn_kwargs"
_tf_keras_metric
(:&@2Adam/m/c0/kernel
(:&@2Adam/v/c0/kernel
:@2Adam/m/c0/bias
:@2Adam/v/c0/bias
):'@А2Adam/m/c3/kernel
):'@А2Adam/v/c3/kernel
:А2Adam/m/c3/bias
:А2Adam/v/c3/bias
": 
А1А2Adam/m/d7/kernel
": 
А1А2Adam/v/d7/kernel
:А2Adam/m/d7/bias
:А2Adam/v/d7/bias
!:	А
2Adam/m/d9/kernel
!:	А
2Adam/v/d9/kernel
:
2Adam/m/d9/bias
:
2Adam/v/d9/bias
эBъ
"__inference__update_step_xla_20772gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
"__inference__update_step_xla_20777gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
"__inference__update_step_xla_20782gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
"__inference__update_step_xla_20787gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
"__inference__update_step_xla_20792gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
"__inference__update_step_xla_20797gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
"__inference__update_step_xla_20802gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
"__inference__update_step_xla_20807gradientvariable"н
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
trackable_dict_wrapperд
"__inference__update_step_xla_20772~xвu
nвk
!К
gradient@
<Т9	%в"
·@
А
p
` VariableSpec 
`рого▐╦?
к "
 М
"__inference__update_step_xla_20777f`в]
VвS
К
gradient@
0Т-	в
·@
А
p
` VariableSpec 
`р▓Мп▐╦?
к "
 з
"__inference__update_step_xla_20782Аzвw
pвm
"К
gradient@А
=Т:	&в#
·@А
А
p
` VariableSpec 
`рТФо▐╦?
к "
 О
"__inference__update_step_xla_20787hbв_
XвU
К
gradientА
1Т.	в
·А
А
p
` VariableSpec 
`рШФо▐╦?
к "
 Ш
"__inference__update_step_xla_20792rlвi
bв_
К
gradient
А1А
6Т3	в
·
А1А
А
p
` VariableSpec 
`р▓╪н▐╦?
к "
 О
"__inference__update_step_xla_20797hbв_
XвU
К
gradientА
1Т.	в
·А
А
p
` VariableSpec 
`рк╪н▐╦?
к "
 Ц
"__inference__update_step_xla_20802pjвg
`в]
К
gradient	А

5Т2	в
·	А

А
p
` VariableSpec 
`рФ н▐╦?
к "
 М
"__inference__update_step_xla_20807f`в]
VвS
К
gradient

0Т-	в
·

А
p
` VariableSpec 
`рЫ н▐╦?
к "
 Т
 __inference__wrapped_model_20111n01LM[\9в6
/в,
*К'
c0_input         
к "'к$
"
d9К
d9         
┤
=__inference_c0_layer_call_and_return_conditional_losses_20829s7в4
-в*
(К%
inputs         
к "4в1
*К'
tensor_0         @
Ъ О
"__inference_c0_layer_call_fn_20816h7в4
-в*
(К%
inputs         
к ")К&
unknown         @╡
=__inference_c3_layer_call_and_return_conditional_losses_20888t017в4
-в*
(К%
inputs         @
к "5в2
+К(
tensor_0         А
Ъ П
"__inference_c3_layer_call_fn_20875i017в4
-в*
(К%
inputs         @
к "*К'
unknown         Аж
=__inference_d7_layer_call_and_return_conditional_losses_20958eLM0в-
&в#
!К
inputs         А1
к "-в*
#К 
tensor_0         А
Ъ А
"__inference_d7_layer_call_fn_20945ZLM0в-
&в#
!К
inputs         А1
к ""К
unknown         Ае
=__inference_d9_layer_call_and_return_conditional_losses_21007d[\0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         

Ъ 
"__inference_d9_layer_call_fn_20994Y[\0в-
&в#
!К
inputs         А
к "!К
unknown         
╡
>__inference_dr2_layer_call_and_return_conditional_losses_20861s;в8
1в.
(К%
inputs         @
p
к "4в1
*К'
tensor_0         @
Ъ ╡
>__inference_dr2_layer_call_and_return_conditional_losses_20866s;в8
1в.
(К%
inputs         @
p 
к "4в1
*К'
tensor_0         @
Ъ П
#__inference_dr2_layer_call_fn_20844h;в8
1в.
(К%
inputs         @
p
к ")К&
unknown         @П
#__inference_dr2_layer_call_fn_20849h;в8
1в.
(К%
inputs         @
p 
к ")К&
unknown         @╖
>__inference_dr5_layer_call_and_return_conditional_losses_20920u<в9
2в/
)К&
inputs         А
p
к "5в2
+К(
tensor_0         А
Ъ ╖
>__inference_dr5_layer_call_and_return_conditional_losses_20925u<в9
2в/
)К&
inputs         А
p 
к "5в2
+К(
tensor_0         А
Ъ С
#__inference_dr5_layer_call_fn_20903j<в9
2в/
)К&
inputs         А
p
к "*К'
unknown         АС
#__inference_dr5_layer_call_fn_20908j<в9
2в/
)К&
inputs         А
p 
к "*К'
unknown         Аз
>__inference_dr8_layer_call_and_return_conditional_losses_20980e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ з
>__inference_dr8_layer_call_and_return_conditional_losses_20985e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ Б
#__inference_dr8_layer_call_fn_20963Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АБ
#__inference_dr8_layer_call_fn_20968Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ак
=__inference_f6_layer_call_and_return_conditional_losses_20936i8в5
.в+
)К&
inputs         А
к "-в*
#К 
tensor_0         А1
Ъ Д
"__inference_f6_layer_call_fn_20930^8в5
.в+
)К&
inputs         А
к ""К
unknown         А1@
__inference_loss_fn_0_21012!в

в 
к "К
unknown @
__inference_loss_fn_1_21017!в

в 
к "К
unknown @
__inference_loss_fn_2_21022!в

в 
к "К
unknown @
__inference_loss_fn_3_21027!в

в 
к "К
unknown @
__inference_loss_fn_4_21032!в

в 
к "К
unknown @
__inference_loss_fn_5_21037!в

в 
к "К
unknown @
__inference_loss_fn_6_21042!в

в 
к "К
unknown @
__inference_loss_fn_7_21047!в

в 
к "К
unknown ч
=__inference_m1_layer_call_and_return_conditional_losses_20839еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ┴
"__inference_m1_layer_call_fn_20834ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ч
=__inference_m4_layer_call_and_return_conditional_losses_20898еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ┴
"__inference_m4_layer_call_fn_20893ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ─
E__inference_sequential_layer_call_and_return_conditional_losses_20276{01LM[\Aв>
7в4
*К'
c0_input         
p

 
к ",в)
"К
tensor_0         

Ъ ─
E__inference_sequential_layer_call_and_return_conditional_losses_20329{01LM[\Aв>
7в4
*К'
c0_input         
p 

 
к ",в)
"К
tensor_0         

Ъ ┬
E__inference_sequential_layer_call_and_return_conditional_losses_20720y01LM[\?в<
5в2
(К%
inputs         
p

 
к ",в)
"К
tensor_0         

Ъ ┬
E__inference_sequential_layer_call_and_return_conditional_losses_20767y01LM[\?в<
5в2
(К%
inputs         
p 

 
к ",в)
"К
tensor_0         

Ъ Ю
*__inference_sequential_layer_call_fn_20389p01LM[\Aв>
7в4
*К'
c0_input         
p

 
к "!К
unknown         
Ю
*__inference_sequential_layer_call_fn_20448p01LM[\Aв>
7в4
*К'
c0_input         
p 

 
к "!К
unknown         
Ь
*__inference_sequential_layer_call_fn_20631n01LM[\?в<
5в2
(К%
inputs         
p

 
к "!К
unknown         
Ь
*__inference_sequential_layer_call_fn_20652n01LM[\?в<
5в2
(К%
inputs         
p 

 
к "!К
unknown         
б
#__inference_signature_wrapper_20602z01LM[\EвB
в 
;к8
6
c0_input*К'
c0_input         "'к$
"
d9К
d9         

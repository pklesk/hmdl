╟й	
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
 И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758╗ж
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
Adam/v/d6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/v/d6/bias
m
"Adam/v/d6/bias/Read/ReadVariableOpReadVariableOpAdam/v/d6/bias*
_output_shapes
:(*
dtype0
t
Adam/m/d6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameAdam/m/d6/bias
m
"Adam/m/d6/bias/Read/ReadVariableOpReadVariableOpAdam/m/d6/bias*
_output_shapes
:(*
dtype0
|
Adam/v/d6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/v/d6/kernel
u
$Adam/v/d6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d6/kernel*
_output_shapes

:@(*
dtype0
|
Adam/m/d6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*!
shared_nameAdam/m/d6/kernel
u
$Adam/m/d6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d6/kernel*
_output_shapes

:@(*
dtype0
t
Adam/v/d4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/v/d4/bias
m
"Adam/v/d4/bias/Read/ReadVariableOpReadVariableOpAdam/v/d4/bias*
_output_shapes
:@*
dtype0
t
Adam/m/d4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/m/d4/bias
m
"Adam/m/d4/bias/Read/ReadVariableOpReadVariableOpAdam/m/d4/bias*
_output_shapes
:@*
dtype0
}
Adam/v/d4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А @*!
shared_nameAdam/v/d4/kernel
v
$Adam/v/d4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/d4/kernel*
_output_shapes
:	А @*
dtype0
}
Adam/m/d4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А @*!
shared_nameAdam/m/d4/kernel
v
$Adam/m/d4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/d4/kernel*
_output_shapes
:	А @*
dtype0
t
Adam/v/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/c0/bias
m
"Adam/v/c0/bias/Read/ReadVariableOpReadVariableOpAdam/v/c0/bias*
_output_shapes
:*
dtype0
t
Adam/m/c0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/c0/bias
m
"Adam/m/c0/bias/Read/ReadVariableOpReadVariableOpAdam/m/c0/bias*
_output_shapes
:*
dtype0
Д
Adam/v/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameAdam/v/c0/kernel
}
$Adam/v/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/c0/kernel*&
_output_shapes
:		*
dtype0
Д
Adam/m/c0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameAdam/m/c0/kernel
}
$Adam/m/c0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/c0/kernel*&
_output_shapes
:		*
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
#__inference_signature_wrapper_81118

NoOpNoOp
О?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╔>
value┐>B╝> B╡>
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
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
е
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator* 
О
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
ж
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
е
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_random_generator* 
ж
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias*
.
0
1
32
43
B4
C5*
.
0
1
32
43
B4
C5*
,
D0
E1
F2
G3
H4
I5* 
░
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
6
Strace_0
Ttrace_1
Utrace_2
Vtrace_3* 
* 
Б
W
_variables
X_iterations
Y_learning_rate
Z_index_dict
[
_momentums
\_velocities
]_update_step_xla*

^serving_default* 

0
1*

0
1*

D0
E1* 
У
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
YS
VARIABLE_VALUE	c0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
С
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ktrace_0* 

ltrace_0* 
* 
* 
* 
С
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

rtrace_0
strace_1* 

ttrace_0
utrace_1* 
* 
* 
* 
* 
С
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

{trace_0* 

|trace_0* 

30
41*

30
41*

F0
G1* 
Х
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

Вtrace_0* 

Гtrace_0* 
YS
VARIABLE_VALUE	d4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

Йtrace_0
Кtrace_1* 

Лtrace_0
Мtrace_1* 
* 

B0
C1*

B0
C1*

H0
I1* 
Ш
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

Тtrace_0* 

Уtrace_0* 
YS
VARIABLE_VALUE	d6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

Фtrace_0* 

Хtrace_0* 

Цtrace_0* 

Чtrace_0* 

Шtrace_0* 

Щtrace_0* 
* 
5
0
1
2
3
4
5
6*

Ъ0
Ы1*
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
n
X0
Ь1
Э2
Ю3
Я4
а5
б6
в7
г8
д9
е10
ж11
з12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
Ь0
Ю1
а2
в3
д4
ж5*
4
Э0
Я1
б2
г3
е4
з5*
V
иtrace_0
йtrace_1
кtrace_2
лtrace_3
мtrace_4
нtrace_5* 
* 
* 
* 
* 

D0
E1* 
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
F0
G1* 
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
<
о	variables
п	keras_api

░total

▒count*
M
▓	variables
│	keras_api

┤total

╡count
╢
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
VARIABLE_VALUEAdam/m/d4/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/d4/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/d4/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/d4/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/d6/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/d6/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/d6/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/d6/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 

░0
▒1*

о	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

┤0
╡1*

▓	variables*
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
√
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	d4/kerneld4/bias	d6/kerneld6/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/d4/kernelAdam/v/d4/kernelAdam/m/d4/biasAdam/v/d4/biasAdam/m/d6/kernelAdam/v/d6/kernelAdam/m/d6/biasAdam/v/d6/biastotal_1count_1totalcountConst*%
Tin
2*
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
__inference__traced_save_81612
Ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	c0/kernelc0/bias	d4/kerneld4/bias	d6/kerneld6/bias	iterationlearning_rateAdam/m/c0/kernelAdam/v/c0/kernelAdam/m/c0/biasAdam/v/c0/biasAdam/m/d4/kernelAdam/v/d4/kernelAdam/m/d4/biasAdam/v/d4/biasAdam/m/d6/kernelAdam/v/d6/kernelAdam/m/d6/biasAdam/v/d6/biastotal_1count_1totalcount*$
Tin
2*
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
!__inference__traced_restore_81694чо
Ш
?
#__inference_dr5_layer_call_fn_81376

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
>__inference_dr5_layer_call_and_return_conditional_losses_80892`
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
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_80756

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
╚
+
__inference_loss_fn_0_81420
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
╕
?
#__inference_dr2_layer_call_fn_81316

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
>__inference_dr2_layer_call_and_return_conditional_losses_80880h
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
█п
щ
__inference__traced_save_81612
file_prefix:
 read_disablecopyonread_c0_kernel:		.
 read_1_disablecopyonread_c0_bias:5
"read_2_disablecopyonread_d4_kernel:	А @.
 read_3_disablecopyonread_d4_bias:@4
"read_4_disablecopyonread_d6_kernel:@(.
 read_5_disablecopyonread_d6_bias:(,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: C
)read_8_disablecopyonread_adam_m_c0_kernel:		C
)read_9_disablecopyonread_adam_v_c0_kernel:		6
(read_10_disablecopyonread_adam_m_c0_bias:6
(read_11_disablecopyonread_adam_v_c0_bias:=
*read_12_disablecopyonread_adam_m_d4_kernel:	А @=
*read_13_disablecopyonread_adam_v_d4_kernel:	А @6
(read_14_disablecopyonread_adam_m_d4_bias:@6
(read_15_disablecopyonread_adam_v_d4_bias:@<
*read_16_disablecopyonread_adam_m_d6_kernel:@(<
*read_17_disablecopyonread_adam_v_d6_kernel:@(6
(read_18_disablecopyonread_adam_m_d6_bias:(6
(read_19_disablecopyonread_adam_v_d6_bias:(+
!read_20_disablecopyonread_total_1: +
!read_21_disablecopyonread_count_1: )
read_22_disablecopyonread_total: )
read_23_disablecopyonread_count: 
savev2_const
identity_49ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
:		*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:		i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:		t
Read_1/DisableCopyOnReadDisableCopyOnRead read_1_disablecopyonread_c0_bias"/device:CPU:0*
_output_shapes
 Ь
Read_1/ReadVariableOpReadVariableOp read_1_disablecopyonread_c0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_d4_kernel"/device:CPU:0*
_output_shapes
 г
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_d4_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А @*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А @d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	А @t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_d4_bias"/device:CPU:0*
_output_shapes
 Ь
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_d4_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_d6_kernel"/device:CPU:0*
_output_shapes
 в
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_d6_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:@(t
Read_5/DisableCopyOnReadDisableCopyOnRead read_5_disablecopyonread_d6_bias"/device:CPU:0*
_output_shapes
 Ь
Read_5/ReadVariableOpReadVariableOp read_5_disablecopyonread_d6_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:(v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Ъ
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ю
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_adam_m_c0_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_adam_m_c0_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:		*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:		m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:		}
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_adam_v_c0_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_adam_v_c0_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:		*
dtype0v
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:		m
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*&
_output_shapes
:		}
Read_10/DisableCopyOnReadDisableCopyOnRead(read_10_disablecopyonread_adam_m_c0_bias"/device:CPU:0*
_output_shapes
 ж
Read_10/ReadVariableOpReadVariableOp(read_10_disablecopyonread_adam_m_c0_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_adam_v_c0_bias"/device:CPU:0*
_output_shapes
 ж
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_adam_v_c0_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_adam_m_d4_kernel"/device:CPU:0*
_output_shapes
 н
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_adam_m_d4_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А @*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А @f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	А @
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_adam_v_d4_kernel"/device:CPU:0*
_output_shapes
 н
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_adam_v_d4_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А @*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А @f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	А @}
Read_14/DisableCopyOnReadDisableCopyOnRead(read_14_disablecopyonread_adam_m_d4_bias"/device:CPU:0*
_output_shapes
 ж
Read_14/ReadVariableOpReadVariableOp(read_14_disablecopyonread_adam_m_d4_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_15/DisableCopyOnReadDisableCopyOnRead(read_15_disablecopyonread_adam_v_d4_bias"/device:CPU:0*
_output_shapes
 ж
Read_15/ReadVariableOpReadVariableOp(read_15_disablecopyonread_adam_v_d4_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_16/DisableCopyOnReadDisableCopyOnRead*read_16_disablecopyonread_adam_m_d6_kernel"/device:CPU:0*
_output_shapes
 м
Read_16/ReadVariableOpReadVariableOp*read_16_disablecopyonread_adam_m_d6_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@(
Read_17/DisableCopyOnReadDisableCopyOnRead*read_17_disablecopyonread_adam_v_d6_kernel"/device:CPU:0*
_output_shapes
 м
Read_17/ReadVariableOpReadVariableOp*read_17_disablecopyonread_adam_v_d6_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@(*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@(e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:@(}
Read_18/DisableCopyOnReadDisableCopyOnRead(read_18_disablecopyonread_adam_m_d6_bias"/device:CPU:0*
_output_shapes
 ж
Read_18/ReadVariableOpReadVariableOp(read_18_disablecopyonread_adam_m_d6_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:(}
Read_19/DisableCopyOnReadDisableCopyOnRead(read_19_disablecopyonread_adam_v_d6_bias"/device:CPU:0*
_output_shapes
 ж
Read_19/ReadVariableOpReadVariableOp(read_19_disablecopyonread_adam_v_d6_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:(v
Read_20/DisableCopyOnReadDisableCopyOnRead!read_20_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_20/ReadVariableOpReadVariableOp!read_20_disablecopyonread_total_1^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_21/DisableCopyOnReadDisableCopyOnRead!read_21_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_21/ReadVariableOpReadVariableOp!read_21_disablecopyonread_count_1^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_total^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_count^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: ·

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*г

valueЩ
BЦ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЯ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B √
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: ╗

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_23/ReadVariableOpRead_23/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
л
J
"__inference__update_step_xla_81264
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
║
O
"__inference__update_step_xla_81259
gradient
variable:	А @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	А @: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	А @
"
_user_specified_name
gradient
╗
П
"__inference_d6_layer_call_fn_81402

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
=__inference_d6_layer_call_and_return_conditional_losses_80854o
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
я
ф
G__inference_sequential_9_layer_call_and_return_conditional_losses_80906
c0_input"
c0_80870:		
c0_80872:
d4_80883:	А @
d4_80885:@
d6_80894:@(
d6_80896:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_80870c0_80872*
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
=__inference_c0_layer_call_and_return_conditional_losses_80779╫
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
=__inference_m1_layer_call_and_return_conditional_losses_80756╤
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
>__inference_dr2_layer_call_and_return_conditional_losses_80880╔
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
=__inference_f3_layer_call_and_return_conditional_losses_80806ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_80883d4_80885*
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
=__inference_d4_layer_call_and_return_conditional_losses_80821╤
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
>__inference_dr5_layer_call_and_return_conditional_losses_80892Є
d6/StatefulPartitionedCallStatefulPartitionedCalldr5/PartitionedCall:output:0d6_80894d6_80896*
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
=__inference_d6_layer_call_and_return_conditional_losses_80854`
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
─
+
__inference_loss_fn_5_81445
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
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_81333

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
у
Ч
"__inference_c0_layer_call_fn_81283

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
=__inference_c0_layer_call_and_return_conditional_losses_80779w
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
┌
я
=__inference_d4_layer_call_and_return_conditional_losses_81366

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
╚
+
__inference_loss_fn_4_81440
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
╤
\
>__inference_dr5_layer_call_and_return_conditional_losses_81393

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
Е
Y
=__inference_m1_layer_call_and_return_conditional_losses_81306

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
╤
\
>__inference_dr5_layer_call_and_return_conditional_losses_80892

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
ё%
└
 __inference__wrapped_model_80750
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
█

]
>__inference_dr2_layer_call_and_return_conditional_losses_81328

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧е
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:         i
IdentityIdentitydropout/SelectV2:output:0*
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
Р	
О
,__inference_sequential_9_layer_call_fn_81158

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
G__inference_sequential_9_layer_call_and_return_conditional_losses_80984o
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
┐
Y
=__inference_f3_layer_call_and_return_conditional_losses_80806

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
К
\
#__inference_dr2_layer_call_fn_81311

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
>__inference_dr2_layer_call_and_return_conditional_losses_80798w
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
Р	
О
,__inference_sequential_9_layer_call_fn_81141

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
G__inference_sequential_9_layer_call_and_return_conditional_losses_80938o
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
н"
а
G__inference_sequential_9_layer_call_and_return_conditional_losses_80867
c0_input"
c0_80780:		
c0_80782:
d4_80822:	А @
d4_80824:@
d6_80855:@(
d6_80857:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallц
c0/StatefulPartitionedCallStatefulPartitionedCallc0_inputc0_80780c0_80782*
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
=__inference_c0_layer_call_and_return_conditional_losses_80779╫
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
=__inference_m1_layer_call_and_return_conditional_losses_80756с
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
>__inference_dr2_layer_call_and_return_conditional_losses_80798╤
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
=__inference_f3_layer_call_and_return_conditional_losses_80806ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_80822d4_80824*
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
=__inference_d4_layer_call_and_return_conditional_losses_80821 
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
>__inference_dr5_layer_call_and_return_conditional_losses_80839·
d6/StatefulPartitionedCallStatefulPartitionedCall$dr5/StatefulPartitionedCall:output:0d6_80855d6_80857*
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
=__inference_d6_layer_call_and_return_conditional_losses_80854`
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
█

]
>__inference_dr2_layer_call_and_return_conditional_losses_80798

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧е
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:         i
IdentityIdentitydropout/SelectV2:output:0*
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
ц
З
#__inference_signature_wrapper_81118
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
 __inference__wrapped_model_80750o
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
ё
\
>__inference_dr2_layer_call_and_return_conditional_losses_80880

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
л
J
"__inference__update_step_xla_81274
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
Ц	
Р
,__inference_sequential_9_layer_call_fn_80999
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_80984o
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
┐
Y
=__inference_f3_layer_call_and_return_conditional_losses_81344

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
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_81296

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
─
+
__inference_loss_fn_1_81425
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
г

]
>__inference_dr5_layer_call_and_return_conditional_losses_80839

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
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Э
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
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentitydropout/SelectV2:output:0*
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
ё"
╔
G__inference_sequential_9_layer_call_and_return_conditional_losses_81244

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
г

]
>__inference_dr5_layer_call_and_return_conditional_losses_81388

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
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Э
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
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentitydropout/SelectV2:output:0*
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
з"
Ю
G__inference_sequential_9_layer_call_and_return_conditional_losses_80938

inputs"
c0_80912:		
c0_80914:
d4_80920:	А @
d4_80922:@
d6_80926:@(
d6_80928:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallвdr2/StatefulPartitionedCallвdr5/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_80912c0_80914*
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
=__inference_c0_layer_call_and_return_conditional_losses_80779╫
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
=__inference_m1_layer_call_and_return_conditional_losses_80756с
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
>__inference_dr2_layer_call_and_return_conditional_losses_80798╤
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
=__inference_f3_layer_call_and_return_conditional_losses_80806ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_80920d4_80922*
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
=__inference_d4_layer_call_and_return_conditional_losses_80821 
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
>__inference_dr5_layer_call_and_return_conditional_losses_80839·
d6/StatefulPartitionedCallStatefulPartitionedCall$dr5/StatefulPartitionedCall:output:0d6_80926d6_80928*
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
=__inference_d6_layer_call_and_return_conditional_losses_80854`
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
─
+
__inference_loss_fn_3_81435
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
ъ
\
#__inference_dr5_layer_call_fn_81371

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
>__inference_dr5_layer_call_and_return_conditional_losses_80839o
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
нe
┘
!__inference__traced_restore_81694
file_prefix4
assignvariableop_c0_kernel:		(
assignvariableop_1_c0_bias:/
assignvariableop_2_d4_kernel:	А @(
assignvariableop_3_d4_bias:@.
assignvariableop_4_d6_kernel:@((
assignvariableop_5_d6_bias:(&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: =
#assignvariableop_8_adam_m_c0_kernel:		=
#assignvariableop_9_adam_v_c0_kernel:		0
"assignvariableop_10_adam_m_c0_bias:0
"assignvariableop_11_adam_v_c0_bias:7
$assignvariableop_12_adam_m_d4_kernel:	А @7
$assignvariableop_13_adam_v_d4_kernel:	А @0
"assignvariableop_14_adam_m_d4_bias:@0
"assignvariableop_15_adam_v_d4_bias:@6
$assignvariableop_16_adam_m_d6_kernel:@(6
$assignvariableop_17_adam_v_d6_kernel:@(0
"assignvariableop_18_adam_m_d6_bias:(0
"assignvariableop_19_adam_v_d6_bias:(%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9¤

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*г

valueЩ
BЦ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHв
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
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
AssignVariableOp_2AssignVariableOpassignvariableop_2_d4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_3AssignVariableOpassignvariableop_3_d4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_4AssignVariableOpassignvariableop_4_d6_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_5AssignVariableOpassignvariableop_5_d6_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_8AssignVariableOp#assignvariableop_8_adam_m_c0_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_9AssignVariableOp#assignvariableop_9_adam_v_c0_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_10AssignVariableOp"assignvariableop_10_adam_m_c0_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_11AssignVariableOp"assignvariableop_11_adam_v_c0_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_12AssignVariableOp$assignvariableop_12_adam_m_d4_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_13AssignVariableOp$assignvariableop_13_adam_v_d4_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_14AssignVariableOp"assignvariableop_14_adam_m_d4_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_15AssignVariableOp"assignvariableop_15_adam_v_d4_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_16AssignVariableOp$assignvariableop_16_adam_m_d6_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_v_d6_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_m_d6_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_19AssignVariableOp"assignvariableop_19_adam_v_d6_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ▀
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ╠
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_23AssignVariableOp_232(
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
╧
V
"__inference__update_step_xla_81249
gradient"
variable:		*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:		: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:		
"
_user_specified_name
gradient
╖
N
"__inference__update_step_xla_81269
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
█
ю
=__inference_d6_layer_call_and_return_conditional_losses_80854

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
╛
Р
"__inference_d4_layer_call_fn_81353

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
=__inference_d4_layer_call_and_return_conditional_losses_80821o
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
┴1
╔
G__inference_sequential_9_layer_call_and_return_conditional_losses_81208

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
:         b
dr2/dropout/ShapeShapem1/MaxPool:output:0*
T0*
_output_shapes
::э╧н
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
:         X
dr2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    л
dr2/dropout/SelectV2SelectV2dr2/dropout/GreaterEqual:z:0dr2/dropout/Mul:z:0dr2/dropout/Const_1:output:0*
T0*/
_output_shapes
:         Y
f3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       z

f3/ReshapeReshapedr2/dropout/SelectV2:output:0f3/Const:output:0*
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
:         @d
dr5/dropout/ShapeShaped4/Relu:activations:0*
T0*
_output_shapes
::э╧б
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
:         @X
dr5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    г
dr5/dropout/SelectV2SelectV2dr5/dropout/GreaterEqual:z:0dr5/dropout/Mul:z:0dr5/dropout/Const_1:output:0*
T0*'
_output_shapes
:         @z
d6/MatMul/ReadVariableOpReadVariableOp!d6_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0Ж
	d6/MatMulMatMuldr5/dropout/SelectV2:output:0 d6/MatMul/ReadVariableOp:value:0*
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
Ц	
Р
,__inference_sequential_9_layer_call_fn_80953
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_80938o
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
г
>
"__inference_m1_layer_call_fn_81301

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
=__inference_m1_layer_call_and_return_conditional_losses_80756Г
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
┌
я
=__inference_d4_layer_call_and_return_conditional_losses_80821

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
щ
т
G__inference_sequential_9_layer_call_and_return_conditional_losses_80984

inputs"
c0_80958:		
c0_80960:
d4_80966:	А @
d4_80968:@
d6_80972:@(
d6_80974:(
identityИвc0/StatefulPartitionedCallвd4/StatefulPartitionedCallвd6/StatefulPartitionedCallф
c0/StatefulPartitionedCallStatefulPartitionedCallinputsc0_80958c0_80960*
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
=__inference_c0_layer_call_and_return_conditional_losses_80779╫
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
=__inference_m1_layer_call_and_return_conditional_losses_80756╤
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
>__inference_dr2_layer_call_and_return_conditional_losses_80880╔
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
=__inference_f3_layer_call_and_return_conditional_losses_80806ё
d4/StatefulPartitionedCallStatefulPartitionedCallf3/PartitionedCall:output:0d4_80966d4_80968*
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
=__inference_d4_layer_call_and_return_conditional_losses_80821╤
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
>__inference_dr5_layer_call_and_return_conditional_losses_80892Є
d6/StatefulPartitionedCallStatefulPartitionedCalldr5/PartitionedCall:output:0d6_80972d6_80974*
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
=__inference_d6_layer_call_and_return_conditional_losses_80854`
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
╚
+
__inference_loss_fn_2_81430
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
л
J
"__inference__update_step_xla_81254
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
╛
Ў
=__inference_c0_layer_call_and_return_conditional_losses_80779

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
█
ю
=__inference_d6_layer_call_and_return_conditional_losses_81415

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
и
>
"__inference_f3_layer_call_fn_81338

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
=__inference_f3_layer_call_and_return_conditional_losses_80806a
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
serving_default_c0_input:0         @@6
d60
StatefulPartitionedCall:0         (tensorflow/serving/predict:├ь
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
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator"
_tf_keras_layer
е
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
╝
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_random_generator"
_tf_keras_layer
╗
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias"
_tf_keras_layer
J
0
1
32
43
B4
C5"
trackable_list_wrapper
J
0
1
32
43
B4
C5"
trackable_list_wrapper
J
D0
E1
F2
G3
H4
I5"
trackable_list_wrapper
╩
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
█
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32Ё
,__inference_sequential_9_layer_call_fn_80953
,__inference_sequential_9_layer_call_fn_80999
,__inference_sequential_9_layer_call_fn_81141
,__inference_sequential_9_layer_call_fn_81158╡
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
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
╟
Strace_0
Ttrace_1
Utrace_2
Vtrace_32▄
G__inference_sequential_9_layer_call_and_return_conditional_losses_80867
G__inference_sequential_9_layer_call_and_return_conditional_losses_80906
G__inference_sequential_9_layer_call_and_return_conditional_losses_81208
G__inference_sequential_9_layer_call_and_return_conditional_losses_81244╡
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
 zStrace_0zTtrace_1zUtrace_2zVtrace_3
╠B╔
 __inference__wrapped_model_80750c0_input"Ш
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
W
_variables
X_iterations
Y_learning_rate
Z_index_dict
[
_momentums
\_velocities
]_update_step_xla"
experimentalOptimizer
,
^serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
н
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
▄
dtrace_02┐
"__inference_c0_layer_call_fn_81283Ш
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
 zdtrace_0
ў
etrace_02┌
=__inference_c0_layer_call_and_return_conditional_losses_81296Ш
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
 zetrace_0
#:!		2	c0/kernel
:2c0/bias
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
н
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
▄
ktrace_02┐
"__inference_m1_layer_call_fn_81301Ш
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
 zktrace_0
ў
ltrace_02┌
=__inference_m1_layer_call_and_return_conditional_losses_81306Ш
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
 zltrace_0
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
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
н
rtrace_0
strace_12Ў
#__inference_dr2_layer_call_fn_81311
#__inference_dr2_layer_call_fn_81316й
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
 zrtrace_0zstrace_1
у
ttrace_0
utrace_12м
>__inference_dr2_layer_call_and_return_conditional_losses_81328
>__inference_dr2_layer_call_and_return_conditional_losses_81333й
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
 zttrace_0zutrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
▄
{trace_02┐
"__inference_f3_layer_call_fn_81338Ш
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
 z{trace_0
ў
|trace_02┌
=__inference_f3_layer_call_and_return_conditional_losses_81344Ш
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
 z|trace_0
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
п
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
▐
Вtrace_02┐
"__inference_d4_layer_call_fn_81353Ш
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
 zВtrace_0
∙
Гtrace_02┌
=__inference_d4_layer_call_and_return_conditional_losses_81366Ш
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
 zГtrace_0
:	А @2	d4/kernel
:@2d4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
▒
Йtrace_0
Кtrace_12Ў
#__inference_dr5_layer_call_fn_81371
#__inference_dr5_layer_call_fn_81376й
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
 zЙtrace_0zКtrace_1
ч
Лtrace_0
Мtrace_12м
>__inference_dr5_layer_call_and_return_conditional_losses_81388
>__inference_dr5_layer_call_and_return_conditional_losses_81393й
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
 zЛtrace_0zМtrace_1
"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
▓
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
▐
Тtrace_02┐
"__inference_d6_layer_call_fn_81402Ш
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
 zТtrace_0
∙
Уtrace_02┌
=__inference_d6_layer_call_and_return_conditional_losses_81415Ш
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
 zУtrace_0
:@(2	d6/kernel
:(2d6/bias
╬
Фtrace_02п
__inference_loss_fn_0_81420П
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
annotationsк *в zФtrace_0
╬
Хtrace_02п
__inference_loss_fn_1_81425П
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
annotationsк *в zХtrace_0
╬
Цtrace_02п
__inference_loss_fn_2_81430П
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
annotationsк *в zЦtrace_0
╬
Чtrace_02п
__inference_loss_fn_3_81435П
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
annotationsк *в zЧtrace_0
╬
Шtrace_02п
__inference_loss_fn_4_81440П
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
annotationsк *в zШtrace_0
╬
Щtrace_02п
__inference_loss_fn_5_81445П
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
annotationsк *в zЩtrace_0
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
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
їBЄ
,__inference_sequential_9_layer_call_fn_80953c0_input"╡
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
їBЄ
,__inference_sequential_9_layer_call_fn_80999c0_input"╡
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
,__inference_sequential_9_layer_call_fn_81141inputs"╡
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
,__inference_sequential_9_layer_call_fn_81158inputs"╡
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
РBН
G__inference_sequential_9_layer_call_and_return_conditional_losses_80867c0_input"╡
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
РBН
G__inference_sequential_9_layer_call_and_return_conditional_losses_80906c0_input"╡
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_81208inputs"╡
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_81244inputs"╡
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
К
X0
Ь1
Э2
Ю3
Я4
а5
б6
в7
г8
д9
е10
ж11
з12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
P
Ь0
Ю1
а2
в3
д4
ж5"
trackable_list_wrapper
P
Э0
Я1
б2
г3
е4
з5"
trackable_list_wrapper
╡
иtrace_0
йtrace_1
кtrace_2
лtrace_3
мtrace_4
нtrace_52К
"__inference__update_step_xla_81249
"__inference__update_step_xla_81254
"__inference__update_step_xla_81259
"__inference__update_step_xla_81264
"__inference__update_step_xla_81269
"__inference__update_step_xla_81274п
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
 0zиtrace_0zйtrace_1zкtrace_2zлtrace_3zмtrace_4zнtrace_5
╦B╚
#__inference_signature_wrapper_81118c0_input"Ф
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
D0
E1"
trackable_list_wrapper
 "
trackable_dict_wrapper
╠B╔
"__inference_c0_layer_call_fn_81283inputs"Ш
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
=__inference_c0_layer_call_and_return_conditional_losses_81296inputs"Ш
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
"__inference_m1_layer_call_fn_81301inputs"Ш
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
=__inference_m1_layer_call_and_return_conditional_losses_81306inputs"Ш
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
#__inference_dr2_layer_call_fn_81311inputs"й
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
#__inference_dr2_layer_call_fn_81316inputs"й
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
>__inference_dr2_layer_call_and_return_conditional_losses_81328inputs"й
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
>__inference_dr2_layer_call_and_return_conditional_losses_81333inputs"й
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
"__inference_f3_layer_call_fn_81338inputs"Ш
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
=__inference_f3_layer_call_and_return_conditional_losses_81344inputs"Ш
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
F0
G1"
trackable_list_wrapper
 "
trackable_dict_wrapper
╠B╔
"__inference_d4_layer_call_fn_81353inputs"Ш
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
=__inference_d4_layer_call_and_return_conditional_losses_81366inputs"Ш
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
#__inference_dr5_layer_call_fn_81371inputs"й
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
#__inference_dr5_layer_call_fn_81376inputs"й
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
>__inference_dr5_layer_call_and_return_conditional_losses_81388inputs"й
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
>__inference_dr5_layer_call_and_return_conditional_losses_81393inputs"й
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
H0
I1"
trackable_list_wrapper
 "
trackable_dict_wrapper
╠B╔
"__inference_d6_layer_call_fn_81402inputs"Ш
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
=__inference_d6_layer_call_and_return_conditional_losses_81415inputs"Ш
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
__inference_loss_fn_0_81420"П
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
__inference_loss_fn_1_81425"П
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
__inference_loss_fn_2_81430"П
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
__inference_loss_fn_3_81435"П
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
__inference_loss_fn_4_81440"П
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
__inference_loss_fn_5_81445"П
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
о	variables
п	keras_api

░total

▒count"
_tf_keras_metric
c
▓	variables
│	keras_api

┤total

╡count
╢
_fn_kwargs"
_tf_keras_metric
(:&		2Adam/m/c0/kernel
(:&		2Adam/v/c0/kernel
:2Adam/m/c0/bias
:2Adam/v/c0/bias
!:	А @2Adam/m/d4/kernel
!:	А @2Adam/v/d4/kernel
:@2Adam/m/d4/bias
:@2Adam/v/d4/bias
 :@(2Adam/m/d6/kernel
 :@(2Adam/v/d6/kernel
:(2Adam/m/d6/bias
:(2Adam/v/d6/bias
эBъ
"__inference__update_step_xla_81249gradientvariable"н
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
"__inference__update_step_xla_81254gradientvariable"н
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
"__inference__update_step_xla_81259gradientvariable"н
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
"__inference__update_step_xla_81264gradientvariable"н
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
"__inference__update_step_xla_81269gradientvariable"н
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
"__inference__update_step_xla_81274gradientvariable"н
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
░0
▒1"
trackable_list_wrapper
.
о	variables"
_generic_user_object
:  (2total
:  (2count
0
┤0
╡1"
trackable_list_wrapper
.
▓	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperд
"__inference__update_step_xla_81249~xвu
nвk
!К
gradient		
<Т9	%в"
·		
А
p
` VariableSpec 
`рШЬАУы?
к "
 М
"__inference__update_step_xla_81254f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`рВЬАУы?
к "
 Ц
"__inference__update_step_xla_81259pjвg
`в]
К
gradient	А @
5Т2	в
·	А @
А
p
` VariableSpec 
`рЭНБУы?
к "
 М
"__inference__update_step_xla_81264f`в]
VвS
К
gradient@
0Т-	в
·@
А
p
` VariableSpec 
`рЪГАУы?
к "
 Ф
"__inference__update_step_xla_81269nhвe
^в[
К
gradient@(
4Т1	в
·@(
А
p
` VariableSpec 
`р─чАУы?
к "
 М
"__inference__update_step_xla_81274f`в]
VвS
К
gradient(
0Т-	в
·(
А
p
` VariableSpec 
`р┼▒АУы?
к "
 Р
 __inference__wrapped_model_80750l34BC9в6
/в,
*К'
c0_input         @@
к "'к$
"
d6К
d6         (┤
=__inference_c0_layer_call_and_return_conditional_losses_81296s7в4
-в*
(К%
inputs         @@
к "4в1
*К'
tensor_0         @@
Ъ О
"__inference_c0_layer_call_fn_81283h7в4
-в*
(К%
inputs         @@
к ")К&
unknown         @@е
=__inference_d4_layer_call_and_return_conditional_losses_81366d340в-
&в#
!К
inputs         А 
к ",в)
"К
tensor_0         @
Ъ 
"__inference_d4_layer_call_fn_81353Y340в-
&в#
!К
inputs         А 
к "!К
unknown         @д
=__inference_d6_layer_call_and_return_conditional_losses_81415cBC/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         (
Ъ ~
"__inference_d6_layer_call_fn_81402XBC/в,
%в"
 К
inputs         @
к "!К
unknown         (╡
>__inference_dr2_layer_call_and_return_conditional_losses_81328s;в8
1в.
(К%
inputs         
p
к "4в1
*К'
tensor_0         
Ъ ╡
>__inference_dr2_layer_call_and_return_conditional_losses_81333s;в8
1в.
(К%
inputs         
p 
к "4в1
*К'
tensor_0         
Ъ П
#__inference_dr2_layer_call_fn_81311h;в8
1в.
(К%
inputs         
p
к ")К&
unknown         П
#__inference_dr2_layer_call_fn_81316h;в8
1в.
(К%
inputs         
p 
к ")К&
unknown         е
>__inference_dr5_layer_call_and_return_conditional_losses_81388c3в0
)в&
 К
inputs         @
p
к ",в)
"К
tensor_0         @
Ъ е
>__inference_dr5_layer_call_and_return_conditional_losses_81393c3в0
)в&
 К
inputs         @
p 
к ",в)
"К
tensor_0         @
Ъ 
#__inference_dr5_layer_call_fn_81371X3в0
)в&
 К
inputs         @
p
к "!К
unknown         @
#__inference_dr5_layer_call_fn_81376X3в0
)в&
 К
inputs         @
p 
к "!К
unknown         @й
=__inference_f3_layer_call_and_return_conditional_losses_81344h7в4
-в*
(К%
inputs         
к "-в*
#К 
tensor_0         А 
Ъ Г
"__inference_f3_layer_call_fn_81338]7в4
-в*
(К%
inputs         
к ""К
unknown         А @
__inference_loss_fn_0_81420!в

в 
к "К
unknown @
__inference_loss_fn_1_81425!в

в 
к "К
unknown @
__inference_loss_fn_2_81430!в

в 
к "К
unknown @
__inference_loss_fn_3_81435!в

в 
к "К
unknown @
__inference_loss_fn_4_81440!в

в 
к "К
unknown @
__inference_loss_fn_5_81445!в

в 
к "К
unknown ч
=__inference_m1_layer_call_and_return_conditional_losses_81306еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ┴
"__inference_m1_layer_call_fn_81301ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ─
G__inference_sequential_9_layer_call_and_return_conditional_losses_80867y34BCAв>
7в4
*К'
c0_input         @@
p

 
к ",в)
"К
tensor_0         (
Ъ ─
G__inference_sequential_9_layer_call_and_return_conditional_losses_80906y34BCAв>
7в4
*К'
c0_input         @@
p 

 
к ",в)
"К
tensor_0         (
Ъ ┬
G__inference_sequential_9_layer_call_and_return_conditional_losses_81208w34BC?в<
5в2
(К%
inputs         @@
p

 
к ",в)
"К
tensor_0         (
Ъ ┬
G__inference_sequential_9_layer_call_and_return_conditional_losses_81244w34BC?в<
5в2
(К%
inputs         @@
p 

 
к ",в)
"К
tensor_0         (
Ъ Ю
,__inference_sequential_9_layer_call_fn_80953n34BCAв>
7в4
*К'
c0_input         @@
p

 
к "!К
unknown         (Ю
,__inference_sequential_9_layer_call_fn_80999n34BCAв>
7в4
*К'
c0_input         @@
p 

 
к "!К
unknown         (Ь
,__inference_sequential_9_layer_call_fn_81141l34BC?в<
5в2
(К%
inputs         @@
p

 
к "!К
unknown         (Ь
,__inference_sequential_9_layer_call_fn_81158l34BC?в<
5в2
(К%
inputs         @@
p 

 
к "!К
unknown         (Я
#__inference_signature_wrapper_81118x34BCEвB
в 
;к8
6
c0_input*К'
c0_input         @@"'к$
"
d6К
d6         (
       �K"	  @����Abrain.Event:2�!���U      ]=�	o�p����A"ƫ
�
-global_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@global_step*
dtype0*
_output_shapes
: 
�
#global_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
global_step/Initializer/zerosFill-global_step/Initializer/zeros/shape_as_tensor#global_step/Initializer/zeros/Const*
T0	*

index_type0*
_class
loc:@global_step*
_output_shapes
: 
�
global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
_output_shapes
: *
T0	
�
!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
_output_shapes
: : *
T0

a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
T0
*
_output_shapes
: 
_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
T0
*
_output_shapes
: 
h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
T0
*
_output_shapes
: 
b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
T0	*
_output_shapes
: 
�
global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
_class
loc:@global_step*
_output_shapes
: : *
T0	
�
global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
N*
_output_shapes
: : *
T0	
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
T0	*
_output_shapes
: 
�
tensors/component_0Const"/device:CPU:0*�
value�B�x"�ffffff@ffffff
@      @      �?333333�?�������?      �?ffffff@������@      �?333333@������@333333�?333333�?������@������@������@ffffff@333333@      �?������@ffffff@�������?ffffff@333333@      �?      @ffffff@      @      @������@      @      @      @333333@������@ffffff@ffffff@      �?ffffff�?������@ffffff@������@ffffff�?�������?      @�������?������@      �?      @ffffff�?������@      @333333�?������@333333@ffffff@ffffff@������@      @333333@ffffff@      @      @      @333333@�������?ffffff@      @      �?�������?ffffff@�������?ffffff@������@      �?ffffff@������@      @333333@      �?������@ffffff@������@ffffff@      �?333333@      @ffffff@�������?������@������@ffffff�?      �?333333@ffffff�?ffffff�?������@333333@�������?ffffff�?ffffff
@�������?ffffff@�������?ffffff�?      �?ffffff�?������@�������?������@ffffff@�������?      @�������?������@������@ffffff�?ffffff�?������@*
dtype0*
_output_shapes
:x
�
tensors/component_1Const"/device:CPU:0*�
value�B�x"�������@      �?333333�?�������?333333�?�������?�������?ffffff@ffffff�?�������?ffffff�?      �?�������?�������?������@�������?ffffff@������ @ffffff@�������?������ @������ @�������?ffffff�?�������?�������?333333�?�������?      �?333333�?�������?�������?      �?������ @�������?ffffff@       @333333@333333�?333333�?       @ffffff�?ffffff@�������?�������?      �?�������?ffffff@�������?�������?�������?ffffff�?�������?�������?ffffff�?�������?ffffff�?ffffff�?ffffff@      �?�������?333333@ffffff�?�������?      �?�������?�������?      @�������?�������?�������?�������?�������?ffffff@�������?�������?�������?�������?      �?ffffff�?�������?ffffff�?      �?�������?333333@�������?ffffff�?�������?�������?333333�?       @333333�?333333�?�������?������@333333�?�������?       @�������?�������?�������?      �?333333�?      �?�������?�������?�������?�������?�������?�������?�������?�������?�������?      @333333�?333333�?333333�?�������?�������?      �?*
dtype0*
_output_shapes
:x
�
tensors/component_2Const"/device:CPU:0*�
value�B�x"�������@      @������@������@������@������@������@������@������@ffffff@������@������@333333@������@������@333333@333333@ffffff@������@������@������@������@������@ffffff@������@������@333333@������@������@������@333333@ffffff@������@333333@������@������@      @333333@ffffff@333333@������@333333@������@ffffff@������@      @333333@������@ffffff@������@      @ffffff@ffffff@      @      @      @������@333333@������@      @ffffff@������@333333@������@������@333333@������@������@      @      @������@ffffff@      @������@ffffff@������@      @������@      @������@������@ffffff@      @ffffff@333333@������@333333@������@      @      @      @ffffff@ffffff@ffffff@      @ffffff@ffffff@������@������@ffffff@������@������@      @333333@      @������@333333@      @ffffff@333333@333333@������@      @333333@      @      @������@������@333333@      @*
dtype0*
_output_shapes
:x
�
tensors/component_3Const"/device:CPU:0*�
value�B�x"�ffffff@ffffff@      @������@ffffff@������	@333333@������@������@������@������@������@      @333333@ffffff@ffffff
@������	@      @������	@������@ffffff
@ffffff@333333@������@      @      @������@      @      @      @ffffff@      @������	@      @ffffff@������	@������	@ffffff@ffffff@      @ffffff@������@      @ffffff@������	@������@333333@������@������@������	@ffffff
@      @ffffff@������	@������	@      @ffffff@������@333333@       @      @������@      @������@������@333333@      @������@      @333333@������	@333333@      @      @      @������@333333@333333@333333@������@������@333333@ffffff@������@333333@������@ffffff@ffffff@������@      @      @ffffff@      @������@      @333333@������	@ffffff@������	@ffffff@      @333333@ffffff@������@333333@333333@������@������@333333@������@������@ffffff@      @ffffff
@      @������@      @333333@      @333333@*
dtype0*
_output_shapes
:x
�
tensors/component_4Const"/device:CPU:0*�
value�B�	x"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  *
dtype0	*
_output_shapes
:x
]
buffer_sizeConst"/device:CPU:0*
value
B	 R�*
dtype0	*
_output_shapes
: 
U
seedConst"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
V
seed2Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
_
countConst"/device:CPU:0*
valueB	 R
���������*
dtype0	*
_output_shapes
: 
[

batch_sizeConst"/device:CPU:0*
value	B	 Rd*
dtype0	*
_output_shapes
: 
�
OneShotIteratorOneShotIterator"/device:CPU:0*-
dataset_factoryR
_make_dataset_7dc53701*
shared_name *^
output_shapesM
K:���������:���������:���������:���������:���������*
	container *
output_types	
2	*
_output_shapes
: 
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
�
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*
output_types	
2	*^
output_shapesM
K:���������:���������:���������:���������:���������*_
_output_shapesM
K:���������:���������:���������:���������:���������
�
Ednn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Adnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims
ExpandDimsIteratorGetNextEdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloatCastAdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeShape>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ddnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeJdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
�
Fdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ddnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
>dnn/input_from_feature_columns/input_layer/PetalLength/ReshapeReshape>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloatDdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
�
Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims
ExpandDimsIteratorGetNext:1Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloatCast@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeShape=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Cdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeIdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
Ednn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
=dnn/input_from_feature_columns/input_layer/PetalWidth/ReshapeReshape=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloatCdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
�
Ednn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Adnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims
ExpandDimsIteratorGetNext:2Ednn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
>dnn/input_from_feature_columns/input_layer/SepalLength/ToFloatCastAdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeShape>dnn/input_from_feature_columns/input_layer/SepalLength/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ddnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeJdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
Fdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ddnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
>dnn/input_from_feature_columns/input_layer/SepalLength/ReshapeReshape>dnn/input_from_feature_columns/input_layer/SepalLength/ToFloatDdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Ddnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims
ExpandDimsIteratorGetNext:3Ddnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloatCast@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeShape=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloat*
out_type0*
_output_shapes
:*
T0
�
Idnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Cdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeIdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
�
Ednn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
=dnn/input_from_feature_columns/input_layer/SepalWidth/ReshapeReshape=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloatCdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
1dnn/input_from_feature_columns/input_layer/concatConcatV2>dnn/input_from_feature_columns/input_layer/PetalLength/Reshape=dnn/input_from_feature_columns/input_layer/PetalWidth/Reshape>dnn/input_from_feature_columns/input_layer/SepalLength/Reshape=dnn/input_from_feature_columns/input_layer/SepalWidth/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:���������
�
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *   �*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *   ?*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:*

seed *
T0
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
T0
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
�
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:*
T0
�
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:
�
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
�
?dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
�
5dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/ConstConst*
valueB
 *    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosFill?dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/shape_as_tensor5dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
�
dnn/hiddenlayer_0/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0*
_output_shapes

:
�
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
data_formatNHWC*'
_output_shapes
:���������*
T0
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:���������
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*'
_output_shapes
:���������*
T0
x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*'
_output_shapes
:���������*

DstT0
h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
_output_shapes
: *
T0
�
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: *
T0
�
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *�Kƾ*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *�K�>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:*

seed *
T0
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: *
T0
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
�
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
�
dnn/hiddenlayer_1/kernel/part_0
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container 
�
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
�
?dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
�
5dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/ConstConst*
valueB
 *    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosFill?dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/shape_as_tensor5dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/Const*

index_type0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:*
T0
�
dnn/hiddenlayer_1/bias/part_0
VariableV2*
dtype0*
_output_shapes
:*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:
�
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:*
T0
s
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
T0*
_output_shapes

:
�
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*'
_output_shapes
:���������
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:���������
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 
�
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *���*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *��?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
T0
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
dnn/logits/kernel/part_0
VariableV2*+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:
�
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0
�
8dnn/logits/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
.dnn/logits/bias/part_0/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
(dnn/logits/bias/part_0/Initializer/zerosFill8dnn/logits/bias/part_0/Initializer/zeros/shape_as_tensor.dnn/logits/bias/part_0/Initializer/zeros/Const*

index_type0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
T0
�
dnn/logits/bias/part_0
VariableV2*)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
�
dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes

:
�
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
_output_shapes
:*
T0
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
data_formatNHWC*'
_output_shapes
:���������*
T0
]
dnn/zero_fraction_2/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_2/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_2/zero*
T0*'
_output_shapes
:���������
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

SrcT0
*'
_output_shapes
:���������*

DstT0
j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
_output_shapes
: *
T0
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
�
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
g
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
out_type0*
_output_shapes
:*
T0
k
)dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/class_idsArgMaxdnn/logits/BiasAdd(dnn/head/predictions/class_ids/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
n
#dnn/head/predictions/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*'
_output_shapes
:���������*
shortest( *
	precision���������*
T0	*

fill *

scientific( *
width���������
s
"dnn/head/predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*'
_output_shapes
:���������
i
dnn/head/labels/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/labels/ExpandDims
ExpandDimsIteratorGetNext:4dnn/head/labels/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
o
dnn/head/labels/ShapeShapednn/head/labels/ExpandDims*
T0	*
out_type0*
_output_shapes
:
i
dnn/head/labels/Shape_1Shapednn/logits/BiasAdd*
out_type0*
_output_shapes
:*
T0
k
)dnn/head/labels/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_rank_at_least/ShapeShapednn/head/labels/ExpandDims*
out_type0*
_output_shapes
:*
T0	
[
Sdnn/head/labels/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/labels/assert_rank_at_least/static_checks_determined_all_okNoOp
�
#dnn/head/labels/strided_slice/stackConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB: *
dtype0*
_output_shapes
:
�
%dnn/head/labels/strided_slice/stack_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:
���������*
dtype0*
_output_shapes
:
�
%dnn/head/labels/strided_slice/stack_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/labels/strided_sliceStridedSlicednn/head/labels/Shape_1#dnn/head/labels/strided_slice/stack%dnn/head/labels/strided_slice/stack_1%dnn/head/labels/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
�
dnn/head/labels/concat/values_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/labels/concat/axisConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
value	B : *
dtype0*
_output_shapes
: 
�
dnn/head/labels/concatConcatV2dnn/head/labels/strided_slicednn/head/labels/concat/values_1dnn/head/labels/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:

"dnn/head/labels/assert_equal/EqualEqualdnn/head/labels/concatdnn/head/labels/Shape*
T0*
_output_shapes
:
�
"dnn/head/labels/assert_equal/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB: *
dtype0*
_output_shapes
:
�
 dnn/head/labels/assert_equal/AllAll"dnn/head/labels/assert_equal/Equal"dnn/head/labels/assert_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
)dnn/head/labels/assert_equal/Assert/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
+dnn/head/labels/assert_equal/Assert/Const_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_0ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_0dnn/head/labels/concat1dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/Shape*
	summarize*
T
2
�
dnn/head/labelsIdentitydnn/head/labels/ExpandDimsE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok+^dnn/head/labels/assert_equal/Assert/Assert*'
_output_shapes
:���������*
T0	
]
dnn/head/assert_range/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
&dnn/head/assert_range/assert_less/LessLessdnn/head/labelsdnn/head/assert_range/Const*'
_output_shapes
:���������*
T0	
x
'dnn/head/assert_range/assert_less/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
%dnn/head/assert_range/assert_less/AllAll&dnn/head/assert_range/assert_less/Less'dnn/head/assert_range/assert_less/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
.dnn/head/assert_range/assert_less/Assert/ConstConst*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_1Const*;
value2B0 B*Condition x < y did not hold element-wise:*
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_2Const*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_3Const*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/SwitchSwitch%dnn/head/assert_range/assert_less/All%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: : 
�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_fIdentity;dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_idIdentity%dnn/head/assert_range/assert_less/All*
_output_shapes
: *
T0

�
9dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOpNoOp>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t
�
Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependencyIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:^dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOp*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t*
_output_shapes
: *
T0

�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*;
value2B0 B*Condition x < y did not hold element-wise:*
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/AssertAssertBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2*
	summarize*
T

2		
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchSwitch%dnn/head/assert_range/assert_less/All<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0
*8
_class.
,*loc:@dnn/head/assert_range/assert_less/All*
_output_shapes
: : 
�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/labels<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*"
_class
loc:@dnn/head/labels*:
_output_shapes(
&:���������:���������*
T0	
�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/assert_range/Const<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0	*.
_class$
" loc:@dnn/head/assert_range/Const*
_output_shapes
: : 
�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Identity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f<^dnn/head/assert_range/assert_less/Assert/AssertGuard/Assert*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*
_output_shapes
: *
T0

�
:dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeMergeIdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
q
/dnn/head/assert_range/assert_non_negative/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
Ednn/head/assert_range/assert_non_negative/assert_less_equal/LessEqual	LessEqual/dnn/head/assert_range/assert_non_negative/Constdnn/head/labels*
T0	*'
_output_shapes
:���������
�
Adnn/head/assert_range/assert_non_negative/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllAllEdnn/head/assert_range/assert_non_negative/assert_less_equal/LessEqualAdnn/head/assert_range/assert_non_negative/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Hdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/ConstConst*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_1Const*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_2Const*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/All?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : *
T0

�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fIdentityUdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch*
_output_shapes
: *
T0

�
Vdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_idIdentity?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: 
�
Sdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
�
adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tT^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: *
T0

�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
	summarize*
T
2	
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : 
�
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/labelsVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0	*"
_class
loc:@dnn/head/labels*:
_output_shapes(
&:���������:���������
�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
dnn/head/assert_range/IdentityIdentitydnn/head/labels;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*
T0	*'
_output_shapes
:���������
�
Odnn/head/sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/SqueezeSqueezednn/head/assert_range/Identity*#
_output_shapes
:���������*
squeeze_dims

���������*
T0	
u
0dnn/head/sparse_softmax_cross_entropy_loss/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/sparse_softmax_cross_entropy_loss/xentropy/ShapeShapeOdnn/head/sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/Squeeze*
out_type0*
_output_shapes
:*
T0	
�
<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAddOdnn/head/sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/Squeeze*
T0*6
_output_shapes$
":���������:���������*
Tlabels0	
�
Mdnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ldnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ldnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy*
out_type0*
_output_shapes
:*
T0
�
Kdnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
c
[dnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
.dnn/head/sparse_softmax_cross_entropy_loss/MulMul<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy0dnn/head/sparse_softmax_cross_entropy_loss/Const\^dnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*#
_output_shapes
:���������
b
dnn/head/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/ExpandDims
ExpandDims.dnn/head/sparse_softmax_cross_entropy_loss/Muldnn/head/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
x
3dnn/head/weighted_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
|
9dnn/head/weighted_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
z
8dnn/head/weighted_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
8dnn/head/weighted_loss/assert_broadcastable/values/shapeShapednn/head/ExpandDims*
out_type0*
_output_shapes
:*
T0
y
7dnn/head/weighted_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
O
Gdnn/head/weighted_loss/assert_broadcastable/static_scalar_check_successNoOp
�
"dnn/head/weighted_loss/ToFloat_1/xConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/weighted_loss/MulMuldnn/head/ExpandDims"dnn/head/weighted_loss/ToFloat_1/x*
T0*'
_output_shapes
:���������
�
dnn/head/weighted_loss/ConstConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/weighted_loss/SumSumdnn/head/weighted_loss/Muldnn/head/weighted_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
k
dnn/head/ones_like/ShapeShapednn/head/ExpandDims*
out_type0*
_output_shapes
:*
T0
]
dnn/head/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/ones_likeFilldnn/head/ones_like/Shapednn/head/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
S
dnn/head/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
i
dnn/head/mulMuldnn/head/mul/xdnn/head/ones_like*'
_output_shapes
:���������*
T0
_
dnn/head/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
o
dnn/head/SumSumdnn/head/muldnn/head/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
f
dnn/head/truedivRealDivdnn/head/weighted_loss/Sumdnn/head/Sum*
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
]
lossScalarSummary	loss/tagsdnn/head/weighted_loss/Sum*
_output_shapes
: *
T0
^
average_loss/tagsConst*
valueB Baverage_loss*
dtype0*
_output_shapes
: 
c
average_lossScalarSummaryaverage_loss/tagsdnn/head/truediv*
_output_shapes
: *
T0
V
dnn/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
\
dnn/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
{
dnn/gradients/FillFilldnn/gradients/Shapednn/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
;dnn/gradients/dnn/head/weighted_loss/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
5dnn/gradients/dnn/head/weighted_loss/Sum_grad/ReshapeReshapednn/gradients/Fill;dnn/gradients/dnn/head/weighted_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
3dnn/gradients/dnn/head/weighted_loss/Sum_grad/ShapeShapednn/head/weighted_loss/Mul*
out_type0*
_output_shapes
:*
T0
�
2dnn/gradients/dnn/head/weighted_loss/Sum_grad/TileTile5dnn/gradients/dnn/head/weighted_loss/Sum_grad/Reshape3dnn/gradients/dnn/head/weighted_loss/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
3dnn/gradients/dnn/head/weighted_loss/Mul_grad/ShapeShapednn/head/ExpandDims*
T0*
out_type0*
_output_shapes
:
x
5dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Cdnn/gradients/dnn/head/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape5dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1dnn/gradients/dnn/head/weighted_loss/Mul_grad/MulMul2dnn/gradients/dnn/head/weighted_loss/Sum_grad/Tile"dnn/head/weighted_loss/ToFloat_1/x*
T0*'
_output_shapes
:���������
�
1dnn/gradients/dnn/head/weighted_loss/Mul_grad/SumSum1dnn/gradients/dnn/head/weighted_loss/Mul_grad/MulCdnn/gradients/dnn/head/weighted_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
5dnn/gradients/dnn/head/weighted_loss/Mul_grad/ReshapeReshape1dnn/gradients/dnn/head/weighted_loss/Mul_grad/Sum3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Mul_1Muldnn/head/ExpandDims2dnn/gradients/dnn/head/weighted_loss/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Sum_1Sum3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Mul_1Ednn/gradients/dnn/head/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1Reshape3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Sum_15dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>dnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/group_depsNoOp6^dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape8^dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1
�
Fdnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/control_dependencyIdentity5dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape?^dnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape*'
_output_shapes
:���������
�
Hdnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/control_dependency_1Identity7dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1?^dnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/group_deps*J
_class@
><loc:@dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
�
,dnn/gradients/dnn/head/ExpandDims_grad/ShapeShape.dnn/head/sparse_softmax_cross_entropy_loss/Mul*
out_type0*
_output_shapes
:*
T0
�
.dnn/gradients/dnn/head/ExpandDims_grad/ReshapeReshapeFdnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/control_dependency,dnn/gradients/dnn/head/ExpandDims_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
Gdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ShapeShape<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy*
T0*
out_type0*
_output_shapes
:
�
Idnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Wdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ShapeIdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ednn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/MulMul.dnn/gradients/dnn/head/ExpandDims_grad/Reshape0dnn/head/sparse_softmax_cross_entropy_loss/Const*#
_output_shapes
:���������*
T0
�
Ednn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/SumSumEdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/MulWdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Idnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ReshapeReshapeEdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/SumGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
Gdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Mul_1Mul<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy.dnn/gradients/dnn/head/ExpandDims_grad/Reshape*
T0*#
_output_shapes
:���������
�
Gdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Sum_1SumGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Mul_1Ydnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Kdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1ReshapeGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Sum_1Idnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
Rdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpJ^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ReshapeL^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Zdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityIdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ReshapeS^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������
�
\dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityKdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1S^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*^
_classT
RPloc:@dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
�
dnn/gradients/zeros_like	ZerosLike>dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy:1*'
_output_shapes
:���������*
T0
�
_dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/PreventGradientPreventGradient>dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:���������
�
^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Zdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims
ExpandDimsZdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Sdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mulMulZdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims_dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/PreventGradient*
T0*'
_output_shapes
:���������
�
1dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGradBiasAddGradSdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul*
T0*
data_formatNHWC*
_output_shapes
:
�
6dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_depsNoOpT^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul2^dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad
�
>dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencyIdentitySdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul7^dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*f
_class\
ZXloc:@dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul*'
_output_shapes
:���������*
T0
�
@dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1Identity1dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad7^dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
+dnn/gradients/dnn/logits/MatMul_grad/MatMulMatMul>dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencydnn/logits/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
-dnn/gradients/dnn/logits/MatMul_grad/MatMul_1MatMuldnn/hiddenlayer_1/Relu>dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
5dnn/gradients/dnn/logits/MatMul_grad/tuple/group_depsNoOp,^dnn/gradients/dnn/logits/MatMul_grad/MatMul.^dnn/gradients/dnn/logits/MatMul_grad/MatMul_1
�
=dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencyIdentity+dnn/gradients/dnn/logits/MatMul_grad/MatMul6^dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*>
_class4
20loc:@dnn/gradients/dnn/logits/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
�
?dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1Identity-dnn/gradients/dnn/logits/MatMul_grad/MatMul_16^dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*@
_class6
42loc:@dnn/gradients/dnn/logits/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
2dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGradReluGrad=dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencydnn/hiddenlayer_1/Relu*
T0*'
_output_shapes
:���������
�
8dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradBiasAddGrad2dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
=dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad9^dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad
�
Ednn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad>^dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad*'
_output_shapes
:���������*
T0
�
Gdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1Identity8dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad>^dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
2dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulMatMulEdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_1/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
4dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1MatMuldnn/hiddenlayer_0/ReluEdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
<dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul5^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1
�
Ddnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul=^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Fdnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1Identity4dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1=^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
2dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGradReluGradDdnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencydnn/hiddenlayer_0/Relu*
T0*'
_output_shapes
:���������
�
8dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradBiasAddGrad2dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
=dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad9^dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad
�
Ednn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad>^dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad*'
_output_shapes
:���������
�
Gdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1Identity8dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad>^dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
2dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulMatMulEdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_0/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
4dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1MatMul1dnn/input_from_feature_columns/input_layer/concatEdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
<dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul5^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1
�
Ddnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul=^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
�
Fdnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1Identity4dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1=^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
=dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:
�
+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:
�
2dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad=dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:
�
0dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
�
;dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
�
)dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad
VariableV2*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
�
0dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/AssignAssign)dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad;dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:
�
.dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/readIdentity)dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:*
T0
�
=dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
�
+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
2dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad=dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:
�
0dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
�
;dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
�
)dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
0dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/AssignAssign)dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad;dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:
�
.dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/readIdentity)dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:*
T0
�
6dnn/dnn/logits/kernel/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
�
$dnn/dnn/logits/kernel/part_0/Adagrad
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container 
�
+dnn/dnn/logits/kernel/part_0/Adagrad/AssignAssign$dnn/dnn/logits/kernel/part_0/Adagrad6dnn/dnn/logits/kernel/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:
�
)dnn/dnn/logits/kernel/part_0/Adagrad/readIdentity$dnn/dnn/logits/kernel/part_0/Adagrad*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0
�
4dnn/dnn/logits/bias/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
"dnn/dnn/logits/bias/part_0/Adagrad
VariableV2*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
�
)dnn/dnn/logits/bias/part_0/Adagrad/AssignAssign"dnn/dnn/logits/bias/part_0/Adagrad4dnn/dnn/logits/bias/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
�
'dnn/dnn/logits/bias/part_0/Adagrad/readIdentity"dnn/dnn/logits/bias/part_0/Adagrad*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
T0
^
dnn/Adagrad/learning_rateConst*
valueB
 *��L=*
dtype0*
_output_shapes
: 
�
?dnn/Adagrad/update_dnn/hiddenlayer_0/kernel/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/kernel/part_0+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagraddnn/Adagrad/learning_rateFdnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:*
use_locking( *
T0
�
=dnn/Adagrad/update_dnn/hiddenlayer_0/bias/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/bias/part_0)dnn/dnn/hiddenlayer_0/bias/part_0/Adagraddnn/Adagrad/learning_rateGdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
�
?dnn/Adagrad/update_dnn/hiddenlayer_1/kernel/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/kernel/part_0+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagraddnn/Adagrad/learning_rateFdnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:*
use_locking( *
T0
�
=dnn/Adagrad/update_dnn/hiddenlayer_1/bias/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/bias/part_0)dnn/dnn/hiddenlayer_1/bias/part_0/Adagraddnn/Adagrad/learning_rateGdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:
�
8dnn/Adagrad/update_dnn/logits/kernel/part_0/ApplyAdagradApplyAdagraddnn/logits/kernel/part_0$dnn/dnn/logits/kernel/part_0/Adagraddnn/Adagrad/learning_rate?dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
use_locking( *
T0
�
6dnn/Adagrad/update_dnn/logits/bias/part_0/ApplyAdagradApplyAdagraddnn/logits/bias/part_0"dnn/dnn/logits/bias/part_0/Adagraddnn/Adagrad/learning_rate@dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
use_locking( *
T0
�
dnn/Adagrad/updateNoOp@^dnn/Adagrad/update_dnn/hiddenlayer_0/kernel/part_0/ApplyAdagrad>^dnn/Adagrad/update_dnn/hiddenlayer_0/bias/part_0/ApplyAdagrad@^dnn/Adagrad/update_dnn/hiddenlayer_1/kernel/part_0/ApplyAdagrad>^dnn/Adagrad/update_dnn/hiddenlayer_1/bias/part_0/ApplyAdagrad9^dnn/Adagrad/update_dnn/logits/kernel/part_0/ApplyAdagrad7^dnn/Adagrad/update_dnn/logits/bias/part_0/ApplyAdagrad
�
dnn/Adagrad/valueConst^dnn/Adagrad/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
dnn/Adagrad	AssignAddglobal_stepdnn/Adagrad/value*
use_locking( *
T0	*
_class
loc:@global_step*
_output_shapes
: �3
�3
-
_make_dataset_7dc53701
BatchDataset��
&TensorSliceDataset/tensors/component_0Const*�
value�B�x"�ffffff@ffffff
@      @      �?333333�?�������?      �?ffffff@������@      �?333333@������@333333�?333333�?������@������@������@ffffff@333333@      �?������@ffffff@�������?ffffff@333333@      �?      @ffffff@      @      @������@      @      @      @333333@������@ffffff@ffffff@      �?ffffff�?������@ffffff@������@ffffff�?�������?      @�������?������@      �?      @ffffff�?������@      @333333�?������@333333@ffffff@ffffff@������@      @333333@ffffff@      @      @      @333333@�������?ffffff@      @      �?�������?ffffff@�������?ffffff@������@      �?ffffff@������@      @333333@      �?������@ffffff@������@ffffff@      �?333333@      @ffffff@�������?������@������@ffffff�?      �?333333@ffffff�?ffffff�?������@333333@�������?ffffff�?ffffff
@�������?ffffff@�������?ffffff�?      �?ffffff�?������@�������?������@ffffff@�������?      @�������?������@������@ffffff�?ffffff�?������@*
dtype0�
&TensorSliceDataset/tensors/component_1Const*�
value�B�x"�������@      �?333333�?�������?333333�?�������?�������?ffffff@ffffff�?�������?ffffff�?      �?�������?�������?������@�������?ffffff@������ @ffffff@�������?������ @������ @�������?ffffff�?�������?�������?333333�?�������?      �?333333�?�������?�������?      �?������ @�������?ffffff@       @333333@333333�?333333�?       @ffffff�?ffffff@�������?�������?      �?�������?ffffff@�������?�������?�������?ffffff�?�������?�������?ffffff�?�������?ffffff�?ffffff�?ffffff@      �?�������?333333@ffffff�?�������?      �?�������?�������?      @�������?�������?�������?�������?�������?ffffff@�������?�������?�������?�������?      �?ffffff�?�������?ffffff�?      �?�������?333333@�������?ffffff�?�������?�������?333333�?       @333333�?333333�?�������?������@333333�?�������?       @�������?�������?�������?      �?333333�?      �?�������?�������?�������?�������?�������?�������?�������?�������?�������?      @333333�?333333�?333333�?�������?�������?      �?*
dtype0�
&TensorSliceDataset/tensors/component_2Const*�
value�B�x"�������@      @������@������@������@������@������@������@������@ffffff@������@������@333333@������@������@333333@333333@ffffff@������@������@������@������@������@ffffff@������@������@333333@������@������@������@333333@ffffff@������@333333@������@������@      @333333@ffffff@333333@������@333333@������@ffffff@������@      @333333@������@ffffff@������@      @ffffff@ffffff@      @      @      @������@333333@������@      @ffffff@������@333333@������@������@333333@������@������@      @      @������@ffffff@      @������@ffffff@������@      @������@      @������@������@ffffff@      @ffffff@333333@������@333333@������@      @      @      @ffffff@ffffff@ffffff@      @ffffff@ffffff@������@������@ffffff@������@������@      @333333@      @������@333333@      @ffffff@333333@333333@������@      @333333@      @      @������@������@333333@      @*
dtype0�
&TensorSliceDataset/tensors/component_3Const*�
value�B�x"�ffffff@ffffff@      @������@ffffff@������	@333333@������@������@������@������@������@      @333333@ffffff@ffffff
@������	@      @������	@������@ffffff
@ffffff@333333@������@      @      @������@      @      @      @ffffff@      @������	@      @ffffff@������	@������	@ffffff@ffffff@      @ffffff@������@      @ffffff@������	@������@333333@������@������@������	@ffffff
@      @ffffff@������	@������	@      @ffffff@������@333333@       @      @������@      @������@������@333333@      @������@      @333333@������	@333333@      @      @      @������@333333@333333@333333@������@������@333333@ffffff@������@333333@������@ffffff@ffffff@������@      @      @ffffff@      @������@      @333333@������	@ffffff@������	@ffffff@      @333333@ffffff@������@333333@333333@������@������@333333@������@������@ffffff@      @ffffff
@      @������@      @333333@      @333333@*
dtype0�
&TensorSliceDataset/tensors/component_4Const*�
value�B�	x"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  *
dtype0	�
TensorSliceDatasetTensorSliceDataset/TensorSliceDataset/tensors/component_0:output:0/TensorSliceDataset/tensors/component_1:output:0/TensorSliceDataset/tensors/component_2:output:0/TensorSliceDataset/tensors/component_3:output:0/TensorSliceDataset/tensors/component_4:output:0*
Toutput_types	
2	*
output_shapes

: : : : : E
ShuffleDataset/buffer_sizeConst*
value
B	 R�*
dtype0	=
ShuffleDataset/seedConst*
value	B	 R *
dtype0	>
ShuffleDataset/seed2Const*
value	B	 R *
dtype0	�
ShuffleDatasetShuffleDatasetTensorSliceDataset:handle:0#ShuffleDataset/buffer_size:output:0ShuffleDataset/seed:output:0ShuffleDataset/seed2:output:0*
reshuffle_each_iteration(*
output_types	
2	*
output_shapes

: : : : : F
RepeatDataset/countConst*
valueB	 R
���������*
dtype0	�
RepeatDatasetRepeatDatasetShuffleDataset:handle:0RepeatDataset/count:output:0*
output_types	
2	*
output_shapes

: : : : : A
BatchDataset/batch_sizeConst*
value	B	 Rd*
dtype0	�
BatchDatasetBatchDatasetRepeatDataset:handle:0 BatchDataset/batch_size:output:0*
output_types	
2	*^
output_shapesM
K:���������:���������:���������:���������:���������"%
BatchDatasetBatchDataset:handle:0"ŉQ3D�     >�/3	R^r����AJ��
�&�%
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
�
ApplyAdagrad
var"T�
accum"T�
lr"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
�
AsString

input"T

output"
Ttype:
	2	
"
	precisionint���������"

scientificbool( "
shortestbool( "
widthint���������"
fillstring 
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
�
IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0�
C
IteratorToStringHandle
resource_handle
string_handle�
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneShotIterator

handle"
dataset_factoryfunc"
output_types
list(type)(0" 
output_shapeslist(shape)(0"
	containerstring "
shared_namestring �
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
\
	RefSwitch
data"T�
pred

output_false"T�
output_true"T�"	
Ttype�
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.7.02unknownƫ
�
-global_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@global_step*
dtype0*
_output_shapes
: 
�
#global_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
global_step/Initializer/zerosFill-global_step/Initializer/zeros/shape_as_tensor#global_step/Initializer/zeros/Const*
T0	*

index_type0*
_class
loc:@global_step*
_output_shapes
: 
�
global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
_output_shapes
: *
T0	
�
!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
T0
*
_output_shapes
: : 
a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
T0
*
_output_shapes
: 
_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
_output_shapes
: *
T0

h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
T0
*
_output_shapes
: 
b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
T0	*
_output_shapes
: 
�
global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 
�
global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
T0	*
N*
_output_shapes
: : 
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
_output_shapes
: *
T0	
�
tensors/component_0Const"/device:CPU:0*�
value�B�x"�ffffff@ffffff
@      @      �?333333�?�������?      �?ffffff@������@      �?333333@������@333333�?333333�?������@������@������@ffffff@333333@      �?������@ffffff@�������?ffffff@333333@      �?      @ffffff@      @      @������@      @      @      @333333@������@ffffff@ffffff@      �?ffffff�?������@ffffff@������@ffffff�?�������?      @�������?������@      �?      @ffffff�?������@      @333333�?������@333333@ffffff@ffffff@������@      @333333@ffffff@      @      @      @333333@�������?ffffff@      @      �?�������?ffffff@�������?ffffff@������@      �?ffffff@������@      @333333@      �?������@ffffff@������@ffffff@      �?333333@      @ffffff@�������?������@������@ffffff�?      �?333333@ffffff�?ffffff�?������@333333@�������?ffffff�?ffffff
@�������?ffffff@�������?ffffff�?      �?ffffff�?������@�������?������@ffffff@�������?      @�������?������@������@ffffff�?ffffff�?������@*
dtype0*
_output_shapes
:x
�
tensors/component_1Const"/device:CPU:0*�
value�B�x"�������@      �?333333�?�������?333333�?�������?�������?ffffff@ffffff�?�������?ffffff�?      �?�������?�������?������@�������?ffffff@������ @ffffff@�������?������ @������ @�������?ffffff�?�������?�������?333333�?�������?      �?333333�?�������?�������?      �?������ @�������?ffffff@       @333333@333333�?333333�?       @ffffff�?ffffff@�������?�������?      �?�������?ffffff@�������?�������?�������?ffffff�?�������?�������?ffffff�?�������?ffffff�?ffffff�?ffffff@      �?�������?333333@ffffff�?�������?      �?�������?�������?      @�������?�������?�������?�������?�������?ffffff@�������?�������?�������?�������?      �?ffffff�?�������?ffffff�?      �?�������?333333@�������?ffffff�?�������?�������?333333�?       @333333�?333333�?�������?������@333333�?�������?       @�������?�������?�������?      �?333333�?      �?�������?�������?�������?�������?�������?�������?�������?�������?�������?      @333333�?333333�?333333�?�������?�������?      �?*
dtype0*
_output_shapes
:x
�
tensors/component_2Const"/device:CPU:0*�
value�B�x"�������@      @������@������@������@������@������@������@������@ffffff@������@������@333333@������@������@333333@333333@ffffff@������@������@������@������@������@ffffff@������@������@333333@������@������@������@333333@ffffff@������@333333@������@������@      @333333@ffffff@333333@������@333333@������@ffffff@������@      @333333@������@ffffff@������@      @ffffff@ffffff@      @      @      @������@333333@������@      @ffffff@������@333333@������@������@333333@������@������@      @      @������@ffffff@      @������@ffffff@������@      @������@      @������@������@ffffff@      @ffffff@333333@������@333333@������@      @      @      @ffffff@ffffff@ffffff@      @ffffff@ffffff@������@������@ffffff@������@������@      @333333@      @������@333333@      @ffffff@333333@333333@������@      @333333@      @      @������@������@333333@      @*
dtype0*
_output_shapes
:x
�
tensors/component_3Const"/device:CPU:0*�
value�B�x"�ffffff@ffffff@      @������@ffffff@������	@333333@������@������@������@������@������@      @333333@ffffff@ffffff
@������	@      @������	@������@ffffff
@ffffff@333333@������@      @      @������@      @      @      @ffffff@      @������	@      @ffffff@������	@������	@ffffff@ffffff@      @ffffff@������@      @ffffff@������	@������@333333@������@������@������	@ffffff
@      @ffffff@������	@������	@      @ffffff@������@333333@       @      @������@      @������@������@333333@      @������@      @333333@������	@333333@      @      @      @������@333333@333333@333333@������@������@333333@ffffff@������@333333@������@ffffff@ffffff@������@      @      @ffffff@      @������@      @333333@������	@ffffff@������	@ffffff@      @333333@ffffff@������@333333@333333@������@������@333333@������@������@ffffff@      @ffffff
@      @������@      @333333@      @333333@*
dtype0*
_output_shapes
:x
�
tensors/component_4Const"/device:CPU:0*�
value�B�	x"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  *
dtype0	*
_output_shapes
:x
]
buffer_sizeConst"/device:CPU:0*
value
B	 R�*
dtype0	*
_output_shapes
: 
U
seedConst"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
V
seed2Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
_
countConst"/device:CPU:0*
valueB	 R
���������*
dtype0	*
_output_shapes
: 
[

batch_sizeConst"/device:CPU:0*
value	B	 Rd*
dtype0	*
_output_shapes
: 
�
OneShotIteratorOneShotIterator"/device:CPU:0*
	container *
output_types	
2	*
_output_shapes
: *-
dataset_factoryR
_make_dataset_7dc53701*
shared_name *^
output_shapesM
K:���������:���������:���������:���������:���������
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
�
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*_
_output_shapesM
K:���������:���������:���������:���������:���������*
output_types	
2	*^
output_shapesM
K:���������:���������:���������:���������:���������
�
Ednn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Adnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims
ExpandDimsIteratorGetNextEdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloatCastAdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims*'
_output_shapes
:���������*

DstT0*

SrcT0
�
<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeShape>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ddnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeJdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
Fdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ddnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
>dnn/input_from_feature_columns/input_layer/PetalLength/ReshapeReshape>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloatDdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
�
Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims
ExpandDimsIteratorGetNext:1Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloatCast@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeShape=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Cdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeIdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
�
Ednn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
=dnn/input_from_feature_columns/input_layer/PetalWidth/ReshapeReshape=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloatCdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
�
Ednn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Adnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims
ExpandDimsIteratorGetNext:2Ednn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
>dnn/input_from_feature_columns/input_layer/SepalLength/ToFloatCastAdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims*'
_output_shapes
:���������*

DstT0*

SrcT0
�
<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeShape>dnn/input_from_feature_columns/input_layer/SepalLength/ToFloat*
out_type0*
_output_shapes
:*
T0
�
Jdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ddnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeJdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
Fdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ddnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
>dnn/input_from_feature_columns/input_layer/SepalLength/ReshapeReshape>dnn/input_from_feature_columns/input_layer/SepalLength/ToFloatDdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
�
Ddnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims
ExpandDimsIteratorGetNext:3Ddnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloatCast@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims*'
_output_shapes
:���������*

DstT0*

SrcT0
�
;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeShape=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Cdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeIdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
Ednn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
=dnn/input_from_feature_columns/input_layer/SepalWidth/ReshapeReshape=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloatCdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
1dnn/input_from_feature_columns/input_layer/concatConcatV2>dnn/input_from_feature_columns/input_layer/PetalLength/Reshape=dnn/input_from_feature_columns/input_layer/PetalWidth/Reshape>dnn/input_from_feature_columns/input_layer/SepalLength/Reshape=dnn/input_from_feature_columns/input_layer/SepalWidth/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:���������
�
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *   �*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *   ?*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
T0
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
�
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:*
T0
�
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:
�
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:
�
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
�
?dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
�
5dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/ConstConst*
valueB
 *    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosFill?dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/shape_as_tensor5dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
�
dnn/hiddenlayer_0/bias/part_0
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container 
�
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:*
T0
s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
_output_shapes

:*
T0
�
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes
:*
T0
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:���������
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:���������
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*'
_output_shapes
:���������*
T0
x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 
�
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: *
T0
�
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *�Kƾ*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *�K�>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: *
T0
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:*
T0
�
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:*
T0
�
dnn/hiddenlayer_1/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:*
T0
�
?dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
�
5dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/ConstConst*
valueB
 *    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosFill?dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/shape_as_tensor5dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/Const*

index_type0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:*
T0
�
dnn/hiddenlayer_1/bias/part_0
VariableV2*
dtype0*
_output_shapes
:*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:
�
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:
s
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
_output_shapes

:*
T0
�
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
data_formatNHWC*'
_output_shapes
:���������*
T0
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:���������
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
_output_shapes
: *
T0
�
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: *
T0
�
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *���*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *��?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
dnn/logits/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:
�
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
8dnn/logits/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
.dnn/logits/bias/part_0/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
(dnn/logits/bias/part_0/Initializer/zerosFill8dnn/logits/bias/part_0/Initializer/zeros/shape_as_tensor.dnn/logits/bias/part_0/Initializer/zeros/Const*

index_type0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
T0
�
dnn/logits/bias/part_0
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:
�
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
_output_shapes

:*
T0
�
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
_output_shapes
:*
T0
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*
data_formatNHWC*'
_output_shapes
:���������
]
dnn/zero_fraction_2/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_2/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_2/zero*'
_output_shapes
:���������*
T0
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
�
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
g
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
k
)dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/class_idsArgMaxdnn/logits/BiasAdd(dnn/head/predictions/class_ids/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
n
#dnn/head/predictions/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*
T0	*

fill *

scientific( *
width���������*'
_output_shapes
:���������*
shortest( *
	precision���������
s
"dnn/head/predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*'
_output_shapes
:���������*
T0
i
dnn/head/labels/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/labels/ExpandDims
ExpandDimsIteratorGetNext:4dnn/head/labels/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0	
o
dnn/head/labels/ShapeShapednn/head/labels/ExpandDims*
T0	*
out_type0*
_output_shapes
:
i
dnn/head/labels/Shape_1Shapednn/logits/BiasAdd*
out_type0*
_output_shapes
:*
T0
k
)dnn/head/labels/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_rank_at_least/ShapeShapednn/head/labels/ExpandDims*
T0	*
out_type0*
_output_shapes
:
[
Sdnn/head/labels/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/labels/assert_rank_at_least/static_checks_determined_all_okNoOp
�
#dnn/head/labels/strided_slice/stackConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB: *
dtype0*
_output_shapes
:
�
%dnn/head/labels/strided_slice/stack_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:
���������*
dtype0*
_output_shapes
:
�
%dnn/head/labels/strided_slice/stack_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/labels/strided_sliceStridedSlicednn/head/labels/Shape_1#dnn/head/labels/strided_slice/stack%dnn/head/labels/strided_slice/stack_1%dnn/head/labels/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
�
dnn/head/labels/concat/values_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/labels/concat/axisConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
value	B : *
dtype0*
_output_shapes
: 
�
dnn/head/labels/concatConcatV2dnn/head/labels/strided_slicednn/head/labels/concat/values_1dnn/head/labels/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:

"dnn/head/labels/assert_equal/EqualEqualdnn/head/labels/concatdnn/head/labels/Shape*
_output_shapes
:*
T0
�
"dnn/head/labels/assert_equal/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB: *
dtype0*
_output_shapes
:
�
 dnn/head/labels/assert_equal/AllAll"dnn/head/labels/assert_equal/Equal"dnn/head/labels/assert_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
)dnn/head/labels/assert_equal/Assert/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
+dnn/head/labels/assert_equal/Assert/Const_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_0ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_0dnn/head/labels/concat1dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/Shape*
	summarize*
T
2
�
dnn/head/labelsIdentitydnn/head/labels/ExpandDimsE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok+^dnn/head/labels/assert_equal/Assert/Assert*'
_output_shapes
:���������*
T0	
]
dnn/head/assert_range/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
&dnn/head/assert_range/assert_less/LessLessdnn/head/labelsdnn/head/assert_range/Const*
T0	*'
_output_shapes
:���������
x
'dnn/head/assert_range/assert_less/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
%dnn/head/assert_range/assert_less/AllAll&dnn/head/assert_range/assert_less/Less'dnn/head/assert_range/assert_less/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
.dnn/head/assert_range/assert_less/Assert/ConstConst*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_1Const*;
value2B0 B*Condition x < y did not hold element-wise:*
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_2Const*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_3Const*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/SwitchSwitch%dnn/head/assert_range/assert_less/All%dnn/head/assert_range/assert_less/All*
_output_shapes
: : *
T0

�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_fIdentity;dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch*
_output_shapes
: *
T0

�
<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_idIdentity%dnn/head/assert_range/assert_less/All*
_output_shapes
: *
T0

�
9dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOpNoOp>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t
�
Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependencyIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:^dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOp*
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*;
value2B0 B*Condition x < y did not hold element-wise:*
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/AssertAssertBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2*
	summarize*
T

2		
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchSwitch%dnn/head/assert_range/assert_less/All<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*8
_class.
,*loc:@dnn/head/assert_range/assert_less/All*
_output_shapes
: : *
T0

�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/labels<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*"
_class
loc:@dnn/head/labels*:
_output_shapes(
&:���������:���������*
T0	
�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/assert_range/Const<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0	*.
_class$
" loc:@dnn/head/assert_range/Const*
_output_shapes
: : 
�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Identity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f<^dnn/head/assert_range/assert_less/Assert/AssertGuard/Assert*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*
_output_shapes
: *
T0

�
:dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeMergeIdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
q
/dnn/head/assert_range/assert_non_negative/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
Ednn/head/assert_range/assert_non_negative/assert_less_equal/LessEqual	LessEqual/dnn/head/assert_range/assert_non_negative/Constdnn/head/labels*
T0	*'
_output_shapes
:���������
�
Adnn/head/assert_range/assert_non_negative/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllAllEdnn/head/assert_range/assert_non_negative/assert_less_equal/LessEqualAdnn/head/assert_range/assert_non_negative/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Hdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/ConstConst*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_1Const*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_2Const*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/All?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: : 
�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fIdentityUdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch*
_output_shapes
: *
T0

�
Vdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_idIdentity?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: 
�
Sdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
�
adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tT^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: *
T0

�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*)
value B Bx (dnn/head/labels:0) = *
dtype0*
_output_shapes
: 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
	summarize*
T
2	
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : 
�
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/labelsVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0	*"
_class
loc:@dnn/head/labels*:
_output_shapes(
&:���������:���������
�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
dnn/head/assert_range/IdentityIdentitydnn/head/labels;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*
T0	*'
_output_shapes
:���������
�
Odnn/head/sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/SqueezeSqueezednn/head/assert_range/Identity*
squeeze_dims

���������*
T0	*#
_output_shapes
:���������
u
0dnn/head/sparse_softmax_cross_entropy_loss/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/sparse_softmax_cross_entropy_loss/xentropy/ShapeShapeOdnn/head/sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/Squeeze*
out_type0*
_output_shapes
:*
T0	
�
<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAddOdnn/head/sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/Squeeze*6
_output_shapes$
":���������:���������*
Tlabels0	*
T0
�
Mdnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ldnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ldnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy*
out_type0*
_output_shapes
:*
T0
�
Kdnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
c
[dnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
.dnn/head/sparse_softmax_cross_entropy_loss/MulMul<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy0dnn/head/sparse_softmax_cross_entropy_loss/Const\^dnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*#
_output_shapes
:���������
b
dnn/head/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/ExpandDims
ExpandDims.dnn/head/sparse_softmax_cross_entropy_loss/Muldnn/head/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
x
3dnn/head/weighted_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
|
9dnn/head/weighted_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
z
8dnn/head/weighted_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
8dnn/head/weighted_loss/assert_broadcastable/values/shapeShapednn/head/ExpandDims*
T0*
out_type0*
_output_shapes
:
y
7dnn/head/weighted_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
O
Gdnn/head/weighted_loss/assert_broadcastable/static_scalar_check_successNoOp
�
"dnn/head/weighted_loss/ToFloat_1/xConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/weighted_loss/MulMuldnn/head/ExpandDims"dnn/head/weighted_loss/ToFloat_1/x*'
_output_shapes
:���������*
T0
�
dnn/head/weighted_loss/ConstConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/weighted_loss/SumSumdnn/head/weighted_loss/Muldnn/head/weighted_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
k
dnn/head/ones_like/ShapeShapednn/head/ExpandDims*
out_type0*
_output_shapes
:*
T0
]
dnn/head/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/ones_likeFilldnn/head/ones_like/Shapednn/head/ones_like/Const*

index_type0*'
_output_shapes
:���������*
T0
S
dnn/head/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
i
dnn/head/mulMuldnn/head/mul/xdnn/head/ones_like*
T0*'
_output_shapes
:���������
_
dnn/head/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
o
dnn/head/SumSumdnn/head/muldnn/head/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
f
dnn/head/truedivRealDivdnn/head/weighted_loss/Sumdnn/head/Sum*
_output_shapes
: *
T0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
]
lossScalarSummary	loss/tagsdnn/head/weighted_loss/Sum*
_output_shapes
: *
T0
^
average_loss/tagsConst*
valueB Baverage_loss*
dtype0*
_output_shapes
: 
c
average_lossScalarSummaryaverage_loss/tagsdnn/head/truediv*
_output_shapes
: *
T0
V
dnn/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
\
dnn/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
{
dnn/gradients/FillFilldnn/gradients/Shapednn/gradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
�
;dnn/gradients/dnn/head/weighted_loss/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
5dnn/gradients/dnn/head/weighted_loss/Sum_grad/ReshapeReshapednn/gradients/Fill;dnn/gradients/dnn/head/weighted_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
3dnn/gradients/dnn/head/weighted_loss/Sum_grad/ShapeShapednn/head/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
�
2dnn/gradients/dnn/head/weighted_loss/Sum_grad/TileTile5dnn/gradients/dnn/head/weighted_loss/Sum_grad/Reshape3dnn/gradients/dnn/head/weighted_loss/Sum_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
�
3dnn/gradients/dnn/head/weighted_loss/Mul_grad/ShapeShapednn/head/ExpandDims*
out_type0*
_output_shapes
:*
T0
x
5dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Cdnn/gradients/dnn/head/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape5dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1dnn/gradients/dnn/head/weighted_loss/Mul_grad/MulMul2dnn/gradients/dnn/head/weighted_loss/Sum_grad/Tile"dnn/head/weighted_loss/ToFloat_1/x*
T0*'
_output_shapes
:���������
�
1dnn/gradients/dnn/head/weighted_loss/Mul_grad/SumSum1dnn/gradients/dnn/head/weighted_loss/Mul_grad/MulCdnn/gradients/dnn/head/weighted_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
5dnn/gradients/dnn/head/weighted_loss/Mul_grad/ReshapeReshape1dnn/gradients/dnn/head/weighted_loss/Mul_grad/Sum3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Mul_1Muldnn/head/ExpandDims2dnn/gradients/dnn/head/weighted_loss/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Sum_1Sum3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Mul_1Ednn/gradients/dnn/head/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1Reshape3dnn/gradients/dnn/head/weighted_loss/Mul_grad/Sum_15dnn/gradients/dnn/head/weighted_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>dnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/group_depsNoOp6^dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape8^dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1
�
Fdnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/control_dependencyIdentity5dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape?^dnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape*'
_output_shapes
:���������
�
Hdnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/control_dependency_1Identity7dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1?^dnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/group_deps*J
_class@
><loc:@dnn/gradients/dnn/head/weighted_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
�
,dnn/gradients/dnn/head/ExpandDims_grad/ShapeShape.dnn/head/sparse_softmax_cross_entropy_loss/Mul*
T0*
out_type0*
_output_shapes
:
�
.dnn/gradients/dnn/head/ExpandDims_grad/ReshapeReshapeFdnn/gradients/dnn/head/weighted_loss/Mul_grad/tuple/control_dependency,dnn/gradients/dnn/head/ExpandDims_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
Gdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ShapeShape<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy*
out_type0*
_output_shapes
:*
T0
�
Idnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Wdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ShapeIdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ednn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/MulMul.dnn/gradients/dnn/head/ExpandDims_grad/Reshape0dnn/head/sparse_softmax_cross_entropy_loss/Const*#
_output_shapes
:���������*
T0
�
Ednn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/SumSumEdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/MulWdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Idnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ReshapeReshapeEdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/SumGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
Gdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Mul_1Mul<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy.dnn/gradients/dnn/head/ExpandDims_grad/Reshape*#
_output_shapes
:���������*
T0
�
Gdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Sum_1SumGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Mul_1Ydnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Kdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1ReshapeGdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Sum_1Idnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Rdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOpJ^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ReshapeL^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Zdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentityIdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/ReshapeS^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������
�
\dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1IdentityKdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1S^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*^
_classT
RPloc:@dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
�
dnn/gradients/zeros_like	ZerosLike>dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy:1*'
_output_shapes
:���������*
T0
�
_dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/PreventGradientPreventGradient>dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy:1*'
_output_shapes
:���������*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
�
^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Zdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims
ExpandDimsZdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
Sdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mulMulZdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/ExpandDims_dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/PreventGradient*'
_output_shapes
:���������*
T0
�
1dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGradBiasAddGradSdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul*
data_formatNHWC*
_output_shapes
:*
T0
�
6dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_depsNoOpT^dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul2^dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad
�
>dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencyIdentitySdnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul7^dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*f
_class\
ZXloc:@dnn/gradients/dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy_grad/mul*'
_output_shapes
:���������*
T0
�
@dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1Identity1dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad7^dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
+dnn/gradients/dnn/logits/MatMul_grad/MatMulMatMul>dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencydnn/logits/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
-dnn/gradients/dnn/logits/MatMul_grad/MatMul_1MatMuldnn/hiddenlayer_1/Relu>dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
5dnn/gradients/dnn/logits/MatMul_grad/tuple/group_depsNoOp,^dnn/gradients/dnn/logits/MatMul_grad/MatMul.^dnn/gradients/dnn/logits/MatMul_grad/MatMul_1
�
=dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencyIdentity+dnn/gradients/dnn/logits/MatMul_grad/MatMul6^dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*>
_class4
20loc:@dnn/gradients/dnn/logits/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
�
?dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1Identity-dnn/gradients/dnn/logits/MatMul_grad/MatMul_16^dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@dnn/gradients/dnn/logits/MatMul_grad/MatMul_1*
_output_shapes

:
�
2dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGradReluGrad=dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencydnn/hiddenlayer_1/Relu*'
_output_shapes
:���������*
T0
�
8dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradBiasAddGrad2dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
=dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad9^dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad
�
Ednn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad>^dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_1/Relu_grad/ReluGrad*'
_output_shapes
:���������*
T0
�
Gdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1Identity8dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad>^dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
2dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulMatMulEdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
4dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1MatMuldnn/hiddenlayer_0/ReluEdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
<dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul5^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1
�
Ddnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul=^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Fdnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1Identity4dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1=^dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
2dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGradReluGradDdnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencydnn/hiddenlayer_0/Relu*'
_output_shapes
:���������*
T0
�
8dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradBiasAddGrad2dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
=dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad9^dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad
�
Ednn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad>^dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_0/Relu_grad/ReluGrad*'
_output_shapes
:���������*
T0
�
Gdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1Identity8dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad>^dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
2dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulMatMulEdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_0/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
4dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1MatMul1dnn/input_from_feature_columns/input_layer/concatEdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
<dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_depsNoOp3^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul5^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1
�
Ddnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependencyIdentity2dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul=^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*E
_class;
97loc:@dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
�
Fdnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1Identity4dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1=^dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1*
_output_shapes

:
�
=dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:
�
+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container 
�
2dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad=dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:
�
0dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
�
;dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
�
)dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container 
�
0dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/AssignAssign)dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad;dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/Initializer/Const*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
.dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/readIdentity)dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:*
T0
�
=dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
�
+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:
�
2dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad=dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:
�
0dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:*
T0
�
;dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
�
)dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container 
�
0dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/AssignAssign)dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad;dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/Initializer/Const*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
.dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/readIdentity)dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:
�
6dnn/dnn/logits/kernel/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
�
$dnn/dnn/logits/kernel/part_0/Adagrad
VariableV2*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:
�
+dnn/dnn/logits/kernel/part_0/Adagrad/AssignAssign$dnn/dnn/logits/kernel/part_0/Adagrad6dnn/dnn/logits/kernel/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:
�
)dnn/dnn/logits/kernel/part_0/Adagrad/readIdentity$dnn/dnn/logits/kernel/part_0/Adagrad*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0
�
4dnn/dnn/logits/bias/part_0/Adagrad/Initializer/ConstConst*
valueB*���=*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
"dnn/dnn/logits/bias/part_0/Adagrad
VariableV2*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
�
)dnn/dnn/logits/bias/part_0/Adagrad/AssignAssign"dnn/dnn/logits/bias/part_0/Adagrad4dnn/dnn/logits/bias/part_0/Adagrad/Initializer/Const*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
�
'dnn/dnn/logits/bias/part_0/Adagrad/readIdentity"dnn/dnn/logits/bias/part_0/Adagrad*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
T0
^
dnn/Adagrad/learning_rateConst*
valueB
 *��L=*
dtype0*
_output_shapes
: 
�
?dnn/Adagrad/update_dnn/hiddenlayer_0/kernel/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/kernel/part_0+dnn/dnn/hiddenlayer_0/kernel/part_0/Adagraddnn/Adagrad/learning_rateFdnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:*
use_locking( *
T0
�
=dnn/Adagrad/update_dnn/hiddenlayer_0/bias/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/bias/part_0)dnn/dnn/hiddenlayer_0/bias/part_0/Adagraddnn/Adagrad/learning_rateGdnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
�
?dnn/Adagrad/update_dnn/hiddenlayer_1/kernel/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/kernel/part_0+dnn/dnn/hiddenlayer_1/kernel/part_0/Adagraddnn/Adagrad/learning_rateFdnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
�
=dnn/Adagrad/update_dnn/hiddenlayer_1/bias/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/bias/part_0)dnn/dnn/hiddenlayer_1/bias/part_0/Adagraddnn/Adagrad/learning_rateGdnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:*
use_locking( *
T0
�
8dnn/Adagrad/update_dnn/logits/kernel/part_0/ApplyAdagradApplyAdagraddnn/logits/kernel/part_0$dnn/dnn/logits/kernel/part_0/Adagraddnn/Adagrad/learning_rate?dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
6dnn/Adagrad/update_dnn/logits/bias/part_0/ApplyAdagradApplyAdagraddnn/logits/bias/part_0"dnn/dnn/logits/bias/part_0/Adagraddnn/Adagrad/learning_rate@dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
�
dnn/Adagrad/updateNoOp@^dnn/Adagrad/update_dnn/hiddenlayer_0/kernel/part_0/ApplyAdagrad>^dnn/Adagrad/update_dnn/hiddenlayer_0/bias/part_0/ApplyAdagrad@^dnn/Adagrad/update_dnn/hiddenlayer_1/kernel/part_0/ApplyAdagrad>^dnn/Adagrad/update_dnn/hiddenlayer_1/bias/part_0/ApplyAdagrad9^dnn/Adagrad/update_dnn/logits/kernel/part_0/ApplyAdagrad7^dnn/Adagrad/update_dnn/logits/bias/part_0/ApplyAdagrad
�
dnn/Adagrad/valueConst^dnn/Adagrad/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
dnn/Adagrad	AssignAddglobal_stepdnn/Adagrad/value*
_class
loc:@global_step*
_output_shapes
: *
use_locking( *
T0	�3
�3
-
_make_dataset_7dc53701
BatchDataset��
&TensorSliceDataset/tensors/component_0Const*�
value�B�x"�ffffff@ffffff
@      @      �?333333�?�������?      �?ffffff@������@      �?333333@������@333333�?333333�?������@������@������@ffffff@333333@      �?������@ffffff@�������?ffffff@333333@      �?      @ffffff@      @      @������@      @      @      @333333@������@ffffff@ffffff@      �?ffffff�?������@ffffff@������@ffffff�?�������?      @�������?������@      �?      @ffffff�?������@      @333333�?������@333333@ffffff@ffffff@������@      @333333@ffffff@      @      @      @333333@�������?ffffff@      @      �?�������?ffffff@�������?ffffff@������@      �?ffffff@������@      @333333@      �?������@ffffff@������@ffffff@      �?333333@      @ffffff@�������?������@������@ffffff�?      �?333333@ffffff�?ffffff�?������@333333@�������?ffffff�?ffffff
@�������?ffffff@�������?ffffff�?      �?ffffff�?������@�������?������@ffffff@�������?      @�������?������@������@ffffff�?ffffff�?������@*
dtype0�
&TensorSliceDataset/tensors/component_1Const*�
value�B�x"�������@      �?333333�?�������?333333�?�������?�������?ffffff@ffffff�?�������?ffffff�?      �?�������?�������?������@�������?ffffff@������ @ffffff@�������?������ @������ @�������?ffffff�?�������?�������?333333�?�������?      �?333333�?�������?�������?      �?������ @�������?ffffff@       @333333@333333�?333333�?       @ffffff�?ffffff@�������?�������?      �?�������?ffffff@�������?�������?�������?ffffff�?�������?�������?ffffff�?�������?ffffff�?ffffff�?ffffff@      �?�������?333333@ffffff�?�������?      �?�������?�������?      @�������?�������?�������?�������?�������?ffffff@�������?�������?�������?�������?      �?ffffff�?�������?ffffff�?      �?�������?333333@�������?ffffff�?�������?�������?333333�?       @333333�?333333�?�������?������@333333�?�������?       @�������?�������?�������?      �?333333�?      �?�������?�������?�������?�������?�������?�������?�������?�������?�������?      @333333�?333333�?333333�?�������?�������?      �?*
dtype0�
&TensorSliceDataset/tensors/component_2Const*�
value�B�x"�������@      @������@������@������@������@������@������@������@ffffff@������@������@333333@������@������@333333@333333@ffffff@������@������@������@������@������@ffffff@������@������@333333@������@������@������@333333@ffffff@������@333333@������@������@      @333333@ffffff@333333@������@333333@������@ffffff@������@      @333333@������@ffffff@������@      @ffffff@ffffff@      @      @      @������@333333@������@      @ffffff@������@333333@������@������@333333@������@������@      @      @������@ffffff@      @������@ffffff@������@      @������@      @������@������@ffffff@      @ffffff@333333@������@333333@������@      @      @      @ffffff@ffffff@ffffff@      @ffffff@ffffff@������@������@ffffff@������@������@      @333333@      @������@333333@      @ffffff@333333@333333@������@      @333333@      @      @������@������@333333@      @*
dtype0�
&TensorSliceDataset/tensors/component_3Const*�
value�B�x"�ffffff@ffffff@      @������@ffffff@������	@333333@������@������@������@������@������@      @333333@ffffff@ffffff
@������	@      @������	@������@ffffff
@ffffff@333333@������@      @      @������@      @      @      @ffffff@      @������	@      @ffffff@������	@������	@ffffff@ffffff@      @ffffff@������@      @ffffff@������	@������@333333@������@������@������	@ffffff
@      @ffffff@������	@������	@      @ffffff@������@333333@       @      @������@      @������@������@333333@      @������@      @333333@������	@333333@      @      @      @������@333333@333333@333333@������@������@333333@ffffff@������@333333@������@ffffff@ffffff@������@      @      @ffffff@      @������@      @333333@������	@ffffff@������	@ffffff@      @333333@ffffff@������@333333@333333@������@������@333333@������@������@ffffff@      @ffffff
@      @������@      @333333@      @333333@*
dtype0�
&TensorSliceDataset/tensors/component_4Const*�
value�B�	x"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  *
dtype0	�
TensorSliceDatasetTensorSliceDataset/TensorSliceDataset/tensors/component_0:output:0/TensorSliceDataset/tensors/component_1:output:0/TensorSliceDataset/tensors/component_2:output:0/TensorSliceDataset/tensors/component_3:output:0/TensorSliceDataset/tensors/component_4:output:0*
Toutput_types	
2	*
output_shapes

: : : : : E
ShuffleDataset/buffer_sizeConst*
value
B	 R�*
dtype0	=
ShuffleDataset/seedConst*
value	B	 R *
dtype0	>
ShuffleDataset/seed2Const*
value	B	 R *
dtype0	�
ShuffleDatasetShuffleDatasetTensorSliceDataset:handle:0#ShuffleDataset/buffer_size:output:0ShuffleDataset/seed:output:0ShuffleDataset/seed2:output:0*
reshuffle_each_iteration(*
output_types	
2	*
output_shapes

: : : : : F
RepeatDataset/countConst*
valueB	 R
���������*
dtype0	�
RepeatDatasetRepeatDatasetShuffleDataset:handle:0RepeatDataset/count:output:0*
output_types	
2	*
output_shapes

: : : : : A
BatchDataset/batch_sizeConst*
value	B	 Rd*
dtype0	�
BatchDatasetBatchDatasetRepeatDataset:handle:0 BatchDataset/batch_size:output:0*
output_types	
2	*^
output_shapesM
K:���������:���������:���������:���������:���������"%
BatchDatasetBatchDataset:handle:0""�
	variables��
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kernel  "2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias "21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel  "25dnn/logits/kernel/part_0/Initializer/random_uniform:0
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0
�
-dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad:02dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/read:0"2
$dnn/hiddenlayer_0/kernel/t_0/Adagrad  "2?dnn/dnn/hiddenlayer_0/kernel/part_0/Adagrad/Initializer/Const:0
�
+dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad:00dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/Assign0dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/read:0"-
"dnn/hiddenlayer_0/bias/t_0/Adagrad "2=dnn/dnn/hiddenlayer_0/bias/part_0/Adagrad/Initializer/Const:0
�
-dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad:02dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/read:0"2
$dnn/hiddenlayer_1/kernel/t_0/Adagrad  "2?dnn/dnn/hiddenlayer_1/kernel/part_0/Adagrad/Initializer/Const:0
�
+dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad:00dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/Assign0dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/read:0"-
"dnn/hiddenlayer_1/bias/t_0/Adagrad "2=dnn/dnn/hiddenlayer_1/bias/part_0/Adagrad/Initializer/Const:0
�
&dnn/dnn/logits/kernel/part_0/Adagrad:0+dnn/dnn/logits/kernel/part_0/Adagrad/Assign+dnn/dnn/logits/kernel/part_0/Adagrad/read:0"+
dnn/logits/kernel/t_0/Adagrad  "28dnn/dnn/logits/kernel/part_0/Adagrad/Initializer/Const:0
�
$dnn/dnn/logits/bias/part_0/Adagrad:0)dnn/dnn/logits/bias/part_0/Adagrad/Assign)dnn/dnn/logits/bias/part_0/Adagrad/read:0"&
dnn/logits/bias/t_0/Adagrad "26dnn/dnn/logits/bias/part_0/Adagrad/Initializer/Const:0"

savers "z
lossesp
n
0dnn/head/sparse_softmax_cross_entropy_loss/Mul:0
dnn/head/weighted_loss/Sum:0
dnn/head/weighted_loss/Sum:0" 
global_step

global_step:0"2
global_step_read_op_cache

global_step/add:0"�-
cond_context�-�-
�
global_step/cond/cond_textglobal_step/cond/pred_id:0global_step/cond/switch_t:0 *�
global_step/cond/pred_id:0
global_step/cond/read/Switch:1
global_step/cond/read:0
global_step/cond/switch_t:0
global_step:08
global_step/cond/pred_id:0global_step/cond/pred_id:0:
global_step/cond/switch_t:0global_step/cond/switch_t:0/
global_step:0global_step/cond/read/Switch:1
�
global_step/cond/cond_text_1global_step/cond/pred_id:0global_step/cond/switch_f:0*�
global_step/Initializer/zeros:0
global_step/cond/Switch_1:0
global_step/cond/Switch_1:1
global_step/cond/pred_id:0
global_step/cond/switch_f:0:
global_step/cond/switch_f:0global_step/cond/switch_f:0>
global_step/Initializer/zeros:0global_step/cond/Switch_1:08
global_step/cond/pred_id:0global_step/cond/pred_id:0
�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/cond_text>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0 *�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency:0
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0�
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
�
@dnn/head/assert_range/assert_less/Assert/AssertGuard/cond_text_1>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0*�

dnn/head/assert_range/Const:0
'dnn/head/assert_range/assert_less/All:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch:0
Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1:0
Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4:0
Kdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1:0
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0
dnn/head/labels:0�
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0[
dnn/head/labels:0Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1:0g
dnn/head/assert_range/Const:0Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2:0o
'dnn/head/assert_range/assert_less/All:0Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch:0
�
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/cond_textXdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0 *�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency:0
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0�
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0�
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0
�
Zdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/cond_text_1Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0*�
Adnn/head/assert_range/assert_non_negative/assert_less_equal/All:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
`dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:0
ednn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1:0
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0
dnn/head/labels:0�
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0�
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0u
dnn/head/labels:0`dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0�
Adnn/head/assert_range/assert_non_negative/assert_less_equal/All:0^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch:0"
train_op

dnn/Adagrad"�	
trainable_variables�	�	
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kernel  "2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias "21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel  "25dnn/logits/kernel/part_0/Initializer/random_uniform:0
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"�
	summaries�
�
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0
loss:0
average_loss:0��
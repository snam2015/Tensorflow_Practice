       �K"	   %ŵ�Abrain.Event:2O7 �y     $~�o	L'%ŵ�A"��
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
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
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
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
�
tensors/component_0Const"/device:CPU:0*�
value�B�"�������@������@333333�?      @      @333333@ffffff�?ffffff@ffffff@333333@������@�������?������@      @      @ffffff�?      @ffffff�?      �?      @333333�?      @ffffff@������@333333@ffffff@      �?������@������@333333@*
dtype0*
_output_shapes
:
�
tensors/component_1Const"/device:CPU:0*�
value�B�"�      �?������ @      �?�������?�������?�������?�������?      �?�������?�������?������ @�������?       @�������?      �?�������?      �?�������?�������?       @�������?      �?�������?      �?333333�?ffffff�?�������?      �?      @�������?*
dtype0*
_output_shapes
:
�
tensors/component_2Const"/device:CPU:0*�
value�B�"�������@������@ffffff@      @      @������@      @333333@ffffff@������@ffffff@333333@ffffff@      @      @ffffff@������@333333@ffffff@������@������@ffffff@333333@333333@333333@ffffff@������@������@������@������@*
dtype0*
_output_shapes
:
�
tensors/component_3Const"/device:CPU:0*�
value�B�"�      @������@ffffff
@333333@      @333333@������@ffffff@      @      @      @      @ffffff@ffffff@������@      @������@333333@333333@      @333333@      @333333@      @������@      @ffffff@������@ffffff
@333333@*
dtype0*
_output_shapes
:
�
tensors/component_4Const"/device:CPU:0*�
value�B�	"�                                                                                                                                                                                                                          *
dtype0	*
_output_shapes
:
[

batch_sizeConst"/device:CPU:0*
value	B	 Rd*
dtype0	*
_output_shapes
: 
�
OneShotIteratorOneShotIterator"/device:CPU:0*
_output_shapes
: *-
dataset_factoryR
_make_dataset_52536cc0*
shared_name *^
output_shapesM
K:���������:���������:���������:���������:���������*
	container *
output_types	
2	
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
Ddnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeJdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
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
>dnn/input_from_feature_columns/input_layer/PetalLength/ReshapeReshape>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloatDdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims
ExpandDimsIteratorGetNext:1Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloatCast@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims*'
_output_shapes
:���������*

DstT0*

SrcT0
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
Ednn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
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
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
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
Cdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeIdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2*
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
=dnn/input_from_feature_columns/input_layer/SepalWidth/ReshapeReshape=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloatCdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
1dnn/input_from_feature_columns/input_layer/concatConcatV2>dnn/input_from_feature_columns/input_layer/PetalLength/Reshape=dnn/input_from_feature_columns/input_layer/PetalWidth/Reshape>dnn/input_from_feature_columns/input_layer/SepalLength/Reshape=dnn/input_from_feature_columns/input_layer/SepalWidth/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*'
_output_shapes
:���������*

Tidx0*
T0
�
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"   
   *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *b�'�*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *b�'?*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:
*

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
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
T0
�
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container 
�
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
�
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
T0
�
?dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:
*0
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
:

�
dnn/hiddenlayer_0/bias/part_0
VariableV2*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container *
shape:
*
dtype0*
_output_shapes
:

�
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:

�
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:

s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes
:
*
T0
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:���������

k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*'
_output_shapes
:���������
*
T0
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*'
_output_shapes
:���������

x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*'
_output_shapes
:���������
*

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
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *�7�*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *�7?*2
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

:

*

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

:


�
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:


�
dnn/hiddenlayer_1/kernel/part_0
VariableV2*
shape
:

*
dtype0*
_output_shapes

:

*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container 
�
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:


�
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:


�
?dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:
*0
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
:
*
T0
�
dnn/hiddenlayer_1/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name 
�
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:

�
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:

s
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
_output_shapes

:

*
T0
�
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
_output_shapes
:
*
T0
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
data_formatNHWC*'
_output_shapes
:���������
*
T0
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������

]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*'
_output_shapes
:���������
*
T0
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*'
_output_shapes
:���������
*

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
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: *
T0
�
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *��-�*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *��-?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:
*

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
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
*
T0
�
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

�
dnn/logits/kernel/part_0
VariableV2*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:

�
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
*
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
(dnn/logits/bias/part_0/Initializer/zerosFill8dnn/logits/bias/part_0/Initializer/zeros/shape_as_tensor.dnn/logits/bias/part_0/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
�
dnn/logits/bias/part_0
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container 
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

:
*
T0
�
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
T0*
_output_shapes
:
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
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: *
T0
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
dnn/head/predictions/class_idsArgMaxdnn/logits/BiasAdd(dnn/head/predictions/class_ids/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
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
:���������*
	precision���������*
shortest( *
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
ExpandDimsIteratorGetNext:4dnn/head/labels/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0	
o
dnn/head/labels/ShapeShapednn/head/labels/ExpandDims*
out_type0*
_output_shapes
:*
T0	
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
dnn/head/labels/strided_sliceStridedSlicednn/head/labels/Shape_1#dnn/head/labels/strided_slice/stack%dnn/head/labels/strided_slice/stack_1%dnn/head/labels/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
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
dnn/head/labels/concatConcatV2dnn/head/labels/strided_slicednn/head/labels/concat/values_1dnn/head/labels/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
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
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_0dnn/head/labels/concat1dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/Shape*
T
2*
	summarize
�
dnn/head/labelsIdentitydnn/head/labels/ExpandDimsE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok+^dnn/head/labels/assert_equal/Assert/Assert*
T0	*'
_output_shapes
:���������
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
%dnn/head/assert_range/assert_less/AllAll&dnn/head/assert_range/assert_less/Less'dnn/head/assert_range/assert_less/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
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
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
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
T

2		*
	summarize
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
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/assert_range/Const<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*.
_class$
" loc:@dnn/head/assert_range/Const*
_output_shapes
: : *
T0	
�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Identity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f<^dnn/head/assert_range/assert_less/Assert/AssertGuard/Assert*
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*
_output_shapes
: 
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
?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllAllEdnn/head/assert_range/assert_non_negative/assert_less_equal/LessEqualAdnn/head/assert_range/assert_non_negative/assert_less_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
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
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
T
2	*
	summarize
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : 
�
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/labelsVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*"
_class
loc:@dnn/head/labels*:
_output_shapes(
&:���������:���������*
T0	
�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: *
T0

�
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
dnn/head/assert_range/IdentityIdentitydnn/head/labels;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*'
_output_shapes
:���������*
T0	
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
.dnn/head/sparse_softmax_cross_entropy_loss/MulMul<dnn/head/sparse_softmax_cross_entropy_loss/xentropy/xentropy0dnn/head/sparse_softmax_cross_entropy_loss/Const\^dnn/head/sparse_softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*#
_output_shapes
:���������*
T0
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
�
Ednn/head/metrics/average_loss/total/Initializer/zeros/shape_as_tensorConst*
valueB *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
;dnn/head/metrics/average_loss/total/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/total/Initializer/zerosFillEdnn/head/metrics/average_loss/total/Initializer/zeros/shape_as_tensor;dnn/head/metrics/average_loss/total/Initializer/zeros/Const*

index_type0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: *
T0
�
#dnn/head/metrics/average_loss/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
	container *
shape: 
�
*dnn/head/metrics/average_loss/total/AssignAssign#dnn/head/metrics/average_loss/total5dnn/head/metrics/average_loss/total/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
validate_shape(*
_output_shapes
: 
�
(dnn/head/metrics/average_loss/total/readIdentity#dnn/head/metrics/average_loss/total*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
�
Ednn/head/metrics/average_loss/count/Initializer/zeros/shape_as_tensorConst*
valueB *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
;dnn/head/metrics/average_loss/count/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/count/Initializer/zerosFillEdnn/head/metrics/average_loss/count/Initializer/zeros/shape_as_tensor;dnn/head/metrics/average_loss/count/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/count
VariableV2*
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
*dnn/head/metrics/average_loss/count/AssignAssign#dnn/head/metrics/average_loss/count5dnn/head/metrics/average_loss/count/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
validate_shape(*
_output_shapes
: 
�
(dnn/head/metrics/average_loss/count/readIdentity#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: *
T0
h
#dnn/head/metrics/average_loss/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Rdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/ExpandDims*
T0*
out_type0*
_output_shapes
:
�
Pdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
h
`dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ShapeShapednn/head/ExpandDimsa^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ConstConsta^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/metrics/average_loss/broadcast_weights/ones_likeFill?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Shape?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Const*

index_type0*'
_output_shapes
:���������*
T0
�
/dnn/head/metrics/average_loss/broadcast_weightsMul#dnn/head/metrics/average_loss/Const9dnn/head/metrics/average_loss/broadcast_weights/ones_like*'
_output_shapes
:���������*
T0
�
!dnn/head/metrics/average_loss/MulMuldnn/head/ExpandDims/dnn/head/metrics/average_loss/broadcast_weights*'
_output_shapes
:���������*
T0
v
%dnn/head/metrics/average_loss/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
!dnn/head/metrics/average_loss/SumSum/dnn/head/metrics/average_loss/broadcast_weights%dnn/head/metrics/average_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
v
%dnn/head/metrics/average_loss/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
�
#dnn/head/metrics/average_loss/Sum_1Sum!dnn/head/metrics/average_loss/Mul%dnn/head/metrics/average_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
'dnn/head/metrics/average_loss/AssignAdd	AssignAdd#dnn/head/metrics/average_loss/total#dnn/head/metrics/average_loss/Sum_1*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
�
)dnn/head/metrics/average_loss/AssignAdd_1	AssignAdd#dnn/head/metrics/average_loss/count!dnn/head/metrics/average_loss/Sum"^dnn/head/metrics/average_loss/Mul*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
�
%dnn/head/metrics/average_loss/truedivRealDiv(dnn/head/metrics/average_loss/total/read(dnn/head/metrics/average_loss/count/read*
_output_shapes
: *
T0
{
8dnn/head/metrics/average_loss/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
s
.dnn/head/metrics/average_loss/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
(dnn/head/metrics/average_loss/zeros_likeFill8dnn/head/metrics/average_loss/zeros_like/shape_as_tensor.dnn/head/metrics/average_loss/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
�
%dnn/head/metrics/average_loss/GreaterGreater(dnn/head/metrics/average_loss/count/read(dnn/head/metrics/average_loss/zeros_like*
_output_shapes
: *
T0
�
#dnn/head/metrics/average_loss/valueSelect%dnn/head/metrics/average_loss/Greater%dnn/head/metrics/average_loss/truediv(dnn/head/metrics/average_loss/zeros_like*
_output_shapes
: *
T0
�
'dnn/head/metrics/average_loss/truediv_1RealDiv'dnn/head/metrics/average_loss/AssignAdd)dnn/head/metrics/average_loss/AssignAdd_1*
T0*
_output_shapes
: 
}
:dnn/head/metrics/average_loss/zeros_like_1/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
u
0dnn/head/metrics/average_loss/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
*dnn/head/metrics/average_loss/zeros_like_1Fill:dnn/head/metrics/average_loss/zeros_like_1/shape_as_tensor0dnn/head/metrics/average_loss/zeros_like_1/Const*

index_type0*
_output_shapes
: *
T0
�
'dnn/head/metrics/average_loss/Greater_1Greater)dnn/head/metrics/average_loss/AssignAdd_1*dnn/head/metrics/average_loss/zeros_like_1*
T0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/update_opSelect'dnn/head/metrics/average_loss/Greater_1'dnn/head/metrics/average_loss/truediv_1*dnn/head/metrics/average_loss/zeros_like_1*
_output_shapes
: *
T0
[
dnn/head/metrics/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/EqualEqualdnn/head/predictions/ExpandDimsdnn/head/assert_range/Identity*
T0	*'
_output_shapes
:���������
y
dnn/head/metrics/ToFloatCastdnn/head/metrics/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

�
Adnn/head/metrics/accuracy/total/Initializer/zeros/shape_as_tensorConst*
valueB *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
7dnn/head/metrics/accuracy/total/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
1dnn/head/metrics/accuracy/total/Initializer/zerosFillAdnn/head/metrics/accuracy/total/Initializer/zeros/shape_as_tensor7dnn/head/metrics/accuracy/total/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
�
dnn/head/metrics/accuracy/total
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
	container 
�
&dnn/head/metrics/accuracy/total/AssignAssigndnn/head/metrics/accuracy/total1dnn/head/metrics/accuracy/total/Initializer/zeros*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
$dnn/head/metrics/accuracy/total/readIdentitydnn/head/metrics/accuracy/total*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
�
Adnn/head/metrics/accuracy/count/Initializer/zeros/shape_as_tensorConst*
valueB *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
7dnn/head/metrics/accuracy/count/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
1dnn/head/metrics/accuracy/count/Initializer/zerosFillAdnn/head/metrics/accuracy/count/Initializer/zeros/shape_as_tensor7dnn/head/metrics/accuracy/count/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: 
�
dnn/head/metrics/accuracy/count
VariableV2*
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/accuracy/count/AssignAssigndnn/head/metrics/accuracy/count1dnn/head/metrics/accuracy/count/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
validate_shape(*
_output_shapes
: 
�
$dnn/head/metrics/accuracy/count/readIdentitydnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: *
T0
�
Ndnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/metrics/ToFloat*
out_type0*
_output_shapes
:*
T0
�
Ldnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
d
\dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ShapeShapednn/head/metrics/ToFloat]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ConstConst]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/accuracy/broadcast_weights/ones_likeFill;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Shape;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Const*

index_type0*'
_output_shapes
:���������*
T0
�
+dnn/head/metrics/accuracy/broadcast_weightsMuldnn/head/metrics/Const5dnn/head/metrics/accuracy/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
dnn/head/metrics/accuracy/MulMuldnn/head/metrics/ToFloat+dnn/head/metrics/accuracy/broadcast_weights*'
_output_shapes
:���������*
T0
p
dnn/head/metrics/accuracy/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/SumSum+dnn/head/metrics/accuracy/broadcast_weightsdnn/head/metrics/accuracy/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
r
!dnn/head/metrics/accuracy/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/Sum_1Sumdnn/head/metrics/accuracy/Mul!dnn/head/metrics/accuracy/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/AssignAdd	AssignAdddnn/head/metrics/accuracy/totaldnn/head/metrics/accuracy/Sum_1*
use_locking( *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
�
%dnn/head/metrics/accuracy/AssignAdd_1	AssignAdddnn/head/metrics/accuracy/countdnn/head/metrics/accuracy/Sum^dnn/head/metrics/accuracy/Mul*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: *
use_locking( *
T0
�
!dnn/head/metrics/accuracy/truedivRealDiv$dnn/head/metrics/accuracy/total/read$dnn/head/metrics/accuracy/count/read*
_output_shapes
: *
T0
w
4dnn/head/metrics/accuracy/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
o
*dnn/head/metrics/accuracy/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$dnn/head/metrics/accuracy/zeros_likeFill4dnn/head/metrics/accuracy/zeros_like/shape_as_tensor*dnn/head/metrics/accuracy/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
�
!dnn/head/metrics/accuracy/GreaterGreater$dnn/head/metrics/accuracy/count/read$dnn/head/metrics/accuracy/zeros_like*
T0*
_output_shapes
: 
�
dnn/head/metrics/accuracy/valueSelect!dnn/head/metrics/accuracy/Greater!dnn/head/metrics/accuracy/truediv$dnn/head/metrics/accuracy/zeros_like*
_output_shapes
: *
T0
�
#dnn/head/metrics/accuracy/truediv_1RealDiv#dnn/head/metrics/accuracy/AssignAdd%dnn/head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
y
6dnn/head/metrics/accuracy/zeros_like_1/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
q
,dnn/head/metrics/accuracy/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/accuracy/zeros_like_1Fill6dnn/head/metrics/accuracy/zeros_like_1/shape_as_tensor,dnn/head/metrics/accuracy/zeros_like_1/Const*

index_type0*
_output_shapes
: *
T0
�
#dnn/head/metrics/accuracy/Greater_1Greater%dnn/head/metrics/accuracy/AssignAdd_1&dnn/head/metrics/accuracy/zeros_like_1*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/update_opSelect#dnn/head/metrics/accuracy/Greater_1#dnn/head/metrics/accuracy/truediv_1&dnn/head/metrics/accuracy/zeros_like_1*
T0*
_output_shapes
: 
�
,mean/total/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
"mean/total/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
mean/total/Initializer/zerosFill,mean/total/Initializer/zeros/shape_as_tensor"mean/total/Initializer/zeros/Const*

index_type0*
_class
loc:@mean/total*
_output_shapes
: *
T0
�

mean/total
VariableV2*
_class
loc:@mean/total*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
g
mean/total/readIdentity
mean/total*
_class
loc:@mean/total*
_output_shapes
: *
T0
�
,mean/count/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
"mean/count/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
mean/count/Initializer/zerosFill,mean/count/Initializer/zeros/shape_as_tensor"mean/count/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@mean/count*
_output_shapes
: 
�

mean/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@mean/count*
	container 
�
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: 
g
mean/count/readIdentity
mean/count*
_class
loc:@mean/count*
_output_shapes
: *
T0
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*
_output_shapes
: *

DstT0*

SrcT0
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
u
mean/SumSumdnn/head/weighted_loss/Sum
mean/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
_class
loc:@mean/total*
_output_shapes
: *
use_locking( *
T0
�
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^dnn/head/weighted_loss/Sum*
_class
loc:@mean/count*
_output_shapes
: *
use_locking( *
T0
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
b
mean/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
Z
mean/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_likeFillmean/zeros_like/shape_as_tensormean/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
_output_shapes
: *
T0
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
T0*
_output_shapes
: 
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
d
!mean/zeros_like_1/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
\
mean/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_like_1Fill!mean/zeros_like_1/shape_as_tensormean/zeros_like_1/Const*

index_type0*
_output_shapes
: *
T0
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
_output_shapes
: *
T0
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
_output_shapes
: *
T0
s

group_depsNoOp$^dnn/head/metrics/accuracy/update_op(^dnn/head/metrics/average_loss/update_op^mean/update_op
�
+eval_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@eval_step*
dtype0*
_output_shapes
: 
�
!eval_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
eval_step/Initializer/zerosFill+eval_step/Initializer/zeros/shape_as_tensor!eval_step/Initializer/zeros/Const*

index_type0*
_class
loc:@eval_step*
_output_shapes
: *
T0	
�
	eval_step
VariableV2*
_class
loc:@eval_step*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name 
�
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
_output_shapes
: *
T0	
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
_output_shapes
: *
use_locking(*
T0	
U
readIdentity	eval_step^group_deps
^AssignAdd*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
T0	*
_output_shapes
: 
�
initNoOp^global_step/Assign'^dnn/hiddenlayer_0/kernel/part_0/Assign%^dnn/hiddenlayer_0/bias/part_0/Assign'^dnn/hiddenlayer_1/kernel/part_0/Assign%^dnn/hiddenlayer_1/bias/part_0/Assign ^dnn/logits/kernel/part_0/Assign^dnn/logits/bias/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized#dnn/head/metrics/average_loss/total*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/head/metrics/accuracy/total*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_13"/device:CPU:0*

axis *
N*
_output_shapes
:*
T0

�
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0B#dnn/head/metrics/average_loss/totalB#dnn/head/metrics/average_loss/countBdnn/head/metrics/accuracy/totalBdnn/head/metrics/accuracy/countB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0
�
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
Tshape0*
_output_shapes
:*
T0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:���������*
T0

�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:���������*
squeeze_dims
*
T0	
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:���������
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_6"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
�
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
�
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
N*
_output_shapes
:*

Tidx0*
T0
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:���������
�
init_2NoOp+^dnn/head/metrics/average_loss/total/Assign+^dnn/head/metrics/average_loss/count/Assign'^dnn/head/metrics/accuracy/total/Assign'^dnn/head/metrics/accuracy/count/Assign^mean/total/Assign^mean/count/Assign^eval_step/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_all_tables^init_3
�
Merge/MergeSummaryMergeSummary-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_1da4411216da40d086aefb74c6e35db7/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*]
valueTBRB10 0,10B4 10 0,4:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
	2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*

axis *
N*
_output_shapes
:*
T0
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*]
valueTBRB10 0,10B4 10 0,4:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
	2	*H
_output_shapes6
4:
:
:
:

::
:
�
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:

�
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2:1*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
�
save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2:2*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
�
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2:3*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:

*
use_locking(*
T0
�
save/Assign_4Assigndnn/logits/bias/part_0save/RestoreV2:4*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assigndnn/logits/kernel/part_0save/RestoreV2:5*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:

�
save/Assign_6Assignglobal_stepsave/RestoreV2:6*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
-
save/restore_allNoOp^save/restore_shard�
�
-
_make_dataset_52536cc0
BatchDataset��
&TensorSliceDataset/tensors/component_0Const*�
value�B�"�������@������@333333�?      @      @333333@ffffff�?ffffff@ffffff@333333@������@�������?������@      @      @ffffff�?      @ffffff�?      �?      @333333�?      @ffffff@������@333333@ffffff@      �?������@������@333333@*
dtype0�
&TensorSliceDataset/tensors/component_1Const*�
value�B�"�      �?������ @      �?�������?�������?�������?�������?      �?�������?�������?������ @�������?       @�������?      �?�������?      �?�������?�������?       @�������?      �?�������?      �?333333�?ffffff�?�������?      �?      @�������?*
dtype0�
&TensorSliceDataset/tensors/component_2Const*�
value�B�"�������@������@ffffff@      @      @������@      @333333@ffffff@������@ffffff@333333@ffffff@      @      @ffffff@������@333333@ffffff@������@������@ffffff@333333@333333@333333@ffffff@������@������@������@������@*
dtype0�
&TensorSliceDataset/tensors/component_3Const*�
value�B�"�      @������@ffffff
@333333@      @333333@������@ffffff@      @      @      @      @ffffff@ffffff@������@      @������@333333@333333@      @333333@      @333333@      @������@      @ffffff@������@ffffff
@333333@*
dtype0�
&TensorSliceDataset/tensors/component_4Const*�
value�B�	"�                                                                                                                                                                                                                          *
dtype0	�
TensorSliceDatasetTensorSliceDataset/TensorSliceDataset/tensors/component_0:output:0/TensorSliceDataset/tensors/component_1:output:0/TensorSliceDataset/tensors/component_2:output:0/TensorSliceDataset/tensors/component_3:output:0/TensorSliceDataset/tensors/component_4:output:0*
output_shapes

: : : : : *
Toutput_types	
2	A
BatchDataset/batch_sizeConst*
value	B	 Rd*
dtype0	�
BatchDatasetBatchDatasetTensorSliceDataset:handle:0 BatchDataset/batch_size:output:0*
output_types	
2	*^
output_shapesM
K:���������:���������:���������:���������:���������"%
BatchDatasetBatchDataset:handle:0"���q�     Le+�	^{)%ŵ�AJ��
�(�(
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
�
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
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


LogicalNot
x

y

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
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
D
Relu
features"T
activations"T"
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
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
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
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
E
Where

input"T	
index	"%
Ttype0
:
2	
*1.7.02unknown��
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
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
_output_shapes
: *
T0	
�
tensors/component_0Const"/device:CPU:0*�
value�B�"�������@������@333333�?      @      @333333@ffffff�?ffffff@ffffff@333333@������@�������?������@      @      @ffffff�?      @ffffff�?      �?      @333333�?      @ffffff@������@333333@ffffff@      �?������@������@333333@*
dtype0*
_output_shapes
:
�
tensors/component_1Const"/device:CPU:0*�
value�B�"�      �?������ @      �?�������?�������?�������?�������?      �?�������?�������?������ @�������?       @�������?      �?�������?      �?�������?�������?       @�������?      �?�������?      �?333333�?ffffff�?�������?      �?      @�������?*
dtype0*
_output_shapes
:
�
tensors/component_2Const"/device:CPU:0*�
value�B�"�������@������@ffffff@      @      @������@      @333333@ffffff@������@ffffff@333333@ffffff@      @      @ffffff@������@333333@ffffff@������@������@ffffff@333333@333333@333333@ffffff@������@������@������@������@*
dtype0*
_output_shapes
:
�
tensors/component_3Const"/device:CPU:0*�
value�B�"�      @������@ffffff
@333333@      @333333@������@ffffff@      @      @      @      @ffffff@ffffff@������@      @������@333333@333333@      @333333@      @333333@      @������@      @ffffff@������@ffffff
@333333@*
dtype0*
_output_shapes
:
�
tensors/component_4Const"/device:CPU:0*�
value�B�	"�                                                                                                                                                                                                                          *
dtype0	*
_output_shapes
:
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
_make_dataset_52536cc0*
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
ExpandDimsIteratorGetNextEdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloatCastAdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeShape>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloat*
out_type0*
_output_shapes
:*
T0
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
Ddnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeJdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2*
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
>dnn/input_from_feature_columns/input_layer/PetalLength/ReshapeReshape>dnn/input_from_feature_columns/input_layer/PetalLength/ToFloatDdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims
ExpandDimsIteratorGetNext:1Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloatCast@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeShape=dnn/input_from_feature_columns/input_layer/PetalWidth/ToFloat*
out_type0*
_output_shapes
:*
T0
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
Ednn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
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
>dnn/input_from_feature_columns/input_layer/SepalLength/ToFloatCastAdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
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
Ddnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeJdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2*
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
Fdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ddnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
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
Cdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeIdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2*
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
=dnn/input_from_feature_columns/input_layer/SepalWidth/ReshapeReshape=dnn/input_from_feature_columns/input_layer/SepalWidth/ToFloatCdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
1dnn/input_from_feature_columns/input_layer/concatConcatV2>dnn/input_from_feature_columns/input_layer/PetalLength/Reshape=dnn/input_from_feature_columns/input_layer/PetalWidth/Reshape>dnn/input_from_feature_columns/input_layer/SepalLength/Reshape=dnn/input_from_feature_columns/input_layer/SepalWidth/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*'
_output_shapes
:���������*

Tidx0*
T0
�
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"   
   *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *b�'�*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *b�'?*2
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

:
*

seed *
T0
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
T0
�
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container 
�
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
�
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
T0
�
?dnn/hiddenlayer_0/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:
*0
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
:

�
dnn/hiddenlayer_0/bias/part_0
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
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
:
*
use_locking(*
T0
�
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:

s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes
:
*
T0
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:���������

k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:���������

[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*'
_output_shapes
:���������

x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*'
_output_shapes
:���������
*

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
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *�7�*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *�7?*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:

*

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
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:


�
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:

*
T0
�
dnn/hiddenlayer_1/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape
:

*
dtype0*
_output_shapes

:

*
shared_name 
�
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:

*
use_locking(*
T0
�
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:


�
?dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/shape_as_tensorConst*
valueB:
*0
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
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosFill?dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/shape_as_tensor5dnn/hiddenlayer_1/bias/part_0/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:

�
dnn/hiddenlayer_1/bias/part_0
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:

�
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
�
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:

s
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
T0*
_output_shapes

:


�
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
_output_shapes
:
*
T0
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
data_formatNHWC*'
_output_shapes
:���������
*
T0
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������

]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*'
_output_shapes
:���������
*
T0
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*'
_output_shapes
:���������
*

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
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *��-�*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *��-?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:

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

:

�
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

�
dnn/logits/kernel/part_0
VariableV2*+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
�
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

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
(dnn/logits/bias/part_0/Initializer/zerosFill8dnn/logits/bias/part_0/Initializer/zeros/shape_as_tensor.dnn/logits/bias/part_0/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
�
dnn/logits/bias/part_0
VariableV2*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
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
dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
T0
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes

:

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
dnn/zero_fraction_2/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_2/zero*
T0*'
_output_shapes
:���������
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
dnn/head/predictions/class_idsArgMaxdnn/logits/BiasAdd(dnn/head/predictions/class_ids/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
n
#dnn/head/predictions/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0	
�
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*
T0	*

fill *

scientific( *
width���������*'
_output_shapes
:���������*
	precision���������*
shortest( 
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
dnn/head/labels/ShapeShapednn/head/labels/ExpandDims*
out_type0*
_output_shapes
:*
T0	
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
:*
T0*
Index0*
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
dnn/head/labels/concatConcatV2dnn/head/labels/strided_slicednn/head/labels/concat/values_1dnn/head/labels/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
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
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_0dnn/head/labels/concat1dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/Shape*
T
2*
	summarize
�
dnn/head/labelsIdentitydnn/head/labels/ExpandDimsE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok+^dnn/head/labels/assert_equal/Assert/Assert*
T0	*'
_output_shapes
:���������
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
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
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
T

2		*
	summarize
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
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fIdentityUdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Vdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_idIdentity?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: *
T0

�
Sdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
�
adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tT^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
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
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
T
2	*
	summarize
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : 
�
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/labelsVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*"
_class
loc:@dnn/head/labels*:
_output_shapes(
&:���������:���������*
T0	
�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: *
T0

�
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
dnn/head/assert_range/IdentityIdentitydnn/head/labels;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*'
_output_shapes
:���������*
T0	
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
9dnn/head/sparse_softmax_cross_entropy_loss/xentropy/ShapeShapeOdnn/head/sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/Squeeze*
T0	*
out_type0*
_output_shapes
:
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
�
Ednn/head/metrics/average_loss/total/Initializer/zeros/shape_as_tensorConst*
valueB *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
;dnn/head/metrics/average_loss/total/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/total/Initializer/zerosFillEdnn/head/metrics/average_loss/total/Initializer/zeros/shape_as_tensor;dnn/head/metrics/average_loss/total/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/total
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
	container 
�
*dnn/head/metrics/average_loss/total/AssignAssign#dnn/head/metrics/average_loss/total5dnn/head/metrics/average_loss/total/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
validate_shape(*
_output_shapes
: 
�
(dnn/head/metrics/average_loss/total/readIdentity#dnn/head/metrics/average_loss/total*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
�
Ednn/head/metrics/average_loss/count/Initializer/zeros/shape_as_tensorConst*
valueB *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
;dnn/head/metrics/average_loss/count/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/count/Initializer/zerosFillEdnn/head/metrics/average_loss/count/Initializer/zeros/shape_as_tensor;dnn/head/metrics/average_loss/count/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/count
VariableV2*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
*dnn/head/metrics/average_loss/count/AssignAssign#dnn/head/metrics/average_loss/count5dnn/head/metrics/average_loss/count/Initializer/zeros*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
(dnn/head/metrics/average_loss/count/readIdentity#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: *
T0
h
#dnn/head/metrics/average_loss/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Rdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/ExpandDims*
out_type0*
_output_shapes
:*
T0
�
Pdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
h
`dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ShapeShapednn/head/ExpandDimsa^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ConstConsta^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/metrics/average_loss/broadcast_weights/ones_likeFill?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Shape?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
/dnn/head/metrics/average_loss/broadcast_weightsMul#dnn/head/metrics/average_loss/Const9dnn/head/metrics/average_loss/broadcast_weights/ones_like*'
_output_shapes
:���������*
T0
�
!dnn/head/metrics/average_loss/MulMuldnn/head/ExpandDims/dnn/head/metrics/average_loss/broadcast_weights*'
_output_shapes
:���������*
T0
v
%dnn/head/metrics/average_loss/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
!dnn/head/metrics/average_loss/SumSum/dnn/head/metrics/average_loss/broadcast_weights%dnn/head/metrics/average_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
v
%dnn/head/metrics/average_loss/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
�
#dnn/head/metrics/average_loss/Sum_1Sum!dnn/head/metrics/average_loss/Mul%dnn/head/metrics/average_loss/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/AssignAdd	AssignAdd#dnn/head/metrics/average_loss/total#dnn/head/metrics/average_loss/Sum_1*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: *
use_locking( *
T0
�
)dnn/head/metrics/average_loss/AssignAdd_1	AssignAdd#dnn/head/metrics/average_loss/count!dnn/head/metrics/average_loss/Sum"^dnn/head/metrics/average_loss/Mul*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: *
use_locking( *
T0
�
%dnn/head/metrics/average_loss/truedivRealDiv(dnn/head/metrics/average_loss/total/read(dnn/head/metrics/average_loss/count/read*
T0*
_output_shapes
: 
{
8dnn/head/metrics/average_loss/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
s
.dnn/head/metrics/average_loss/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
(dnn/head/metrics/average_loss/zeros_likeFill8dnn/head/metrics/average_loss/zeros_like/shape_as_tensor.dnn/head/metrics/average_loss/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
�
%dnn/head/metrics/average_loss/GreaterGreater(dnn/head/metrics/average_loss/count/read(dnn/head/metrics/average_loss/zeros_like*
_output_shapes
: *
T0
�
#dnn/head/metrics/average_loss/valueSelect%dnn/head/metrics/average_loss/Greater%dnn/head/metrics/average_loss/truediv(dnn/head/metrics/average_loss/zeros_like*
_output_shapes
: *
T0
�
'dnn/head/metrics/average_loss/truediv_1RealDiv'dnn/head/metrics/average_loss/AssignAdd)dnn/head/metrics/average_loss/AssignAdd_1*
T0*
_output_shapes
: 
}
:dnn/head/metrics/average_loss/zeros_like_1/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
u
0dnn/head/metrics/average_loss/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
*dnn/head/metrics/average_loss/zeros_like_1Fill:dnn/head/metrics/average_loss/zeros_like_1/shape_as_tensor0dnn/head/metrics/average_loss/zeros_like_1/Const*
T0*

index_type0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/Greater_1Greater)dnn/head/metrics/average_loss/AssignAdd_1*dnn/head/metrics/average_loss/zeros_like_1*
_output_shapes
: *
T0
�
'dnn/head/metrics/average_loss/update_opSelect'dnn/head/metrics/average_loss/Greater_1'dnn/head/metrics/average_loss/truediv_1*dnn/head/metrics/average_loss/zeros_like_1*
T0*
_output_shapes
: 
[
dnn/head/metrics/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/EqualEqualdnn/head/predictions/ExpandDimsdnn/head/assert_range/Identity*
T0	*'
_output_shapes
:���������
y
dnn/head/metrics/ToFloatCastdnn/head/metrics/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

�
Adnn/head/metrics/accuracy/total/Initializer/zeros/shape_as_tensorConst*
valueB *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
7dnn/head/metrics/accuracy/total/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
1dnn/head/metrics/accuracy/total/Initializer/zerosFillAdnn/head/metrics/accuracy/total/Initializer/zeros/shape_as_tensor7dnn/head/metrics/accuracy/total/Initializer/zeros/Const*

index_type0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: *
T0
�
dnn/head/metrics/accuracy/total
VariableV2*
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
	container *
shape: *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/accuracy/total/AssignAssigndnn/head/metrics/accuracy/total1dnn/head/metrics/accuracy/total/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
validate_shape(*
_output_shapes
: 
�
$dnn/head/metrics/accuracy/total/readIdentitydnn/head/metrics/accuracy/total*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
�
Adnn/head/metrics/accuracy/count/Initializer/zeros/shape_as_tensorConst*
valueB *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
7dnn/head/metrics/accuracy/count/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
1dnn/head/metrics/accuracy/count/Initializer/zerosFillAdnn/head/metrics/accuracy/count/Initializer/zeros/shape_as_tensor7dnn/head/metrics/accuracy/count/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: 
�
dnn/head/metrics/accuracy/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
	container 
�
&dnn/head/metrics/accuracy/count/AssignAssigndnn/head/metrics/accuracy/count1dnn/head/metrics/accuracy/count/Initializer/zeros*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
$dnn/head/metrics/accuracy/count/readIdentitydnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: *
T0
�
Ndnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/metrics/ToFloat*
out_type0*
_output_shapes
:*
T0
�
Ldnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
d
\dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ShapeShapednn/head/metrics/ToFloat]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ConstConst]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/accuracy/broadcast_weights/ones_likeFill;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Shape;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
+dnn/head/metrics/accuracy/broadcast_weightsMuldnn/head/metrics/Const5dnn/head/metrics/accuracy/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
dnn/head/metrics/accuracy/MulMuldnn/head/metrics/ToFloat+dnn/head/metrics/accuracy/broadcast_weights*
T0*'
_output_shapes
:���������
p
dnn/head/metrics/accuracy/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/SumSum+dnn/head/metrics/accuracy/broadcast_weightsdnn/head/metrics/accuracy/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
r
!dnn/head/metrics/accuracy/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/Sum_1Sumdnn/head/metrics/accuracy/Mul!dnn/head/metrics/accuracy/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/AssignAdd	AssignAdddnn/head/metrics/accuracy/totaldnn/head/metrics/accuracy/Sum_1*
use_locking( *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
�
%dnn/head/metrics/accuracy/AssignAdd_1	AssignAdddnn/head/metrics/accuracy/countdnn/head/metrics/accuracy/Sum^dnn/head/metrics/accuracy/Mul*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: *
use_locking( *
T0
�
!dnn/head/metrics/accuracy/truedivRealDiv$dnn/head/metrics/accuracy/total/read$dnn/head/metrics/accuracy/count/read*
_output_shapes
: *
T0
w
4dnn/head/metrics/accuracy/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
o
*dnn/head/metrics/accuracy/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$dnn/head/metrics/accuracy/zeros_likeFill4dnn/head/metrics/accuracy/zeros_like/shape_as_tensor*dnn/head/metrics/accuracy/zeros_like/Const*

index_type0*
_output_shapes
: *
T0
�
!dnn/head/metrics/accuracy/GreaterGreater$dnn/head/metrics/accuracy/count/read$dnn/head/metrics/accuracy/zeros_like*
_output_shapes
: *
T0
�
dnn/head/metrics/accuracy/valueSelect!dnn/head/metrics/accuracy/Greater!dnn/head/metrics/accuracy/truediv$dnn/head/metrics/accuracy/zeros_like*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/truediv_1RealDiv#dnn/head/metrics/accuracy/AssignAdd%dnn/head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
y
6dnn/head/metrics/accuracy/zeros_like_1/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
q
,dnn/head/metrics/accuracy/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/accuracy/zeros_like_1Fill6dnn/head/metrics/accuracy/zeros_like_1/shape_as_tensor,dnn/head/metrics/accuracy/zeros_like_1/Const*
T0*

index_type0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/Greater_1Greater%dnn/head/metrics/accuracy/AssignAdd_1&dnn/head/metrics/accuracy/zeros_like_1*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/update_opSelect#dnn/head/metrics/accuracy/Greater_1#dnn/head/metrics/accuracy/truediv_1&dnn/head/metrics/accuracy/zeros_like_1*
_output_shapes
: *
T0
�
,mean/total/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
"mean/total/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
mean/total/Initializer/zerosFill,mean/total/Initializer/zeros/shape_as_tensor"mean/total/Initializer/zeros/Const*

index_type0*
_class
loc:@mean/total*
_output_shapes
: *
T0
�

mean/total
VariableV2*
shared_name *
_class
loc:@mean/total*
	container *
shape: *
dtype0*
_output_shapes
: 
�
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: 
g
mean/total/readIdentity
mean/total*
T0*
_class
loc:@mean/total*
_output_shapes
: 
�
,mean/count/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
"mean/count/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
mean/count/Initializer/zerosFill,mean/count/Initializer/zeros/shape_as_tensor"mean/count/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@mean/count*
_output_shapes
: 
�

mean/count
VariableV2*
shared_name *
_class
loc:@mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: 
g
mean/count/readIdentity
mean/count*
T0*
_class
loc:@mean/count*
_output_shapes
: 
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*
_output_shapes
: *

DstT0*

SrcT0
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
u
mean/SumSumdnn/head/weighted_loss/Sum
mean/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
_class
loc:@mean/total*
_output_shapes
: *
use_locking( *
T0
�
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^dnn/head/weighted_loss/Sum*
_class
loc:@mean/count*
_output_shapes
: *
use_locking( *
T0
Z
mean/truedivRealDivmean/total/readmean/count/read*
_output_shapes
: *
T0
b
mean/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
Z
mean/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_likeFillmean/zeros_like/shape_as_tensormean/zeros_like/Const*

index_type0*
_output_shapes
: *
T0
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
_output_shapes
: *
T0
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
_output_shapes
: *
T0
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
_output_shapes
: *
T0
d
!mean/zeros_like_1/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
\
mean/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_like_1Fill!mean/zeros_like_1/shape_as_tensormean/zeros_like_1/Const*
T0*

index_type0*
_output_shapes
: 
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
_output_shapes
: *
T0
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
_output_shapes
: *
T0
s

group_depsNoOp$^dnn/head/metrics/accuracy/update_op(^dnn/head/metrics/average_loss/update_op^mean/update_op
�
+eval_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@eval_step*
dtype0*
_output_shapes
: 
�
!eval_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
eval_step/Initializer/zerosFill+eval_step/Initializer/zeros/shape_as_tensor!eval_step/Initializer/zeros/Const*

index_type0*
_class
loc:@eval_step*
_output_shapes
: *
T0	
�
	eval_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@eval_step*
	container 
�
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
_output_shapes
: *
T0	
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
_output_shapes
: *
use_locking(*
T0	
U
readIdentity	eval_step^group_deps
^AssignAdd*
_output_shapes
: *
T0	
;
IdentityIdentityread*
_output_shapes
: *
T0	
�
initNoOp^global_step/Assign'^dnn/hiddenlayer_0/kernel/part_0/Assign%^dnn/hiddenlayer_0/bias/part_0/Assign'^dnn/hiddenlayer_1/kernel/part_0/Assign%^dnn/hiddenlayer_1/bias/part_0/Assign ^dnn/logits/kernel/part_0/Assign^dnn/logits/bias/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized#dnn/head/metrics/average_loss/total*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/head/metrics/accuracy/total*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_13"/device:CPU:0*

axis *
N*
_output_shapes
:*
T0

�
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0B#dnn/head/metrics/average_loss/totalB#dnn/head/metrics/average_loss/countBdnn/head/metrics/accuracy/totalBdnn/head/metrics/accuracy/countB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
�
3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0
�
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:���������*
T0

�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:���������
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_6"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
�
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
�
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
�
5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*

axis *
N*
_output_shapes
:*
T0
�
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
N*
_output_shapes
:*

Tidx0*
T0
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze"/device:CPU:0*#
_output_shapes
:���������*
Tindices0	*
validate_indices(*
Tparams0
�
init_2NoOp+^dnn/head/metrics/average_loss/total/Assign+^dnn/head/metrics/average_loss/count/Assign'^dnn/head/metrics/accuracy/total/Assign'^dnn/head/metrics/accuracy/count/Assign^mean/total/Assign^mean/count/Assign^eval_step/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_all_tables^init_3
�
Merge/MergeSummaryMergeSummary-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_1da4411216da40d086aefb74c6e35db7/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*]
valueTBRB10 0,10B4 10 0,4:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
	2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*]
valueTBRB10 0,10B4 10 0,4:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*H
_output_shapes6
4:
:
:
:

::
:*
dtypes
	2	
�
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:

�
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2:1*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
�
save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2:2*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
�
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2:3*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:

*
use_locking(*
T0
�
save/Assign_4Assigndnn/logits/bias/part_0save/RestoreV2:4*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assigndnn/logits/kernel/part_0save/RestoreV2:5*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:

�
save/Assign_6Assignglobal_stepsave/RestoreV2:6*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
-
save/restore_allNoOp^save/restore_shard�
�
-
_make_dataset_52536cc0
BatchDataset��
&TensorSliceDataset/tensors/component_0Const*�
value�B�"�������@������@333333�?      @      @333333@ffffff�?ffffff@ffffff@333333@������@�������?������@      @      @ffffff�?      @ffffff�?      �?      @333333�?      @ffffff@������@333333@ffffff@      �?������@������@333333@*
dtype0�
&TensorSliceDataset/tensors/component_1Const*�
value�B�"�      �?������ @      �?�������?�������?�������?�������?      �?�������?�������?������ @�������?       @�������?      �?�������?      �?�������?�������?       @�������?      �?�������?      �?333333�?ffffff�?�������?      �?      @�������?*
dtype0�
&TensorSliceDataset/tensors/component_2Const*�
value�B�"�������@������@ffffff@      @      @������@      @333333@ffffff@������@ffffff@333333@ffffff@      @      @ffffff@������@333333@ffffff@������@������@ffffff@333333@333333@333333@ffffff@������@������@������@������@*
dtype0�
&TensorSliceDataset/tensors/component_3Const*�
value�B�"�      @������@ffffff
@333333@      @333333@������@ffffff@      @      @      @      @ffffff@ffffff@������@      @������@333333@333333@      @333333@      @333333@      @������@      @ffffff@������@ffffff
@333333@*
dtype0�
&TensorSliceDataset/tensors/component_4Const*�
value�B�	"�                                                                                                                                                                                                                          *
dtype0	�
TensorSliceDatasetTensorSliceDataset/TensorSliceDataset/tensors/component_0:output:0/TensorSliceDataset/tensors/component_1:output:0/TensorSliceDataset/tensors/component_2:output:0/TensorSliceDataset/tensors/component_3:output:0/TensorSliceDataset/tensors/component_4:output:0*
output_shapes

: : : : : *
Toutput_types	
2	A
BatchDataset/batch_sizeConst*
value	B	 Rd*
dtype0	�
BatchDatasetBatchDatasetTensorSliceDataset:handle:0 BatchDataset/batch_size:output:0*
output_types	
2	*^
output_shapesM
K:���������:���������:���������:���������:���������"%
BatchDatasetBatchDataset:handle:0""�
local_variables��
�
%dnn/head/metrics/average_loss/total:0*dnn/head/metrics/average_loss/total/Assign*dnn/head/metrics/average_loss/total/read:027dnn/head/metrics/average_loss/total/Initializer/zeros:0
�
%dnn/head/metrics/average_loss/count:0*dnn/head/metrics/average_loss/count/Assign*dnn/head/metrics/average_loss/count/read:027dnn/head/metrics/average_loss/count/Initializer/zeros:0
�
!dnn/head/metrics/accuracy/total:0&dnn/head/metrics/accuracy/total/Assign&dnn/head/metrics/accuracy/total/read:023dnn/head/metrics/accuracy/total/Initializer/zeros:0
�
!dnn/head/metrics/accuracy/count:0&dnn/head/metrics/accuracy/count/Assign&dnn/head/metrics/accuracy/count/read:023dnn/head/metrics/accuracy/count/Initializer/zeros:0
T
mean/total:0mean/total/Assignmean/total/read:02mean/total/Initializer/zeros:0
T
mean/count:0mean/count/Assignmean/count/read:02mean/count/Initializer/zeros:0
P
eval_step:0eval_step/Assigneval_step/read:02eval_step/Initializer/zeros:0"!
local_init_op

group_deps_2"�

	variables�	�	
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel
  "
2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias
 "
21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kernel

  "

2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias
 "
21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel
  "
25dnn/logits/kernel/part_0/Initializer/random_uniform:0
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"
ready_op


concat:0"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"\
lossesR
P
0dnn/head/sparse_softmax_cross_entropy_loss/Mul:0
dnn/head/weighted_loss/Sum:0" 
global_step

global_step:0"&

summary_op

Merge/MergeSummary:0"�
	summaries�
�
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"�	
trainable_variables�	�	
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel
  "
2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias
 "
21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kernel

  "

2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias
 "
21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel
  "
25dnn/logits/kernel/part_0/Initializer/random_uniform:0
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"
init_op

group_deps_1"
	eval_step

eval_step:0"�&
cond_context�&�&
�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/cond_text>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0 *�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency:0
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0�
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0
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
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0�
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
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
Adnn/head/assert_range/assert_non_negative/assert_less_equal/All:0^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch:0"�
metric_variables�
�
%dnn/head/metrics/average_loss/total:0
%dnn/head/metrics/average_loss/count:0
!dnn/head/metrics/accuracy/total:0
!dnn/head/metrics/accuracy/count:0
mean/total:0
mean/count:0���@       (��	�)%ŵ�Ad*3

average_lossƿu>

accuracy��n?

loss�c�@H��
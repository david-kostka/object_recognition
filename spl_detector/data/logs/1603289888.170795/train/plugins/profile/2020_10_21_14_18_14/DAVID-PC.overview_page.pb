�%  *     �@@4333�=A2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[1]::ParallelMapV2�	h"l�x@!��q"�X@)�	h"l�x@1��q"�X@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2�%��x@!`"�iL�X@)6�;Nё�?1+^F�^ż?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenatee�`TR�x@!�(CC
�X@)�?Ƭ?1AZ�+��?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate<Nё\�x@!L�a�X@)M�St$�?1�E��6N�?:Preprocessing2F
Iterator::Model�b�=y�?!m�zg��?)�MbX9�?1��=B�]�?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[1]::ParallelMapV2::FlatMap[0]::TFRecordL7�A`�?!3=� ��?)L7�A`�?13=� ��?:Advanced file read2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[1]::ParallelMapV2::FlatMap��j+���?!�J9�\͍?)a��+e�?1�"���y?:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::Shuffle��o_�x@!Ȑ:�X@)K�=�U�?1Y�<�%�o?:Preprocessing2�
XIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate2�%��x@!�m���X@)�+e�X�?1�A<�g?:Preprocessing2�
xIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate�i�q��x@!i+�D�X@)Ǻ����?1�IBjg?:Preprocessing2�
hIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenateh��s��x@!Cg��o�X@)/�$��?1)e�ӧe?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate�lV}�x@!od�5�X@)/�$��?1)e�ӧe?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenateı.n��x@!"�h�[�X@)Έ����?1��?.c?:Preprocessing2P
Iterator::Model::Prefetch�St$���?!>��PCa?)�St$���?1>��PCa?:Preprocessing2
HIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate/�$�x@!�&&��X@)�St$���?1>��PCa?:Preprocessing2o
8Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate㥛� �x@!O����X@)9��v��z?1W���Z?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q2>0����?"�
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.CDAVID-PC: Failed to load libcupti (is it installed and accessible?)
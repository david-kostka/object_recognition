�%  *43333�L@hfff�k�@2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[1]::ParallelMapV2V-��X@!0s� [�X@)V-��X@10s� [�X@:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate��9#J�X@!"oN��X@)��S㥛�?1��;��?:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2�� ��X@!U�����X@)��~j�t�?1rӦ͐p�?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[1]::ParallelMapV2::FlatMap[0]::TFRecord�u����?!���趞�?)�u����?1���趞�?:Advanced file read2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate"�uq�X@!��ɩ��X@)���9#J�?1.B#�ID�?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[1]::ParallelMapV2::FlatMap���o_�?!�
�GmZ�?)�o_��?1��H�#�?:Preprocessing2F
Iterator::Modeln���?!�@[h��?)��镲�?1��	�?:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::Shuffle�1�%�X@!��m���X@)Q�|a2�?1*�PY�-�?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate���9#�X@!ڀ�9��X@)?�ܵ�|�?1���x�?:Preprocessing2�
hIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate�|a2�X@!�T����X@)�~j�t��?11,0��?:Preprocessing2�
xIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate�8��m�X@!�gl�X@)n���?1�@[h��?:Preprocessing2
HIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenateu�V�X@!i�B��X@);�O��n�?1Q�$!�j�?:Preprocessing2�
XIterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate�(\���X@!C���h�X@)/n���?1D*W��?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::ConcatenateyX�5��X@!'���s�X@)����Mb�?1� ��^�?:Preprocessing2o
8Iterator::Model::Prefetch::BatchV2::Shuffle::Concatenate;pΈ��X@!���x�X@)ŏ1w-!?1��
I�?:Preprocessing2P
Iterator::Model::Prefetch��0�*x?!
mck%x?)��0�*x?1
mck%x?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q� ��E@"�
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
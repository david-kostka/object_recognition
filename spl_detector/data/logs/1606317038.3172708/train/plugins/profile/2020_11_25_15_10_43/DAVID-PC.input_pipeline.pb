  *fffffFX@����	�@2q
:Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2����U@!P�:\<(T@)����U@1P�:\<(T@:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::ParallelMapV2d]�F�3@!����2@)d]�F�3@1����2@:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::ShuffleTt$���U@!�����0T@)�{�Pk�?1�h���D�?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::ParallelMapV2::FlatMap[0]::TFRecord�q��۸?!��VSN�?)�q��۸?1��VSN�?:Advanced file read2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::ConcatenateM�O�3@!
�g��2@)����K�?1�a���׵?:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2<Nё\�U@!$�U �5T@)��g��s�?1~~:��?:Preprocessing2~
GIterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate�>W[�?4@!s����2@)X9��v��?1��?S3í?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::ParallelMapV2::FlatMapvOjM�?!{uL(��?)t$���~�?1c�u�5ǩ?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate��j+�4@!	��?��2@)X�5�;N�?1f��s�9�?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate�����4@!<��g�2@)O��e�c�?10{
�獛?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate���1�4@!��i_�2@)��6��?1����?:Preprocessing2F
Iterator::Model7�[ A�?!�#�ug-�?)p_�Q�?1� ����?:Preprocessing2�
�Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate�g��s4@!� �h��2@)-C��6�?1OI�#���?:Preprocessing2�
gIterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::ConcatenateDio��)4@!�X�U�2@)V}��b�?1NF�>��?:Preprocessing2�
wIterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate[0]::Concatenate[0]::Concatenate���&#4@!��ҹ[�2@)a��+e�?1��B\ϗ?:Preprocessing2�
WIterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Concatenate[0]::Concatenate����/4@!y��B=�2@)�(��0�?1r RJ4��?:Preprocessing2P
Iterator::Model::Prefetch����Mb�?!��,��~?)����Mb�?1��,��~?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.
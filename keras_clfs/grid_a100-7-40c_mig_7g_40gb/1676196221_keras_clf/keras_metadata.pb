
�wroot"_tf_keras_sequential*�w{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "c0_input"}}, {"class_name": "Conv2D", "config": {"name": "c0", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "c1", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "m2", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dr3", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "rate": 0.125, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "c4", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "c5", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "m6", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dr7", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "rate": 0.125, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "f8", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "d9", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096]}, "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dr10", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "rate": 0.125, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "d11", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 37, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "float32", "c0_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "float32", "c0_input"]}, "keras_version": "2.12.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "c0_input"}, "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "c0", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 3}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 4}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5}, {"class_name": "Conv2D", "config": {"name": "c1", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 8}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 9}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "MaxPooling2D", "config": {"name": "m2", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "dr3", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "rate": 0.125, "noise_shape": null, "seed": null}, "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "c4", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 15}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 16}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "c5", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 20}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 21}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22}, {"class_name": "MaxPooling2D", "config": {"name": "m6", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dropout", "config": {"name": "dr7", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "rate": 0.125, "noise_shape": null, "seed": null}, "shared_object_id": 24}, {"class_name": "Flatten", "config": {"name": "f8", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "d9", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096]}, "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 28}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 29}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 30}, {"class_name": "Dropout", "config": {"name": "dr10", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "rate": 0.125, "noise_shape": null, "seed": null}, "shared_object_id": 31}, {"class_name": "Dense", "config": {"name": "d11", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 34}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 35}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 39}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": true, "is_legacy_optimizer": false, "learning_rate": 0.0010000000474974513, "beta_1": 0.9, "beta_2": 0.9, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "c0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "c0", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 3}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 4}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "c1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "c1", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 8}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 9}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}2
�root.layer-2"_tf_keras_layer*�{"name": "m2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MaxPooling2D", "config": {"name": "m2", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}2
�root.layer-3"_tf_keras_layer*�{"name": "dr3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dr3", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "rate": 0.125, "noise_shape": null, "seed": null}, "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "c4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "c4", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 32]}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 15}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 16}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "c5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "c5", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 20}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 21}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}2
�root.layer-6"_tf_keras_layer*�{"name": "m6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MaxPooling2D", "config": {"name": "m6", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}2
�root.layer-7"_tf_keras_layer*�{"name": "dr7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dr7", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "rate": 0.125, "noise_shape": null, "seed": null}, "shared_object_id": 24, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}2
�	root.layer-8"_tf_keras_layer*�{"name": "f8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Flatten", "config": {"name": "f8", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 64]}, "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}2
�

root.layer_with_weights-4"_tf_keras_layer*�	{"name": "d9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "d9", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096]}, "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 28}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 29}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096]}}2
�
�
root.layer_with_weights-5"_tf_keras_layer*�	{"name": "d11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "d11", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 34}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}, "shared_object_id": 35}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 39}2
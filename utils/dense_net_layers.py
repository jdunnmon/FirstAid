#annhe

def _count_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))

def conv2d(_input, out_features, kernel_size,
            strides=[1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = self.weight_variable_msra(
        [kernel_size, kernel_size, in_features, out_features],
        name='kernel')
    output = tf.nn.conv2d(_input, kernel, strides, padding)
    return output

def avg_pool(_input, k):
    ksize = [1, k, k, 1]
    strides = [1, k, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(_input, ksize, strides, padding)
    return output

def batch_norm(_input, is_training):
    output = tf.contrib.layers.batch_norm(
        _input, scale=True, is_training=is_training,
        updates_collections=None)
    return output

def dropout(_input):
    if keep_prob < 1:
        output = tf.cond(
            self.is_training,
            lambda: tf.nn.dropout(_input, keep_prob),
            lambda: _input
        )
    else:
        output = _input
    return output

def weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer())

def weight_variable_xavier(shape, name):
    return tf.get_variable(
        name,
        shape=shape,
        initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)

def composite_function(_input, is_training, out_features, kernel_size=3):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        # BN
        output = self.batch_norm(_input, is_training)
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = self.conv2d(
            output, out_features=out_features, kernel_size=kernel_size)
        # dropout(in case of training and in case it is no 1.0)
        output = self.dropout(output)
    return output

def bottleneck(_input, is_training, out_features):
    with tf.variable_scope("bottleneck"):
        output = self.batch_norm(_input, is_training)
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = self.conv2d(
            output, out_features=inter_features, kernel_size=1,
            padding='VALID')
        output = self.dropout(output)
    return output

def add_internal_layer(_input, is_training, growth_rate):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not self.bc_mode:
        comp_out = composite_function(
            _input, out_features=growth_rate, kernel_size=3)
    elif self.bc_mode:
        bottleneck_out = bottleneck(_input, is_training, out_features=growth_rate)
        comp_out = composite_function(
            bottleneck_out, out_features=growth_rate, kernel_size=3)
    # concatenate _input with out from composite function
    if TF_VERSION >= 1.0:
        output = tf.concat(axis=3, values=(_input, comp_out))
    else:
        output = tf.concat(3, (_input, comp_out))
    return output

def add_block(_input, growth_rate, layers_per_block):
    """Add N H_l internal layers"""
    output = _input
    for layer in range(layers_per_block):
        with tf.variable_scope("layer_%d" % layer):
            output = add_internal_layer(output, growth_rate)
    return output

def transition_layer(_input):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    # ANN: what is self.reduction???
    out_features = int(int(_input.get_shape()[-1]) * self.reduction)
    output = composite_function(
        _input, out_features=out_features, kernel_size=1)
    # run average pooling
    output = self.avg_pool(output, k=2)
    return output

def transition_layer_to_classes(_input):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide average pooling
    - FC layer multiplication
    """
    # BN
    output = self.batch_norm(_input)
    # ReLU
    output = tf.nn.relu(output)
    # average pooling
    last_pool_kernel = int(output.get_shape()[-2])
    output = self.avg_pool(output, k=last_pool_kernel)
    # FC
    features_total = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    W = self.weight_variable_xavier(
        [features_total, self.n_classes], name='W')
    bias = self.bias_variable([self.n_classes])
    logits = tf.matmul(output, W) + bias
    return logits

def Dense_Net(_input, is_training, growth_rate, layers_per_block, first_output_features, total_blocks, keep_prob=1.0, reduction=1.0):
    # first - initial 3 x 3 conv to first_output_features
    with tf.variable_scope("Initial_convolution"):
        output = self.conv2d(
            _input,
            out_features=first_output_features,
            kernel_size=3)

    # add N required blocks
    for block in range(total_blocks):
        with tf.variable_scope("Block_%d" % block):
            output = add_block(output, growth_rate, layers_per_block)
        # last block exist without transition layer
        if block != total_blocks - 1:
            with tf.variable_scope("Transition_after_block_%d" % block):
                output = self.transition_layer(output)

    with tf.variable_scope("Transition_to_classes"):
        logits = transition_layer_to_classes(output)
    prediction = tf.nn.softmax(logits)
    return logits

#annhe
import tensorflow as tf

def _count_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))

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

def conv2d(_input, out_features, kernel_size,
            strides=[1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = weight_variable_msra(
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
    #print "TYPE OF IS TRAINING ", type(is_training)
    #print "SHAPE OF IS TRAINING ", tf.shape(is_training)
    output = tf.contrib.layers.batch_norm(
        _input, scale=True, is_training=is_training,
        updates_collections=None)
    return output

def dropout(_input, is_training, keep_prob):
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda: tf.nn.dropout(_input, keep_prob),
            lambda: _input
        )
    else:
        output = _input
    return output

def composite_function(_input, is_training, out_features, keep_prob, kernel_size=3):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        # BN
        output = batch_norm(_input, is_training)
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = conv2d(
            output, out_features=out_features, kernel_size=kernel_size)
        # dropout(in case of training and in case it is no 1.0)
        output = dropout(output, is_training, keep_prob)
    return output

def bottleneck(_input, is_training, out_features):
    with tf.variable_scope("bottleneck"):
        output = batch_norm(_input, is_training)
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = conv2d(
            output, out_features=inter_features, kernel_size=1,
            padding='VALID')
        output = dropout(output, is_training)
    return output

def add_internal_layer(_input, is_training, growth_rate, bc_mode, keep_prob):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not bc_mode:
        comp_out = composite_function(
            _input, is_training, out_features=growth_rate, keep_prob=keep_prob, kernel_size=3)
    elif bc_mode:
        bottleneck_out = bottleneck(_input, is_training, out_features=growth_rate)
        comp_out = composite_function(
            bottleneck_out, out_features=growth_rate, keep_prob=keep_prob, kernel_size=3)
    # concatenate _input with out from composite function
    output = tf.concat(axis=3, values=(_input, comp_out))
    return output

def add_block(_input, is_training, growth_rate, layers_per_block, bc_mode, keep_prob):
    """Add N H_l internal layers"""
    output = _input
    for layer in range(layers_per_block):
        with tf.variable_scope("layer_%d" % layer):
            output = add_internal_layer(output, is_training, growth_rate, bc_mode, keep_prob)
    return output

def transition_layer(_input, is_training, reduction, keep_prob):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    # ANN: what is self.reduction???
    out_features = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(
        _input, is_training, out_features=out_features, keep_prob=keep_prob, kernel_size=1)
    # run average pooling
    output = avg_pool(output, k=2)
    return output

def transition_layer_to_classes(_input, is_training, n_classes):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide average pooling
    - FC layer multiplication
    """
    # BN
    output = batch_norm(_input, is_training)
    # ReLU
    output = tf.nn.relu(output)
    # average pooling
    last_pool_kernel = int(output.get_shape()[-2])
    output = avg_pool(output, k=last_pool_kernel)
    # FC
    features_total = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    W = weight_variable_xavier(
        [features_total, n_classes], name='W')
    bias = bias_variable([n_classes])
    logits = tf.matmul(output, W) + bias
    return logits

def max_pool(layer, k=2, stride=None, padding='SAME'):
    """
    A simple 2-dimensional max pooling layer.
    Strides and size of max pool kernel is constrained to be the same.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - k: (int) size of the max_filter to be made.
    - stride: (int) size of stride
    """
    if stride==None:
        stride=k
    # Doing the Max Pool
    stride_var = [1, stride, stride, 1]
    kernel_var = [1, k, k, 1]

    max_layer = tf.nn.max_pool(layer, kernel_var, stride_var, padding)
    return max_layer

def Dense_Net(_input, is_training, growth_rate, layers_per_block, first_output_features, total_blocks, keep_prob=1.0, reduction=1.0, bc_mode=0, n_classes=2):
    # first - initial 3 x 3 conv to first_output_features
    #print "FROM DENSE NET TYPE OF IS TRAINING ", type(is_training)
    #print "FROM DENSE NET SHAPE OF IS TRAINING ", tf.shape(is_training)
    is_training = tf.cast(is_training, tf.bool)
    # if is_training is not None:
    #     is_training = True
    # else:
    #     is_training = False
    with tf.variable_scope("Initial_convolution"):
        if _input.shape[1] == 32:
            output = conv2d(
                _input,
                out_features=first_output_features,
                kernel_size=3)
        elif _input.shape[1] == 224:
            output = conv2d(
                _input,
                out_features=first_output_features,
                kernel_size=7, strides=[1,2,2,1], padding='SAME')
            output = batch_norm(output, is_training)
            output = tf.nn.relu(output)
            output = max_pool(output, k=3, stride=2, padding='SAME')


    # add N required blocks
    for block in range(total_blocks):
        with tf.variable_scope("Block_%d" % block):
            output = add_block(output, is_training, growth_rate, layers_per_block, bc_mode, keep_prob)
        # last block exist without transition layer
        if block != total_blocks - 1:
            with tf.variable_scope("Transition_after_block_%d" % block):
                output = transition_layer(output, is_training, reduction, keep_prob)
    #print "SHAPE OF LAST OUTPUT ", tf.shape(output)
    with tf.variable_scope("Transition_to_classes"):
        logits = transition_layer_to_classes(output, is_training, n_classes)
    prediction = tf.nn.softmax(logits)
    #print "SHAPE OF LOGITS ", tf.shape(logits)
    #print "SHAPE OF PREDICTION ", tf.shape(prediction)
    return logits

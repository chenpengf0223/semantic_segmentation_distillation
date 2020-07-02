import tensorflow as tf


def Huber_loss(x,y):
    with tf.name_scope('Huber_loss'):
        return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x-y), 1.), 
                                        tf.square(x-y)/2, tf.abs(x-y)-1/2))
    
def distance_wise_potential(x):
    with tf.name_scope('DwP'):
        x_square = tf.reduce_sum(tf.square(x), -1)
        prod = tf.matmul(x, x, transpose_b=True)
        distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square,1) + 
            tf.expand_dims(x_square,0) - 2*prod, 1e-12))
        mu = tf.reduce_sum(distance) / tf.reduce_sum(tf.where(distance > 0., 
            tf.ones_like(distance), tf.zeros_like(distance)))
        return distance / (mu + 1e-8)
    
def angle_wise_potential(x):
    with tf.name_scope('AwP'):
        e = tf.expand_dims(x, 0) - tf.expand_dims(x, 1)
        e_norm = tf.nn.l2_normalize(e, 2)
    return tf.matmul(e_norm, e_norm, transpose_b=True)

def relational_knowledge_distillation(source, target):
    '''
    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.  
    relational knowledge distillation.
    arXiv preprint arXiv:1904.05068, 2019.
    '''
    with tf.name_scope('RKD'):
        source = tf.nn.l2_normalize(source, 1)
        target = tf.nn.l2_normalize(target, 1)
        distance_loss = Huber_loss(distance_wise_potential(source), distance_wise_potential(target))
        angle_loss    = Huber_loss(   angle_wise_potential(source),    angle_wise_potential(target))
        return distance_loss, angle_loss

def pairwise_suprvise(student_feature, teacher_feature, distillation_mask):
    def map_fn_pairwise_suprvise(student, teacher, mask):
        student_feature_shape = student.get_shape()
        if student_feature_shape.ndims != 3:
            raise ValueError('student_feature must be of size [height, width, c]')
        teacher_feature_shape = teacher.get_shape()
        if teacher_feature_shape.ndims != 3:
            raise ValueError('teacher_feature must be of size [height, width, c]')
        student = tf.reshape(student,
         (student_feature_shape[0]*student_feature_shape[1], student_feature_shape[2]))
        teacher = tf.reshape(teacher,
         (teacher_feature_shape[0]*teacher_feature_shape[1], teacher_feature_shape[2]))

        if mask.get_shape().ndims != 0:
            raise ValueError('teacher_feature must be of size [height, width, c]')
        
        return tf.cond(pred=tf.equal(mask, 1),
            true_fn=lambda: relational_knowledge_distillation(student, teacher),
            false_fn=lambda: (0.0, 0.0))
        #return relational_knowledge_distillation(student, teacher)

    distance_loss, angle_loss = tf.map_fn(
        fn=lambda x: map_fn_pairwise_suprvise(student=x[0], teacher=x[1], mask=x[2]),
        elems=(student_feature, teacher_feature, distillation_mask),
        dtype=(tf.float32, tf.float32)
        )
    return tf.reduce_mean(distance_loss), tf.reduce_mean(angle_loss)

def fake_soft_logits(student, teacher, T = 2):
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531, 2015.
    '''
    with tf.name_scope('KL'):
        return tf.reduce_mean(
            tf.reduce_sum(
               tf.sigmoid(teacher/T) * (
                   tf.log(tf.sigmoid(teacher/T)) - tf.log(tf.sigmoid(student/T))
                   ),
                   1 
               )
             )

def soft_logits(student, teacher, T = 2):
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531, 2015.
    '''
    with tf.name_scope('KD'):
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.softmax(teacher/T)*(tf.nn.log_softmax(teacher/T)-tf.nn.log_softmax(student/T)),
                 1
                 )
                )


def pixelwise_suprvise(student_feature, teacher_feature, T):
    student_feature_shape = student_feature.get_shape()
    if student_feature_shape.ndims != 2:
        raise ValueError('student_feature must be of size [n, c]')
    teacher_feature_shape = teacher_feature.get_shape()
    if teacher_feature_shape.ndims != 2:
        raise ValueError('teacher_feature must be of size [n, c]')
    KL_loss = soft_logits(student_feature, teacher_feature, T) #fake_soft_logits
    return tf.reduce_mean(KL_loss)

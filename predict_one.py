from PIL import Image
import tensorflow as tf
import eye_model_predict

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/eye_train',
                           """Directory where to read model checkpoints.""")
width = 32
height = 32
categories = [ "AMD","BRVO","CSC","DR","ENormal","FOthers","GPM" ]
filename = "AMD_6.png" # absolute path to input image
input_img = tf.image.decode_png(tf.read_file(filename), channels=3)
tf_cast = tf.cast(input_img, tf.float32)
float_image = tf.image.resize_image_with_crop_or_pad(tf_cast, height, width)
float_image = tf.image.per_image_whitening(float_image)
images = tf.expand_dims(float_image, 0)
logits = eye_model_predict.inference(images)
logits = tf.nn.softmax(logits)
_, top_k_pred = tf.nn.top_k(logits, k=5)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    variable_averages = tf.train.ExponentialMovingAverage(eye_model_predict.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/eye_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
    _, top_indices = sess.run([_, top_k_pred])
    for key, value in enumerate(top_indices[0]):
        print (categories[value] + ", " + str(_[0][key]))

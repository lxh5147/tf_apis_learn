import os

# os.environ['KERAS_BACKEND']= 'tensorflow'
# import keras.backend as K
# from keras.losses import  categorical_crossentropy
# from keras.layers import Dropout,Dense
import tensorflow as tf
import numpy as np


def build_model(x_data, y_data):
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    return loss


def training(loss, learning_rate, is_chief, replicas_to_aggregate=1):
    # Create the gradient descent optimizer with the given learning rate.
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=replicas_to_aggregate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = opt.minimize(loss, global_step=global_step)
    # Hook which handles initialization and queues. Without this hook,training will not work
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    return train_op, global_step, [sync_replicas_hook]


#python /home/lxh5147/projects/tf_apis_learn/train_distributed.py > work_0.log 2>&1 &
#python /home/lxh5147/projects/tf_apis_learn/train_distributed.py --task_index 1 > work_1.log 2>&1 &
#python /home/lxh5147/projects/tf_apis_learn/train_distributed.py --job_name ps > ps_0.log 2>&1 &

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--ps_hosts",
                        default="localhost:2224",
                        help="ps hosts separated by ','")

    parser.add_argument("--worker_hosts",
                        default="localhost:2222,localhost:2223",
                        help="worker hosts separated by ','")

    parser.add_argument("--job_name",
                        default="worker",
                        help="worker or ps")

    parser.add_argument("--task_index",
                        default=0,
                        type=int,
                        help="Index of task within the job")

    parser.add_argument("--learning_rate",
                        default=0.01,
                        type=float,
                        help="Learning rate")

    args = parser.parse_args()

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster_spec,
                             job_name=args.job_name,
                             task_index=args.task_index)

    if args.job_name == 'ps':
        server.join()

    worker_device = "/job:worker/task:%d" % args.task_index

    device = tf.train.replica_device_setter(
        worker_device=worker_device,
        cluster=cluster_spec)

    is_chief = (args.task_index == 0)

    # build the model
    with tf.device(device):
        x_data = tf.placeholder(tf.float32, [100])
        y_data = tf.placeholder(tf.float32, [100])
        loss = build_model(x_data, y_data)
        train_op, global_step, hooks = training(loss,
                                                learning_rate=args.learning_rate,
                                                is_chief=is_chief,
                                                replicas_to_aggregate=2, )

    # other hooks
    hooks.extend([tf.train.StopAtStepHook(last_step=10000000),
                  tf.train.NanTensorHook(loss), ])


    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
    )

    with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            hooks=hooks,
            config=sess_config) as sess:

        while not sess.should_stop():
            x_data_v = np.random.rand(100).astype(np.float32)
            y_data_v = x_data_v * 0.1 + 0.3
            _, step, loss_v = sess.run([train_op, global_step, loss],
                                       feed_dict={x_data: x_data_v, y_data: y_data_v})
            if step % 100 == 0:
                print "step: %d, loss: %f" % (step, loss_v)

        print "Optimization finished."

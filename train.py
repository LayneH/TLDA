import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from cifar_data import *

class Model():
    def __init__(self):
        pass

    def build(self, Xs, Xt, y, shape=[3072, 10, 2, 10, 3072]):
        # encoder layers
        with tf.variable_scope('encoder1') as scope:
            h = self.fc_sigmoid(Xs, shape[:2])
            scope.reuse_variables()
            t = self.fc_sigmoid(Xt, shape[:2])

        # KL divergence between distributions of source and target domains
        dist_h = tf.reduce_mean(h, axis=0)
        dist_t = tf.reduce_mean(t, axis=0)
        KL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_h, labels=dist_t))
        KL += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_t, labels=dist_h))
        tf.summary.scalar('KL', KL)

        with tf.variable_scope('encoder2') as scope:
            h = self.fc_sigmoid(h, shape[1:3])
            scope.reuse_variables()
            t = self.fc_sigmoid(t, shape[1:3])
        # cross entropy of source domain
        xe = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y))
        tf.summary.scalar('xe', xe)
        # accuracy op
        hit = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
        acc_op = tf.reduce_mean(tf.cast(hit, tf.float32))
        tf.summary.scalar('accuracy', acc_op)

        # decoder layers
        with tf.variable_scope('decoder1') as scope:
            h = self.fc_sigmoid(h, shape[2:4])
            scope.reuse_variables()
            t = self.fc_sigmoid(t, shape[2:4])

        with tf.variable_scope('decoder2') as scope:
            h = self.fc_sigmoid(h, shape[3:])
            scope.reuse_variables()
            t = self.fc_sigmoid(t, shape[3:])

        # L2 regularization term
        reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        tf.summary.scalar('reg', reg)'
        # reconstruction error term
        rec_loss = tf.losses.mean_squared_error(Xs, h) + tf.losses.mean_squared_error(Xt, t)
        tf.summary.scalar('rec_loss', rec_loss)
        return KL, xe, reg, rec_loss, acc_op

    def fc(self, X, shape):
        initializer = tf.truncated_normal_initializer(stddev=0.01, mean=0)
        weight = tf.get_variable('kernel', shape, initializer=initializer)
        bias = tf.get_variable('bias', shape[-1:], initializer=initializer)
        return tf.matmul(X, weight) + bias

    def fc_sigmoid(self, X, shape):
        return tf.nn.sigmoid(self.fc(X, shape))

def train(src, targ, param, summaries_dir):
    alpha = param.pop('alpha', 1e-2)
    beta = param.pop('beta', 1e-2)
    gamma = param.pop('gamma', 1e-2)
    max_epochs = 1000
    verbose = 50

    g = tf.Graph()
    with g.as_default():
        Xs = tf.placeholder(tf.float32, [None, 3072])
        Xt = tf.placeholder(tf.float32, [None, 3072])
        y = tf.placeholder(tf.float32, [None, 2])

        net = Model()
        KL, xe, reg, rec_op, acc_op = net.build(Xs, Xt, y)

        # objective of TLDA
        loss_op = rec_op + alpha * KL + beta * xe + gamma * reg
        tf.summary.scalar('loss', loss_op)
        # training op of plain autoencoder
        ae_op = tf.train.AdamOptimizer().minimize(rec_op)
        # training op of TLDA
        train_op = tf.train.AdamOptimizer().minimize(loss_op)

    config = tf.ConfigProto()
    # comment next line if you want tensorflow to use all available GPU memory
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g, config=config) as sess:
        tlda_writer = tf.summary.FileWriter(summaries_dir + '/tlda', sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        # first pre-train the autoencoder
        for i in xrange(200):
            xx, yy = src.next_batch(64)
            xxx, yyy = targ.next_batch(64)
            sess.run(ae_op, feed_dict={Xs:xx, Xt:xxx})

        for i in xrange(max_epochs):
            summary, loss, _ = sess.run([merged, loss_op, train_op], feed_dict={Xs:src.images, Xt:targ.images, y:src.labels})
            if verbose > 0 and i > 0 and (i % verbose == 0 or verbose+1 == max_epochs):
                tlda_writer.add_summary(summary, i)
                print '%d / %d'%(i, max_epochs), sess.run(acc_op, feed_dict={Xs:targ.images, y:targ.labels})
        return sess.run(acc_op, feed_dict={Xs:targ.images, y:targ.labels})

def get_problems():
    problems = []
    for pos1 in data[types[0]]:
        for pos2 in data[types[0]]:
            if pos1 == pos2:
                continue
            for neg1 in data[types[1]]:
                for neg2 in data[types[1]]:
                    if neg1 == neg2:
                        continue
                    name = "%s-%s-vs-%s-%s"%(pos1, neg1, pos2, neg2)
                    problems.append(name)
    return problems

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--cifar_path', dest='cifar_path', metavar='CIFAR_PATH',
        help='directory of CIFAR-100 dataset', default=DEFAULT_CIFAR_PATH)
    parser.add_argument('--outfile', dest='outfile', metavar='OUTPUTFILE',
        help='file that save the result', default=DEFAULT_OUTFILE)
    return parser

DEFAULT_OUTFILE = 'result.csv'
DEFAULT_CIFAR_PATH = 'datasets/cifar-100-python'
alpha = 1e-3
beta = 1e2
gamma = 1e-5
verbose = 50
types = ["fruit_and_vegetables", "household_electrical_devices"]

def main():
    parser = get_parser()
    option = parser.parse_args()
    assert os.path.isdir(option.cifar_path), "You may forget to download CIFAR-100 "\
                                        "dataset, or please specify the path "\
                                        "by --cifar_path option if you have the "\
                                        "dataset already but not in default path %s."%DEFAULT_CIFAR_PATH
    data = load_data(types, option.cifar_path)
    problems = get_problems(data)
    param = {'alpha':alpha, 'beta':beta, 'gamma':gamma, 'verbose':verbose}
    rr = np.zeros((len(problems), ))
    for i, name in enumerate(problems):
        pos1, neg1, _, pos2, neg2 = name.split('-')
        Xs, ys = get_XY(data[types[0]][pos1], data[types[1]][neg1])
        Xt, yt = get_XY(data[types[0]][pos2], data[types[1]][neg2])
        src = data_holder(Xs, ys)
        targ = data_holder(Xt, yt)
        print 'Train classifier for ', name
        print 'alpha %.6f, beta %.6f, gamma %.6f'%(alpha, beta, gamma)
        rr[i] = train(src, targ, param, 'summary')
        print 'Test Accuracy is ', rr[i]
        with open(option.outfile, 'rb') as f:
            np.savetxt(f, rr, delimiter=',', fmt='%.4f')
    print 'The result is stored in %s and mean accuracy is %.4f'%(option.outfile, np.mean(rr, axis=0))

if __name__ == '__main__':
    main()

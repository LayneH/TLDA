import numpy as np
import tensorflow as tf
from cifar_data import *

%load_ext autoreload
%autoreload 2

data = load_data()
types = ["fruit_and_vegetables", "household_electrical_devices"]

problems = []
i = 0
for pos1 in data[types[0]]:
    data[types[0]][pos1] = data[types[0]][pos1].reshape((600, -1)) 
    for pos2 in data[types[0]]:
        if pos1 == pos2:
            continue
        for neg1 in data[types[1]]:
            data[types[1]][neg1] = data[types[1]][neg1].reshape((600, -1))
            for neg2 in data[types[1]]:
                if neg1 == neg2:
                    continue
                name = "%s-%s-vs-%s-%s"%(pos1, neg1, pos2, neg2)
                problems.append(name)
                
train_problems = ['pear-keyboard-vs-sweet_pepper-clock', 'mushroom-lamp-vs-pear-television',
        'orange-clock-vs-apple-keyboard', 'orange-telephone-vs-mushroom-lamp',
        'pear-telephone-vs-sweet_pepper-television', 'apple-telephone-vs-orange-keyboard',
        'orange-clock-vs-pear-keyboard', 'apple-telephone-vs-mushroom-lamp',
        'pear-lamp-vs-apple-television', 'pear-lamp-vs-orange-television']

class model():
    def __init__(self):
        pass
    
    def build(self, Xs, Xt, y, shape=[3072, 10, 2, 10, 3072]):
        with tf.variable_scope('encoder1') as scope:
            h = self.fc_sigmoid(Xs, shape[:2])
            scope.reuse_variables() 
            t = self.fc_sigmoid(Xt, shape[:2])
        
        dist_h = tf.reduce_mean(h, axis=0)
        dist_t = tf.reduce_mean(t, axis=0)
        KL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_h, labels=dist_t))
        KL += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_t, labels=dist_h))
        
        with tf.variable_scope('encoder2') as scope:
            h = self.fc_sigmoid(h, shape[1:3])
            scope.reuse_variables() 
            t = self.fc_sigmoid(t, shape[1:3])
        
        xe = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y))
        tf.summary.scalar('xe', xe)
        hit = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
        acc_op = tf.reduce_mean(tf.cast(hit, tf.float32))
        tf.summary.scalar('accuracy', acc_op)
        
        with tf.variable_scope('decoder1') as scope:
            h = self.fc_sigmoid(h, shape[2:4])
            scope.reuse_variables()
            t = self.fc_sigmoid(t, shape[2:4])
        dist_h = tf.reduce_mean(h, axis=0)
        dist_t = tf.reduce_mean(t, axis=0)
        KL += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_h, labels=dist_t))
        KL += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_t, labels=dist_h))
        tf.summary.scalar('KL', KL)
        with tf.variable_scope('decoder2') as scope:
            h = self.fc(h, shape[3:])
            scope.reuse_variables()
            t = self.fc_sigmoid(t, shape[3:])
        
        reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        tf.summary.scalar('reg', reg)
        rec_loss = tf.losses.mean_squared_error(Xs, tf.nn.sigmoid(h)) + tf.losses.mean_squared_error(Xt, t)
        rec_sum = tf.summary.scalar('rec_loss', rec_loss)
        return KL, xe, reg, rec_loss, acc_op, rec_sum
    
    
    def inference(self, rec_loss, param):
        alpha = param.pop('alpha', 1e-2)
        beta = param.pop('beta', 1e-2)
        gamma = param.pop('gamma', 1e-2)
        return rec_loss + alpha * KL + beta * xe + gamma * reg
    
    def conv(self, X, shape):
        initializer = tf.truncated_normal_initializer(stddev=0.01, mean=0)
        try:
            kernel = tf.get_variable('kernel', [3, 3] + shape, initializer=initializer, reuse=True)
            bias = tf.get_variable('bias', shape[-1:], initializer=initializer, reuse=True)
        except ValueError:
            kernel = tf.get_variable('kernel', [3, 3] + shape, initializer=initializer)
            bias = tf.get_variable('bias', shape[-1:], initializer=initializer)
        return tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME') + bias
    
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
        net = model()
        KL, xe, reg, rec_op, acc_op, rec_sum = net.build(Xs, Xt, y)
        loss_op = rec_op + alpha * KL + beta * xe + gamma * reg
        tf.summary.scalar('loss', loss_op)
        ae_op = tf.train.AdamOptimizer().minimize(rec_op)
        train_op = tf.train.AdamOptimizer().minimize(loss_op)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g, config=config) as sess:
        ae_writer = tf.summary.FileWriter(summaries_dir + '/ae', sess.graph)
        tlda_writer = tf.summary.FileWriter(summaries_dir + '/tlda', sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        # first pretrain the autoencoder
        for i in xrange(200):
            xx, yy = src.next_batch(64)
            xxx, yyy = targ.next_batch(64)
            rec_loss, _ = sess.run([rec_sum, ae_op], feed_dict={Xs:xx, Xt:xxx})
            # ae_writer.add_summary(rec_loss, i)
        #print 'Reconstruction loss is ', rec_loss
        
        for i in xrange(max_epochs):
            summary, _ = sess.run([merged, train_op], feed_dict={Xs:src.images, Xt:targ.images, y:src.labels})
            if verbose > 0 and i > 0 and (i % verbose == 0 or verbose+1 == max_epochs):
                #tlda_writer.add_summary(summary, i)
                print '%d / %d'%(i, max_epochs), sess.run(acc_op, feed_dict={Xs:targ.images, y:targ.labels})
        return sess.run(acc_op, feed_dict={Xs:targ.images, y:targ.labels})

alphas = np.logspace(-3, -1, 5)
betas = np.logspace(1, 3, 5)
gammas = np.logspace(-5, 2, 1)
verbose = 50
rr = np.zeros((len(train_problems), len(alphas), len(betas), len(gammas)))
for i, name in enumerate(train_problems):
    pos1, neg1, _, pos2, neg2 = name.split('-')
    Xs, ys = get_XY(data[types[0]][pos1] / 255., data[types[1]][neg1] / 255.)
    Xt, yt = get_XY(data[types[0]][pos2] / 255., data[types[1]][neg2] / 255.)
    src = data_holder(Xs, ys)
    targ = data_holder(Xt, yt)
    print 'Train classifier for ', name
    for j, alpha in enumerate(alphas):
        for k, beta in enumerate(betas):
            for l, g1amma in enumerate(gammas):
                print 'alpha %.6f, beta %.6f, gamma %.6f'%(alpha, beta, gamma)
                param = {'alpha':alpha, 'beta':beta, 'gamma':gamma, 'verbose':verbose}
                rr[i, j, k, l] = train(src, targ, param, 'summary')
                print 'Test Accuracy is ', rr[i, j, k, l]
    print rr[i]
    np.save('grid.npy', rr)
print rr
print np.mean(rr, axis=0)
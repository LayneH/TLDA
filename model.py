import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model():
    def __init__(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.Xs = tf.placeholder(tf.float32, [None, 3072])
            self.Xt = tf.placeholder(tf.float32, [None, 3072])
            self.y = tf.placeholder(tf.float32, [None, 2])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.sess.close()
        except Exception:
            pass

    def build(self, shape=[3072, 10, 2, 10, 3072]):
        with self.g.as_default():
            # encoder layers
            with tf.variable_scope('encoder1') as scope:
                h = self.fc_sigmoid(self.Xs, shape[:2])
                scope.reuse_variables()
                t = self.fc_sigmoid(self.Xt, shape[:2])

            # KL divergence between distributions of source and target domains
            dist_h = tf.reduce_mean(h, axis=0)
            dist_t = tf.reduce_mean(t, axis=0)
            self.KL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_h, labels=dist_t))
            self.KL += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_t, labels=dist_h))
            tf.summary.scalar('KL', self.KL)

            with tf.variable_scope('encoder2') as scope:
                h = self.fc_sigmoid(h, shape[1:3])
                scope.reuse_variables()
                t = self.fc_sigmoid(t, shape[1:3])
            # cross entropy of source domain
            self.xe = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=self.y))
            tf.summary.scalar('xe', self.xe)
            # accuracy op
            hit = tf.equal(tf.argmax(self.y, 1), tf.argmax(h, 1))
            self.acc_op = tf.reduce_mean(tf.cast(hit, tf.float32))
            tf.summary.scalar('accuracy', self.acc_op)

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
            self.reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            tf.summary.scalar('reg', self.reg)
            # reconstruction error term
            self.rec_loss = tf.losses.mean_squared_error(self.Xs, h) + tf.losses.mean_squared_error(self.Xt, t)
            tf.summary.scalar('rec_loss', self.rec_loss)

    def fc(self, X, shape):
        initializer = tf.truncated_normal_initializer(stddev=0.01, mean=0)
        weight = tf.get_variable('kernel', shape, initializer=initializer)
        bias = tf.get_variable('bias', shape[-1:], initializer=initializer)
        return tf.matmul(X, weight) + bias

    def fc_sigmoid(self, X, shape):
        return tf.nn.sigmoid(self.fc(X, shape))

    def inference(self, alpha, beta, gamma):
        with self.g.as_default():
            # objective of TLDA
            self.loss_op = self.rec_loss + alpha * self.KL + beta * self.xe + gamma * self.reg
            tf.summary.scalar('loss', self.loss_op)
            # training op of plain autoencoder
            self.ae_op = tf.train.AdamOptimizer().minimize(self.rec_loss)
            # training op of TLDA
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)

    def train(self, src, targ, max_epochs=1000, verbose=50, summaries_dir='summary'):
        config = tf.ConfigProto()
        # comment next line if you want tensorflow to use all available GPU memory
        config.gpu_options.allow_growth = True
        with self.g.as_default():
            self.sess = tf.Session(graph=self.g, config=config)
            tlda_writer = tf.summary.FileWriter(summaries_dir + '/tlda', self.g)
            merged = tf.summary.merge_all()
            self.sess.run(tf.global_variables_initializer())
            # first pre-train the autoencoder
            for i in xrange(200):
                xx, yy = src.next_batch(64)
                xxx, yyy = targ.next_batch(64)
                self.sess.run(self.ae_op, feed_dict={self.Xs:xx, self.Xt:xxx})

            for i in xrange(max_epochs):
                summary, loss, _ = self.sess.run([merged, self.loss_op, self.train_op],
                    feed_dict={self.Xs:src.images, self.Xt:targ.images, self.y:src.labels})
                if verbose > 0 and i > 0 and (i % verbose == 0 or verbose+1 == max_epochs):
                    tlda_writer.add_summary(summary, i)
                    print '%4d / %4d, test accuracy is %.4f'%(i, max_epochs, self.test(targ))

    def test(self, targ):
        return self.sess.run(self.acc_op, feed_dict={self.Xs:targ.images,
                            self.y:targ.labels})

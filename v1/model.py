import tensorflow as tf
import numpy as np

import crawler
import preprocess
import csv

class LstmRNN():
    def __init__(self, sess, lstm_size=256,
                 num_layers=1,
                 num_steps=5,
                 epochs = 50,
                 batch_size = 64,
                 input_size=11,
                 output_size=11):

        self.sess = sess
        self.epochs = epochs
        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.num_steps = num_steps
        self.input_size = input_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.alpha = 3

        self.build_graph()

    def build_graph(self):

        self.X = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name='input2')
        self.Y = tf.placeholder(tf.float32, [None, self.output_size], name='output')
        self.global_step = tf.Variable(0, name='global_step')

        self.add_input_hidden()
        self.add_cell()
        self.add_output_hidden()
        self.add_optimizer()



    def add_input_hidden(self):
        l_in_x = tf.reshape(self.X, shape=[-1, self.input_size], name='3D-2')
        Ws_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32)
        bs_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
        self.Ws_in = tf.get_variable(shape=[self.input_size, self.lstm_size], initializer=Ws_initializer, name='Ws_in')
        self.bs_in = tf.get_variable(shape=[self.lstm_size,], initializer= bs_initializer, name='bs_in')
        l_in_y = tf.matmul(l_in_x, self.Ws_in) + self.bs_in
        self.l_in_y = tf.reshape(l_in_y, shape=[-1, self.num_steps, self.lstm_size], name='l_in_y')

    def add_cell(self):
        self.lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, initializer=tf.orthogonal_initializer())
        self.multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell]*self.num_layers, state_is_tuple=True)
        self.initial_state = self.multi_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        val, self.final_state = tf.nn.dynamic_rnn(self.multi_lstm_cell, self.l_in_y, dtype=tf.float32, time_major=False)
        cell_output = tf.transpose(val, [1,0,2])
        self.cell_last = tf.gather(cell_output, int(cell_output.shape[0])-1, name='cell_last_output')

    def add_output_hidden(self):
        Ws_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32)
        bs_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
        self.Ws_out = tf.get_variable(shape=[self.lstm_size, self.output_size], initializer=Ws_initializer, name='Ws_out')
        self.bs_out = tf.get_variable(shape=[self.output_size], initializer=bs_initializer, name='bs_out')
        l_out_y = tf.matmul(self.cell_last, self.Ws_out) + self.bs_out
        self.l_out_y = tf.reshape(l_out_y, shape=[-1, self.output_size], name='pred_value')
    def add_optimizer(self):
        self.close = tf.gather(tf.transpose(self.l_out_y, [1,0]), 3)
        self.close_y = tf.gather(tf.transpose(self.Y,[1,0]), 3)
        self.cost = tf.reduce_mean(tf.square(self.l_out_y - self.Y)) + self.alpha*tf.reduce_mean(tf.square(self.close-self.close_y))
        tf.summary.scalar(tensor=self.cost, name='loss')
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.08).minimize(self.cost, global_step=self.global_step)

    def fit(self):
        seq, seq_y = preprocess.getSeq2(self.num_steps, '20140101', '20141231')
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('./event/', self.sess.graph)
        merged = tf.summary.merge_all()
        first = True
        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)
        for i in range(self.epochs):
            sample_index = np.arange(len(seq))
            np.random.shuffle(sample_index)
            for j in range(int(len(seq)/self.batch_size)):
                tr, gt= preprocess.getBatch(seq, seq_y, sample_index, j, self.batch_size)
                if i == 0 and first:
                    feed_dict = {
                        self.X: tr,
                        self.Y: gt
                    }
                    first = False
                else:
                    feed_dict = {
                        self.X: tr,
                        self.Y: gt,
                        self.initial_state: state
                    }
                _, state, step, cost, pred= self.sess.run([self.train_op, self.initial_state, self.global_step, self.cost, self.l_out_y], feed_dict=feed_dict)
                if j % 10 == 0:
                    summary_result = self.sess.run(merged, feed_dict)
                    print('Epochs {} cost {}'.format(i, cost))
                    writer.add_summary(summary_result, step)
        saver.save(self.sess, './check/model.ckpt')

    def predict(self, stockNo, date='20180518'):
        saver = tf.train.Saver()
        first = True
        with open('./data/result_'+date+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            columns = ['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice', 'Wed_ud', 'Wed_cprice', 'Thu_ud', 'Thu_cprice', 'Fri_ud', 'Fri_cprice']
            writer.writerow(columns)
            seq, closes = crawler.getTarget(stockNo, self.num_steps, date)
            for num, target in enumerate(seq):
                saver.restore(self.sess, './check/model.ckpt')
                result = [stockNo[num]]
                feed_dict = {
                    self.X: np.array(target).reshape([1,self.num_steps,self.input_size])
                }
                pred = self.sess.run([self.l_out_y], feed_dict=feed_dict)[0].reshape([-1])
                if pred[3] > 0:
                    result.append(1)
                elif pred[3] == 0:
                    result.append(0)
                elif pred[3] < 0:
                    result.append(-1)
                result.append(closes[num] + pred[3])
                closes[num] = closes[num] + pred[3]
                for j in range(4):
                    target = target[1:]
                    new_data = [pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7], pred[8], pred[9], pred[10]]
                    target.append(new_data)
                    feed_dict = {
                        self.X: np.array(target).reshape([1,self.num_steps,self.input_size])
                    }
                    pred = self.sess.run([self.l_out_y], feed_dict = feed_dict)[0].reshape([-1])
                    if pred[3] > 0:
                        result.append(1)
                    elif pred[3] == 0:
                        result.append(0)
                    elif pred[3] < 0:
                        result.append(-1)
                    result.append(closes[num] + pred[3])
                    closes[num] = closes[num] + pred[3]
                writer.writerow(result)

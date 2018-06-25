import model
import tensorflow as tf

if __name__ == '__main__':
    train = False
    if train:
        with tf.Session() as sess:
            lstmRNN = model.LstmRNN(sess)
            lstmRNN.fit()
    else:
        with tf.Session() as sess:
            lstmRNN = model.LstmRNN(sess)
            dates2 = ['20180611', '20180612', '20180613', '20180614', '20180615']
            for date in dates2:
                lstmRNN.predict(stockNo, date)

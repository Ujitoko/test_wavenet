import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from wavenet.layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d)
from wavenet.utils import *
import os
import numpy as np

class Model(object):
    def __init__(self,
                 num_time_samples,
                 num_channels=1,
                 num_classes=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 gpu_num = "0",
                 model_name = "default_name"):
        
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_num = gpu_num
        self.model_name = model_name
        
        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

        h = inputs
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)

        outputs = conv1d(h,
                         num_classes,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs, labels=targets)
        cost = tf.reduce_mean(costs)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list="0"
            )
        )
        sess = tf.Session(config=self.config)
        sess.run(tf.global_variables_initializer())
        self.inputs_ph = inputs
        self.targets_ph = targets
        self.outputs = outputs
        self.hs = hs
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess

        
        # dataset
        self.ad_train = AccelerationDataset("acc_dataset_selected/train")
        self.batch_size = 2
        
        self.generate_init()
        self.count = 0
        

    def _train(self):
        batch_x, batch_y = self.ad_train.getBatchTrain(True, self.num_time_samples, self.batch_size)
        batch_x = batch_x.reshape((self.batch_size, self.num_time_samples, 10))#self.num_input))        
        batch_x_0 = batch_x[:,:,0]#.reshape((self.batch_size, self.num_time_samples,-1))
        bins = np.linspace(-1, 1, 256)
        # Quantize inputs.
                
        inputs_batch, targets_batch = [],[]
        for batch in range(self.batch_size):
            x = batch_x_0[batch]
            y = batch_y[batch]
            inputs = np.digitize(x, bins, right=False) - 1
            inputs = bins[inputs][None, :, None]
            inputs_batch.append(inputs)
        
            # Encode targets as ints.
            targets = (np.digitize(y, bins, right=False) - 1)[None, :]
            targets_batch.append(targets)

        inputs_batch = np.vstack(inputs_batch)
        inputs_batch = np.concatenate((inputs_batch, batch_x[:,:,1:]), axis=2)
        targets_batch = np.vstack(targets_batch)

        # save for test
        if self.count == 0:
            self.test_inputs = inputs_batch
            self.test_targets = targets_batch
            self.count += 1

        feed_dict = {self.inputs_ph: inputs_batch, self.targets_ph: targets_batch}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1                        
            cost = self._train()
            
            if cost < 1e-1 or i > 100000:
                terminal = True
            losses.append(cost)
            
            if i % 1000 == 0:
                self.generate_run(self.test_inputs[0,:,:][np.newaxis,:,:], i)
                print(cost)

                show_wave(losses, dirname=self.model_name, filename="losses_" + str(i), y_lim=7)
                #plt.plot(losses)
                #plt.ylim(0, 7)
                #plt.show()
                
                
    def generate_init(self, batch_size=1):
        self.bins = np.linspace(-1, 1, self.num_classes)
        input_size = self.num_channels #self.num_time_samples
        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        for b in range(self.num_blocks):
            for i in range(self.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = self.num_channels
                else:
                    state_size = self.num_hidden
                    
                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops
        
        # Initialize queues.
        self.sess.run(self.init_ops)

    def generate_run(self, inputs, i):
        predictions = []
        
        num_samples = inputs.shape[1] # time_steps
        for step in range(num_samples):
            input = inputs[:,step,:].reshape([1,-1])
            feed_dict = {self.inputs: input}
            output = self.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
            value = np.argmax(output[0, :])

            #input = np.array(self.bins[value])[None, None] # TODO
            value_binned = np.array(self.bins[value])[None, None]
            predictions.append(value_binned)

        #if step % 10000 == 0:
        predictions_ = np.concatenate(predictions, axis=1)
        show_wave(predictions_[0, :], dirname=self.model_name, filename="gen_"+str(i))
        #plt.plot(predictions_[0, :], label='pred')
        #plt.legend()
        #plt.xlabel('samples from start')
        #plt.ylabel('signal')
        #plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_        
        
        
"""
class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        self.bins = np.linspace(-1, 1, self.model.num_classes)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = self.model.num_hidden
                    
                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops
        
        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
            value = np.argmax(output[0, :])

            input = np.array(self.bins[value])[None, None] # TODO
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plt.plot(predictions_[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_
"""

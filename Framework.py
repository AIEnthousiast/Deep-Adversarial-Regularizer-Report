import numpy as np
import os
from abc import ABC
import numpy as np
import os
import odl
import odl.contrib.tensorflow
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
import util as ut

RECORD_RATE = 10
CHECKPOINT_RATE = 25
def smooth_total_variation(x, epsilon=1e-6):
    """
    Computes a smooth approximation of the isotropic TV norm.
    x: 4D tensor [batch, height, width, channels]
    """
    dx = x[:, 1:, :, :] - x[:, :-1, :, :]  # vertical differences
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # horizontal differences

    dx2 = tf.square(dx)
    dy2 = tf.square(dy)

    tv = tf.reduce_sum(tf.sqrt(dx2[:, :, :-1, :] + dy2[:, :-1, :, :] + epsilon))
    return tv
    
class GenericFramework(ABC):
    model_name = 'no_model'
    experiment_name = 'default_experiment'

    # set the noise level used for experiments
    noise_level = 0.02

    @abstractmethod
    def get_network(self, size, colors):
        # returns an object of the network class. Used to set the network used
        pass

    @abstractmethod
    def get_Data_pip(self, path):
        # returns an object of the data_pip class.
        pass

    @abstractmethod
    def get_model(self, size):
        # Returns an object of the forward_model class.
        pass

    def __init__(self, data_path, saves_path):
        self.data_pip = self.get_Data_pip(data_path)
        self.colors = self.data_pip.colors
        self.image_size = self.data_pip.image_size
        self.network = self.get_network(self.image_size, self.colors)
        self.model = self.get_model(self.image_size)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()
        # finding the correct path for saving models
        self.path = saves_path+'Saves/{}/{}/{}/{}/'.format(self.model.name, self.data_pip.name,
                                                           self.model_name, self.experiment_name)

        # generate needed folder structure
        ut.create_single_folder(self.path + 'Data')
        ut.create_single_folder(self.path + 'Logs')

    def generate_training_data(self, batch_size, training_data=True):
        # method to generate training data given the current model type
        y = np.empty((batch_size, self.measurement_space[0], self.measurement_space[1], self.colors), dtype='float32')
        x_true = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
        fbp = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')

        for i in range(batch_size):
            if training_data:
                image = self.data_pip.load_data(training_data=True)
            else:
                image = self.data_pip.load_data(training_data=False)
            data = self.model.forward_operator(image)

            # add white Gaussian noise
            noisy_data = data + self.noise_level*np.random.normal(size=(self.measurement_space[0],
                                                                        self.measurement_space[1],
                                                                        self.colors))
            fbp[i, ...] = self.model.inverse(noisy_data)
            x_true[i, ...] = image
            y[i, ...] = noisy_data
        return y, x_true, fbp

    def save(self, global_step):
        # saver = tf.train.Saver()
        # saver.save(self.sess, self.path+'Data/model', global_step=global_step)
        # print('Progress saved')
        pass

    def load(self):
        # saver = tf.train.Saver()
        # if os.listdir(self.path+'Data/'):
        #     saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
        #     print('Save restored')
        # else:
        #     print('No save found')
        pass

    def end(self):
        # tf.reset_default_graph()
        # self.sess.close()
        pass

    @abstractmethod
    def evaluate(self, guess, measurement):
        # apply the model to data
        pass


class AdversarialRegulariser(GenericFramework):
    model_name = 'Adversarial_Regulariser'
    # the absolut noise level
    batch_size = 16
    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 1.5
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0001
    # default step size for picture optimization
    step_size = 1
    # the amount of steps of gradient descent taken on loss functional
    total_steps_default = 30
    # default sampling pattern
    starting_point = 'Mini'

    def set_total_steps(self, steps):
        self.total_steps = steps

    # sets up the network architecture
    def __init__(self, data_path, saves_path):
        # call superclass init
        super(AdversarialRegulariser, self).__init__(data_path, saves_path)
        self.total_steps = self.total_steps_default

        ### Training the regulariser ###

        # placeholders for NN
        self.gen_im = tf.Variable(tf.zeros(shape=[self.image_space[0], self.image_space[1], self.colors],
                                     dtype=tf.float32))
        self.true_im = tf.Variable(tf.zeros(shape=[None, self.image_space[0], self.image_space[1], self.colors],
                                      dtype=tf.float32))
        self.random_uint = tf.Variable(tf.zeros(shape=[None],
                                          dtype=tf.float32))

        # the network outputs
        self.gen_was = self.network.net(self.gen_im)
        self.data_was = self.network.net(self.true_im)

        # Wasserstein loss
        self.wasserstein_loss = tf.reduce_mean(self.data_was - self.gen_was)

        # intermediate point
        random_uint_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.random_uint, axis=1), axis=1), axis=1)
        self.inter = tf.multiply(self.gen_im, random_uint_exp) + \
                     tf.multiply(self.true_im, 1 - random_uint_exp)
        self.inter_was = self.network.net(self.inter)

        # calculate derivative at intermediate point
        self.gradient_was = tf.gradients(self.inter_was, self.inter)[0]

        # take the L2 norm of that derivative
        self.norm_gradient = tf.sqrt(tf.reduce_sum(tf.square(self.gradient_was), axis=(1, 2, 3)))
        self.regulariser_was = tf.reduce_mean(tf.square(tf.nn.relu(self.norm_gradient - 1)))

        # Overall Net Training loss
        self.loss_was = self.wasserstein_loss + self.lmb * self.regulariser_was

        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_was,
                                                                                global_step=self.global_step)

        ### The reconstruction network ###

        # placeholders
        self.reconstruction = tf.placeholder(shape=[None, self.image_space[0], self.image_space[0], self.colors],
                                             dtype=tf.float32)
        self.data_term = tf.placeholder(shape=[None, self.measurement_space[0], self.measurement_space[1], self.colors],
                                        dtype=tf.float32)
        self.mu = tf.placeholder(dtype=tf.float32)

        # data loss
        self.ray = self.model.tensorflow_operator(self.reconstruction)
        data_mismatch = tf.square(self.ray - self.data_term)
        self.data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch, axis=(1, 2, 3)))

        # the loss functional
        self.was_output = tf.reduce_mean(self.network.net(self.reconstruction))
        self.full_error = self.mu * self.was_output + self.data_error

        # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
        # averaged quantities already. Makes gradients scaling batch size inveriant
        batch_s = tf.cast(tf.shape(self.reconstruction)[0], tf.float32)

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.full_error * batch_s, self.reconstruction)

        # Measure quality of reconstruction
        self.cut_reco = tf.clip_by_value(self.reconstruction, 0.0, 1.0)
        self.ground_truth = tf.placeholder(shape=[None, self.image_space[0], self.image_space[0], self.colors],
                                           dtype=tf.float32)
        self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.ground_truth - self.reconstruction),
                                                            axis=(1, 2, 3))))

        # logging tools
        with tf.name_scope('Network_Optimization'):
            dd = tf.summary.scalar('Data_Difference', self.wasserstein_loss)
            lr = tf.summary.scalar('Lipschitz_Regulariser', self.regulariser_was)
            ol = tf.summary.scalar('Overall_Net_Loss', self.loss_was)
            self.merged_network = tf.summary.merge([dd, lr, ol])
        with tf.name_scope('Picture_Optimization'):
            data_loss = tf.summary.scalar('Data_Loss', self.data_error)
            wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_output)
            recon = tf.summary.image('Reconstruction', self.cut_reco, max_outputs=1)
            ground_truth = tf.summary.image('Ground_truth', self.ground_truth, max_outputs=1)
            quality_assesment = tf.summary.scalar('L2_to_ground_truth', self.quality)
            self.merged_pic = tf.summary.merge([data_loss, wasser_loss, quality_assesment, recon, ground_truth])
        with tf.name_scope('Reconstruction_Quality'):
            data_loss = tf.summary.scalar('Data_Loss', self.data_error)
            wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_output)
            recon = tf.summary.image('Reconstruction', self.cut_reco, max_outputs=1)
            ground_truth = tf.summary.image('Ground_truth', self.ground_truth, max_outputs=1)
            quality_assesment = tf.summary.scalar('L2_to_ground_truth', self.quality)
            self.training_eval = tf.summary.merge([data_loss, wasser_loss, quality_assesment, recon, ground_truth])

        # set up the logger
        self.writer = tf.summary.FileWriter(self.path + 'Logs/Network_Optimization/')

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def update_pic(self, steps, stepsize, measurement, guess, mu):
        # updates the guess to come closer to the solution of the variational problem.
        for k in range(steps):
            gradient = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: guess,
                                                               self.data_term: measurement,
                                                               self.mu: mu})
            guess = guess - stepsize * gradient[0]
        return guess

    def unreg_mini(self, y, fbp):
        # unregularised minimization. In case the method to compute the pseudo inverse returns images that
        # are far from the data minimizer, it might be benificial to do some steps of gradient descent on the data
        # term before applying the adversarial regularizer algorithm. Set starting_point to 'Mini' and define the amount
        # of steps and step size to be performed before any training or reconstruction on the data term here.
        return self.update_pic(10, y, fbp, 0)

    def log_minimum(self):
        # Finds the optimumt of reconstruction and logs the values of this point only
        # This method is meant for quality evaluation during training
        # It uses the default values set in the class file for step size, total steps and regularisation
        # parameter mu.
        y, x_true, fbp = self.generate_training_data(self.batch_size, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        k = 0
        guess_update = self.update_pic(self.total_steps,  y, guess, self.mu_default)
        logs, step = self.sess.run([self.training_eval, self.global_step], feed_dict={self.reconstruction: guess,
                                                                                      self.data_term: y,
                                                                                      self.ground_truth: x_true,
                                                                                      self.mu: self.mu_default})
        self.writer.add_summary(logs, step)

    def log_network_training(self):
        # evaluates and prints the network performance.
        y, x_true, fbp = self.generate_training_data(batch_size=self.batch_size, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp=fbp)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=self.batch_size)
        logs, step = self.sess.run([self.merged_network, self.global_step],
                                   feed_dict={self.gen_im: guess, self.true_im: x_true,
                                              self.random_uint: epsilon})
        self.writer.add_summary(logs, step)

    def log_optimization(self, batch_size=None, steps=None, step_s=None,
                         mu=None, starting_point=None):
        # Logs every step of picture optimization.
        # Can be used to play with the variational formulation once training is complete
        if batch_size is None:
            batch_size = self.batch_size
        if steps is None:
            steps = self.total_steps
        if step_s is None:
            step_s = self.step_size
        if mu is None:
            mu = self.mu_default
        if starting_point is None:
            starting_point = self.starting_point
        y, x_true, fbp = self.generate_training_data(batch_size, training_data=False)
        guess = np.copy(fbp)
        if starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        writer = tf.summary.FileWriter(self.path + '/Logs/Picture_Opt/mu_{}_step_s_{}'.format(mu, step_s))
        for k in range(steps+1):
            summary = self.sess.run(self.merged_pic,
                                    feed_dict={self.reconstruction: guess,
                                               self.data_term: y,
                                               self.ground_truth: x_true,
                                               self.mu: mu})
            writer.add_summary(summary, k)
            guess = self.update_pic(1, step_s, y, guess, mu)
        writer.close()

    def train(self, steps):
        # the training routine
        for k in range(steps):
            if k % 100 == 0:
                self.log_network_training()
                self.log_minimum()
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            guess = np.copy(fbp)
            if self.starting_point == 'Mini':
                guess = self.unreg_mini(y, fbp=fbp)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=self.batch_size)
            # optimize network
            self.sess.run(self.optimizer,
                          feed_dict={self.gen_im: guess, self.true_im: x_true, self.random_uint: epsilon})
        self.save(self.global_step)

    def find_good_lambda(self, sample=64):
        # Method to estimate a good value of the regularisation paramete.
        # This is done via estimation of 2 ||K^t (Kx-y)||_2 where x is the ground truth
        y, x_true, fbp = self.generate_training_data(sample)
        gradient_truth = self.sess.run(self.pic_grad, {self.reconstruction: x_true,
                                       self.data_term: y,
                                       self.ground_truth: x_true,
                                       self.mu: 0})
        print('Value of mu around equilibrium: ' + str(np.mean(np.sqrt(np.sum(
              np.square(gradient_truth[0]), axis=(1, 2, 3))))))

    def evaluate(self, guess, measurement):
        fbp = np.copy(guess)
        if self.starting_point == 'Mini':
            fbp = self.unreg_mini(measurement, fbp)
        return self.update_pic(steps=self.total_steps, measurement=measurement, guess=fbp,
                                mu=self.mu_default)


class PostProcessing(GenericFramework):
    # Framework for postprocessing
    model_name = 'PostProcessing'
    # learning rate for Adams
    learning_rate = 0.001
    # The batch size
    batch_size = 16
    # noise level
    noise_level = 0.02

    def __init__(self, data_path, saves_path):
        # call superclass init
        super(PostProcessing, self).__init__(data_path, saves_path)

        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        self.fbp = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        # network output
        self.out = self.network.net(self.fbp)
        # compute loss
        data_mismatch = tf.square(self.out - self.true)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))
        # optimizer
        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step)
        # logging tools
        tf.summary.scalar('Loss', self.loss)
        tf.summary.image('Reconstruction', self.out, max_outputs=1)
        tf.summary.image('Input', self.fbp, max_outputs=1)
        tf.summary.image('GroundTruth', self.true, max_outputs=1)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        
        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def log(self, x_true, fbp):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.true: x_true,
                                                 self.fbp: fbp})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.true: x_true,
                                                    self.fbp: fbp})
            if k % 50 == 0:
                y, x_true, fbp = self.generate_training_data(self.batch_size, training_data=True)
                self.log(x_true, fbp)
        self.save(self.global_step)

    def evaluate(self, guess, measurement):
        output = self.sess.run(self.out, feed_dict={self.fbp: guess})
        return output

    def evaluate_red(self, y, initial_guess, step_size, reg_para, steps):
        # implements the RED method with the denoising neural network as denoising model.
        guess = initial_guess
        for j in range(steps):
            gradient_data = np.zeros(shape=(guess.shape[0], self.image_size[0], self.image_space[1], self.colors))
            for k in range(guess.shape[0]):
                data_misfit = self.model.forward_operator(guess[k, ...]) - y[k, ...]
                gradient_data[k, ...] = self.model.forward_operator_adjoint(data_misfit)
            gradient_reg = guess - self.evaluate(y, guess)
            gradient = gradient_data + reg_para*gradient_reg
            guess = guess - step_size * gradient
        return guess
    
    
class AdversarialRegulariserKeras(GenericFramework):
    model_name = 'Adversarial_Regulariser'
    # the absolut noise level
    batch_size = 16
    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 1.5
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0001
    # default step size for picture optimization
    step_size = 1
    # the amount of steps of gradient descent taken on loss functional
    total_steps_default = 30
    # default sampling pattern
    starting_point = 'Mini'

    update_mini_steps = 10
    tv_coeff = 10e-6

    def set_total_steps(self, steps):
        self.total_steps = steps

    # sets up the network architecture
    def __init__(self, data_path, saves_path,restore=True):
        # call superclass init
        super().__init__(data_path, saves_path)
        
        self.global_step = tf.Variable(0, name = 'global_step', trainable=False)
        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        self.writer = tf.summary.create_file_writer(os.path.join(self.path +'Logs/Network_Optimization/'))
        self.writer_picture = tf.summary.create_file_writer(os.path.join(self.path + "Logs/Picture_Optimization"))
        

        self._tv_coeff = self.tv_coeff
        
        
        self.set_total_steps(self.total_steps_default)
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.network.netw, step=self.global_step)
        
        ut.create_single_folder(self.path + 'checkpoints')
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.path+'checkpoints', max_to_keep=3)
        
        if restore:
            self.restore_checkpoint()
        

    def toggle_tv(self):
        assert (self.tv_coeff == self._tv_coeff or self.tv_coeff == 0)
        self.tv_coeff = self._tv_coeff - self.tv_coeff
    
    def find_good_lambda(self, sample=64):
        y, x_true, fbp = self.generate_training_data(sample)
        
        gradient_truth = self._get_pic_grad(x_true, y, 0)
        print('Value of mu around equilibrium: ' + str(np.mean(np.sqrt(np.sum(
              np.square(gradient_truth), axis=(1, 2, 3))))))
        
    @tf.function
    def _evaluate_loss(self, reconstruction, data_term,mu, tv_coeff = 0, gradient= tf.constant(False)):
        
        def only_loss():  
            ray = self.model.tensorflow_operator(reconstruction)
            data_mismatch = tf.square(ray - data_term) #L2
            
            data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch,axis=(1,2,3)))
            was_output = tf.reduce_mean(self.network.net(reconstruction))
            batch_size = tf.cast(tf.shape(reconstruction)[0], tf.float32)
            tv_coeff_tensor = tf.convert_to_tensor(tv_coeff, dtype=tf.float32)
            
            def with_tv():
                tv_loss = tv_coeff_tensor * smooth_total_variation(reconstruction)
                return (mu * was_output + data_error + tv_loss) * batch_size
        
            def without_tv():
                return (mu * was_output + data_error) * batch_size
        
            full_error = tf.cond(tv_coeff_tensor > 0, with_tv, without_tv)
            return full_error,tf.zeros_like(reconstruction) 

        def loss_with_gradient():
            with tf.GradientTape() as tape:
                tape.watch(reconstruction)
                ray = self.model.tensorflow_operator(reconstruction)
                data_mismatch = tf.square(ray - data_term) #L2
                data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch,axis=(1,2,3)))
               
                was_output = tf.reduce_mean(self.network.net(reconstruction))
                batch_size = tf.cast(tf.shape(reconstruction)[0], tf.float32)
            
                tv_coeff_tensor = tf.convert_to_tensor(tv_coeff, dtype=tf.float32)
            
                def with_tv():
                    tv_loss = tv_coeff_tensor * smooth_total_variation(reconstruction)
                    return (mu * was_output + data_error + tv_loss) * batch_size
            
                def without_tv():
                    return (mu * was_output + data_error) * batch_size
            
                full_error = tf.cond(tv_coeff_tensor > 0, with_tv, without_tv)
            pic_gradient = tape.gradient(full_error, reconstruction)
            return full_error, pic_gradient
            
        return tf.cond(gradient,loss_with_gradient, only_loss)
    
    @tf.function
    def _get_pic_grad(self, reconstruction, data_term, mu):
        reconstruction = tf.convert_to_tensor(reconstruction,dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(reconstruction)
            full_error = self._evaluate_loss(reconstruction, data_term, mu)
            
        pic_gradient = tape.gradient(full_error, reconstruction)
        return pic_gradient
    
    @tf.function 
    def _backtracking_line_search(self, reconstruction, loss,grad, data_term, mu, alpha0 = 1, c = 0.0001, rho=0.5, itermax = 100):
        alpha = tf.convert_to_tensor(alpha0, dtype=tf.float32)
        m = tf.norm(grad) 
        direction = - grad 
        t = -c*m**2
        k = tf.constant(0)
        
         
        next_step = reconstruction + alpha * direction 
        loss_next_step,_ = self._evaluate_loss(next_step,data_term, mu)
        
        def cond(alpha, k, loss_next_step):
            armijo = (loss_next_step - loss) > (alpha * t)
            return tf.logical_and(armijo, k < itermax)

        def body(alpha, k, loss_next_step):
            alpha = alpha * rho
            k = k+1
            next_step = reconstruction + alpha * direction 
            loss_next_step,_ = self._evaluate_loss(next_step, data_term, mu)
            return alpha, k, loss_next_step

        alpha, k, loss_next_step = tf.while_loop(cond, body, [alpha, k, loss_next_step])
        #tf.print(loss_next_step < loss)
        return alpha
    
    @tf.function
    def update_pic(self, steps,  measurement, guess, mu, tv_coeff = 0.0, TOL= 1e-8):
        # updates the guess to come closer to the solution of the variational problem.
                
        guess = tf.convert_to_tensor(guess)
        k = tf.constant(0)
        TOL = tf.constant(TOL, dtype=guess.dtype)
        loss, grad = self._evaluate_loss(guess, measurement, mu, tv_coeff = tv_coeff, gradient = tf.constant(True))
        norm_grad = tf.norm(grad)

        def cond(k, guess, loss, grad, norm_grad):
            return tf.logical_and(k < steps, norm_grad > TOL)
        
        def body(k, guess, loss, grad, norm_grad):
            direction = - grad 
            alpha = self._backtracking_line_search(guess, loss, grad, measurement, mu)
            guess = guess + alpha * direction
            loss, grad = self._evaluate_loss(guess, measurement, mu,tv_coeff = tv_coeff, gradient = tf.constant(True))
            #tf.print("loss:",loss ,"-alpha:", alpha)
            return k + 1, guess, loss, grad, norm_grad 

        k, guess, loss, grad, norm_grad = tf.while_loop(
                cond, body, [k, guess, loss, grad, norm_grad])
       
        return guess


    def unreg_mini(self, y, fbp):
        # unregularised minimization. In case the method to compute the pseudo inverse returns images that
        # are far from the data minimizer, it might be benificial to do some steps of gradient descent on the data
        # term before applying the adversarial regularizer algorithm. Set starting_point to 'Mini' and define the amount
        # of steps and step size to be performed before any training or reconstruction on the data term here.
        return self.update_pic(10,y,fbp, 0, tv_coeff = self.tv_coeff)
    
    
    
    @tf.function
    def _calculate_network_training_losses(self,guess,x_true,inter):
        gen_was = self.network.net(guess)
        data_was = self.network.net(x_true)
        
        wasserstein_loss = tf.reduce_mean(data_was - gen_was)
        
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(inter)
            inter_was = self.network.net(inter)
        gradient_was = inner_tape.gradient(inter_was, inter)
        norm_gradient = tf.sqrt(tf.reduce_sum(tf.square(gradient_was),axis=(1,2,3)))
        regularizer_was = tf.reduce_mean(tf.square(tf.nn.relu(norm_gradient - 1)))
        
        loss_was = wasserstein_loss + self.lmb * regularizer_was
    
    
        return loss_was

    def log_network_training(self):
        # evaluates and prints the network performance.
        y, x_true, fbp = self.generate_training_data(batch_size=self.batch_size, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp=fbp)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=self.batch_size)
        epsilon = tf.convert_to_tensor(epsilon,dtype=tf.float32)
        epsilon = tf.expand_dims(tf.expand_dims(epsilon,axis=1),axis=1)
        epsilon = tf.expand_dims(epsilon,axis=1)
        inter = tf.multiply(guess, epsilon) + \
            tf.multiply(x_true,1 - epsilon)
        
        step = self.global_step.numpy()
        
        gen_was = self.network.net(guess)
        data_was = self.network.net(x_true)
        
        wasserstein_loss = tf.reduce_mean(data_was - gen_was)
        
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(inter)
            inter_was = self.network.net(inter)
        
        gradient_was = inner_tape.gradient(inter_was, inter)
        norm_gradient = tf.sqrt(tf.reduce_sum(tf.square(gradient_was),axis=(1,2,3)))
        regularizer_was = tf.reduce_mean(tf.square(tf.nn.relu(norm_gradient - 1)))
        
        loss_was = wasserstein_loss + self.lmb * regularizer_was
        
        
        with self.writer.as_default():
            tf.summary.scalar("Network_Optimization/STD_Ground_Truth_Wass",tf.math.reduce_std(data_was),step=step)
            tf.summary.scalar("Network_Optimization/STD_Reconstruction_Wass",tf.math.reduce_std(gen_was),step=step)
            tf.summary.scalar("Network_Optimization/Average_Reconstruction_Wass", tf.reduce_mean(gen_was),step=step)
            tf.summary.scalar("Network_Optimization/Average_Ground_Truth_Wass", tf.reduce_mean(data_was),step=step)
            tf.summary.scalar("Network_Optimization/Average_Inter_Wass", tf.reduce_mean(inter_was),step=step)
            tf.summary.scalar('Network_Optimization/Data_Difference', wasserstein_loss, step=step)
            tf.summary.scalar('Network_Optimization/Lipschitz_Regulariser', regularizer_was, step=step)
            tf.summary.scalar('Network_Optimization/Overall_Net_Loss', loss_was, step=step)

        
        
    
    def restore_checkpoint(self):
        latest_ckpt = self.checkpoint_manager.latest_checkpoint
        print(latest_ckpt)
        if latest_ckpt:
            self.checkpoint.restore(latest_ckpt)
            print(f"Restored from {latest_ckpt}, step={self.global_step.numpy()}")
        else:
            print("Initializing from scratch.")
            

    @tf.function
    def train_step(self, y, x_true, guess):
        epsilon = np.random.uniform(size=self.batch_size)
        epsilon = tf.convert_to_tensor(epsilon,dtype=tf.float32)
        epsilon = tf.expand_dims(tf.expand_dims(epsilon,axis=1),axis=1)
        epsilon = tf.expand_dims(epsilon,axis=1)
        inter = tf.multiply(guess, epsilon) + \
            tf.multiply(x_true,1 - epsilon)
        with tf.GradientTape() as tape:
            loss_was = self._calculate_network_training_losses(guess,x_true, inter)
        
        gradients = tape.gradient(loss_was, self.network.netw.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.network.netw.trainable_variables))

    def train(self, steps):

        for k in range(steps):

            y, x_true , fbp = self.generate_training_data(self.batch_size)
            guess = np.copy(fbp)
            if self.starting_point == "Mini":                    
                guess = self.unreg_mini(y, fbp)

            self.train_step(y, x_true, guess)
            self.global_step.assign_add(1)

            if k % CHECKPOINT_RATE == 0:
                self.checkpoint_manager.save()
            if k % RECORD_RATE == 0:
                self.log_network_training()
                self.log_minimum()
                self.log_update_mini()
                self.log_optimization(batch_size=1)
    
  
    def evaluate(self, guess, measurement,step_size=None,total_steps=None,step_rho=0.9):
        fbp = np.copy(guess)
        if self.starting_point == "Mini":
            fbp = self.unreg_mini(measurement, fbp)
        return self.update_pic(steps=self.total_steps if total_steps is None else total_steps, measurement=measurement, guess=fbp, stepsize=self.step_size if step_size is None else step_size, mu = self.mu_default,step_rho=0.9)


    def reconstruct(self, measurements, total_steps):
        fbp = self.model.inverse(measurements)
        if self.starting_point == "Mini":
            fbp = self.unreg_mini(measurements, fbp)
        return self.update_pic(steps=total_steps, measurement=measurements, guess = fbp, mu = self.mu_default)
    
    def log_minimum(self):

        y, x_true, fbp = self.generate_training_data(1, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        guess = self.update_pic(self.total_steps,  y, guess, self.mu_default)
        
        step = self.global_step.numpy()
        cut_reco = tf.clip_by_value(guess, 0.0,1.0)
        quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x_true - guess),
                                                            axis=(1, 2, 3))))
        ray = self.model.tensorflow_operator(guess)
        data_mismatch = tf.square(ray - y)

        if self.tv_coeff > 0:
            tv = tf.reduce_mean(smooth_total_variation(guess))
        
        data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch,axis=(1,2,3)))
        was_output = tf.reduce_mean(self.network.net(guess))

        psnr = tf.image.psnr(x_true, guess, max_val=1.0)
        ssim = tf.image.ssim(x_true, guess, max_val=1.0)
        with self.writer.as_default():
            tf.summary.scalar('Reconstruction_Quality/Mean_Data_Loss', data_error, step=step)
            tf.summary.scalar('Reconstruction_Quality/Mean_Wasserstein_Loss', was_output, step=step)
            tf.summary.scalar('Reconstruction_Quality/Mean_L2_to_ground_truth', quality, step=step)
            tf.summary.scalar('Reconstruction_Quality/Mean_PSNR', tf.reduce_mean(psnr), step=step)
            tf.summary.scalar('Reconstruction_Quality/Mean_SSIM', tf.reduce_mean(ssim), step=step)
            if self.tv_coeff > 0:
                tf.summary.scalar('Reconstruction_Quality/Mean_TV', tv, step=step)
            tf.summary.image('Reconstruction_Quality/Reconstruction', cut_reco, step=step, max_outputs=1)
            tf.summary.image("Reconstruction_Quality/Noisy_reconstruction",fbp,step=step, max_outputs=1)
            tf.summary.image('Reconstruction_Quality/Ground_truth', x_true, step=step, max_outputs=1)
            
            

    def log_update_mini(self):
        
        y, x_true, fbp = self.generate_training_data(1, training_data=False)
        guess = np.copy(fbp)
        
        for k in range(self.update_mini_steps+1):
            k += 1
            
            cut_reco = tf.clip_by_value(guess, 0.0,1.0)
            quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x_true - guess),
                                                                axis=(1, 2, 3))))
            reconstuction_to_guess = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(guess - fbp))))
            ray = self.model.tensorflow_operator(guess)
            data_mismatch = tf.square(ray - y)

            if self.tv_coeff > 0:
                tv = smooth_total_variation(guess)
        
            data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch,axis=(1,2,3)))
            was_output = tf.reduce_mean(self.network.net(guess))
            
            with self.writer_picture.as_default():
                tf.summary.scalar('Mini_steps/Data_Loss', data_error, step=k)
                tf.summary.scalar('Mini_steps/Wasserstein_Loss', was_output, step=k)
                tf.summary.scalar('Mini_steps/L2_to_ground_truth', quality, step=k)
                tf.summary.scalar('Mini_steps/L2_to_noisy_reconstruction', reconstuction_to_guess,step=k)
                if self.tv_coeff > 0:
                    tf.summary.scalar('Mini_steps/Total_Variation', tv,step=k)
                tf.summary.image('Mini_steps/Reconstruction', cut_reco, step=k, max_outputs=1)
                tf.summary.image("Mini_steps/Noisy_reconstruction",fbp,step=k, max_outputs=1)
                tf.summary.image('Mini_steps/Ground_truth', x_true, step=k, max_outputs=1)
            guess = self.update_pic(1, y, guess, 0, tv_coeff = self.tv_coeff)
            
    def log_optimization(self, batch_size=None, steps=None, step_s=None,
                         mu=None, starting_point=None):
        # Logs every step of picture optimization.
        # Can be used to play with the variational formulation once training is complete
        if batch_size is None:
            batch_size = self.batch_size
        if steps is None:
            steps = self.total_steps
        if step_s is None:
            step_s = self.step_size
        if mu is None:
            mu = self.mu_default
        if starting_point is None:
            starting_point = self.starting_point
        y, x_true, fbp = self.generate_training_data(batch_size, training_data=False)
        guess = np.copy(fbp)
        if starting_point == 'Mini':
            guess = self.unreg_mini(y, fbp)
        for k in range(steps+1):
            k += 1
            
            step = self.global_step.numpy()
            cut_reco = tf.clip_by_value(guess, 0.0,1.0)
            quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x_true - guess),
                                                                axis=(1, 2, 3))))
            reconstuction_to_guess = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(guess - fbp))))
            ray = self.model.tensorflow_operator(guess)
            data_mismatch = tf.square(ray - y)

            if self.tv_coeff > 0:
                tv = tf.reduce_mean(smooth_total_variation(guess))

        
            data_error = tf.reduce_mean(tf.reduce_sum(data_mismatch,axis=(1,2,3)))
            was_output = tf.reduce_mean(self.network.net(guess))
            #tf.print("loss_func:", data_error + mu * was_output)
            
            with self.writer_picture.as_default():
                tf.summary.scalar('mu_{}_step_s_{}/Data_Loss'.format(mu,step_s), data_error, step=k)
                tf.summary.scalar('mu_{}_step_s_{}/Wasserstein_Loss'.format(mu,step_s), was_output, step=k)
                tf.summary.scalar('mu_{}_step_s_{}/L2_to_ground_truth'.format(mu,step_s), quality, step=k)
                tf.summary.scalar('mu_{}_step_s_{}/L2_to_noisy_reconstruction'.format(mu,step_s), reconstuction_to_guess,step=k)
                if self.tv_coeff > 0:
                    tf.summary.scalar('mu_{}_step_s_{}/Total_Variation'.format(mu,step_s), tv,step=k)
                tf.summary.image('mu_{}_step_s_{}/Reconstruction'.format(mu,step_s), cut_reco, step=k, max_outputs=1)
                tf.summary.image("mu_{}_step_s_{}/Noisy_reconstruction".format(mu,step_s),fbp,step=k, max_outputs=1)
                tf.summary.image('mu_{}_step_s_{}/Ground_truth'.format(mu,step_s), x_true, step=k, max_outputs=1)
            guess = self.update_pic(1, y, guess, mu)

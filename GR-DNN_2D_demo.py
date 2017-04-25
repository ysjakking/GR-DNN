from __future__ import print_function
import six.moves.cPickle as pickle
import gzip

import os
import sys
import timeit

import numpy
import scipy.io as sio

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler


from GR_DNN_class import GR_DNN

def load_data(dataset):
    ''' Load the dataset (here MNIST)
    :param dataset: the path to the dataset 
    '''
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_graph(dataset,K=30,n_c=1000):
    ''' construct the pre-constructed KNN graph (here from MNIST)
        :param dataset: the path to the dataset
                     K: the K of the KNN
                     n_: number of anchor points
        '''
    data_dir, data_file = os.path.split(dataset)
    AG_file='AG_%s_%d_%d.mat' % (data_file,K,n_c)
    if os.path.isfile(AG_file):
        # load pre-trained graph from mat
        print('... loading %s '%(AG_file))
        mat_contents = sio.loadmat(AG_file)
        train_graph = mat_contents['train_AG']
        val_graph = mat_contents['val_AG']

    else:
        print('... construct %s ' % (AG_file))
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

        data_x, data_y = train_set
        data_x_valid, data_y_valid = valid_set

        c_result = KMeans(n_clusters=n_c, n_jobs=-1).fit(data_x)
        centers = c_result.cluster_centers_

        # train_graph
        sim = rbf_kernel(data_x, centers) #calculate similarity
        s_sim = numpy.sort(sim)
        s_sim_k_value = s_sim[:, -K]
        ss = s_sim_k_value.reshape(s_sim_k_value.shape[0], 1)
        ss = numpy.dot(ss, numpy.ones((1, n_c)))

        select_knn = (sim - ss) > 0 #select K neighbour

        train_graph = sim * select_knn

        # valid_graph
        sim_valid = rbf_kernel(data_x_valid, centers) #calculate similarity
        s_sim_valid = numpy.sort(sim_valid)
        s_sim_k_value = s_sim_valid[:, -K]
        ss = s_sim_k_value.reshape(s_sim_k_value.shape[0], 1)
        ss = numpy.dot(ss, numpy.ones((1, n_c)))

        select_knn = (sim_valid - ss) > 0 #select K neighbour

        val_graph = sim_valid * select_knn


        sio.savemat(AG_file,{'train_AG': train_graph, 'val_AG': val_graph})


    train_graph = theano.shared(numpy.asarray(train_graph,dtype=theano.config.floatX),borrow=True)
    val_graph = theano.shared(numpy.asarray(val_graph,dtype=theano.config.floatX),borrow=True)

    return train_graph,val_graph


def test_GRDNN(model_version_name,data_path_sizes,graph_path_sizes,finetune_lr=0.05,finetune_lr_GR =0.01, pretraining_epochs=50,
             pretrain_lr=0.01, training_epochs=1000,
             dataset='./mnist.pkl.gz', batch_size=50,corruption=0.2):


    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #contruct KNN graph
    [train_graph, val_graph]=load_graph(dataset)



    # compute number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')
    # construct the GR-DNN class
    sda=GR_DNN(
        numpy_rng,
        n_ins_data=784,
        n_ins_graph=1000,
        lamuda=1,
        hidden_layers_sizes=data_path_sizes,
        graph_hidden_sizes= graph_path_sizes)




    max_L_da= sda.n_layers_data if sda.n_layers_data >= sda.n_layers_graph else sda.n_layers_graph

    corruption_levels = corruption * numpy.ones(max_L_da,dtype=numpy.float32)


    ###Step 1: The layer-wise pre-training
    print('... getting the layer-wise pretraining functions')
    pretraining_fns_data = sda.pretraining_functions_data(train_set_x=train_set_x,batch_size=batch_size)
    pretraining_fns_graph = sda.pretraining_functions_graph(train_set_x=train_graph,batch_size=batch_size)

    #############
    print('... layer-wise pre-training the data-path')

    start_time = timeit.default_timer()

    for i in range(sda.n_layers_data):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns_data[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()

    print(('The layer-wise pre-training the data-path ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    #############

    print('... layer-wise pre-training the graph path')
    start_time = timeit.default_timer()

    for i in range(sda.n_layers_graph):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns_graph[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()

    print(('The layer-wise pre-training the graph-path ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    ###Step 2: The path-wise fine-tuning

    print('... getting the finetuning functions')
    train_fn_data, validate_model_data, test_model_data ,generate_code= sda.build_finetune_functions_data(
        train_set_x=train_set_x,
        valid_set_x=valid_set_x ,
        test_set_x=test_set_x,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    train_fn_graph, validate_model_graph = sda.build_finetune_functions_graph(
        train_set_x=train_graph,
        valid_set_x=val_graph ,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the data path')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.998  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        batch_order=range(n_train_batches)
        numpy.random.shuffle(batch_order)
        for minibatch_index in batch_order:
            minibatch_avg_cost = train_fn_data(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model_data()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, training_cost %f , validation error %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       minibatch_avg_cost* 1,
                       this_validation_loss * 1.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # generate the final deep auto-encoder code and the reconstructed samples of the test set
                    [test_codes,test_reconstruct ]=generate_code()
                    test_codes= numpy.asarray(test_codes)
                    test_reconstruct = numpy.asarray(test_reconstruct)
                    sio.savemat('test_codes_data_%s.mat'%(model_version_name), {'test_codes':test_codes,'test_reconstruct':test_reconstruct})

                    # test it on the test set
                    test_losses = test_model_data()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f ') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 1.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f '
        )
        % (best_validation_loss, best_iter + 1, test_score)
    )
    print(('The path-wise fine-tuning data path ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


    #############
    print('... finetunning the graph path')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.998  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1

        batch_order=range(n_train_batches)
        numpy.random.shuffle(batch_order)

        for minibatch_index in batch_order:
            minibatch_avg_cost = train_fn_graph(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model_graph()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, training_cost %f , validation error %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       minibatch_avg_cost* 1,
                       this_validation_loss * 1.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter


            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print(('The path-wise fine-tuning graph path ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    #################################################
    ###Step 3: The joint fine-tuning of GR-DNN
    print('...joint finetunning the GR_DNN')
    train_fn_DNN, validate_model_DNN = sda.build_finetune_functions_GRDNN(
        train_set_x=train_set_x,
        valid_set_x=valid_set_x ,
        train_graph=train_graph,
        val_graph=val_graph,
        batch_size=batch_size,
        learning_rate=finetune_lr_GR
    )
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 10.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.998  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        batch_order=range(n_train_batches)
        numpy.random.shuffle(batch_order)
        for minibatch_index in batch_order:
            minibatch_avg_cost = train_fn_DNN(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model_DNN()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, training_cost %f , validation error %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       minibatch_avg_cost* 1,
                       this_validation_loss * 1.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    [test_codes,test_reconstruct ]=generate_code()
                    test_codes= numpy.asarray(test_codes)
                    test_reconstruct = numpy.asarray(test_reconstruct)
                    sio.savemat('test_codes_GRDNN_%s.mat'%(model_version_name), {'test_codes':test_codes,'test_reconstruct':test_reconstruct})

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print(('The joint fine-tuning of GR-DNN ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    test_GRDNN('model1',data_path_sizes=[1000,500,250,2],graph_path_sizes=[1200,500,100,2]) #provide a model_version_name for each training

    '''
    model1 : hidden_layers_sizes [1000,500,250,2]           graph_hidden_sizes [1200,500,100,2]           currupt=0.2  lamuda=1
    '''
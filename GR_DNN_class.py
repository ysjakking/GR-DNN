import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from dA import dA

class GR_DNN(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins_data=784,
        n_ins_graph=1000,
        lamuda=1,
        hidden_layers_sizes=[500, 500],
        graph_hidden_sizes=[500,500]

    ):
        self.lamuda=lamuda
        ##data pathway
        self.sigmoid_layers_data = []
        self.dA_layers_data = []
        self.params_data = []
        self.n_layers_data = len(hidden_layers_sizes)

        ##graph pathway
        self.sigmoid_layers_graph = []
        self.dA_layers_graph = []
        self.params_graph = []
        self.n_layers_graph = len(graph_hidden_sizes)

        ##for joint training
        self.sigmoid_layers_DNN = []
        self.params_DNN = []

        assert self.n_layers_data > 0
        assert self.n_layers_graph > 0
        assert hidden_layers_sizes[-1] == graph_hidden_sizes[-1]

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic input variables for the data-path and graph-path
        self.x = T.matrix('x')
        self.graph = T.matrix('graph')


        ###################### building data-path #######################
        # construct the encoder of the data-path
        for i in range(self.n_layers_data):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins_data
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input =  self.sigmoid_layers_data[-1].output

            if i==self.n_layers_data-1:
                temp_activation=None
            else:
                temp_activation=T.nnet.sigmoid

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=temp_activation)
            # add the layer to the list of layers
            self.sigmoid_layers_data.append(sigmoid_layer)
            self.params_data.extend(sigmoid_layer.params)

            # Construct an auto-encoder that shared weights with this
            # layer, which is used for layer-wise pre-training
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers_data.append(dA_layer)

        #define the final output code layer of GR-DNN
        self.out_put_codes = self.sigmoid_layers_data[-1].output

        # construct the decoder of the data-path
        temp_range=range(self.n_layers_data)
        temp_range=temp_range[::-1]
        for i in temp_range:
            temp_dA=self.dA_layers_data[i]

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=self.sigmoid_layers_data[-1].output,
                                        n_in=temp_dA.n_hidden,
                                        n_out=temp_dA.n_visible,
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers_data.append(sigmoid_layer)
            self.params_data.extend(sigmoid_layer.params)

        # define the reconstructed output of the data-path
        z_data= self.sigmoid_layers_data[-1].output
        self.out_reconstruct_data = z_data

        # define the reconstructed loss (binary cross-entropy) of the data-path
        L_data = - T.sum(self.x * T.log(z_data) + (1 - self.x) * T.log(1 - z_data), axis=1)
        self.finetune_cost_data = T.mean(L_data)


        ###################### building graph path ########################
        # construct graph encoder
        for i in range(self.n_layers_graph):

            if i == 0:
                input_size = n_ins_graph
            else:
                input_size = graph_hidden_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.graph
            else:
                layer_input = self.sigmoid_layers_graph[-1].output

            if i==self.n_layers_graph-1:
                temp_activation=None
            else:
                temp_activation=T.nnet.sigmoid

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=graph_hidden_sizes[i],
                                        activation=temp_activation)
            # add the layer to the list of layers
            self.sigmoid_layers_graph.append(sigmoid_layer)
            self.params_graph.extend(sigmoid_layer.params)

            # Construct an autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=graph_hidden_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers_graph.append(dA_layer)

        # construct graph-decoder
        temp_range=range(self.n_layers_graph)
        temp_range=temp_range[::-1]
        ii=0
        for i in temp_range:

            ii=ii+1

            temp_dA=self.dA_layers_graph[i]

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=self.sigmoid_layers_graph[-1].output,
                                        n_in=temp_dA.n_hidden,
                                        n_out=temp_dA.n_visible,
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers_graph.append(sigmoid_layer)
            self.params_graph.extend(sigmoid_layer.params)

            # construct the shared graph-decoder for joint training
            if ii == 1:
                layer_input_DNN=self.out_put_codes
            else:
                layer_input_DNN = self.sigmoid_layers_DNN[-1].output

            # note that the params W and b are shared
            sigmoid_layer_DNN = HiddenLayer(rng=numpy_rng,
                                        input=layer_input_DNN,
                                        n_in=temp_dA.n_hidden,
                                        n_out=temp_dA.n_visible,
                                        W=sigmoid_layer.W,
                                        b=sigmoid_layer.b,
                                        activation=T.nnet.sigmoid)

            self.sigmoid_layers_DNN.append(sigmoid_layer_DNN)
            self.params_DNN.extend(sigmoid_layer_DNN.params)

        # define the reconstructed output and the loss of the graph-path
        z_graph= self.sigmoid_layers_graph[-1].output
        L_graph = - T.sum(self.graph * T.log(z_graph) + (1 - self.graph) * T.log(1 - z_graph), axis=1)

        self.finetune_cost_graph = T.mean(L_graph)
        self.out_reconstruct_graph = z_graph

        ##GR-DNN joint training cost
        z_graph_DNN= self.sigmoid_layers_DNN[-1].output
        L_graph_DNN = - T.sum(self.graph * T.log(z_graph_DNN) + (1 - self.graph) * T.log(1 - z_graph_DNN), axis=1)

        #add the params of data-path for joint training of the GR-DNN
        self.params_DNN.extend(self.params_data)
        #the final objective
        self.finetune_cost_graphDNN = self.lamuda * T.mean(L_graph_DNN) + self.finetune_cost_data


    ##construct layer-wise pretraining functions
    def pretraining_functions_data(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')  # % corruption rate of the denoising auto-encoder
        learning_rate = T.scalar('lr')  # learning rate
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers_data:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    corruption_level,
                    learning_rate
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def pretraining_functions_graph(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')  # corruption rate
        learning_rate = T.scalar('lr')  # learning rate
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers_graph:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    corruption_level,
                    learning_rate
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.graph: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    ##construct path-wise pretraining functions
    def build_finetune_functions_data(self, train_set_x, valid_set_x , test_set_x, batch_size, learning_rate):

        #init params of the decoder from the layer-wise pre-trained auto-encoders
        for i in range(self.n_layers_data):
            self.sigmoid_layers_data[-1-i].W.set_value(self.sigmoid_layers_data[i].W.get_value().T.astype(theano.config.floatX))
            self.sigmoid_layers_data[-1-i].b.set_value(self.dA_layers_data[i].b_prime.get_value().astype(theano.config.floatX))


        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size

        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost_data, self.params_data)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params_data, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_data,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_data,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]

            },
            name='test'
        )

        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_data,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        generate_code = theano.function(
            inputs=[],
            outputs=[self.out_put_codes,self.out_reconstruct_data],
            givens={
                self.x: test_set_x
            },
            name='code'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score, generate_code

    def build_finetune_functions_graph(self, train_set_x, valid_set_x , batch_size, learning_rate):

        #init weights
        for i in range(self.n_layers_graph):
            self.sigmoid_layers_graph[-1-i].W.set_value(self.sigmoid_layers_graph[i].W.get_value().T.astype(theano.config.floatX))
            self.sigmoid_layers_graph[-1-i].b.set_value(self.dA_layers_graph[i].b_prime.get_value().astype(theano.config.floatX))

        # compute number of minibatches for validation
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost_graph, self.params_graph)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params_graph, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_graph,
            updates=updates,
            givens={
                self.graph: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )


        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_graph,
            givens={
                self.graph: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        return train_fn, valid_score

    def build_finetune_functions_GRDNN(self, train_set_x, valid_set_x ,train_graph, val_graph ,batch_size, learning_rate):
        # compute the number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost_graphDNN, self.params_DNN)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params_DNN, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_graphDNN,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.graph: train_graph[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )


        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_graphDNN,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.graph: val_graph[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        return train_fn, valid_score
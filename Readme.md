# GR-DNN
A demo code for paper 'A Graph Regularized Deep Neural Network (GR-DNN) for Unsupervised Image
Representation Learning', which appears in CVPR2017. This project qualitatively shows the learned subspace embedding by projecting them into a 2D subspace.<br>

The code is written in Theano. <br>
The modules are implemented based on the DeepLearning 0.1 documentation (http://deeplearning.net/tutorial/SdA.html).<br>
To use it you will also need: `cPickle`, `scikit-learn`, `matplotlib` and `PIL`.<br>
Before running the code make sure that you have set floatX to float32 in Theano settings.<br>

To train the model, simply run:<br>
* python GR-DNN_2D_demo.py<br>
When we first run the code, the Anchor Graph `AG_mnist.pkl.gz_30_1000.mat` will be created.<br>

This step will create files `test_codes_data_model1.mat` and `test_codes_GRDNN_model1.mat`, which contain the 2D code learned by regular stacked DAE and GR-DNN respectively.<br>

Then, to generate the image visualization of the learned 2D subspace and the reconstructed samples, run:<br>
* python plot_2D.py<br>

The resulting images will be saved in the folder './result/'.<br>



## Reference


If you found this code or our paper useful, please consider citing the following paper:<br>

@inproceedings{GRDNN17,<br>
    author    = {Shijie Yang, Liang Li, Shuhui Wang, Weigang Zhang, Qingming Huang},<br>
    title     = {A Graph Regularized Deep Neural Network for Unsupervised Image Representation Learning},<br>
    booktitle = {CVPR},<br>
    year      = {2017}<br>
}<br>




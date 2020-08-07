### Implemented Apollo space traffic_light det/rec models by python

#### Prepare
1. Install caffe envs  caffe-apollo[apollo branche]
    + https://github.com/ApolloAuto/caffe/tree/apollo

#### Run test
1. firstly, change the 'caffe_root' in the code.

+ run `python det_test.py` to get the det results.
+ run `python rec_test.py` to get the recognization results.

2. the code is easy, the preprocess and postprocess  is clearly.
3. I only test on the vertical model, you can change the horizon model in the  './detection'





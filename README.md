# BinaryClassify

Implementation of a binary classify By pytorch

### DataSets

  Prepare a dataset for positive and negative samples. Such as Eye Dataset, determine open eyes as positive samples, closed eyes as negative samples.

  1.put positive samples to `./data/positive/`

  2.put negative samples to `./data/negative/` 

  3.run `./data/train_test_data.py`
  
  ~~~shell
   $ cd ./data 
   $ python3 train_test_data.py
  ~~~
 
### training & testing

  training :

  ~~~shell
   $ sh train.sh
  ~~~

 testing:
 
 ~~~shell
  $ python3 test_img.py
 ~~~

### pytorch -> onnx -> ncnn

**Pytorch -> onnx -> onnx_sim**  

make sure pip3 install onnx-simplifier

 ~~~~shell
  python3 pytorch2onnx.py
  python3 -m onnxsim model.onnx model_sim.onnx
 ~~~~

**onnx_sim -> ncnn**  

how to build :https://github.com/Tencent/ncnn/wiki/how-to-build

 ~~~shell
  cd ncnn/build/tools/onnx
  ./onnx2ncnn model_sim.onnx model_sim.param model_sim.bin
 ~~~

### TODO:

- [x] ncnn inference
- [ ] train on FocalLoss
- [ ] train on multi-class model
- [ ] fix bugs

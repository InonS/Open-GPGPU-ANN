Open Source GPGPU support for Artificial Neural Networks
========================================================
**The benchmarks used are based on the Sentdex (PythonProgramming.net) Neural Networks Machine Learning Tutorial**

Artificial Neural Network implementation alternatives in OpenCL ('cause GPGPU != nVidia), and if that isn't enough, try Micro$oft Windoze:
1. TF:
    1. [Luke Iwanski's OpenCL support for TensorFlow via SYCL](https://github.com/lukeiwanski/tensorflow)
    2. [Benoit Steiner's OpenCL support for TensorFlow](https://github.com/benoitsteiner/tensorflow-opencl)
    1. Hugh Perkins's <a href="https://github.com/hughperkins/tf-coriander">TensorFlow on Coriander</a>.
        * No windows support 
        * No native GPU-based convolutions (see corainder-dnn WIP)
    2. Benoit Steiner's <a href="https://github.com/benoitsteiner/tensorflow-opencl/blob/master/tensorflow/g3doc/get_started/os_setup.md#optional-install-opencl-experimental-linux-only">tensorflow-opencl</a>
    3. [TensorFlow.js](https://github.com/tensorflow/tfjs-core/blob/master/src/kernels/webgl/gpgpu_math.ts) on WebGL
2. Theano:
    1. <a href="http://deeplearning.net/software/theano/tutorial/using_gpu.html#gpuarray-backend">Theano using libgpuarray</a>
    2. Jonas Jarutis's <a href="https://gist.github.com/jarutis/ff28bca8cfb9ce0c8b1a">"Theano and Keras setup on ubuntu with OpenCL on AMD card"</a> (see also <a href="https://gist.github.com/fabriciorsf/b911963d8b71987a236401c49f1b75d6">Fabrício Silva's version</a>)
3. Caffe: <a href="https://github.com/BVLC/caffe/tree/opencl#opencl-backend">OpenCL Caffe</a>
    1. <a href="https://github.com/BVLC/caffe/issues/4929#issuecomment-267226532">requires</a> the <a href="https://github.com/viennacl/viennacl-dev">ViennaCL developer repo</a> to install on Windows. see also <a href="https://github.com/BVLC/caffe/tree/windows">Caffe on Micro$oft Windoze</a>
    2. <a href="https://github.com/01org/caffe/wiki/clCaffe">Using Intel's</a> spatial convolution engine
    3. <a href="https://bitbucket.org/multicoreware/hccaffe">HCC Caffe backend</a>
4. <a href="https://github.com/ARM-software/ComputeLibrary">AMD software ComputeLibrary</a> (TODO)
5. [ArrayFire](https://github.com/arrayfire/arrayfire/tree/master/examples/machine_learning)
6. [Codeplay's ComputeCpp (SYCL implementation)](https://developer.codeplay.com/computecppce)
5. Torch:
    1. Hugh Perkins's <a href="https://github.com/hughperkins/pycltorch">pyCLTorch</a>. [Requires][pycltorch_cmake] CMake to install on Micro$oft Windoze.
    2. <a href="https://bitbucket.org/multicoreware/hctorch">HCC Torch backend</a>
5. DeepCL: Hugh Perkins's <a href="https://github.com/hughperkins/DeepCL">DeepCL</a>. (Works! still trying to get the <a href="https://pypi.python.org/pypi/DeepCL">Python bindings</a> working on Micro$oft Windoze, though)
6. <a href="http://singa.incubator.apache.org/en/docs/installation.html#compile-singa-with-use-opencl-on">Apache SINGA</a> (TODO)
6. FANN: Steffen Nissen's <a href="http://leenissen.dk/fann/wp/">FANN</a> (see <a href="https://github.com/martin-steinegger/fann-opencl">OpenCL backend with C API</a> and <a href="https://github.com/FutureLinkCorporation/fann2">non-OpenCL Python bindings<a/>)
7. Nengo OpenCL (nengo_ocl)
8. Encog: Jeff Heaton's <a href="https://github.com/encog">encog</a> (no Python. For example <a href="https://search.maven.org/#artifactdetails%7Corg.encog%7Cencog-core%7C3.3.0%7Cjar">Maven Central POM for encog 3.3.0</a>) (Jeff's reply to <a href="https://github.com/encog/encog-java-core/issues/245">my inquiry</a> was that he gave up support of OpenCL since Encog 3). My attempt is <a href="https://github.com/InonS/encog-opencl-test">here</a>.
9. Veles: Samsung's <a href="https://velesnet.ml/">Veles</a> (Ubuntu only)
10. Neural Networks: Ivan Vasilev's <a href="https://github.com/ivan-vasilev/neuralnetworks">Neural Networks</a> (Works!)
11. Spark: Max Grossman's <a href="https://github.com/agrippa/spark-swat">Spark SWAT</a>

[pycltorch_cmake]: https://github.com/torch/distro/tree/master/win-files "PyCLTorch requires CMake to install on Micro$oft Windoze."

DevOps
------
1. <tt>docker cp $AMP_APP_SDK_SRC_DIR\AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.
bz2 my-ubuntu:$AMP_APP_SDK_TGT_DIR</tt>

(not "<tt>docker-machine scp</tt>", which accesses the virtual machine itself, rather than the docker container we're running in it)

2. <tt>tar -xjvf AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2</tt>
3. <tt>./AMD-APP-SDK-v3.0.130.136-GA-linux64.sh</tt>
4. <tt>apt install ocl-icd-opencl-dev</tt>
 

Artificial Neural Networks video resources
==========================================
1. <a href="https://pythonprogramming.net/neural-networks-machine-learning-tutorial/">Sentdex</a>
2. <a href="https://www.youtube.com/playlist?list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu">Nando Freias</a>
3. <a href="https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9">Hinton Coursera</a>
4. <a href="https://www.youtube.com/playlist?list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So">Beginner</a>
5. <a href="https://www.youtube.com/playlist?list=PL29C61214F2146796">C#</a>

Keras video resources
---------------------
1. <a href="https://www.youtube.com/watch?v=FrkYu2zVUyM">Valerio Maggio (PyDataLondon '17)</a>
2. <a href="https://www.youtube.com/playlist?list=PLVBorYCcu-xX3Ppjb_sqBd_Xf6GqagQyl">playlist2</a>
3. <a href="https://www.youtube.com/playlist?list=PLFxrZqbLojdKuK7Lm6uamegEFGW2wki6P">playlist3</a>

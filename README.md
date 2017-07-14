Sentdex (PythonProgramming.net) Neural Networks Machine Learning Tutorial
=========================================================================

Artificial Neural Network implementation alternatives in OpenCL:
1. TF: Hugh Perkins's <a href="https://github.com/hughperkins/tf-coriander">TensorFlow on Coriander</a>. No windows support.
2. Torch: Hugh Perkins's <a href="https://github.com/hughperkins/pycltorch">pyCLTorch</a>. See [1].
3. Hugh Perkins's <a href="https://github.com/hughperkins/DeepCL">DeepCL</a>. (Works! still trying to get the Python bindings working on Micro$oft Windoze, though)
4. Caffe: <a href="https://github.com/BVLC/caffe/tree/opencl">OpenCL Caffe</a> (<a href="https://github.com/BVLC/caffe/issues/4929#issuecomment-267226532">requires</a> the <a href="https://github.com/viennacl/viennacl-dev">ViennaCL developer repo</a> to install on Windows. see also <a href="https://github.com/BVLC/caffe/tree/windows">Caffe on Micro$oft Windoze</a>)
5. Jeff Heaton's <a href="https://github.com/encog">encog</a> (no Python. For example <a href="https://search.maven.org/#artifactdetails%7Corg.encog%7Cencog-core%7C3.3.0%7Cjar">Maven Central POM for encog 3.3.0</a>) (Jeff's reply to my inquiry wa that he gave up support of OpenCL since Encog 3)
6. Samsung's <a href="https://velesnet.ml/">Veles</a> (Ubuntu only)
7. Steffen Nissen's <a href="http://leenissen.dk/fann/wp/">FANN</a> (see <a href="https://github.com/martin-steinegger/fann-opencl">OpenCL backend with C API</a> and <a href="https://github.com/FutureLinkCorporation/fann2">non-OpenCL Python bindings<a/>)
8. Ivan Vasilev's <a href="https://github.com/ivan-vasilev/neuralnetworks">Neural Networks</a> (Works!)

[1] <a href="https://github.com/torch/distro/tree/master/win-files">Requires</a> CMake to install on Micro$oft Windoze.

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
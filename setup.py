from distutils.core import setup

setup(
    name='Open_GPGPU_ANN',
    packages=['Open_GPGPU_ANN', 'sentdex_nn_ml_tutorial', 'testkeras'],
    url='https://github.com/InonS/Open_GPGPU_ANN',
    author='Inon Sharony',
    author_email='Inon.Sharony@gmail.com',
    description='Open Source GPGPU support for Artificial Neural Networks.'
                'The benchmarks used are based on the Sentdex (PythonProgramming.net) '
                'Neural Networks Machine Learning Tutorial.',
    keywords=['artificial-neural-networks', 'GPGPU', 'OpenCL'],
    requires=['keras', 'tensorflow', 'numpy', 'pandas', 'tqdm', 'matplotlib', 'nengo', 'nltk', 'pydot', 'scipy',
              'pyopencl', 'nengo_ocl', 'nengo_dl', 'plaidml', 'opencv-python'],
)

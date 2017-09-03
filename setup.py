from distutils.core import setup

setup(
    name='Open-GPGPU-ANN',
    packages=['Open-GPGPU-ANN', 'sentdex-nn-ml-tutorial', 'testkeras'],
    url='https://github.com/InonS/Open-GPGPU-ANN',
    author='Inon Sharony',
    author_email='Inon.Sharony@gmail.com',
    description='Open Source GPGPU support for Artificial Neural Networks.'
                'The benchmarks used are based on the Sentdex (PythonProgramming.net) '
                'Neural Networks Machine Learning Tutorial.',
    keywords=['artificial-neural-networks', 'GPGPU', 'OpenCL'],
    requires=['keras', 'tensorflow', 'numpy', 'pandas', 'tqdm', 'matplotlib', 'nengo', 'nengo_ocl', 'nengo_dl', 'nltk',
              'pydot'],
)

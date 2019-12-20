from distutils.core import setup
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup

def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_deps = []

if get_dist('tensorflow') is None and get_dist('tensorflow_gpu') is None:
    install_deps.append('tensorflow')

install_deps.extend(['numpy', 'opencv-contrib-python', 'h5py', 'matplotlib'])

setup(
  name = 'odddML',   
  packages = ['odddML', 'odddML.resnet'],
  version = 'v0.1.1-alpha',
  license='MIT',
  description = 'Easy ML for Devs, out of the box ML tools from ODDD Technologies',   
  author = 'Nick Koutroumpinis, ODDD Technologies',                 
  author_email = 'nick@odddtech.com',      
  url = 'https://github.com/ODDDTechnologies/BinianML',   
  download_url = 'https://github.com/ODDDTechnologies/BinianML/archive/v0.1.1-alpha.tar.gz',    
  keywords = ['Easy', 'Deep Learning', 'Machine Learning'],   
  install_requires=install_deps,
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.7',
  ]
)
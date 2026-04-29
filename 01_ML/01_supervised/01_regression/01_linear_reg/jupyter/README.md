# Linear Regression - Python

## Pre-requisites
- Anaconda/Miniconda or Python 3
- Jupyter

## Setup

- Virtual Env: `venv`
- Jupyter Setup
- Delete Old Jupyter Kernels
- Add Jupyter Kernel `venv`
- PIP install dependencies
- run Jupyter and select `venv` as kernel

```sh
# install virtual env
$ virtualenv venv --system-site-packages
$ source venv/bin/activate
...
# install jupyter
$ which jupyter
$ pip install jupyter --target=venv/bin
$ ls -la venv/bin
$ which jupyter
$ which venv/bin/jupyter.py
$ python venv/bin/jupyter.py --paths
# delete old jupyter kernels
$ jupyter-kernelspec uninstall venv
# add new jupyter kernel
$ python -m ipykernel install --user --name=venv
# install dependencies
$ which pip
$ pip list
$ pip install -r requirements.txt
$ pip list
# run jupyter and select venv as kernel
$ python venv/bin/jupyter.py notebook
...
$ deactivate
```

### Troubleshoot
- https://www.codingforentrepreneurs.com/blog/install-jupyter-notebooks-virtualenv
- https://ipython.readthedocs.io/en/stable/install/kernel_install.html


## Resources
- https://mlu-explain.github.io/linear-regression/
- https://youtu.be/sGy8yWq9O1g
- https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp/blob/main/notebooks/MLA-NLP-Lecture2-Linear-Regression.ipynb
# Install useful Jupyter Lab Extensions

# Check for latest Jupyter Lab install
conda install -y -c conda-forge jupyterlab

# Table of Contents
jupyter labextension install @jupyterlab/toc

# Flake8 integration
conda install -y flake8
jupyter labextension install jupyterlab-flake8

# Bokeh Integration
jupyter labextension install jupyterlab_bokeh

# Git GUI
jupyter labextension install @jupyterlab/git
#pip install jupyterlab-git
#jupyter serverextension enable --py jupyterlab_git


# Notebook Templates
pip install jupyterlab_templates
jupyter labextension install jupyterlab_templates
jupyter serverextension enable --py jupyterlab_templates

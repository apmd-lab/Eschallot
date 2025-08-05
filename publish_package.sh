## Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate

## Install Packaging Tools
pip install --upgrade pip
pip install setuptools wheel twine

## Generate Distribution Files
python setup.py sdist bdist_wheel

## Upload package
twine upload dist/*

## Exit Virtual Environment
deactivate

## Example Project Structure
## my_project/
## ������ setup.py
## ������ README.md
## ������ LICENSE
## ������ MANIFEST.in
## ������ eschallot/
##     ������ __init__.py
##     ������ mie/
##     ��   ������ __init__.py
##     ��   ������ module.py
##     ������ optimization/
##         ������ __init__.py
##         ������ module.py


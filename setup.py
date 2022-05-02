import setuptools

exec(open('lgbmbuilder/version.py').read())

setuptools.setup(
    name="lgbmbuilder", 
    version=__version__,
    author="DJ",
    author_email="willofdeepak@gmail.com",
    description="A package designed to facilitate building and evaluation of LightGBM models",
    packages=['lgbmbuilder'],
    python_requires = '>=3.7',
    install_requires = ['pandas>=0.24.0', 'numpy', 'optuna', 'matplotlib',
                        'sklearn', 'shap', 'pyarrow>=0.12.0', 'lightgbm']
)

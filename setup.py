from setuptools import setup, find_packages

setup(
    name='stock_prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.5',
        'numpy>=1.21.6',
        'scikit-learn>=0.24.2',
        'statsmodels>=0.13.5',
        'matplotlib>=3.5.3',
        'tensorflow>=2.11.0',
        'yfinance>=0.2.18',
        'pyyaml>=6.0',
        'seaborn>=0.12.2',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A stock price prediction project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/stock_prediction',
)
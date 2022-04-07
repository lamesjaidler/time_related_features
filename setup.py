# Authors: James Laidler <jlaidler@paypal.com>
# License: BSD 3 clause
import setuptools
setuptools.setup(
    name="time_related_features",
    version="0.0.0",
    author="James Laidler",
    packages=setuptools.find_packages(exclude=['examples']),
    install_requires=[
        'pandas==1.3.4', 'numpy==1.21.4', 'scikit-learn==1.0.1',
        'pytest==6.2.5', 'nbmake==1.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

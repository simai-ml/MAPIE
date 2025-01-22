.. -*- mode: rst -*-

|GitHubActions| |Codecov| |ReadTheDocs| |License| |PythonVersion| |PyPi| |Conda| |Release| |Commits| |DOI|

.. |GitHubActions| image:: https://github.com/scikit-learn-contrib/MAPIE/actions/workflows/test.yml/badge.svg
    :target: https://github.com/scikit-learn-contrib/MAPIE/actions

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1
   :target: https://codecov.io/gh/scikit-learn-contrib/MAPIE

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mapie/badge/?version=stable
    :target: https://mapie.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. |License| image:: https://img.shields.io/github/license/scikit-learn-contrib/MAPIE
    :target: https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/mapie
    :target: https://pypi.org/project/mapie/

.. |PyPi| image:: https://img.shields.io/pypi/v/mapie
    :target: https://pypi.org/project/mapie/

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/mapie
    :target: https://anaconda.org/conda-forge/mapie

.. |Release| image:: https://img.shields.io/github/v/release/scikit-learn-contrib/mapie
    :target: https://github.com/scikit-learn-contrib/MAPIE/releases

.. |Commits| image:: https://img.shields.io/github/commits-since/scikit-learn-contrib/mapie/latest/master
    :target: https://github.com/scikit-learn-contrib/MAPIE/commits/master

.. |DOI| image:: https://img.shields.io/badge/10.48550/arXiv.2207.12274-B31B1B.svg
    :target: https://arxiv.org/abs/2207.12274

.. image:: https://github.com/scikit-learn-contrib/MAPIE/raw/master/doc/images/mapie_logo_nobg_cut.png
    :width: 400
    :align: center



MAPIE - Model Agnostic Prediction Interval Estimator
====================================================

**MAPIE** is an open-source Python library for quantifying uncertainties and controlling the risks of machine learning models.
It is a scikit-learn-contrib project that allows you to:

- Easily **compute conformal prediction intervals** (or prediction sets) with controlled (or guaranteed) marginal coverage rate
  for regression [3,4,8], classification (binary and multi-class) [5-7] and time series [9].
- Easily **control risks** of more complex tasks such as multi-label classification,
  semantic segmentation in computer vision (probabilistic guarantees on recall, precision, ...) [10-12].
- Easily **wrap any model (scikit-learn, tensorflow, pytorch, ...) with, if needed, a scikit-learn-compatible wrapper**
  for the purposes just mentioned.

Here's a quick instantiation of MAPIE models for regression and classification problems related to uncertainty quantification
(more details in the Quickstart section):

.. code:: python

    # Uncertainty quantification for regression problem
    from mapie_v1.regression import SplitConformalRegressor
    mapie_regressor = SplitConformalRegressor(estimator=regressor)

.. code:: python

    # Uncertainty quantification for classification problem
    from mapie.classification import MapieClassifier
    mapie_classifier = MapieClassifier(estimator=classifier, method='score', cv=5)

Implemented methods in **MAPIE** respect three fundamental pillars:

- They are **model and use case agnostic**, 
- They possess **theoretical guarantees** under minimal assumptions on the data and the model,
- They are based on **peer-reviewed algorithms** and respect programming standards.

**MAPIE** relies notably on the field of *Conformal Prediction* and *Distribution-Free Inference*.


🔗 Requirements
===============

- **MAPIE** runs on Python 3.7+.
- **MAPIE** stands on the shoulders of giants. Its only internal dependencies are `scikit-learn <https://scikit-learn.org/stable/>`_ and `numpy=>1.21 <https://numpy.org/>`_.


🛠 Installation
===============

**MAPIE** can be installed in different ways:

.. code:: sh

    $ pip install mapie  # installation via `pip`
    $ conda install -c conda-forge mapie  # or via `conda`
    $ pip install git+https://github.com/scikit-learn-contrib/MAPIE  # or directly from the github repository


⚡ Quickstart
=============

Here we propose two basic uncertainty quantification problems for regression and classification tasks with scikit-learn.

As **MAPIE** is compatible with the standard scikit-learn API, you can see that with just these few lines of code:

- How easy it is **to wrap your favorite scikit-learn-compatible model** around your model.
- How easy it is **to follow the standard sequential** ``fit`` and ``predict`` process like any scikit-learn estimator.

.. code:: python

    # Uncertainty quantification for regression problem
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from mapie_v1.regression import SplitConformalRegressor

    X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)
    X_train_conformalize, X_test, y_train_conformalize, y_test = train_test_split(X, y, test_size=0.5)
    X_train, X_conformalize, y_train, y_conformalize = train_test_split(X_train_conformalize, y_train_conformalize, test_size=0.5)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    mapie_regressor = SplitConformalRegressor(
        regressor,
        confidence_level=[0.95, 0.68],
    )
    mapie_regressor.conformalize(X_conformalize, y_conformalize)

    y_pred, y_pred_intervals = mapie_regressor.predict_interval(X_test)

.. code:: python

    # Uncertainty quantification for classification problem
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    from mapie.classification import MapieClassifier


    X, y = make_blobs(n_samples=500, n_features=2, centers=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    classifier = LogisticRegression()

    mapie_classifier = MapieClassifier(estimator=classifier, method='score', cv=5)

    mapie_classifier = mapie_classifier.fit(X_train, y_train)
    y_pred, y_pis = mapie_classifier.predict(X_test, alpha=[0.05, 0.32])


📘 Documentation
================

The full documentation can be found `on this link <https://mapie.readthedocs.io/en/latest/>`_.


📝 Contributing
===============

You are welcome to propose and contribute new ideas.
We encourage you to `open an issue <https://github.com/scikit-learn-contrib/MAPIE/issues>`_ so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.
For more information on the contribution process, please go `here <CONTRIBUTING.rst>`_.


🤝  Affiliations
================

MAPIE has been developed through a collaboration between Capgemini, Quantmetry, Michelin, ENS Paris-Saclay,
and with the financial support from Région Ile de France and Confiance.ai.

|Capgemini| |Quantmetry| |Michelin| |ENS| |Confiance.ai| |IledeFrance|

.. |Capgemini| image:: https://www.capgemini.com/wp-content/themes/capgemini2020/assets/images/logo.svg
    :height: 35px
    :width: 140px
    :target: https://www.capgemini.com/

.. |Quantmetry| image:: https://www.quantmetry.com/wp-content/uploads/2020/08/08-Logo-quant-Texte-noir.svg
    :height: 35px
    :width: 140px
    :target: https://www.quantmetry.com/

.. |Michelin| image:: https://agngnconpm.cloudimg.io/v7/https://dgaddcosprod.blob.core.windows.net/corporate-production/attachments/cls05tqdd9e0o0tkdghwi9m7n-clooe1x0c3k3x0tlu4cxi6dpn-bibendum-salut.full.png
    :height: 50px
    :width: 45px
    :target: https://www.michelin.com/en/

.. |ENS| image:: https://www.ens.psl.eu/sites/default/files/logo_ens_psl_en_png.png
    :height: 35px
    :width: 140px
    :target: https://ens-paris-saclay.fr/en/

.. |Confiance.ai| image:: https://pbs.twimg.com/profile_images/1443838558549258264/EvWlv1Vq_400x400.jpg
    :height: 45px
    :width: 45px
    :target: https://www.confiance.ai/

.. |IledeFrance| image:: https://www.iledefrance.fr/sites/default/files/logo/2024-02/logoGagnerok.svg
    :height: 35px
    :width: 140px
    :target: https://www.iledefrance.fr/


🔍  References
==============

[1] Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer Nature, 2022.

[2] Angelopoulos, Anastasios N., and Stephen Bates. "Conformal prediction: A gentle introduction." Foundations and Trends® in Machine Learning 16.4 (2023): 494-591.

[3] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. "Predictive inference with the jackknife+." Ann. Statist., 49(1):486–507, (2021).

[4] Kim, Byol, Chen Xu, and Rina Barber. "Predictive inference is free with the jackknife+-after-bootstrap." Advances in Neural Information Processing Systems 33 (2020): 4138-4149.

[5] Sadinle, Mauricio, Jing Lei, and Larry Wasserman. "Least ambiguous set-valued classifiers with bounded error levels." Journal of the American Statistical Association 114.525 (2019): 223-234.

[6] Romano, Yaniv, Matteo Sesia, and Emmanuel Candes. "Classification with valid and adaptive coverage." Advances in Neural Information Processing Systems 33 (2020): 3581-3591.

[7] Angelopoulos, Anastasios, et al. "Uncertainty sets for image classifiers using conformal prediction." International Conference on Learning Representations (2021).

[8] Romano, Yaniv, Evan Patterson, and Emmanuel Candes. "Conformalized quantile regression." Advances in neural information processing systems 32 (2019).

[9] Xu, Chen, and Yao Xie. "Conformal prediction interval for dynamic time-series." International Conference on Machine Learning. PMLR, (2021).

[10] Bates, Stephen, et al. "Distribution-free, risk-controlling prediction sets." Journal of the ACM (JACM) 68.6 (2021): 1-34.

[11] Angelopoulos, Anastasios N., Stephen, Bates, Adam, Fisch, Lihua, Lei, and Tal, Schuster. "Conformal Risk Control." (2022).

[12] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al. "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022).


📝 License
==========

MAPIE is free and open-source software licensed under the `license <https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE>`_.


📚 Citation
===========

If you use MAPIE in your research, please cite using:

.. code:: latex

    @inproceedings{Cordier_Flexible_and_Systematic_2023,
    author = {Cordier, Thibault and Blot, Vincent and Lacombe, Louis and Morzadec, Thomas and Capitaine, Arnaud and Brunel, Nicolas},
    booktitle = {Conformal and Probabilistic Prediction with Applications},
    title = {{Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library}},
    year = {2023}
    }

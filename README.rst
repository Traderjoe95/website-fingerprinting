======================
Website Fingerprinting
======================

This repository contains an extensible evaluation suite for
Website Fingerprinting attacks, defenses,and feature sets.
As such, the project is heavily influenced by Giovanni Cherubins
2017 paper `"Bayes, not Naïve: Security Bounds on Website
Fingerprinting Defenses"`__, and the accompanying code, which is
available in `this repository`_.

__ https://www.degruyter.com/downloadpdf/j/popets.2017.2017.issue-4/popets-2017-0046/popets-2017-0046.pdf
.. _this repository: https://github.com/gchers/wfes

However, this project aims at replacing the WEKA-based implementations of older attacks by something more modern (and
also more pythonic), such as |scikit-learn|_ for ML-related tasks and
|pandas|_ for data management and manipulation.

.. |scikit-learn| replace:: ``scikit-learn``
.. _scikit-learn: https://scikit-learn.org/stable/

.. |pandas| replace:: ``pandas``
.. _pandas: https://pandas.pydata.org/
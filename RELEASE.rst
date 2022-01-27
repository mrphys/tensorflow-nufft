Release 0.6.0
=============

Major Features and Improvements
-------------------------------

* Added definition for the gradient with respect to the ``points`` input. It is
  now possible to obtain this gradient whenever the ``nufft`` operation is
  recorded by a gradient tape and the ``points`` input is being watched. This
  enables applications such as optimization of sampling patterns.

Bug Fixes and Other Changes
---------------------------

* Fixed a bug with broadcasting that would result in an error if both ``source``
  and ``points`` had non-scalar batch shapes of unequal rank.

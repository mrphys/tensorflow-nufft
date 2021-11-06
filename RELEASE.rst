Release 0.3.2
=============

Bug Fixes and Other Changes
---------------------------

* Fixed a bug in GPU kernel that would ignore internal CUFFT errors and simply
  return an incorrect result. An error will be raised now. 

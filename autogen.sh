#!/bin/sh
# you can either set the environment variables AUTOCONF, AUTOHEADER, AUTOMAKE,
# ACLOCAL, AUTOPOINT and/or LIBTOOLIZE to the right versions, or leave them
# unset and get the defaults

mkdir -p common
touch common/Makefile.am

autoreconf --verbose --force --install --make || {
 echo 'autogen.sh failed';
 exit 1;
}

./configure || {
 echo 'configure failed';
 exit 1;
}

rm -r aclocal.m4 autom4te.cache config.guess config.status config.sub configure install-sh libtool ltmain.sh missing
rm common/config.h.in*  common/Makefile*  common/stamp-h1


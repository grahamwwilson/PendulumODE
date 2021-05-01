#!/bin/sh

VERSION=$1
echo 'Using VERSION '${VERSION}

mv SimDataFile.dat SimDataFile-${VERSION}.dat
gzip SimDataFile-${VERSION}.dat
cp SimDataFile-${VERSION}.dat.gz ~/PendDataSimComp

mv Oscillations.dat Oscillations-${VERSION}.dat
gzip Oscillations-${VERSION}.dat

exit

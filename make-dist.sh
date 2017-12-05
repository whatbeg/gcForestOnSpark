#!/usr/bin/env bash

#
# Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
#

set -e

# Check java
if type -p java>/dev/null; then
    _java=java
else
    echo "Java is not installed"
    exit 1
fi

MVN_OPTS_LIST="-Xmx2g -XX:ReservedCodeCacheSize=512m"

if [[ "$_java" ]]; then
    version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    if [[ "$version" < "1.7" ]]; then
        echo Require a java version not lower than 1.7
        exit 1
    fi
    # For jdk7
    if [[ "$version" < "1.8" ]]; then
        MVN_OPTS_LIST="$MVN_OPTS_LIST -XX:MaxPermSize=1G"
    fi
fi

export MAVEN_OPTS=${MAVEN_OPTS:-"$MVN_OPTS_LIST"}

# Check if mvn installed
MVN_INSTALL=$(which mvn 2>/dev/null | grep mvn | wc -l)
if [ $MVN_INSTALL -eq 0 ]; then
  echo "MVN is not installed. Exit"
  exit 1
fi

mvn clean package -DskipTests $*

BASEDIR=$(dirname "$0")
DIST_DIR=$BASEDIR/dist

if [ ! -d "$DIST_DIR" ]
then
  mkdir $DIST_DIR
else
  rm -r $DIST_DIR
  mkdir $DIST_DIR
fi

#cp -r $BASEDIR/target/gcforest-*-with-dependencies.jar ./dist/
cp $BASEDIR/target/gcforest-*.jar ./dist

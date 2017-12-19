#!/usr/bin/env bash

# run local jar to validate the generated jar.

#spark-submit --master local[*] \
# --class examples.RandomForest.DecisionTreeExample \
# dist/gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar

spark-submit --master local[*] \
 --class examples.Yggdrasil.YggdrasilExample \
 dist/gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar
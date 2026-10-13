#!/bin/bash
# run sf_predict for the settings whose tag matches the regex $1, sequentially; $2 labels the run
cd "$(dirname "$0")"
awk -v pat="$1" '$1 ~ pat' sf_settings.txt | while read tag d a L bbf ns; do
  if [ ! -f sfp_$tag.json ]; then
    echo "start $tag $(date)" >> sfp_run_$2.log
    python3 sf_predict.py $d $a $L $bbf $ns sfp_$tag.json > sfp_$tag.log 2>&1
  fi
done
echo "ALLDONE $(date)" >> sfp_run_$2.log

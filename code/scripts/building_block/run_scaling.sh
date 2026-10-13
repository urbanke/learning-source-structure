#!/bin/bash
# scaling-law grid: building block (ext_grid.py) if missing, then the sampling-free crossing sweep (sf_cross.py).
# $1 = worker id (0 or 1): takes every second line of scaling_settings.txt; $2 = number of workers
cd "$(dirname "$0")"
W=${2:-2}; i=0
grep -v '^#' scaling_settings.txt | while read tag d a L Lmax kmax; do
  if [ $((i % W)) -eq $1 ]; then
    bbtag="d${d}_a$(python3 -c "print(f'{$a:g}')")_L${L}"
    if [ ! -f bb2_$bbtag.json ]; then
      echo "grid $tag $(date)" >> scaling_run_$1.log
      python3 ext_grid.py $d $a $L $Lmax $kmax > ext_grid_$bbtag.log 2>&1
    fi
    if [ -f bb2_$bbtag.json ] && [ ! -f sfx_$tag.json ]; then
      echo "sweep $tag $(date)" >> scaling_run_$1.log
      python3 sf_cross.py $d $a $L bb2_$bbtag.json sfx_$tag.json > sfx_$tag.log 2>&1
    fi
  fi
  i=$((i+1))
done
echo "ALLDONE $(date)" >> scaling_run_$1.log

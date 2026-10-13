#!/bin/bash
# extension settings (Appendix E runs): d alpha Ls Lmax kmax; two workers
cd "$(dirname "$0")"
w1() { for a in "512 1.0 6 30 1440" "512 1.5 6 30 1120" "512 1.5 10 30 256" "256 1.0 6 26 320" "256 1.5 6 26 640" "256 1.5 10 26 64"; do
  python3 ext_grid.py $a > ext_grid_$(echo $a | awk '{print "d"$1"_a"$2"_L"$3}').log 2>&1; done; }
w2() { for a in "512 2.0 3 30 2500" "256 2.0 3 26 2000" "512 2.0 6 30 800" "256 2.0 6 26 512" "512 0.5 6 30 1120" "256 0.5 6 26 1120"; do
  python3 ext_grid.py $a > ext_grid_$(echo $a | awk '{print "d"$1"_a"$2"_L"$3}').log 2>&1; done; }
w1 & w2 & wait; echo ALLDONE

#!/bin/bash
cd "$(dirname "$0")"
sim() { tag=$1; shift; [ -f sim_X2_$tag.json.done ] || { nice -n 10 python3 chain_sim.py "$@" sim_X2_$tag.json > sim_X2_$tag.log 2>&1 && touch sim_X2_$tag.json.done; }; }
sim L3_a1.25_d256 256 1.25 3 176 60 352,704,1056,1232,1408,1584,1760,2112,2816
sim L2_a1.25_d256 256 1.25 2 176 59 1001,2002,3003,3503,4003,4504,5004,6005,8007
echo ALLDONE

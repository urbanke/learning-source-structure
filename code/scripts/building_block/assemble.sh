#!/bin/bash
# final prediction files tab_M_<tag>.json: archived 18 settings -> tab_B (unchanged); extension settings -> tab_B + tab_C merged
cd "$(dirname "$0")"
for t in L2_a0.75_d128 L2_a1_d128 L2_a1.25_d128 L3_a0.75_d128 L3_a1_d128 L3_a1.25_d128 L2_a0.75_d256 L2_a1_d256 L2_a1.25_d256 L3_a0.75_d256 L3_a1_d256 L3_a1.25_d256 L2_a0.75_d512 L2_a1_d512 L2_a1.25_d512 L3_a0.75_d512 L3_a1_d512 L3_a1.25_d512; do cp tab_B_$t.json tab_M_$t.json; done
for t in L6_a0.5_d256 L6_a1_d256 L6_a1.5_d256 L6_a2_d256 L3_a2_d256 L10_a1.5_d256 L6_a0.5_d512 L6_a1_d512 L6_a1.5_d512 L6_a2_d512 L3_a2_d512 L10_a1.5_d512; do
  if [ -f tab_C_$t.json ] && grep -q ALLDONE run_ext_tables2.log 2>/dev/null || [ -f tab_C_$t.json -a "$(python3 -c "import json;t=json.load(open('tab_C_$t.json'));print(len(t['predC'])==t['NT_occ'])")" = "True" ]; then
    python3 merge_tabs.py tab_M_$t.json tab_B_$t.json tab_C_$t.json
  else cp tab_B_$t.json tab_M_$t.json; echo "$t: tab_B only"; fi
done
TAB_PREFIX=tab_M python3 compare_all.py > compare_all.log 2>&1; grep "crossing:\|===" compare_all.log | paste - - | awk '{match($0, /paths R=[0-9]+/); r=substr($0, RSTART+8, RLENGTH-8); match($0, /tables [0-9\/]+/); tb=substr($0, RSTART+7, RLENGTH-7); match($0, /simulation.*/); print $2, "R=" r, "tables", tb, substr($0, RSTART)}'
python3 stats3.py

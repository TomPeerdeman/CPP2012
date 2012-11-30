#!/bin/bash
rm -rf /tmp/hadoop1207*

for i in {65..85}
do
   ssh node0$i "rm -rf /tmp/hadoop1207*"
done
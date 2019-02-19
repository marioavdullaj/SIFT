import numpy as np
import os
import io
import sys
import matplotlib.pyplot as plt
import yaml
import shutil

filename = "results.yaml"

# YAML COMPATIBILITY WITH FILESTORAGE OPENCV FORMAT
with open(filename) as fin:
    lines = fin.readlines()
lines[0] = lines[0].replace('%YAML:1.0\n', '%YAML 1.0\n')

with open(filename, 'w') as fout:
    for line in lines:
        fout.write(line)
fin.close()
fout.close()

with open(filename) as f:
    dataMap = yaml.safe_load(f)

duration_linear = dataMap['duration_linear']

branching = dataMap['branching'].replace('"','').split(",")
leaf_size = dataMap['leaf_size'].replace('"','').split(",")
trees = dataMap['trees'].replace('"','').split(",")

print("Branching values: ")
print(branching)
print("Leaf sizes values: ");
print(leaf_size);
print("trees values: ");
print(trees)

branching_selected = branching[0]
leaf_size_selected = leaf_size[0]
trees_selected = trees[0]

index = str(branching_selected)+'-'+str(leaf_size_selected)+'-'+str(trees_selected)
L_max_selected = dataMap['L_max'+index].replace('"','').split(",")

prec = []
speedup = []

for i in range(len(L_max_selected)):
    ind = index+'-'+L_max_selected[i]
    p = dataMap['precision'+ind]
    dur = dataMap['duration'+ind]
    prec.append(p)
    speedup.append(1.0*duration_linear/dur)

plt.figure(6)
plt.plot(prec, speedup, marker='o', markersize=5)
plt.yscale('log')
plt.xlabel('precision')
plt.ylabel('speedup')
plt.show()

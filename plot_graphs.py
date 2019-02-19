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

# PLOTTING THE TREE-DEPENDENCY GRAPH
print("\n\nTrees graph - select parameters")
print("Select branching value: ")
branching_selected = input()
print("Select leaf size value: ")
leaf_size_selected = input()

plt.figure(1)
for t in trees:
    prec = []
    speedup = []
    index = str(branching_selected)+'-'+str(leaf_size_selected)+'-'+str(t)
    L_max_selected = dataMap['L_max'+index].replace('"','').split(",")
    for i in range(len(L_max_selected)):
        ind = index+'-'+L_max_selected[i]
        p = dataMap['precision'+ind]
        dur = dataMap['duration'+ind]
        prec.append(p)
        speedup.append(1.0*duration_linear/dur)
    plt.plot(prec, speedup, marker='o', markersize=5, label = 'tree = '+str(t))

plt.legend()
plt.xlim(left=0.4)
plt.xlim(right=1)
plt.ylim(bottom=1)
plt.yscale('log')
plt.xlabel('precision')
plt.ylabel('speedup')
plt.show()


# PLOTTING THE BRANCH-DEPENDENCY GRAPH
print("\n\nBranch graph - select parameters")
print("Select tree value: ")
tree_selected = input()
print("Select leaf size value: ")
leaf_size_selected = input()

plt.figure(2)
for b in branching:
    prec = []
    speedup = []
    index = str(b)+'-'+str(leaf_size_selected)+'-'+str(tree_selected)
    L_max_selected = dataMap['L_max'+index].replace('"','').split(",")
    for i in range(len(L_max_selected)):
        ind = index+'-'+L_max_selected[i]
        p = dataMap['precision'+ind]
        dur = dataMap['duration'+ind]
        prec.append(p)
        speedup.append(1.0*duration_linear/dur)
    plt.plot(prec, speedup, marker='o', markersize=5, label = 'branching = '+str(b))

plt.legend()
plt.xlim(left=0.4)
plt.xlim(right=1)
plt.ylim(bottom=1)
plt.yscale('log')
plt.xlabel('precision')
plt.ylabel('speedup')
plt.show()

# PLOTTING THE LEAF SIZE-DEPENDENCY GRAPH
print("\n\nLeaf-size graph - select parameters")
print("Select branch value: ")
branching_selected = input()
print("Select tree value: ")
tree_selected = input()

plt.figure(3)
for l in leaf_size:
    prec = []
    speedup = []
    index = str(branching_selected)+'-'+str(l)+'-'+str(tree_selected)
    L_max_selected = dataMap['L_max'+index].replace('"','').split(",")
    for i in range(len(L_max_selected)):
        ind = index+'-'+L_max_selected[i]
        p = dataMap['precision'+ind]
        dur = dataMap['duration'+ind]
        prec.append(p)
        speedup.append(1.0*duration_linear/dur)
    plt.plot(prec, speedup, marker='o', markersize=5, label = 'leaf size = '+str(l))

plt.legend()
plt.xlim(left=0.4)
plt.xlim(right=1)
plt.ylim(bottom=1)
plt.yscale('log')
plt.xlabel('precision')
plt.ylabel('speedup')
plt.show()

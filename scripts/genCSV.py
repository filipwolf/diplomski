import collections

f = open("/home/filip/Desktop/yeast_data/graphs/graph2.txt", "r")
f2 = open("/home/filip/Desktop/yeast_data/graphs/graph_nodes.csv", "w")
f3 = open("/home/filip/Desktop/yeast_data/graphs/graph_edges.csv", "w")

nodes_list = {}
edge_list = []

f2.write('node_name,node_class\n')
f3.write('node1,node2\n')

for line in f.readlines():
    tab_split = line.split('\t')
    semicol_split1 = tab_split[0].rstrip().split(';')
    semicol_split2 = tab_split[1].rstrip().split(';')

    if semicol_split1[1] not in nodes_list.keys():
        nodes_list[semicol_split1[1]] = 1 if semicol_split1[0] == 'mut' else 0
    if semicol_split2[1] not in nodes_list.keys():
        nodes_list[semicol_split2[1]] = 1 if semicol_split2[0] == 'mut' else 0

    edge_list.append([semicol_split1[1], semicol_split2[1]])

od = collections.OrderedDict(sorted(nodes_list.items(), key=lambda x: int(x[0])))
edge_list = sorted(edge_list, key=lambda x: int(x[0]))

nodes_list = {}
trans_list = {}

for i, (key, value) in enumerate(od.items()):
    nodes_list[i] = value
    trans_list[key] = i

for edge in edge_list:
    f3.write(str(trans_list[edge[0]]) + ',' + str(trans_list[edge[1]]) + '\n')

for key, value in nodes_list.items():
    f2.write(str(key) + ',' + str(value) + '\n')

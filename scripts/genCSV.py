import collections

for i in range(0, 101):
    print(i)

    path = '/media/filip/DA2A5AE02A5AB8E9/diplomski/yeast_data/graphs/'
    path_graphs = path + 'modified_graphs_2/graph' + str(i) + '_modified.txt'
    path_node_features = path + 'node_features/node_features' + str(i) + '.csv'
    path_edge_features = path + 'edge_features/edge_features' + str(i) + '.csv'
    path_edge_lengths = path + 'edge_lengths/edge_lengths' + str(i) + '.txt'

    f = open(path_graphs, "r")
    f2 = open(path_node_features, "w")
    f3 = open(path_edge_features, "w")
    f4 = open(path_edge_lengths, "r")

    nodes_list = {}
    edge_list = []

    f2.write('node_name,node_class\n')
    f3.write('node1,node2,node_length\n')

    edge_lengths = f4.readlines()

    for i, line in enumerate(f.readlines()):
        tab_split = line.split('\t')
        semicol_split1 = tab_split[0].rstrip().split(';')
        semicol_split2 = tab_split[1].rstrip().split(';')

        if semicol_split1[1] not in nodes_list.keys():
            nodes_list[semicol_split1[1]] = 1 if semicol_split1[0] == 'mut' else 0
        if semicol_split2[1] not in nodes_list.keys():
            nodes_list[semicol_split2[1]] = 1 if semicol_split2[0] == 'mut' else 0

        edge_list.append([semicol_split1[1], semicol_split2[1], edge_lengths[i].rstrip()])

    od = collections.OrderedDict(sorted(nodes_list.items(), key=lambda x: int(x[0])))

    nodes_list = {}
    trans_list = {}

    for i, (key, value) in enumerate(od.items()):
        nodes_list[i] = value
        trans_list[key] = i

    for edge in edge_list:
        f3.write(str(trans_list[edge[0]]) + ',' + str(trans_list[edge[1]]) + ',' + str(edge[2]) + '\n')

    for key, value in nodes_list.items():
        f2.write(str(key) + ',' + str(value) + '\n')

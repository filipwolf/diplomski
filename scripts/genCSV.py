import collections
import math
from statistics import mean

import numpy as np

from path_utils import PATH

for i in range(0, 101):
    print(i)

    path = PATH
    path_graphs = path + "modified_graphs_2/graph" + str(i) + "_modified.txt"
    path_node_features = path + "node_features/node_features" + str(i) + ".csv"
    path_edge_features = path + "edge_features/edge_features" + str(i) + ".csv"
    # path_edge_lengths = path + "edge_lengths/edge_lengths" + str(i) + ".txt"
    path_edge_overlaps = path + "edge_overlaps/edge_overlaps" + str(i) + ".txt"

    # path2 = "/media/filip/DA2A5AE02A5AB8E92/diplomski/yeast_data/graphs/larger_dataset/"
    # path_csv = path2 + "graph" + str(i) + "/graph.csv"
    # path_gfa = path2 + "graph" + str(i) + "/graph.gfa"

    f = open(path_graphs, "r")
    f2 = open(path_node_features, "w")
    f3 = open(path_edge_features, "w")
    f4 = open(path_edge_overlaps, "r")
    # f4 = open(path_edge_lengths, "r")
    # f5 = open(path_csv, "r")
    # f6 = open(path_gfa, "r")

    # csv = f5.readlines()
    # gfa = f6.readlines()

    # csv_indexes = []
    #
    # overlap_dict = {}
    #
    # max_index = 0
    #
    # max_value = 0
    # min_value = float("inf")
    #
    # for line in csv:
    #     idx = int(line.split()[0])
    #
    #     if idx < max_index:
    #         max_index += 1
    #         value = math.log(float(line.split()[-1]))
    #         max_value = max(value, max_value)
    #         min_value = min(value, min_value)
    #     else:
    #         max_index = max(max_index, idx)
    #
    # max_value -= min_value
    # max_index = 0
    #
    # for line in csv:
    #     idx = int(line.split()[0])
    #
    #     if idx < max_index:
    #         max_index += 1
    #         idx1 = int(line.split()[1].rstrip("]").lstrip("["))
    #         idx2 = int(line.split()[4].rstrip("]").lstrip("["))
    #         if (idx1, idx2) not in overlap_dict.keys() and (idx2, idx1) not in overlap_dict.keys():
    #             overlap_dict[(idx1, idx2)] = (math.log(float(line.split()[-1])) - min_value) / max_value
    #     else:
    #         csv_indexes.append(idx)
    #         max_index = max(max_index, idx)
    #
    # mapping_dict = {}
    #
    # for idx in range(len(csv_indexes)):
    #     gfa_sign = gfa[int(idx)].split()[0]
    #     if gfa_sign != "S":
    #         print(gfa_sign)
    #         continue
    #     mapping_dict[int(gfa[int(idx)].split()[1].split(",")[0].split("=")[1])] = idx

    nodes_list = {}
    edge_list = []

    f2.write("node_name,node_class\n")
    f3.write("node1,node2,edge_overlap,edge_class\n")

    # edge_lengths = f4.readlines()

    # edge_lengths = [int(x) for x in edge_lengths]
    # edge_lengths_avg = int(mean(edge_lengths))

    edge_overlaps = f4.readlines()

    edge_overlaps = [int(x) for x in edge_overlaps]

    for j, line in enumerate(f.readlines()):
        edge_class = 0
        tab_split = line.split("\t")
        semicol_split1 = tab_split[0].rstrip().split(";")
        semicol_split2 = tab_split[1].rstrip().split(";")

        if semicol_split1[1] not in nodes_list.keys():
            nodes_list[semicol_split1[1]] = 1 if semicol_split1[0] == "mut" else 0
        if semicol_split2[1] not in nodes_list.keys():
            nodes_list[semicol_split2[1]] = 1 if semicol_split2[0] == "mut" else 0
        if semicol_split1[0] == semicol_split2[0]:
            edge_class = 1

        # gfa_edge1 = mapping_dict[int(semicol_split1[1])]
        # gfa_edge2 = mapping_dict[int(semicol_split2[1])]

        # if (gfa_edge1, gfa_edge2) in overlap_dict.keys():
        #     edge_overlap = overlap_dict[(gfa_edge1, gfa_edge2)]
        # elif (gfa_edge2, gfa_edge1) in overlap_dict.keys():
        #     edge_overlap = overlap_dict[(gfa_edge2, gfa_edge1)]
        # else:
        #     edge_overlap = 0

        # if len(edge_lengths) <= j:
        #     edge_list.append([semicol_split1[1], semicol_split2[1], edge_lengths_avg, edge_overlap, edge_class])
        # else:
        edge_list.append([semicol_split1[1], semicol_split2[1], edge_overlaps[j], edge_class])

    od = collections.OrderedDict(sorted(nodes_list.items(), key=lambda x: int(x[0])))

    nodes_list = {}
    trans_list = {}

    for j, (key, value) in enumerate(od.items()):
        nodes_list[j] = value
        trans_list[key] = j

    for edge in edge_list:
        f3.write(
            str(trans_list[edge[0]])
            + ","
            + str(trans_list[edge[1]])
            + ","
            + str(edge[2])
            + ","
            + str(edge[3])
            + "\n"
        )

    for key, value in nodes_list.items():
        f2.write(str(key) + "," + str(value) + "\n")

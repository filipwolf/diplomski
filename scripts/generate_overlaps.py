for i in range(0, 101):

    path = '/media/filip/DA2A5AE02A5AB8E92/diplomski/yeast_data/graphs/simulated_graphs_with_csvs/'
    path_csv = path + 'graph' + str(i) + '/graph.csv'
    path_gfa = path + 'graph' + str(i) + '/graph.gfa'
    path_w = '/media/filip/DA2A5AE02A5AB8E92/diplomski/yeast_data/graphs/edge_overlaps/edge_overlaps' + str(i) + '.txt'

    f1 = open(path_csv, "r")
    f2 = open(path_gfa, "r")
    f3 = open(path_w, "w")

    csv = f1.readlines()
    gfa = f2.readlines()

    csv_indexes = []

    max_index = 0

    for line in csv:
        idx = int(line.split()[0])

        if idx < max_index:
            break

        csv_indexes.append(idx)

        max_index = max(max_index, idx)

    mapping_dict = {}

    for idx in range(len(csv_indexes)):
        gfa_sign = gfa[int(idx)].split()[0]
        if gfa_sign != 'S':
            continue
        mapping_dict[idx] = gfa[int(idx)].split()[1].split(',')[0].split('=')[1]




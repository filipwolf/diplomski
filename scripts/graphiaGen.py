from path_utils import PATH


def gen(path_r, path_w):

    f = open(path_r, "r")
    f2 = open(path_w, "w")

    for line in f.readlines():
        elems = line.rstrip().split(',')
        if elems[1] == 'mutated':
            writeLine = 'mut;' + elems[0] + '\t'
            if len(elems) == 4:
                writeLine += 'mut;'
            else:
                writeLine += 'not;'
            writeLine += elems[2]
        else:
            writeLine = 'not;' + elems[0] + '\t'
            if len(elems) == 3:
                writeLine += 'mut;'
            else:
                writeLine += 'not;'
            writeLine += elems[1]
        writeLine += '\n'
        f2.write(writeLine)


if __name__ == "__main__":

    path = PATH

    # training

    for i in range(0, 101):
        print(i)

        path_r = path + 'modified_graphs/graph' + str(i) + '.txt'
        path_w = path + 'modified_graphs_2/graph' + str(i) + '_modified.txt'

        gen(path_r, path_w)

    # validation

    path += 'chr2/'

    path_r = path + 'graph.txt'
    path_w = path + 'graph_modified.txt'

    gen(path_r, path_w)

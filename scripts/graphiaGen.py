from path_utils import PATH


def gen(path_r, path_w, path_w2):
    """Generate file suitable as input to Graphia.

    Parameters
    ----------
    path_r : str
        Path to modified_graphs folder for reading.
    path_w : str
        Path to modified_graphs_2 folder for writing.
    path_w2 : str
        Path to modified_graphs_separated folder for writing.
    """

    # open folders for reading and writing
    f = open(path_r, "r")
    f2 = open(path_w, "w")
    f3 = open(path_w2, "w")

    # iterate over data
    for line in f.readlines():
        elems = line.rstrip().split(',')
        if elems[1] == 'mutated':
            writeLine = 'mut;' + elems[0] + '\t'
            if len(elems) == 4:
                writeLine += 'mut;'
                writeLine2 = 'mut;' + elems[0] + '\t'
                writeLine2 += 'mut;'
                writeLine2 += elems[2]
                writeLine2 += '\n'
                f3.write(writeLine2)
            else:
                writeLine += 'not;'
            writeLine += elems[2]
        else:
            writeLine = 'not;' + elems[0] + '\t'
            if len(elems) == 3:
                writeLine += 'mut;'
            else:
                writeLine += 'not;'
                writeLine2 = 'not;' + elems[0] + '\t'
                writeLine2 += 'not;'
                writeLine2 += elems[1]
                writeLine2 += '\n'
                f3.write(writeLine2)
            writeLine += elems[1]
        writeLine += '\n'
        f2.write(writeLine)


if __name__ == "__main__":

    path = PATH

    # loop for generating training data
    for i in range(0, 101):
        print(i)

        path_r = path + 'modified_graphs/graph' + str(i) + '.txt'
        path_w = path + 'modified_graphs_2/graph' + str(i) + '_modified.txt'
        path_w2 = path + 'modified_graphs_separated/graph' + str(i) + '_modified.txt'

        gen(path_r, path_w, path_w2)

    # validation data
    path += 'chr2/'

    path_r = path + 'graph.txt'
    path_w = path + 'graph_modified.txt'
    path_w2 = path + 'graph_separated.txt'

    gen(path_r, path_w, path_w2)

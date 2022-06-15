from path_utils import PATH


def gen(prediction):
    """Generate file suitable as input to Graphia and separate haplotypes.

    Parameters
    ----------
    prediction: list
        List of predictions for every class and every edge in a graph.
    """

    path = PATH

    path_r = path + 'modified_graphs_2/graph100_modified.txt'
    path_w = 'val_graph.txt'

    f = open(path_r, "r")
    f2 = open(path_w, "w")

    for i, line in enumerate(f.readlines()):
        elems = line.rstrip().split(',')
        if len(elems) == 4 or len(elems) == 2:
            if prediction[i] == 0:
                f2.write(line)
        else:
            if prediction[i] == 1:
                f2.write(line)

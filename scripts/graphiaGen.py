f = open("/home/filip/Desktop/yeast_data/graphs/graph.txt", "r")
f2 = open("/home/filip/Desktop/yeast_data/graphs/graph2.txt", "w")

for line in f.readlines():
    writeLine = ''
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
    print(writeLine)
    f2.write(writeLine)

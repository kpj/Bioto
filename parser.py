import networkx as nx


def generate_tf_gene_regulation(file):
    # parse file
    data = []
    with open(file, 'r') as fd:
        content = fd.read()

        for line in content.split('\n'):
            if len(line) == 0 or line[0] == '#':
                continue

            parts = line.split()
            entry = {
                'transcription_factor': parts[0],
                'gene': parts[1],
                'effect': parts[3]
            }
            data.append(entry)

    # create graph
    graph = nx.DiGraph()
    for e in data:
        source = e['transcription_factor'].lower()
        sink = e['gene'].lower()

        graph.add_nodes_from([source, sink])
        graph.add_edge(source, sink)

    return graph

def parse_concentration(names, file, conc_range=[0]):
    """ Returns concentrations (at specified point in time) of given entries and list of unprocessable entries
    """
    def parse_line(line):
        """ Reads line and returns concentrations
        """
        conc = line.split()[3]
        return float(conc)
    with open(file, 'r') as fd:
        content = fd.read()

    concs = []
    lines = content.split('\n')
    stop = False
    header = None
    no_match = []
    for n in names:
        for l in lines:
            parts = l.split()
            if len(l) == 0 or l[0] in '#^!':
                continue
            if not header:
                # assume first non-comment row to be header
                header = parts
                continue

            gene = parts[1]
            conc = []
            for i in conc_range:
                try:
                    conc.append(float(parts[2+i]))
                except ValueError:
                    conc.append(None)
            if len(conc) == 1: conc = conc[0] # small fix for compatibility and stuff

            if n.lower() == gene.lower():
                concs.append(conc)
                break
        else:
            #print("Nothing found for", n)
            concs.append(0 if len(conc_range) == 1 else [0]*len(conc_range))
            no_match.append(n)

    return concs, no_match


if __name__ == '__main__':
    #generate_tf_gene_regulation('../data/network_tf_gene.txt')
    parse_concentration(['cyoB'], '../data/GDS4815_full.soft')

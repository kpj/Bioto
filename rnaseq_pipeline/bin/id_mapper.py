import re, sys


if len(sys.argv) != 2:
    print('Usage: %s <count file>' % sys.argv[0])
    sys.exit(1)

# get id-name mapping
id_map = {}
with open('data/ecoli_genome_annotations.gtf', 'r') as fd:
    for line in fd.read().split('\n'):
        id_match = re.search(r'gene_id "(.*?)"', line)
        name_match = re.search(r'gene_name "(.*?)"', line)

        if id_match and name_match:
             i = id_match.group(1)
             n = name_match.group(1)

             id_map[i] = n

print('Found mapping for %d genes' % len(id_map), file=sys.stderr)

# convert count file
with open(sys.argv[1], 'r') as fd:
    for line in fd.read().split('\n'):
        if line.startswith('__') or len(line) == 0: continue
        gid, gcount = line.split()

        if gid in id_map:
            print(id_map[gid], gcount)

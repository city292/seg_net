flist = []
with open('/Users/city/projects/seg_net/flielist.txt', 'r') as f:
    for line in f.readlines():
        # print(line.replace('  ', ' '))
        line = line.strip()
        vs = line.split(' ')
        # print(vs)
        flist.append((vs[0], vs[1]))
print(flist)
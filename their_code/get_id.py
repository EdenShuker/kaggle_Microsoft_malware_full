import os
import pickle

# extract all .asm file names
xid = [i.split('.')[0] for i in os.listdir('train') if '.asm' in i]
Xt_id = [i.split('.')[0] for i in os.listdir('test') if '.asm' in i]
f = open('trainLabels.csv')
f.readline()
name_to_label = {}
for line in f:
    xx = line.split(',')
    file_name = xx[0][1:-1]
    name_to_label[file_name] = int(xx[-1])
f.close()
y = [name_to_label[i] for i in xid]


pickle.dump(xid, open('xid_train.p', 'w'))
pickle.dump(Xt_id, open('xid_test.p', 'w'))
pickle.dump(xid, open('xid.p', 'w'))
pickle.dump(Xt_id, open('Xt_id.p', 'w'))
pickle.dump(y, open('y.p', 'w'))

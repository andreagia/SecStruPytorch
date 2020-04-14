import xml.etree.ElementTree as et
import sys
xtree = et.parse("./Data/uniprot3dannotated.xml")
xroot = xtree.getroot()

#sstype = ["helix", "turn", "strand"]
# sstype = ["helix"]
sstype = ["helix", "strand"]

print(xroot)
out = []

outss = []
outseq = []
outssint = []
outseqfasta = []
for elem in xroot.iter("{http://uniprot.org/uniprot}entry"):
    # print("------INI--------")
    print
    seq = elem.find("{http://uniprot.org/uniprot}sequence")
    # print("-----------------INI------------------------")
    # print(seq.text)
    seqt = seq.text
    seqtf = seqt.replace('\n', '')
    # print(seqtf)
    if seqt:
        # print(seqt.replace('\n', ''))
        ssm = elem.findall("{http://uniprot.org/uniprot}feature")
        for ss in ssm:
            xmlstr = et.tostring(ss)
            # print(xmlstr)
            if ss.get("type") in sstype:
                # print("TYPE "+ss.get("type"))
                # print("EVIDENCE "+ss.get("evidence"))
                begin = ss.find("./{http://uniprot.org/uniprot}location/{http://uniprot.org/uniprot}begin")
                end = ss.find("./{http://uniprot.org/uniprot}location/{http://uniprot.org/uniprot}end")
                ibegin = int(begin.get("position"))
                iend = int(end.get("position"))
                if 20 > (iend - ibegin) > 5:
                    #outseqfasta.append(seqtf[ibegin - 1:iend])

                    outss.append(ss.get("type")[0:1])
                    if ss.get("type")[0:1] == "h":
                        outssint.append(0)
                        outseqfasta.append(seqtf[ibegin - 1:iend])
                    # elif ss.get("type")[0:1] == "t":
                    #     outssint.append(1)
                    #     outseqfasta.append(seqtf[ibegin - 1:iend])
                    # elif ss.get("type")[0:1] == "s":
                    #     outssint.append(2)
                    #     outseqfasta.append(seqtf[ibegin - 1:iend])
                # print("BEGIN ",ibegin,"END", iend)
                # print(seqtf[ibegin-1:iend])
                # print("-----------------END------------------------")
        # itera = itera +1

        # if itera == 100:
        #    sys.exit()

        #            if  ss.get("type") in sstype:
        #                 print(ss.get("type"))
        #                 begin = ss.find("./{http://uniprot.org/uniprot}location/{http://uniprot.org/uniprot}begin")
        #                 end = ss.find("./{http://uniprot.org/uniprot}location/{http://uniprot.org/uniprot}end")
        #                 #print(begin.get("position"),end.get("position"))
        #                 ibegin = int(begin.get("position"))
        #                 iend = int(end.get("position"))
        #                 #outseq.append("k ".join(seqtf[ibegin-1:iend].lower()))
        #                 outseqfasta.append(seqtf[ibegin-1:iend])
        #
        #                 outss.append(ss.get("type")[0:2])
        #                 #print(seqtf[ibegin-1:iend])
        #
        #                 sys.exit()
        out.append(seqt.replace('\n', ''))

#  print("------END--------")

s = "--"
helix = s.join(outseqfasta)

text_file = open("./Data/helix5-20.txt", "w")
n = text_file.write(helix)
text_file.close()

sys.exit()

import numpy as np

#save numpy raw fastadate
npoutseqfasta = np.array(outseqfasta)
npoutss = np.array(outss)
npoutssint = np.array(outssint)
print(npoutseqfasta.shape)
print(npoutss.shape)

from numpy import save

save("/Users/andrea/PycharmProjects/SecStruPytorch/data/npoutseqfasta10.npy",npoutseqfasta)
save("/Users/andrea/PycharmProjects/SecStruPytorch/data/npoutss10.npy",npoutss)
save("/Users/andrea/PycharmProjects/SecStruPytorch/data/npoutssint10.npy",npoutssint)
sys.exit()

fasta = "ACDEFGHIKLMNPQRSTVWY"
len(fasta)
# Encode categorical data
from sklearn import preprocessing

fastaLE = preprocessing.LabelBinarizer()
fastaLE.fit(list(fasta))
fastaLE.classes_

sec = "h"
# sec = "ht"
secLE = preprocessing.LabelBinarizer()
secLE.fit(list(sec))
secLE.classes_
from functools import reduce

iniarray = np.asarray(fastaLE.transform(list(outseqfasta[0]))).reshape((1, -1))

finarray1 = reduce(lambda x, y: np.concatenate((x, np.asarray(fastaLE.transform(list(y))).reshape((1, -1))), axis=0),
                   outseqfasta, iniarray)

finarray2 = np.delete(finarray1, 0, 0)

secarray = np.asarray(secLE.transform(list(outss[0]))).reshape((1, -1))

secarray1 = reduce(lambda x, y: np.concatenate((x, np.asarray(secLE.transform(list(y))).reshape((1, -1))), axis=0),
                   outss, secarray)

secarray2 = np.delete(secarray1, 0, 0)

secarrayint = np.asarray(outssint, dtype='int64')



save("./data/secarray20.npy",secarray)
save("./data/secarrayint20.npy",secarrayint)
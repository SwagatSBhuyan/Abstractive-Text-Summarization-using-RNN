import pandas as pd
import string
import re
import os


root = '../datasetDUC/DUC2001_Summarization_Documents/data/training'
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
# print(dirlist)
loca = '../datasetDUC/DUC2001_Summarization_Documents/data/training'
locations = []
doc_locs = []
for i in range(len(dirlist)):
    # print(loca + '/' + dirlist[i] + '/' + dirlist[i] + dirlist[i][-1])
    locations.append(loca + '/' + dirlist[i] + '/' + dirlist[i] + dirlist[i][-1])
    doc_locs.append(loca + '/' + dirlist[i] + '/docs')
# print(locations)

dir_num = 0
for loc in locations:

    print('---------------------------NEW ITERATION-------------------------------')
    file1 = open(loc + '/perdocs')

    strng = file1.read()
    # print(strng[0])

    sums = []
    for i in range(len(strng)):
        textsum = ''
        if strng[i] == '<':
            # print('found <')
            j = i
            while strng[j] != '>':
                textsum = textsum + strng[j]
                j = j + 1
            # print('found >')
            textsum = textsum[1:]
            sums.append(textsum)
    # print(sums)

    docrefs = []
    for i in sums:
        if len(i) < 8:
            continue
        idx = i.find('DOCREF=')
        j = idx + 8
        docref = ''
        while i[j] != '\n':
            docref = docref + i[j]
            j = j + 1
        docref = docref[:-1]
        docrefs.append(docref)

    strng = strng.replace("</SUM>", " </SUM>")
    strng = strng.replace("</SUM>", "JOTHARO132")
    strng = re.sub('<[^>]+>', '', strng)
    # strng = strng.replace("\n", "")
    # strng = strng.replace("\n\n", "\n")

    text = ''
    paras = []
    for i in strng.split():
        if i == 'JOTHARO132':
            paras.append(text)
            text = ''
        text = text + i + ' '

    for i in range(len(paras)):
        paras[i] = paras[i].replace('JOTHARO132 ', '')

    ori_texts = []
    for dr, p in zip(docrefs, paras):
        if os.path.isfile(doc_locs[dir_num] + '/' + dr):
            file2 = open(doc_locs[dir_num] + '/' + dr)
            test_str = file2.read()
            sub1 = "<TEXT>"
            sub2 = "</TEXT>"
            idx1 = test_str.index(sub1)
            idx2 = test_str.index(sub2)
            res = ''
            for idx in range(idx1 + len(sub1) + 1, idx2):
                res = res + test_str[idx]
            ori_texts.append(str(res))
            # print(ori_texts)
            # printing result
            # print("The extracted string : " + res)
        else:
            docrefs.remove(dr)
            paras.remove(p)



    # print(str(dir_num))
    # print('\niteration #' + str(dir_num))
    print('DOCREF: ' + str(len(docrefs)))
    print('PARAS: ' + str(len(paras)))
    print('TEXTS: ' + str(len(ori_texts)))

    for i in range(len(ori_texts)):
        ori_texts[i] = ori_texts[i].replace('\n', '')

    # print(docrefs)
    # print(paras)

    if len(docrefs) == len(paras) == len(ori_texts):
        doc_dat = pd.DataFrame({
            'docrefs': docrefs,
            'original text': ori_texts,
            'paragraphs': paras
        })
        doc_dat.to_csv('DUCs/{0}.csv'.format(dirlist[dir_num]))
    dir_num = dir_num + 1
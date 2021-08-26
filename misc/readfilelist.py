

def getFileList_all(fileTxt, Dir=''):
    with open(fileTxt) as f:
        lines = f.readlines()
        file_list0 = ['%s/%s' % (Dir, i.split('\n')[0])for i in lines]
        f.close()
    return file_list0

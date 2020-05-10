from os import listdir
from os.path import isfile, join, basename
from urllib.parse import urlparse
from PIL import Image
from numpy import array

def get_folders(path):
    file = open(path)
    doc = file.read()
    folders = list()
    for line in doc.split():
        a = urlparse(line)
        temp = 'pframe_trainvaltest/'+basename(a.path)
        temp = temp[:-4]
        folders.append(temp)
    return folders

def get_img_path(path):
    onlyfiles = dict()
    for folder in get_folders(path):
        onlyfiles[folder] = [f for f in listdir(folder) if isfile(join(folder, f))]
    return onlyfiles
        
def img_generator(images):
    for folder in images.keys():
        img_num = 0
        temp = list(images[folder])
        while img_num < len(images[folder]):
            W, H, U, V, Y = list(), list(), list(), list(), list()
            image_u = Image.open(folder + '/' + temp[img_num])
            image_v = Image.open(folder + '/' + temp[img_num+1])
            image_y = Image.open(folder + '/' + temp[img_num+2]) # 4 times bigger then image U and V
            pix_u = image_u.load()
            pix_v = image_v.load()
            pix_y = image_y.load()
            width, height = image_y.size

            for y in range(height):
                for x in range(width):
                    W.append(x)
                    H.append(y)
                    U.append(pix_u[int(x/2),int(y/2)])
                    V.append(pix_v[int(x/2),int(y/2)])
                    Y.append(pix_y[x,y])

            yield ([[array(W)],[array(H)],[array(U), array(V), array(Y)]])
            img_num = img_num + 3


img_generator(get_img_path('ourkit.txt'))

'''
@comment:用来获取anchor box
@author:
'''
import xml.etree.ElementTree as ET
import os
import pickle
import pdb

classes = ["tyrannosaurus","mazdakey","orange","cup"]

def convert_annotation(image_id,flag,savepath):
    in_file = open(savepath+'/trainImageXML/%s.xml' % (image_id))
    labeltxt = savepath+'/trainImageLabelTxt';
    if os.path.exists(labeltxt) == False:
        os.mkdir(labeltxt);
    out_file = open(savepath+'/trainImageLabelTxt/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))   #获取到对角的坐标
        object_box = [ b[1]-b[0], b[3]-b[2] ]     #宽, 高
        object_boxes.append( object_box )

data_file_path = './pkl_file'
cache_file = os.path.join( data_file_path, 'data_cache_file.pkl' )

if not os.path.exists( data_file_path ):
    os.makedirs( data_file_path )
    print( "pkl file not exists, create it" )



object_boxes = []      #物体大小都放到这个里面

savepath = '.'
idtxt = savepath + "/trainImageId.txt";
pathtxt = savepath + "/trainImagePath.txt" ;
image_ids = open(idtxt).read().strip().split()
list_file = open(pathtxt, 'w')
s = '\xef\xbb\xbf'
# pdb.set_trace(  )
for image_id in image_ids:
    nPos = image_id.find(s)
    if nPos >= 0:
       image_id = image_id[3:]
    # list_file.write('%s/trainImage/%s.jpg\n'%(wd,image_id))
    print(image_id)
    convert_annotation(image_id, 0, savepath)     #中间这个参数0表示是训练数据集
    img_num += 1
list_file.close()

with open( cache_file, 'wb' ) as f:
    pickle.dump( object_boxes, f )    #把结果存到文件里面去

with open( cache_file, 'rb' ) as f:
    b = pickle.load( f )
    print( len( b ) )
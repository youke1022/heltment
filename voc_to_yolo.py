import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 数据集根目录
dataset_root = r'c:\Users\15110\Desktop\安全帽\dataset'
# 输出目录
output_dir = r'c:\Users\15110\Desktop\安全帽\yolo_dataset'
# 类别名称
classes = ['hat', 'person']

# 创建输出目录结构
os.makedirs(output_dir, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# 读取ImageSets/Main中的文件列表
def get_image_ids(split):
    ids_file = os.path.join(dataset_root, 'ImageSets', 'Main', f'{split}.txt')
    with open(ids_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 转换XML标注到YOLO格式
def convert_annotation(image_id, split):
    xml_path = os.path.join(dataset_root, 'Annotations', f'{image_id}.xml')
    txt_path = os.path.join(output_dir, 'labels', split, f'{image_id}.txt')

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        with open(txt_path, 'w') as f:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls_name = obj.find('name').text
                if cls_name not in classes or int(difficult) == 1:
                    continue

                cls_id = classes.index(cls_name)
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                ymin = float(xmlbox.find('ymin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymax = float(xmlbox.find('ymax').text)

                # 转换为YOLO格式 (归一化中心点坐标和宽高)
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                f.write(f'{cls_id} {x_center} {y_center} {w} {h}\n')

        # 复制图片
        src_img = os.path.join(dataset_root, 'JPEGImages', f'{image_id}.jpg')
        dst_img = os.path.join(output_dir, 'images', split, f'{image_id}.jpg')
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f'Warning: Image {src_img} not found')

    except Exception as e:
        print(f'Error processing {xml_path}: {e}')

# 处理所有数据集分割
for split in ['train', 'val', 'test']:
    print(f'Processing {split} split...')
    image_ids = get_image_ids(split)
    for image_id in image_ids:
        convert_annotation(image_id, split)

print('Conversion completed!')
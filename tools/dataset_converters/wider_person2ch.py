import mmcv
import mmengine
import json
import argparse
import os.path as osp

METAINFO = {
    'classes': ('pedestrains', 'riders',
                'partially-visible persons',
                'ignore regions', 'crowd'),
    # palette is a list of color tuples, which is used for visualization.
    'palette': [(220, 20, 60)]
}
def convert_annotations(path):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json = []
    lines = open(path).readlines()
    for i, line in enumerate(lines):
        print(f'{i}/{len(lines)}')
        file_name = f'{line.strip()}.jpg'
        annotations = open(osp.join(osp.dirname(path), 'Annotations', f'{file_name}.txt')).readlines()[1:]
        # We skip training images with 0 pedestrians.
        if osp.basename(path).startswith('train') and \
            len(tuple(filter(lambda x: int(x.split()[0]) == 1, annotations))) == 0:
            print(f'No pedestrians for {file_name}')
            continue
        image = mmcv.imread(osp.join(osp.dirname(path), 'Images', file_name))
        sample_json = {
            'file_name': file_name,
            'width': image.shape[1],
            'height': image.shape[0],
            'ID': line.strip(),
            'gtboxes': []
        }
        for annotation in annotations:
            label, x_min, y_min, x_max, y_max = tuple(map(int, annotation.split()))
            sample_json['gtboxes'].append({
                'iscrowd': label != 1,
                'image_id': img_id,
                'bbox': [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                'area': (x_max - x_min + 1) * (y_max - y_min + 1),
                'segmentation': [[]],
                'tag': METAINFO['classes'][int(label) - 1],
                'id': ann_id
            })
            ann_id += 1
        img_id += 1
        out_json.append(sample_json)
    with open(f'{path[:-4]}.odgt', 'a') as fi:
        for x in out_json:
            fi.write(json.dumps(x) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert WiderPerson to COCO format')
    parser.add_argument('--path', help='WiderPerson data path', default='/home/tzh/Project/WiderPerson/data')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert_annotations(osp.join(args.path, 'train.txt'))
    convert_annotations(osp.join(args.path, 'val.txt'))
    convert_annotations(osp.join(args.path, 'test.txt'))
    # convert_annotations(osp.join(args.path, 'test_one.txt'))


if __name__ == '__main__':
    main()
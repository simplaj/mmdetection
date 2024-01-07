import os
import json
import cv2


classes = ['back', 'ped']

def data_read(json_path):
    with open(json_path, 'r') as fi:
        data = json.load(fi)
    return data


def draw(data, img_prefix):
    for img_dict in data:
        img_id = img_dict['ID']
        img = cv2.imread(os.path.join(img_prefix, img_id + '.jpg'))
        for det in img_dict['dtboxes']:
            if det['score'] > 0.5:
                x, y, w, h = det['box']
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                              (0, 0, 255))
                cv2.putText(img, classes[int(det['tag'])], (int(x + w), int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imwrite(f'./outputs/{img_id}.jpg', img)
                

def main():
    res_json = '/home/tzh/Project/mmdetection/work_dirs/ped_only/out.pkl.json'
    img_prefix = '/home/tzh/Project/WiderPerson/data/Images'
    data = data_read(res_json)
    draw(data, img_prefix)


if __name__ == '__main__':
    main()
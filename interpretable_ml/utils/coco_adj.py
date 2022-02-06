from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle
import argparse
import os


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='count')
    parser.add_argument('--data_dir', type=str, default='data/coco')
    parser.add_argument('--version', type=str, default='2014')
    return parser.parse_args()


def load_coco(data_dir, version):

    ann_file = os.path.join(
        data_dir, f"annotations/instances_train{version}.json")
    coco = COCO(ann_file)

    cats = {k: v for k, v in sorted(
        coco.cats.items(), key=lambda cat: cat[1]['name'])}

    coco_id_to_new_id = {c: i for i, c in enumerate(cats)}
    return coco, coco_id_to_new_id


def adj_by_count(args):

    output_file = os.path.join(args.data_dir, f"coco_adj_{args.version}.pkl")

    if os.path.exists(output_file):
        return

    print("Generating adj matrix by co-occurrences")

    coco, coco_id_to_new_id = load_coco(args.data_dir, args.version)
    adj = np.zeros((80, 80)).astype(int)
    nums = np.zeros(80).astype(int)

    for idx in tqdm(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=idx))
        labels = list(set([coco_id_to_new_id[a["category_id"]]
                           for a in anns if not a['iscrowd']]))
        n = len(labels)
        if n > 0:
            for i in range(n):
                nums[labels[i]] += 1
                for j in range(i+1, n):
                    adj[labels[i]][labels[j]] += 1
                    adj[labels[j]][labels[i]] += 1

    result = {'nums': nums, 'adj': adj}
    pickle.dump(result, open(output_file), "wb")


def adj_x_y(args):
    cof_x, cof_y = args.cof_x, args.cof_y

    args = load_args()
    output_file = os.path.join(
        args.data_dir,  f"coco_adj_{args.version}_{cof_x}_{cof_y}.pkl")

    if os.path.exists(output_file):
        return

    print(f"Generating {cof_x}_{cof_y} adj matrix")

    coco, coco_id_to_new_id = load_coco(args.data_dir, args.version)

    adj = np.zeros((80, 80))
    nums = np.zeros(80).astype(int)

    for idx in tqdm(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=idx))
        labels = [coco_id_to_new_id[a["category_id"]]
                  for a in anns if not a['iscrowd']]
        label_count = Counter(labels)
        labels = list(label_count.keys())
        n = len(labels)
        if n > 0:
            for i in range(n):
                if label_count[labels[i]] >= cof_x:
                    nums[labels[i]] += 1
                for j in range(i+1, n):
                    x = labels[i]
                    y = labels[j]
                    if label_count[x] >= cof_x and label_count[y] >= cof_y:
                        adj[x][y] += 1

                    if label_count[y] >= cof_x and label_count[x] >= cof_y:
                        adj[y][x] += 1

    result = {'nums': nums, 'adj': adj}
    pickle.dump(result, open(output_file, "wb"))


def adj_x_dot_y(args):

    cof_x, cof_y = args.cof_x, args.cof_y

    output_file = os.path.join(
        args.data_dir, f"coco_adj_{args.version}_{cof_x}.{cof_y}.pkl")
    if os.path.exists(output_file):
        return

    print(f"Generating {cof_x}.{cof_y} adj matrix")

    coco, coco_id_to_new_id = load_coco(args.data_dir, args.version)

    adj = np.zeros((80, 80))
    nums = np.zeros(80).astype(int)

    for idx in tqdm(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=idx))
        labels = [coco_id_to_new_id[a["category_id"]]
                  for a in anns if not a['iscrowd']]
        label_count = Counter(labels)
        labels = list(label_count.keys())
        n = len(labels)
        if n > 0:
            for i in range(n):
                if label_count[labels[i]] / cof_x >= 1:
                    nums[labels[i]] += 1
                for j in range(i+1, n):
                    x = labels[i]
                    y = labels[j]
                    if label_count[x] / cof_x >= 1 and label_count[y] >= cof_y / cof_x:
                        adj[x][y] += 1

                    if label_count[y] / cof_y >= 1 and label_count[x] >= cof_x / cof_y:
                        adj[y][x] += 1

    result = {'nums': nums, 'adj': adj}
    pickle.dump(result, open(output_file, "wb"))


if __name__ == "__main__":

    args = load_args()
    if args.mode == "count":
        adj_by_count()
    elif args.mode == "x_y":
        adj_x_y(args)
    elif args.mode == "x.y":
        adj_x_dot_y(args)

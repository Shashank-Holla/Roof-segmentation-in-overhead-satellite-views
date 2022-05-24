import random
import glob
import os
import shutil
import sys

# code to split dataset into train/test in ratio 80:20
def split_train_test_dataset(data_path='./data', test_size=0.2):
    if not os.path.exists(os.path.join(data_path, 'test')):
        os.makedirs(os.path.join(data_path, 'test', 'images'))
        os.makedirs(os.path.join(data_path, 'test', 'gt'))
    data_dir = os.path.join(data_path, 'train')
    sat_path = glob.glob(os.path.join(data_dir, 'images/*'))
    sat_path.sort()
    gt_path = glob.glob(os.path.join(data_dir, 'gt/*'))
    gt_path.sort()
    all_path = list(zip(sat_path, gt_path))
    random.shuffle(all_path)
    test_set_size = round(len(all_path) * test_size)
    test_path = all_path[:test_set_size]
    for sat_im, gt_im in test_path:
        shutil.move(sat_im, os.path.join(data_path, 'test/images'))
        shutil.move(gt_im, os.path.join(data_path, 'test/gt'))


if __name__ == "__main__":
    data_path = sys.argv[1]
    print("Preparing train test split...")
    split_train_test_dataset(data_path, test_size=0.2)
    print("train test split completed.")
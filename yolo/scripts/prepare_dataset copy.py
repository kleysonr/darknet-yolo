#
# Prepara e organiza o dataset da forma que o Darknet precisa
#
import argparse
import os
import shutil
from string import Template

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path for the dataset")
ap.add_argument("-l", "--labels-file", required = True, help = "Relative path for the labels file")
ap.add_argument("-y", "--yolo-size", required = True, help = "Yolo `full` or `tiny`")
args = vars(ap.parse_args())

DATASET_PATH = args['dataset']
LABELS_FILE = args['labels_file']
YOLO_SIZE = args['yolo_size']

# Clean up data directory
shutil.rmtree('../data')
os.makedirs('../data')
os.makedirs('../data/obj')

# Copy the labels file to the data dir
labels_src = os.path.join(DATASET_PATH, LABELS_FILE)
labels_dst = os.path.join('../data', 'obj.names')
shutil.copyfile(labels_src, labels_dst)

# Get the number of classes
with open(labels_dst) as f:
    for i, l in enumerate(f):
        pass

# Yolo parameters
num_classes = i + 1
batch_size = 64
batch_subdivisions = 16  # Subdivison explanation -> https://github.com/pjreddie/darknet/issues/224
max_batches = num_classes*2000
steps1 = .8 * max_batches
steps2 = .9 * max_batches
steps_str = str(steps1)+','+str(steps2)
num_filters = (num_classes + 5) * 3

# Template dictionary
params = {
    'num_classes': num_classes,
    'batch_size': batch_size,
    'batch_subdivisions': batch_subdivisions,  # Subdivison explanation -> https://github.com/pjreddie/darknet/issues/224
    'max_batches': max_batches,
    'steps1': steps1,
    'steps2': steps2,
    'steps_str': steps_str,
    'num_filters': num_filters
}

# Copy images and labels for train and val datasets
train_path = os.path.join(DATASET_PATH, 'train')
val_path = os.path.join(DATASET_PATH, 'val')

print('Copying train dataset...')
if os.path.isdir(train_path):
    for f in os.listdir(train_path):
        shutil.copy(os.path.join(train_path, f), '../data/obj')

print('Copying val files...')
if os.path.isdir(val_path):
    for f in os.listdir(val_path):
        shutil.copy(os.path.join(val_path, f), '../data/obj')

# Create definition file
print('Creating darknet definition file...')
with open('../data/obj.data', 'w') as out:
    out.write('classes = {}\n'.format(params['num_classes']))
    out.write('train = /yolo/data/train.txt\n')
    out.write('valid = /yolo/data/val.txt\n')
    out.write('names = /yolo/data/obj.names\n')
    out.write('backup = /yolo/backup')

# Generating list of train and val image files
print('Creating train.txt and val.txt definition files...')
with open('../data/train.txt', 'w') as out:
  for img in [f for f in os.listdir(train_path) if (f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png'))]:
      print('/yolo/data/obj/' + img)
    # out.write('/yolo/data/obj/' + img + '\n')

with open('../data/val.txt', 'w') as out:
  for img in [f for f in os.listdir(val_path) if (f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png'))]:
    out.write('/yolo/data/obj/' + img + '\n')

# Creating cfg file
print('Dynamically building cfg file...')
with open('../cfg/templates/yolov4-{}.cfg'.format(YOLO_SIZE), 'r') as f:
    cfg_template = f.read()

cfg_file = Template(cfg_template).substitute(params)

with open('../cfg/yolo-obj.cfg', 'w') as f:
    print(cfg_file, file=f)

print('Finished.')
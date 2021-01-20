import tools.dataset as ds
import os
from absl import app, flags, logging
from absl.flags import FLAGS
#python generate_csv.py --raw_images_path=D:/workspace/datasets/simulator/1_1/images/ --csv_path=D:/workspace/datasets/simulator/1_1/labels.csv
# TODO: Use config file like yaml
FLAGS = flags.FLAGS
flags.DEFINE_string('raw_images_path', '', 'Path to images')
flags.DEFINE_string('csv_path', '', 'Path to original CSV')
flags.DEFINE_string('output_folder', '../data/datasets/simulator/', 'Path to TFRecord output folder')
flags.DEFINE_string('full_csv_name', 'labels_with_negatives.csv', 'Name of generated clean labels CSV')
flags.DEFINE_string('train_csv_name', 'single_nao_train.csv', 'Name of train split CSV')
flags.DEFINE_string('test_csv_name', 'single_nao_val.csv', 'Name of test split CSV')

def main(argv):
    clean_csv_path = os.path.join(FLAGS.output_folder, FLAGS.full_csv_name)

    ds.create_clean_csv(FLAGS.csv_path, FLAGS.raw_images_path, clean_csv_path)
    logging.info('Created clean CSV with false negatives in: ' + clean_csv_path)

    ds.split_csv(clean_csv_path, FLAGS.output_folder, FLAGS.train_csv_name, FLAGS.test_csv_name)
    logging.info('Splitting clean CSV into: ' + FLAGS.train_csv_name + ', ' + FLAGS.test_csv_name)
    logging.info('Done creating CSVs!')

if __name__ == '__main__':
    app.run(main)
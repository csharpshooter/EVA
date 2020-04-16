# from src.utils import Utils
import time
import numpy as np
from imageio import imread
from tqdm import tqdm


class TinyImagenetHelper:

    def __init__(self):
        self.output_folder = ''
        self.dictionaryFileName = ''

    def download_dataset(self, folder_path):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        from src.utils import Utils
        folder_path = Utils.download_file(folder_path, url=url)

        return Utils.extract_zip_file(file_path=folder_path, extract_path='data')

    def get_id_dictionary(self, path):
        id_dict = {}
        for i, line in enumerate(open(path + '/wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict

    def get_class_to_id_dict(self, id_dict, path):
        # id_dict = self.get_id_dictionary(path)
        all_classes = {}
        classes = []
        result = {}
        for i, line in enumerate(open(path + '/words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
            classes.append(word)
        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])
        return result, classes

    def get_train_test_labels_data(self, id_dict, path, test_split=0.3):
        print('Starting data loading')
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        t = time.time()
        total_images = 110000
        train_image_count = total_images - (total_images * test_split)
        for key, value in tqdm(id_dict.items()):
            for i in range(500):
                if len(train_data) < train_image_count:
                    # noinspection DuplicatedCode
                    train_data.append(path + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)))
                    # imread(path + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB'))
                    # train_labels_ = np.array([[0] * 200])
                    # train_labels_[:, value] = 1
                    # train_labels += train_labels_.tolist()
                    train_labels.append(id_dict[key])

                else:
                    test_data.append(path + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)))
                    # imread(path + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB'))
                    # test_labels_ = np.array([[0] * 200])
                    # test_labels_[:, value] = 1
                    # test_labels += test_labels_.tolist()
                    test_labels.append(id_dict[key])

        for line in tqdm(open(path + '/val/val_annotations.txt')):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(path + '/val/images/{}'.format(img_name))
            # test_data.append(imread(path + '/val/images/{}'.format(img_name), pilmode='RGB'))
            # test_labels_ = np.array([[0] * 200])
            # test_labels_[0, id_dict[class_id]] = 1
            # test_labels += test_labels_.tolist()
            test_labels.append(id_dict[class_id])

        print('Finished data loading, in {} seconds'.format(time.time() - t))
        return np.array(train_data), train_labels, np.array(test_data), test_labels

    # def get_code_words(self,path):

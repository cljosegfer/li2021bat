from base import BaseEvaluater
from utils import *
import torch.nn as nn
from data_loader.util import *

from tqdm import tqdm

import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def find_best_thresholds(predictions, true_labels_dict, thresholds):
    num_classes = len(predictions[0])
    best_thresholds = [0.5] * num_classes
    best_f1s = [0.0] * num_classes

    for class_idx in (range(num_classes)):
        for thresh in thresholds:
            f1 = f1_score(
                true_labels_dict[class_idx],
                predictions[thresh][class_idx],
                zero_division=0,
            )

            if f1 > best_f1s[class_idx]:
                best_f1s[class_idx] = f1
                best_thresholds[class_idx] = thresh
    
    return best_f1s, best_thresholds

def metrics_table(all_binary_results, all_true_labels):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    num_classes = all_binary_results.shape[-1]
    for class_idx in range(num_classes):
        class_binary_results = all_binary_results[:, class_idx]
        class_true_labels = all_true_labels[:, class_idx]

        accuracy = accuracy_score(class_true_labels, class_binary_results)
        precision = precision_score(class_true_labels, class_binary_results, zero_division=0)
        recall = recall_score(class_true_labels, class_binary_results, zero_division=0)
        f1 = f1_score(class_true_labels, class_binary_results, zero_division=0)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    metrics_dict = {
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1 Score": f1_scores,
    }

    return metrics_dict



class Evaluater_beat_aligned_data(BaseEvaluater):
    """
    Evaluater class
    """

    def __init__(self, model, criterion, metric_ftns, config, data_loader, checkpoint_dir=None, result_dir=None):
        super().__init__(model, criterion, metric_ftns, config, checkpoint_dir, result_dir)

        self.config = config
        # self.beat_length = config["evaluater"]["beat_length"]
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.num_classes = config['arch']['args']['num_classes']
        self.lead_number = config['data_loader']['args']['lead_number']
        self.split_index = config['data_loader']['args']['split_index']
        self.label_dir = config['data_loader']['args']['label_dir']
        self.resample_Fs = config["data_loader"]['args']["resample_Fs"]
        self.window_size = config["data_loader"]['args']["window_size"]
        self.n_segment = config["evaluater"]["n_segment"]
        self.save_dir = config["data_loader"]["args"]["save_dir"]
        self.seg_with_r = config["data_loader"]["args"]["seg_with_r"]
        self.sigmoid = nn.Sigmoid()
        self.data_loader = data_loader

        split_idx = loadmat(self.split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        self.test_index = test_index.reshape((test_index.shape[1], ))

        # self.train_info = np.array(np.load(os.pandasth.join(self.save_dir, 'train_info' + str(self.window_size) + '_' + str(self.resample_Fs) + '_' + str(self.seg_with_r) + '.npy')))

        if self.lead_number == 2:
            # two leads
            self.leads_index = [1, 10]
        elif self.lead_number == 3:
            # three leads
            self.leads_index = [0, 1, 7]
        elif self.lead_number == 6:
            # six leads
            self.leads_index = [0, 1, 2, 3, 4, 5]
        elif self.lead_number == 8:
            # eight leads
            self.leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        else:
            self.leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def _to_np(self, tensor):
        if self.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()

    def evaluate(self):
        """
        Evaluate after training procedure finished
        """
        self.model.eval()

        weights_file = 'weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(self.label_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class,
                                                                           equivalent_classes)
        self.classes = classes

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)
        self.weights = weights

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.

        # Get number of samples for each category
        self.indices = indices
        num_files = len(label_files)

        # output_logits = []
        # targets = []

        # recordings_saved = np.load(os.path.join('/data/ecg/challenge2020/data/recordings_' + str(5000) + '_' + str(
        #     500) + '_' + str(False) + '.npy'))
        # ratio_saved = np.load(os.path.join('/data/ecg/challenge2020/data/info_' + str(5000) + '_' + str(
        #     500) + '_' + str(False) + '.npy'))

        # for i in range(num_files):
        #     recording, header, name = load_challenge_data(label_files[i], self.label_dir)
        #     if i in self.test_index and name[0] != 'A' and name[0] != 'Q':
        #         print('{}/{}'.format(i + 1, num_files))

        #         recording = [recordings_saved[i]]
        #         info2save = ratio_saved[i]
        #         if info2save.sum() <= 0:
        #             continue

        #         if recording is None:
        #             continue

        #         data, info = torch.tensor(recording), torch.tensor(info2save)

        #         data = data[:, self.leads_index, :, :]
        #         # info -= self.train_info.mean()
        #         # info /= self.train_info.std()

        #         data, info = data.to(self.device, dtype=torch.float), info.to(self.device, dtype=torch.float)
        #         output = self.model(data, info)
        #         output_logit = torch.sigmoid(output)
        #         output_logit = output_logit.detach().cpu().numpy()
        #         output_logit = np.mean(output_logit, axis=0)

        #         output_logits.append(output_logit)
        #         targets.append(labels_onehot[i])

        # output_logits = np.array(output_logits)
        # targets = np.array(targets)

        # for met in self.metric_ftns:
        #     self.test_metrics.update(met.__name__, met(output_logits, targets))

        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, ([data, info], target, class_weights) in tqdm(enumerate(self.data_loader.test_data_loader)):
                # data, target, class_weights, info = data.to(self.device), target.to(self.device), class_weights.to(self.device), info.to(self.device)
                data, target, class_weights, info = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float), \
                                                class_weights.to(self.device, dtype=torch.float), info.to(self.device, dtype=torch.float)
                # target_coarse = target_coarse.to(device)
                output = self.model(data, info)

                output_logit = self.sigmoid(output)
                output_logit = self._to_np(output_logit)
                target = self._to_np(target)

                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output_logit, target))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

class Evaluater(BaseEvaluater):
    """
    Evaluater class
    """

    def __init__(self, model, criterion, metric_ftns, config, checkpoint_dir=None, result_dir=None):
        super().__init__(model, criterion, metric_ftns, config, checkpoint_dir, result_dir)
        self.config = config
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.num_classes = config['arch']['args']['num_classes']
        self.lead_number = config['data_loader']['args']['lead_number']
        self.split_index = config['data_loader']['args']['split_index']
        self.label_dir = config['data_loader']['args']['label_dir']
        self.resample_Fs = config["data_loader"]['args']["resample_Fs"]
        self.window_size = config["data_loader"]['args']["window_size"]
        self.n_segment = config["evaluater"]["n_segment"]
        self.sigmoid = nn.Sigmoid()

        split_idx = loadmat(self.split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        self.test_index = test_index.reshape((test_index.shape[1], ))

        if self.lead_number == 2:
            # two leads
            self.leads_index = [1, 10]
        elif self.lead_number == 3:
            # three leads
            self.leads_index = [0, 1, 7]
        elif self.lead_number == 6:
            # six leads
            self.leads_index = [0, 1, 2, 3, 4, 5]
        elif self.lead_number == 8:
            # eight leads
            self.leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        else:
            self.leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def evaluate(self):
        """
        Evaluate after training procedure finished
        """
        self.model.eval()

        weights_file = 'weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(self.label_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class,
                                                                           equivalent_classes)
        self.classes = classes

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)
        self.weights = weights

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.

        # Get number of samples for each category
        self.indices = indices
        num_files = len(label_files)

        output_logits = []
        targets = []

        for i in range(num_files):

            recording, header, name = load_challenge_data(label_files[i], self.label_dir)

            if i in self.test_index and name[0] != 'A' and name[0] != 'Q':
                print('{}/{}'.format(i + 1, num_files))

                output_logit = self.run_my_model(header, recording)
                output_logits.append(output_logit)
                targets.append(labels_onehot[i])

        output_logits = np.array(output_logits)
        targets = np.array(targets)

        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(output_logits, targets))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def run_my_model(self, header, recording):
        recording[np.isnan(recording)] = 0

        # divide ADC_gain and resample
        recording = resample(recording, header, self.resample_Fs)
        recording = recording[self.leads_index, :]

        # slide and cut
        recording = slide_and_cut(recording, self.n_segment, self.window_size, self.resample_Fs)
        data = torch.tensor(recording)
        data = data.to(self.device, dtype=torch.float)
        output = self.model(data)
        output_logit = torch.sigmoid(output)
        output_logit = output_logit.detach().cpu().numpy()
        output_logit = np.mean(output_logit, axis=0)
        return output_logit

class Evaluater_beat_aligned_data_CODE(BaseEvaluater):
    """
    Evaluater class
    """

    def __init__(self, model, criterion, metric_ftns, config, data_loader, checkpoint_dir=None, result_dir=None):
        super().__init__(model, criterion, metric_ftns, config, checkpoint_dir, result_dir)

        self.config = config
        # self.beat_length = config["evaluater"]["beat_length"]
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.num_classes = config['arch']['args']['num_classes']
        self.lead_number = config['data_loader']['args']['lead_number']
        self.split_index = config['data_loader']['args']['split_index']
        self.label_dir = config['data_loader']['args']['label_dir']
        self.resample_Fs = config["data_loader"]['args']["resample_Fs"]
        self.window_size = config["data_loader"]['args']["window_size"]
        self.n_segment = config["evaluater"]["n_segment"]
        self.save_dir = config["data_loader"]["args"]["save_dir"]
        self.seg_with_r = config["data_loader"]["args"]["seg_with_r"]
        self.sigmoid = nn.Sigmoid()
        self.data_loader = data_loader

        split_idx = loadmat(self.split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        self.test_index = test_index.reshape((test_index.shape[1], ))

        # self.train_info = np.array(np.load(os.pandasth.join(self.save_dir, 'train_info' + str(self.window_size) + '_' + str(self.resample_Fs) + '_' + str(self.seg_with_r) + '.npy')))

        if self.lead_number == 2:
            # two leads
            self.leads_index = [1, 10]
        elif self.lead_number == 3:
            # three leads
            self.leads_index = [0, 1, 7]
        elif self.lead_number == 6:
            # six leads
            self.leads_index = [0, 1, 2, 3, 4, 5]
        elif self.lead_number == 8:
            # eight leads
            self.leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        else:
            self.leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def _to_np(self, tensor):
        if self.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()

    def evaluate(self):
        """
        Evaluate after training procedure finished
        """

        num_classes = 6
        thresholds = np.arange(0, 1.01, 0.01)
        predictions = {thresh: [[] for _ in range(num_classes)] for thresh in thresholds}
        true_labels_dict = [[] for _ in range(num_classes)]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, ([data, info], target, class_weights) in tqdm(enumerate(self.data_loader.valid_data_loader)):
                data, target, class_weights, info = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float), \
                                                class_weights.to(self.device, dtype=torch.float), info.to(self.device, dtype=torch.float)
                output = self.model(data, info)

                output_logit = self.sigmoid(output)
                output_logit = self._to_np(output_logit)
                target = self._to_np(target)

                for class_idx in range(num_classes):
                    for thresh in thresholds:
                        predicted_binary = (output_logit[:, class_idx] >= thresh)
                        predictions[thresh][class_idx].extend(
                            predicted_binary
                        )
                    true_labels_dict[class_idx].extend(
                        target[:, class_idx]
                    )
        
        best_f1s, best_thresholds = find_best_thresholds(predictions, true_labels_dict, thresholds)

        all_binary_results = []
        all_true_labels = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, ([data, info], target, class_weights) in tqdm(enumerate(self.data_loader.test_data_loader)):
                data, target, class_weights, info = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float), \
                                                class_weights.to(self.device, dtype=torch.float), info.to(self.device, dtype=torch.float)
                output = self.model(data, info)

                output_logit = self.sigmoid(output)
                output_logit = self._to_np(output_logit)
                target = self._to_np(target)

                binary_result = np.zeros_like(output_logit)
                for i in range(len(best_thresholds)):
                    binary_result[:, i] = (
                        output_logit[:, i] >= best_thresholds[i]
                    )
                
                all_binary_results.append(binary_result)
                all_true_labels.append(target)

        all_binary_results = np.array(all_binary_results)[0, :, :]
        all_true_labels = np.array(all_true_labels)[0, :, :]

        metrics_dict = metrics_table(all_binary_results, all_true_labels)

        pathways = str(self.checkpoint_dir).split('/')
        pathways[2] = 'results'
        result_dir = '/'.join(pathways)
        with open('{}/metrics.json'.format(str(result_dir)), 'w') as f:
            json.dump(metrics_dict, f, indent = 2)

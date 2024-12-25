import os
import json
import data_preprocessing
import models
import numpy as np
import torch
import random
import datetime
import argparse
import socket
from similarity.damerau import Damerau # pip install strsim
import utils
from copy import deepcopy


def iterate_over_generated_suffixes(predictions=None):
    damerau = Damerau()
    nb_worst_situs = 0
    nb_all_situs = 0
    average_damerau_levenshtein_similarity = 0
    average_mae = 0.0
    average_mae_denormalised = 0.0
    eos_token = predictions['eos_token']
    predictions['dls'] = {}
    predictions['mae'] = {}
    predictions['mae_denormalised'] = {}
    predictions['evaluated_ids'] = {}

    for prefix in predictions['ids'].keys():
        predictions['dls'][prefix] = []
        predictions['mae'][prefix] = []
        predictions['mae_denormalised'][prefix] = []
        predictions['evaluated_ids'][prefix] = []

        for i in range(len(predictions['activities']['suffixes']['target'][prefix])):
            target_activity_suffix_padded = predictions['activities']['suffixes']['target'][prefix][i]
            target_activity_suffix = []
            target_time_suffix_padded = predictions['times']['suffixes']['target'][prefix][i]
            target_time_suffix = []
            target_time_denormalised_suffix_padded = predictions['times_denormalised']['suffixes']['target'][prefix][i]
            target_time_denormalised_suffix = []

            # The situ when the original trace was shorter than the given prefix is:
            # prepared for an expanded dimensionality too:
            if (len(target_activity_suffix_padded) > 0) and isinstance(target_activity_suffix_padded[0], list):
                if target_activity_suffix_padded[0][0] == predictions['pad_token']:
                    continue
            else:
                if target_activity_suffix_padded[0] == predictions['pad_token']:
                    continue

            # prepared for an expanded dimensionality too:
            # for the activities:
            target_length = 0
            if (len(target_activity_suffix_padded) > 0) and isinstance(target_activity_suffix_padded[0], list):
                for j in target_activity_suffix_padded:
                    if j[0] != eos_token:
                        target_activity_suffix.append(j[0])
                        target_length += 1
                    else:
                        break
            else:
                for j in target_activity_suffix_padded:
                    if j != eos_token:
                        target_activity_suffix.append(j)
                        target_length += 1
                    else:
                        break
            # for the times:
            if (len(target_time_suffix_padded) > 0) and isinstance(target_time_suffix_padded[0], list):
                for j in range(target_length):
                    target_time_suffix.append(target_time_suffix_padded[j][0])
                    target_time_denormalised_suffix.append(target_time_denormalised_suffix_padded[j][0])
            else:
                for j in range(target_length):
                    target_time_suffix.append(target_time_suffix_padded[j])
                    target_time_denormalised_suffix.append(target_time_denormalised_suffix_padded[j])

            is_the_worst_situ = True
            prediction_activity_suffix_padded = predictions['activities']['suffixes']['prediction'][prefix][i]
            prediction_activity_suffix = []
            prediction_time_suffix_padded = predictions['times']['suffixes']['prediction'][prefix][i]
            prediction_time_suffix = []
            prediction_time_denormalised_suffix_padded = predictions['times_denormalised']['suffixes']['prediction'][prefix][i]
            prediction_time_denormalised_suffix = []

            # In the worst case it stops at the length of the longest suffix
            # prepared for an expanded dimensionality too:
            # for the activities:
            prediction_length = 0
            if (len(prediction_activity_suffix_padded) > 0) and isinstance(prediction_activity_suffix_padded[0], list):
                for j in prediction_activity_suffix_padded:
                    if j[0] != eos_token:
                        prediction_activity_suffix.append(j[0])
                        prediction_length += 1
                    else:
                        is_the_worst_situ = False
                        break
            else:
                for j in prediction_activity_suffix_padded:
                    if j != eos_token:
                        prediction_activity_suffix.append(j)
                        prediction_length += 1
                    else:
                        is_the_worst_situ = False
                        break
            # for the times:
            if (len(prediction_time_suffix_padded) > 0) and isinstance(prediction_time_suffix_padded[0], list):
                for j in range(prediction_length):
                    prediction_time_suffix.append(prediction_time_suffix_padded[j][0])
                    prediction_time_denormalised_suffix.append(prediction_time_denormalised_suffix_padded[j][0])
            else:
                for j in range(prediction_length):
                    prediction_time_suffix.append(prediction_time_suffix_padded[j])
                    prediction_time_denormalised_suffix.append(prediction_time_denormalised_suffix_padded[j])

            if is_the_worst_situ:
                nb_worst_situs += 1

            # The situ when the suffix had an [EOS] position only and that is perfectly predicted:
            if len(prediction_activity_suffix) == len(target_activity_suffix) == 0:
                damerau_levenshtein_similarity = 1.0
                mae = 0.0
                mae_denormalised = 0.0
            else:
                damerau_levenshtein_similarity = 1.0 - damerau.distance(prediction_activity_suffix, target_activity_suffix) / max(len(prediction_activity_suffix), len(target_activity_suffix))
                if len(target_time_suffix) == 0:
                    sum_target_time_suffix = 0.0
                elif len(target_time_suffix) == 1:
                    sum_target_time_suffix = target_time_suffix[0]
                elif len(target_time_suffix) > 1:
                    sum_target_time_suffix = sum(target_time_suffix)
                if len(prediction_time_suffix) == 0:
                    sum_prediction_time_suffix = 0.0
                elif len(prediction_time_suffix) == 1:
                    sum_prediction_time_suffix = prediction_time_suffix[0]
                elif len(prediction_time_suffix) > 1:
                    sum_prediction_time_suffix = sum(prediction_time_suffix)
                if len(target_time_denormalised_suffix) == 0:
                    sum_target_time_denormalised_suffix = 0.0
                elif len(target_time_denormalised_suffix) == 1:
                    sum_target_time_denormalised_suffix = target_time_denormalised_suffix[0]
                elif len(target_time_denormalised_suffix) > 1:
                    sum_target_time_denormalised_suffix = sum(target_time_denormalised_suffix)
                if len(prediction_time_denormalised_suffix) == 0:
                    sum_prediction_time_denormalised_suffix = 0.0
                elif len(prediction_time_denormalised_suffix) == 1:
                    sum_prediction_time_denormalised_suffix = prediction_time_denormalised_suffix[0]
                elif len(prediction_time_denormalised_suffix) > 1:
                    sum_prediction_time_denormalised_suffix = sum(prediction_time_denormalised_suffix)
                mae = abs(sum_target_time_suffix - sum_prediction_time_suffix)
                mae_denormalised = abs(sum_target_time_denormalised_suffix - sum_prediction_time_denormalised_suffix)

            trace_id = predictions['ids'][prefix][i]
            average_damerau_levenshtein_similarity += damerau_levenshtein_similarity
            average_mae += mae
            average_mae_denormalised += mae_denormalised
            predictions['dls'][prefix].append(damerau_levenshtein_similarity) # in-place editing of the input dictionary
            predictions['mae'][prefix].append(mae)  # in-place editing of the input dictionary
            predictions['mae_denormalised'][prefix].append(mae_denormalised)  # in-place editing of the input dictionary
            predictions['evaluated_ids'][prefix].append(trace_id)  # in-place editing of the input dictionary
            nb_all_situs += 1

    # the list (of the prefix) is created in the beggining hence if it remains empty it is deleted now:
    prefix_to_delete = []
    for prefix in predictions['dls'].keys():
        if len(predictions['dls'][prefix]) == 0:
            prefix_to_delete.append(prefix)
    for prefix in prefix_to_delete:
        del predictions['dls'][prefix]
        del predictions['mae'][prefix]
        del predictions['mae_denormalised'][prefix]
        del predictions['evaluated_ids'][prefix]

    average_damerau_levenshtein_similarity /= nb_all_situs
    average_mae /= nb_all_situs
    average_mae_denormalised /= nb_all_situs
    return average_damerau_levenshtein_similarity, average_mae, average_mae_denormalised, nb_worst_situs, nb_all_situs



def rnn_predict(seq_ae_teacher_forcing_ratio, model, model_input_x, model_input_y, temperature=1.0, top_k=None, sample=False, max_length=None):
    # TODO prepare it for activity labels only
    # init model wiht prefix input:
    prefix = model_input_x[0].size(1)
    inp_p = (model_input_x[0], model_input_x[1])
    output = model(inp_p)
    a_decoded = utils.generate(output[0][:, -1, :].unsqueeze(1), temperature=temperature, top_k=top_k, sample=sample)
    prediction = (a_decoded, output[1][:, -1, :].unsqueeze(1))
    input_position = prediction

    # Semi open loop:
    for i in range(prefix, max_length - 1):
        output_position = model(input_position)
        a_decoded = utils.generate(output_position[0], temperature=temperature, top_k=top_k, sample=sample)
        a_pred = torch.cat((prediction[0], a_decoded), dim=1)
        t_pred = torch.cat((prediction[1], output_position[1]), dim=1)
        prediction = (a_pred, t_pred)
        a_inp = a_decoded
        t_inp = output_position[1]
        input_position = (a_inp, t_inp)

    return prediction




# loop for the rnn model:
def iterate_over_prefixes_rnn(log_with_prefixes,
                              model=None,
                              device=None,
                              subset=None,
                              to_wrap_into_torch_dataset=None,
                              max_length=None):

    # TODO prepare it for activity labels only
    predictions = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'ids': {},
                   'eos_token': log_with_prefixes['eos_token'],
                   'sos_token': log_with_prefixes['sos_token'],
                   'pad_token': log_with_prefixes['pad_token'],
                   'max_time_value': log_with_prefixes['max_time_value'],
                   'min_time_value': log_with_prefixes['min_time_value']}

    if not to_wrap_into_torch_dataset:
        # Not implemented
        pass
    else:
        prefixes = list(log_with_prefixes[subset + '_torch_data_loaders'].keys())

        for prefix in prefixes:
            data_loader = log_with_prefixes[subset + '_torch_data_loaders'][prefix]

            a_p_mini_batches = []
            t_p_mini_batches = []
            a_s_t_mini_batches = []
            t_s_t_mini_batches = []
            prediction_a_mini_batches = []
            prediction_t_mini_batches = []

            for mini_batch in iter(data_loader):
                if device == 'GPU':
                    a_p = mini_batch[0].cuda()
                    t_p = mini_batch[1].cuda()
                    a_s_i = mini_batch[2].cuda()
                    t_s_i = mini_batch[3].cuda()
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]
                else:
                    a_p = mini_batch[0]
                    t_p = mini_batch[1]
                    a_s_i = mini_batch[2]
                    t_s_i = mini_batch[3]
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]

                prediction = rnn_predict(seq_ae_teacher_forcing_ratio=0.0,
                                         model=model,
                                         model_input_x=(a_p, t_p),
                                         model_input_y=(a_s_i, t_s_i),
                                         max_length=max_length)

                a_p_mini_batches += a_p.tolist()
                t_p_mini_batches += t_p.tolist()
                a_s_t_mini_batches += a_s_t.tolist()
                t_s_t_mini_batches += t_s_t.tolist()
                prediction_a_mini_batches += prediction[0].tolist()
                prediction_t_mini_batches += prediction[1].tolist()

                del a_p
                del t_p
                del a_s_i
                del t_s_i
                del a_s_t
                del t_s_t
                del mini_batch
                del prediction

            del data_loader

            predictions['activities']['prefixes'][prefix] = a_p_mini_batches
            predictions['times']['prefixes'][prefix] = t_p_mini_batches
            predictions['activities']['suffixes']['target'][prefix] = a_s_t_mini_batches
            predictions['times']['suffixes']['target'][prefix] = t_s_t_mini_batches
            predictions['activities']['suffixes']['prediction'][prefix] = prediction_a_mini_batches
            predictions['times']['suffixes']['prediction'][prefix] = prediction_t_mini_batches
            predictions['ids'][prefix] = log_with_prefixes[subset + '_prefixes_and_suffixes']['ids'][prefix]

        return predictions


def generate_suffixes_rnn(checkpoint_file, log_file, args, path):
    log_with_prefixes = data_preprocessing.create_prefixes(log_file,
                                                           min_prefix=2,
                                                           create_tensors=True,
                                                           add_special_tokens=True,
                                                           pad_sequences=True,
                                                           pad_token=args.pad_token,
                                                           to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                           training_batch_size=args.training_batch_size,
                                                           validation_batch_size=args.validation_batch_size,
                                                           single_position_target=False)

    del log_with_prefixes['training_torch_data_loaders']

    # [EOS], [SOS], [PAD]
    nb_special_tokens = 3
    attributes_meta = {
        0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': log_with_prefixes['vocabulary_size']},
        1: {'min_value': 0.0, 'max_value': 1.0}}

    vars(args)['sos_token'] = log_with_prefixes['sos_token']
    vars(args)['eos_token'] = log_with_prefixes['eos_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = log_with_prefixes['vocabulary_size']
    vars(args)['longest_trace_length'] = log_with_prefixes['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = log_with_prefixes['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.SequentialDecoder(hidden_size=args.hidden_dim,
                                     num_layers=args.n_layers,
                                     dropout_prob=args.dropout_prob,
                                     vocab_size=attributes_meta[0]['vocabulary_size'],
                                     attributes_meta=attributes_meta,
                                     time_attribute_concatenated=args.time_attribute_concatenated,
                                     pad_token=args.pad_token,
                                     nb_special_tokens=attributes_meta[0]['nb_special_tokens'])

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_prefixes_rnn(log_with_prefixes=log_with_prefixes,
                                                model=model,
                                                device=args.device,
                                                subset='validation',
                                                to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                max_length=max_length)
        del log_with_prefixes
        return predictions


def generate_suffixes_rnn_full(checkpoint_file, log_file, args, path):
    log_with_prefixes = data_preprocessing.create_prefixes(log_file,
                                                           min_prefix=2,
                                                           create_tensors=True,
                                                           add_special_tokens=True,
                                                           pad_sequences=True,
                                                           pad_token=args.pad_token,
                                                           to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                           training_batch_size=args.training_batch_size,
                                                           validation_batch_size=args.validation_batch_size,
                                                           single_position_target=False)

    del log_with_prefixes['training_torch_data_loaders']

    # [EOS], [SOS], [PAD], [MASK]
    nb_special_tokens = 4
    attributes_meta = {
        0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': log_with_prefixes['vocabulary_size']},
        1: {'min_value': 0.0, 'max_value': 1.0}}

    vars(args)['sos_token'] = log_with_prefixes['sos_token']
    vars(args)['eos_token'] = log_with_prefixes['eos_token']
    vars(args)['mask_token'] = log_with_prefixes['mask_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = log_with_prefixes['vocabulary_size']
    vars(args)['longest_trace_length'] = log_with_prefixes['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = log_with_prefixes['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.SequentialDecoder(hidden_size=args.hidden_dim,
                                     num_layers=args.n_layers,
                                     dropout_prob=args.dropout_prob,
                                     vocab_size=attributes_meta[0]['vocabulary_size'],
                                     attributes_meta=attributes_meta,
                                     time_attribute_concatenated=args.time_attribute_concatenated,
                                     pad_token=args.pad_token,
                                     nb_special_tokens=attributes_meta[0]['nb_special_tokens'],
                                     architecture='GPT')

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_prefixes_rnn(log_with_prefixes=log_with_prefixes,
                                                model=model,
                                                device=args.device,
                                                subset='validation',
                                                to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                max_length=max_length)
        del log_with_prefixes
        return  predictions


def generate(datetime, model_type, args):
    path = os.path.join('results', model_type)

    # Walk through all the log-result directories:
    for log_directory in sorted(os.scandir(path), key=lambda x: x.name):
        if not os.path.isdir(log_directory): continue

        log_path = log_directory.path
        checkpoint_path = os.path.join(log_path, 'checkpoints')
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))]
        matching_checkpoint_files = [s for s in checkpoint_files if datetime in s]
        checkpoint_file = matching_checkpoint_files[0]
        split_log_file_path = os.path.join(log_path, 'split_log_' + datetime + '.json')

        with open(split_log_file_path) as f_in:
            log_file = json.load(f_in)

        if model_type == 'rnn':
            predictions = generate_suffixes_rnn(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)
        
        del log_file

        predictions['datetime'] = datetime
        predictions['log'] = log_directory.name

        with open(os.path.join(log_path, 'suffix_generation_result_' + str(datetime) + '.json'), 'w') as fp:
            json.dump(predictions, fp)

        del predictions


def evaluate_generation(datetime, model_type):
    path = os.path.join('results', model_type)

    # Walk through all the log-result directories:
    for log_directory in sorted(os.scandir(path), key=lambda x: x.name):
        if not os.path.isdir(log_directory): continue

        log_path = log_directory.path
        print('This is evaluation of: ' + log_path)
        with open(os.path.join(log_path, 'suffix_generation_result_' + str(datetime) + '.json')) as f_in:
            predictions = json.load(f_in)

        # in-place time attribute denormalisation:
        data_preprocessing.Evaluation.denormalise(predictions)

        results = {model_type: {}}
        results[model_type][predictions['log']] = {}

        average_damerau_levenshtein_similarity, average_mae, average_mae_denormalised, nb_worst_situs, nb_all_situs = iterate_over_generated_suffixes(predictions=predictions)

        results[model_type][predictions['log']]['dls'] = "{:.4f}".format(average_damerau_levenshtein_similarity)
        results[model_type][predictions['log']]['mae'] = "{:.4f}".format(average_mae)
        results[model_type][predictions['log']]['mae_denormalised'] = "{:.4f}".format(average_mae_denormalised)
        results[model_type][predictions['log']]['nb_worst_situs'] = nb_worst_situs
        results[model_type][predictions['log']]['nb_all_situs'] = nb_all_situs
        results[model_type][predictions['log']]['dls_per_prefix'] = predictions['dls']
        results[model_type][predictions['log']]['mae_per_prefix'] = predictions['mae']
        results[model_type][predictions['log']]['mae_denormalised_per_prefix'] = predictions['mae_denormalised']
        results[model_type][predictions['log']]['id_per_prefix'] = predictions['evaluated_ids']

        with open(os.path.join(log_path, 'suffix_evaluation_result_dls_mae_' + str(datetime) + '.json'), 'w') as fp:
            json.dump(results[model_type][predictions['log']], fp)

        del results
        del predictions

    # Merge evaluation results (per model type):
    suffix_evaluation_results = {model_type: {}}
    for log_directory in os.scandir(path):
        if not os.path.isdir(log_directory): continue

        log_path = log_directory.path
        with open(os.path.join(log_path, 'suffix_evaluation_result_dls_mae_' + str(datetime) + '.json')) as f_in:
            suffix_evaluation_result = json.load(f_in)

        suffix_evaluation_results[model_type][log_directory.name] = suffix_evaluation_result

    with open(os.path.join(path, 'suffix_evaluation_result_dls_mae_' + str(datetime) + '.json'), 'w') as fp:
        json.dump(suffix_evaluation_results, fp)

    del suffix_evaluation_results
    del suffix_evaluation_result


def main(args, dt_object):
    if not args.random:
        # RANDOM SEEDs:
        random_seed = args.random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        rng = np.random.default_rng(seed=random_seed)
        torch.backends.cudnn.deterministic = True
        random.seed(a=args.random_seed)
    
    model_type = 'rnn'
    datetime = '202411282107'
    vars(args)['datetime'] = datetime
    vars(args)['architecture'] = model_type
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)

if __name__ == '__main__':
    dt_object = datetime.datetime.now()
    print(dt_object)

    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', help='datetime', default=dt_object.strftime("%Y%m%d%H%M"), type=str)
    parser.add_argument('--hidden_dim', help='hidden state dimensions', default=128, type=int)
    parser.add_argument('--n_layers', help='number of layers', default=4, type=int)
    parser.add_argument('--n_heads', help='number of heads', default=4, type=int)
    parser.add_argument('--training_batch_size', help='number of training samples in mini-batch', default=512, type=int)
    parser.add_argument('--validation_batch_size', help='number of validation samples in mini-batch', default=512, type=int)
    parser.add_argument('--training_mlm_method', help='training MLM method', default='BERT', type=str)
    parser.add_argument('--validation_mlm_method', help='validation MLM method', default='fix_masks', type=str)  # we would like to end up with some non-stochastic & at least pseudo likelihood metric
    parser.add_argument('--mlm_masking_prob', help='mlm_masking_prob', default=0.15, type=float)
    parser.add_argument('--dropout_prob', help='dropout_prob', default=0.1, type=float)
    parser.add_argument('--validation_split', help='validation_split', default=0.2, type=float)
    parser.add_argument('--dataset', help='dataset', default='', type=str)
    parser.add_argument('--random_seed', help='random_seed', default=1982, type=int)
    parser.add_argument('--random', help='if random', default=False, type=bool)
    parser.add_argument('--gpu', help='gpu', default=0, type=int)
    parser.add_argument('--validation_indexes', help='list of validation_indexes NO SPACES BETWEEN ITEMS!', default='[0,1,4,10,15]', type=str)
    parser.add_argument('--ground_truth_p', help='ground_truth_p', default=0.0, type=float)
    parser.add_argument('--time_attribute_concatenated', help='time_attribute_concatenated', default=False, type=bool)
    parser.add_argument('--device', help='GPU or CPU', default='CPU', type=str)
    parser.add_argument('--pad_token', help='pad_token', default=0, type=int)
    parser.add_argument('--to_wrap_into_torch_dataset', help='to_wrap_into_torch_dataset', default=True, type=bool)
    parser.add_argument('--seq_ae_teacher_forcing_ratio', help='seq_ae_teacher_forcing_ratio', default=0.0, type=float)
    parser.add_argument('--single_position_target', help='single_position_target', default=False, type=bool)
    parser.add_argument('--bert_order', help='BERT inference order: random or l2r', default='random', type=str)
    parser.add_argument('--dls_evaluation', help='DLS', default=True, type=bool)
    parser.add_argument('--suffix_generation', help='suffix generation', default=True, type=bool)

    args = parser.parse_args()

    vars(args)['hostname'] = str(socket.gethostname())

    if args.device == 'GPU':
        torch.cuda.set_device(args.gpu)
        print('This is evaluation at gpu: ' + str(args.gpu))

    main(args, dt_object)

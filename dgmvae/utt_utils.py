from __future__ import print_function

from dgmvae import main as engine
from dgmvae.enc2dec.decoders import GEN, DecoderRNN, TEACH_FORCE
from dgmvae import utils
from collections import defaultdict, Counter
import logging
import numpy as np
import pickle
import torch
from sklearn import metrics
import os
import json
import math

logger = logging.getLogger()

def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    model.eval()
    old_batch_size = config.batch_size

    if num_batch != None:
        config.batch_size = 3

    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

    evaluator.initialize()
    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs, labels = model((batch, None), mode=GEN, gen_type=config.gen_type)  # todo: config.gen_type

        if DecoderRNN.KEY_LATENT in outputs:
            key_latent = outputs[DecoderRNN.KEY_LATENT]
            key_latent = key_latent.cpu().data.numpy()
        else:
            key_latent = None

        if DecoderRNN.KEY_CLASS in outputs:
            key_class = outputs[DecoderRNN.KEY_CLASS].cpu().data.numpy()
        else:
            key_class = None

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()
        # get attention if possible
        pred_attns = None

        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = engine.get_sent(model, de_tknize, pred_labels,
                                             b_id, attn=pred_attns)
            true_str, _ = engine.get_sent(model, de_tknize, true_labels, b_id)
            evaluator.add_example(true_str, pred_str)
            if dest_f is None:
                logger.info("Target: {}".format(true_str))
                logger.info("Predict: {}".format(pred_str))
                if key_latent is not None and key_class is not None:
                    key_latent_str = "-".join(map(str, key_latent[b_id]))
                    logger.info("Key Latent: {}\n".format(str(key_class[b_id]) + "\t" + key_latent_str))
                logger.info("\n")
            else:
                dest_f.write("Target: {}\n".format(true_str))
                dest_f.write("Predict: {}\n\n".format(pred_str))

    if dest_f is None:
        logging.info(evaluator.get_report(include_error=dest_f is not None))
    else:
        dest_f.write(evaluator.get_report(include_error=dest_f is not None))
    logger.info("Generation Done")
    return evaluator.get_report(include_error=dest_f is not None, get_value=True)

def dump_latent(model, data_feed, config, dest_f, num_batch=1):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Dumping: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    all_zs = []
    all_labels = []
    all_metas = []
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        metas = batch.metas
        # log_qy = results.log_qy.cpu().squeeze(0).data
        # y_ids = results.y_ids.cpu().data
        # dec_init = results.dec_init_state.cpu().squeeze().data

        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            all_labels.append(true_str)
            all_metas.append(metas[b_id])

        all_zs.append({k: v.cpu().squeeze().data.numpy() for k, v in results.items() if k in model.return_latent_key})
        # all_zs.append((log_qy.numpy(), dec_init.numpy(), y_ids.numpy()))

    pickle.dump({'z': all_zs, 'labels': all_labels, "metas": all_metas}, dest_f)
    logger.info("Dumping Done")

def sampling(model, selected_clusters, config, n_sample, dest_f):
    model.eval()
    de_tknize = utils.get_dekenize()

    selected_codes = [list(map(int, item['code'].split("-"))) for item in selected_clusters]
    # print(selected_codes)
    outputs = model.sampling(mode=GEN, gen_type=config.gen_type, selected_codes=selected_codes, n_sample=n_sample)
    # move from GPU to CPU
    pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
    pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)
    pred_attns = None

    code_samplings = [list() for _ in range(len(selected_codes))]
    for b_id in range(pred_labels.shape[0]):
        pred_str, attn = engine.get_sent(model, de_tknize, pred_labels,
                                         b_id, attn=pred_attns)
        code = b_id % len(selected_codes)
        code_samplings[code].append(pred_str)

    for code_idx, sents in enumerate(code_samplings):
        dest_f.write("Code: {}\n".format('-'.join(map(str, selected_codes[code_idx]))))
        for s in sents:
            dest_f.write("Sampling: {}\n".format(s))
        dest_f.write("\n")

    logger.info("Generation Done")
    pass

def exact_sampling(model, max_sampling_num, config, dest_f, sampling_batch_size=30):
    model.eval()
    # de_tknize = utils.get_dekenize()
    de_tknize = lambda w_list: " ".join(w_list)

    # batch_size = config.batch_size
    done_epoch = 0
    sampling_batch_size = sampling_batch_size
    while (done_epoch * sampling_batch_size < max_sampling_num):
        if (max_sampling_num - done_epoch * sampling_batch_size) > sampling_batch_size:
            batch_size = sampling_batch_size
        else:
            batch_size = max_sampling_num - done_epoch * sampling_batch_size
        # sample_y = torch.randint(0, config.k, [batch_size, config.mult_k], dtype=torch.long).cuda()
        # # print(sample_y)
        # # print(torch.arange(config.mult_k))
        # y_index = (model.torch2var(torch.arange(config.mult_k) * config.k) + sample_y).view(-1)
        # # sample_y = model.np2var(sample_y)
        # mean = model.gaussian_mus.view(-1, config.latent_size)[y_index].squeeze()
        # sigma = model.gaussian_vars.view(-1, config.latent_size)[y_index].squeeze()
        # zs = model.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        # zs = zs.view(-1, config.mult_k * config.latent_size)
        # cs = model.torch2var(idx2onehot(sample_y.view(-1), config.k)).view(-1, config.mult_k * config.k)
        # dec_init_state = model.dec_init_connector(torch.cat((cs, zs), dim=1)
        #                                          if config.feed_discrete_variable_into_decoder
        #                                          else zs)
        #
        # _, _, outputs = model.decoder(cs.size(0),
        #                            None, dec_init_state,
        #                            mode=GEN, gen_type=config.gen_type,
        #                            beam_size=config.beam_size)
        outputs = model.sampling(batch_size)
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)
        pred_attns = None

        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = engine.get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
            # print(pred_str)
            dest_f.write(pred_str + "\n")
        done_epoch+=1
        if done_epoch % 10 == 0:
            print("%d/%d epochs done." % (done_epoch, max_sampling_num//config.batch_size + 1))
        # print(done_epoch)
        # print("Generating %d/%d\n", )

    logger.info("Generation Done")



def cond_exact_sampling_lsebm(model, max_sampling_num, config, dest_fp, dest_fn, sampling_batch_size=30):
    model.eval()
    # de_tknize = utils.get_dekenize()
    de_tknize = lambda w_list: " ".join(w_list)

    # batch_size = config.batch_size
    done_epoch = 0
    sampling_batch_size = sampling_batch_size
    while (done_epoch * sampling_batch_size < max_sampling_num):
        if (max_sampling_num - done_epoch * sampling_batch_size) > sampling_batch_size:
            batch_size = sampling_batch_size
        else:
            batch_size = max_sampling_num - done_epoch * sampling_batch_size
        # sample_y = torch.randint(0, config.k, [batch_size, config.mult_k], dtype=torch.long).cuda()
        # # print(sample_y)
        # # print(torch.arange(config.mult_k))
        # y_index = (model.torch2var(torch.arange(config.mult_k) * config.k) + sample_y).view(-1)
        # # sample_y = model.np2var(sample_y)
        # mean = model.gaussian_mus.view(-1, config.latent_size)[y_index].squeeze()
        # sigma = model.gaussian_vars.view(-1, config.latent_size)[y_index].squeeze()
        # zs = model.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        # zs = zs.view(-1, config.mult_k * config.latent_size)
        # cs = model.torch2var(idx2onehot(sample_y.view(-1), config.k)).view(-1, config.mult_k * config.k)
        # dec_init_state = model.dec_init_connector(torch.cat((cs, zs), dim=1)
        #                                          if config.feed_discrete_variable_into_decoder
        #                                          else zs)
        #
        # _, _, outputs = model.decoder(cs.size(0),
        #                            None, dec_init_state,
        #                            mode=GEN, gen_type=config.gen_type,
        #                            beam_size=config.beam_size)

        batch_size_0 = batch_size // 2
        batch_size_1 = batch_size - batch_size_0
        cond_y = torch.tensor([0]*batch_size_0 + [1]*batch_size_1).long()
        outputs = model.cond_sampling(batch_size, cond_y)
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)
        pred_attns = None

        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = engine.get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
            if cond_y[b_id] == 1:
                dest_fp.write(pred_str + "\n")
            else:
                dest_fn.write(pred_str + "\n")
        done_epoch+=1
        if done_epoch % 10 == 0:
            print("%d/%d epochs done." % (done_epoch, max_sampling_num//config.batch_size + 1))
        # print(done_epoch)
        # print("Generating %d/%d\n", )

    logger.info("Generation Done")

def cond_exact_sampling(model, max_sampling_num, config, dest, sampling_batch_size=30):
    model.eval()
    # de_tknize = utils.get_dekenize()
    de_tknize = lambda w_list: " ".join(w_list)

    # batch_size = config.batch_size
    done_epoch = 0
    # sampling_batch_size = sampling_batch_size
    while (done_epoch  < config.num_cls):
        batch_size = sampling_batch_size
        # if (max_sampling_num - done_epoch * sampling_batch_size) > sampling_batch_size:
        #     batch_size = sampling_batch_size
        # else:
        #     batch_size = max_sampling_num - done_epoch * sampling_batch_size
        # sample_y = torch.randint(0, config.k, [batch_size, config.mult_k], dtype=torch.long).cuda()
        # # print(sample_y)
        # # print(torch.arange(config.mult_k))
        # y_index = (model.torch2var(torch.arange(config.mult_k) * config.k) + sample_y).view(-1)
        # # sample_y = model.np2var(sample_y)
        # mean = model.gaussian_mus.view(-1, config.latent_size)[y_index].squeeze()
        # sigma = model.gaussian_vars.view(-1, config.latent_size)[y_index].squeeze()
        # zs = model.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        # zs = zs.view(-1, config.mult_k * config.latent_size)
        # cs = model.torch2var(idx2onehot(sample_y.view(-1), config.k)).view(-1, config.mult_k * config.k)
        # dec_init_state = model.dec_init_connector(torch.cat((cs, zs), dim=1)
        #                                          if config.feed_discrete_variable_into_decoder
        #                                          else zs)
        #
        # _, _, outputs = model.decoder(cs.size(0),
        #                            None, dec_init_state,
        #                            mode=GEN, gen_type=config.gen_type,
        #                            beam_size=config.beam_size)

        cond_y = torch.ones(batch_size).long() * done_epoch
        outputs = model.cond_sampling(batch_size, cond_y)
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)
        pred_attns = None

        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = engine.get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
            if b_id == 0:
                dest.write("\n\n---------> {:d} <--------- \n".format(done_epoch))
            dest.write(pred_str + "\n")
        done_epoch+=1
        if done_epoch % 10 == 0:
            print("%d/%d epochs done." % (done_epoch, max_sampling_num//config.batch_size + 1))
        # print(done_epoch)
        # print("Generating %d/%d\n", )

    logger.info("Generation Done")

def calculate_likelihood(model, data_feed, max_sampling_num, config,
                         every_time_sampling_num=100, sample_type="LL"):

    model.eval()
    original_batch_size = config.batch_size
    config.batch_size = 1
    data_feed.epoch_init(config, shuffle=False, verbose=False)
    config.batch_size = original_batch_size

    ll_collect = []
    elbo_collect = []
    tot_nll = 0.0
    tot_w = 0.0
    done = 0
    while True:
        batch = data_feed.next_batch()
        done += 1
        if done % 200 == 0:
            print(done, "done.", (tot_nll / tot_w))
        if batch is None:
            break

        ll_sum = []
        logll_sum = []
        for split_num in range(max_sampling_num // every_time_sampling_num): # 5:  0,1,2,3,4
            nll  = model.sampling_for_likelihood(config.batch_size, batch, every_time_sampling_num,
                                                     sample_type=sample_type)

            if sample_type == 'logLL':
                ave_logLL = torch.mean(nll, dim=-1).item()
                ll_sum.append(torch.mean(torch.exp(nll), dim=-1).item())
                logll_sum.append(ave_logLL)
            else:
                ll_sum.append(torch.mean(nll, dim=-1).item())

        ll_ave = sum(ll_sum) / len(ll_sum)
        if sample_type == 'logLL':
            logll_ave = sum(logll_sum) / len(logll_sum)

        if ll_ave > 0.0:
            tot_nll += math.log(ll_ave)
        elif sample_type == "logLL":
            tot_nll += logll_ave
        tot_w += 1

    logger.info("Negative Log-likehood %lf" % (tot_nll / tot_w))

    return tot_nll / tot_w

def latent_cluster(model, data_feed, config, num_batch=1, max_samples=5, exclude_sents=None):
    # mult_k = config.mult_k if config.gmm else config.latent_size
    print('%d clusters in totoal.' % np.power(config.k, config.mult_k))
    # if np.power(config.k, mult_k) > 1000000:
    #     logger.info("Skip latent cluster too many states")
    #     return
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Find cluster for: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))

    all_clusters = defaultdict(list)
    cond_y_matrix = np.zeros((config.k, config.k))

    def write(msg):
        logger.info(msg)

    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        metas = batch.metas
        if results.log_qy.size(0) != config.batch_size:
            log_qy = results.log_qy.view(-1, config.mult_k, config.k)
        else:
            log_qy = results.log_qy
        #     log_qy = results.log_qy.unsqueeze(1)
        qy = torch.exp(log_qy)
        qy = qy.cpu().data.numpy()
        
        # if config.gmm:
        #     iter_latent_size = config.mult_k
        # else:
        #     iter_latent_size = config.latent_size
        
        y_ids = results.y_ids.cpu().data.numpy()
        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            cond_y_matrix[y_ids[b_id]] += 1
            code = []
            for y_id in range(config.mult_k):
                for k_id in range(config.k):
                    # print(qy.shape)
                    # print(labels.shape[0])
                    # print(iter_latent_size)
                    # print(config.k)
                    if qy[b_id, y_id, k_id] == np.max(qy[b_id, y_id]):
                        code.append(str(k_id))
                        break
            code = '-'.join(code)
            all_clusters[code].append((true_str, metas[b_id]))

    # show clusters
    keys = all_clusters.keys()
    keys = sorted(keys)
    logger.info("Find {} clusters".format(len(keys)))

    selected_clusters = []
    for symbol in keys:
        sents = all_clusters[symbol]
        if exclude_sents is not None:
            if symbol not in exclude_sents:
                continue
            sents = [s for s in sents if s[0] not in exclude_sents[symbol]['examples']]
        if len(sents) < 5:
            # write("Skip tiny cluster with {} utts - {}".format(len(sents), symbol))
            continue
        write("Symbol {}".format(symbol))
        if len(sents) < max_samples:
            print("Find small cluster with {} utts".format(len(sents)))
            subset_ids = range(len(sents))
            np.random.shuffle(subset_ids)
        else:
            subset_ids = np.random.choice(range(len(sents)), max_samples, replace=False)
        for s_id in subset_ids[0:5]:
            write(sents[s_id][0])
        write("")
        if exclude_sents is not None:
            selected_clusters.append({'code': symbol, 'meaning': exclude_sents[symbol]['meaning'], 'score': "      ",
                                      'examples': [sents[idx][0] for idx in subset_ids]})
        else:
            selected_clusters.append({'code': symbol, 'meaning': '',
                                      'examples': [sents[idx][0] for idx in subset_ids]})


    return selected_clusters


def latent_cluster_lsebm(model, data_feed, config, num_batch=1, max_samples=5, exclude_sents=None):
    print('%d clusters in totoal.' % config.num_cls)
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Find cluster for: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))

    all_clusters = defaultdict(list)
    # cond_y_matrix = np.zeros((config.k, config.k))

    def write(msg):
        logger.info(msg)

    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        metas = batch.metas
        # if results.log_qy.size(0) != config.batch_size:
        #     log_qy = results.log_qy.view(-1, config.num_cls)
        # else:
        #     log_qy = results.log_qy
        assert results.log_qy.size(0) == config.batch_size
        log_qy = results.log_qy.view(-1, config.num_cls)

        qy = torch.exp(log_qy)
        qy = qy.cpu().data.numpy()
                
        y_ids = results.y_ids.cpu().data.numpy()
        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            # cond_y_matrix[y_ids[b_id]] += 1
            # code = []
            # for y_id in range(config.mult_k):
            #     for k_id in range(config.k):
            #         if qy[b_id, y_id, k_id] == np.max(qy[b_id, y_id]):
            #             code.append(str(k_id))
            #             break
            # code = '-'.join(code)
            code = '{:0>3d}'.format(int(y_ids[b_id]))
            all_clusters[code].append((true_str, metas[b_id]))

    # show clusters
    keys = all_clusters.keys()
    keys = sorted(keys)
    logger.info("Find {} clusters".format(len(keys)))

    selected_clusters = []
    for symbol in keys:
        sents = all_clusters[symbol]
        if exclude_sents is not None:
            if symbol not in exclude_sents:
                continue
            sents = [s for s in sents if s[0] not in exclude_sents[symbol]['examples']]
        if len(sents) < 5:
            continue
        write("Symbol {}".format(symbol))
        if len(sents) < max_samples:
            print("Find small cluster with {} utts".format(len(sents)))
            subset_ids = range(len(sents))
            np.random.shuffle(subset_ids)
        else:
            subset_ids = np.random.choice(range(len(sents)), max_samples, replace=False)
        for s_id in subset_ids[0:5]:
            write(sents[s_id][0])
        write("")
        if exclude_sents is not None:
            selected_clusters.append({'code': symbol, 'meaning': exclude_sents[symbol]['meaning'], 'score': "      ",
                                      'examples': [sents[idx][0] for idx in subset_ids]})
        else:
            selected_clusters.append({'code': symbol, 'meaning': '',
                                      'examples': [sents[idx][0] for idx in subset_ids]})


    return selected_clusters

def find_mi(model, data_feed, config, seperate=False):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Find MI for: {} batches".format(data_feed.num_batch))

    all_codes = []
    all_metas = []
    all_labels = []
    meta_keys = set()
    def write(msg):
        logger.info(msg)

    def code2id(code, base):
        idx = 0
        for c_id, c in enumerate(code):
            idx += int(c) * np.power(base, c_id)
        return idx

    iter_latent_size = config.mult_k
    # if config.gmm:
    #     iter_latent_size = config.mult_k
    # else:
    #     iter_latent_size = config.latent_size


    while True:
        batch = data_feed.next_batch()
        if batch is None:
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        metas = batch.metas

        z_labels = batch.get("z_labels", None)
        all_labels.append(z_labels)

        for key in metas[0].keys():
            meta_keys.add(key)
        log_qy = results.log_qy.view(-1, config.mult_k, config.latent_size if ("bmm" in config and config.bmm) else config.k)
        qy = torch.exp(log_qy)
        qy = qy.cpu().data.numpy()
        y_ids = results.y_ids.cpu().data.numpy()
        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            code = []
            for y_id in range(log_qy.size(1)):
                for k_id in range(log_qy.size(2)):
                    if qy[b_id, y_id, k_id] == np.max(qy[b_id, y_id]):
                        code.append(str(k_id))
                        break
            all_codes.append(code)
            # all_codes.append(y_ids[b_id])
            all_metas.append(metas[b_id])

    vec_codes = np.array(all_codes).transpose(0, 1)
    vec_idxes = [code2id(c, config.k) for c in vec_codes]
    vec_vocabs = list(set(vec_idxes))
    vec_idxes = [vec_vocabs.index(v) for v in vec_idxes]
    all_labels = np.concatenate(all_labels, axis=0)

    for key in meta_keys:
        # get all meta about this key
        meta_vals = []
        for m in all_metas:
            if type(m[key]) is list:
                meta_vals.append(" ".join(map(str, m[key])))
            elif type(m[key]) is dict:
                break
            else:
                meta_vals.append(m[key])
        if not meta_vals:
            continue
        meta_vocab = list(set(meta_vals))
        meta_vals = [meta_vocab.index(v) for v in meta_vals]

        mi = metrics.homogeneity_score(meta_vals, vec_idxes)
        write("{} mi with ID is {}".format(key, mi))

        # individual dimension
        for y_id in range(config.mult_k):
            mi = metrics.homogeneity_score(meta_vals, vec_codes[:, y_id])
            write("{} mi with dim {} is {}".format(key, y_id, mi))

    vec_codes = vec_codes.astype(np.int)
    for y_id in range(all_labels.shape[1]):
        acc = np.mean(all_labels[:, y_id] == vec_codes[:, y_id])
        write("{} acc with dim {} is {}".format(["emotion", "act"][y_id], y_id, acc))

def selective_generate(model, data_feed, config, selected_clusters):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, shuffle=False, verbose=False)
    # get all code
    codes = set([d['code'] for d in selected_clusters])

    logger.info("Generation: {} batches".format(data_feed.num_batch))
    data = []
    total_cnt = 0.0
    in_cnt = 0.0

    while True:
        batch = data_feed.next_batch()
        if batch is None:
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)
        # move from GPU to CPU
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.cpu().data.numpy()
        y_ids = outputs[DecoderRNN.KEY_LATENT].cpu().data.numpy()
        if config.gmm:
            pass
        else:
            y_ids = y_ids.reshape(-1, config.latent_size)

        for b_id in range(pred_labels.shape[0]):
            y_id = map(str, y_ids[b_id])
            code = '-'.join(y_id)
            total_cnt +=1
            if code in codes:
                pred_str, attn = engine.get_sent(model, de_tknize, pred_labels,
                                                 b_id, attn=None)
                true_str, _ = engine.get_sent(model, de_tknize, true_labels, b_id)
                in_cnt +=1
                data.append({'target': true_str,
                             'predict': pred_str, 'code': code})

    logger.info("In rate {}".format(in_cnt/total_cnt))
    return data

def draw_pics(model, data_feed, config, epoch, num_batch=10, add_text = False, shuffle=False):

    # sampling some z:
    model.eval()
    data_feed.epoch_init(config, verbose=False, shuffle=shuffle)
    num_batch = min(num_batch, data_feed.num_batch)
    logger.info("Draw pics: {} batches".format(num_batch))

    all_zs = []
    all_ys = []
    all_text = []  # the original sentences.
    while True:
        batch = data_feed.next_batch()
        if batch is None or (data_feed.ptr > num_batch):
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)
        all_ys.append(results['y_ids'])
        all_zs.append(results['z'])
        if add_text:
            all_text.extend([meta["text"] for meta in batch.metas])


    all_zs = torch.cat(all_zs, dim=0).view(-1, config.mult_k, config.latent_size).data.cpu().numpy()
    all_ys = torch.cat(all_ys, dim=0).view(-1, config.mult_k).data.cpu().numpy()
    all_zs = np.transpose(all_zs, (1, 0, 2))
    all_ys = np.transpose(all_ys, (1, 0))
    print('all sample:', all_zs.shape)
    print("all ys:", all_ys.shape)
    print("all text", len(all_text))
    assert all_zs.shape[1] == all_ys.shape[1]
    assert all_zs.shape[0] == all_ys.shape[0]
    # draw pics:
    if not os.path.exists(os.path.join(config.fig_dir, config.time_stamp)):
        os.makedirs(os.path.join(config.fig_dir, config.time_stamp))



    for i in range(config.mult_k):
        X = all_zs[i,:,:]
        y = all_ys[i, :]
        # print(X.shape)

        fout_test_clustering = open(os.path.join(config.fig_dir, config.time_stamp, "tsne-S%i-B%i.txt" % (i, epoch)), "w")

        if config.tsne:
            # X = all_zs
            Means = model.gaussian_mus.data.cpu().numpy()[i, :, :]
            X_tsne, Means_tsne = utils.tsne(X, Means, y, os.path.join(config.fig_dir, config.time_stamp, "tsne-S%i-B%i.png" % (i, epoch)),
                       text_label=all_text if add_text else None)
            # Dumping points aftering TSNE
            json.dump({'text': all_text, 'x': X_tsne.tolist(), 'y': y.tolist(), 'Means': Means_tsne.tolist()}, open(os.path.join(config.fig_dir, config.time_stamp, "tsne-S%i-B%i.json" % (i, epoch)), "w"))

            # Check Clustering:
            cluster2text = defaultdict(list)
            for it in range(len(all_text)):
                cluster2text[y[it]].append(all_text[it])
            for c in cluster2text:
                fout_test_clustering.write("Cluster " + str(c) + "\n")
                for t in cluster2text[c]:
                    fout_test_clustering.write(t + "\n")
                fout_test_clustering.write("\n")

        else:
            # X = all_zs
            sns.set_context("talk", font_scale=1.0,
                            rc={"xtick.labelsize": 0, "axes.linewidth": 2, "ytick.labelsize": 0,
                                })
            sns.set_style("ticks", {"ytick.direction": "in", # "axes.grid": True, "grid.color": '1.0'
                                    "ytick.color": "0.35",
                                    "xtick.direction": "in", "xtick.color": "0.35"})
            plt.rcParams['ytick.major.width'] = plt.rcParams['ytick.minor.width']
            plt.rcParams['ytick.major.size'] = plt.rcParams['ytick.minor.size']

            plt.rcParams['xtick.major.width'] = plt.rcParams['xtick.minor.width']
            plt.rcParams['xtick.major.size'] = plt.rcParams['xtick.minor.size']
            plt.rcParams['xtick.minor.bottom'] = False

            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True

            # print(plt.rcParams)

            df_datapoints = pd.DataFrame(data={'x': X[:, 0], 'y': X[:, 1], 'label': y})
            # g = sns.FacetGrid(tips, col="time", row="smoker")
            g = sns.lmplot(x='x', y='y', data=df_datapoints, fit_reg=False, legend=False,
                           hue='label', )

            g.set_xlabels("")
            g.set_ylabels("")
            g.despine(**{'top': False, 'right': False, 'left': False, 'bottom': False})

            # g.set_xticks(np.arange(-4, 4, 0.2), minor=True)
            # g.set_xticks(np.arange(-4, 4, 0.4), minor=False)

            ax = g.facet_axis(0, 0)
            # center points
            ax.scatter(model.gaussian_mus.data.cpu().numpy()[i, :, 0], model.gaussian_mus.data.cpu().numpy()[i, :, 1],
                       color="DimGray")

            for ii in range(model.gaussian_mus.size(1)):
                polygon = matplotlib.patches.RegularPolygon((model.gaussian_mus.data.cpu().numpy()[i, ii, 0],
                                                             model.gaussian_mus.data.cpu().numpy()[i, ii, 1]),
                                                            3, 10, color="DimGray") #
                # patches.append(polygon)
                # label(grid[3], "Polygon")


            # vars circles
            for ii in range(model.gaussian_mus.size(1)):
                if hasattr(model, "gaussian_vars"):
                    gaussian_vars = model.gaussian_vars
                elif hasattr(model, "gaussian_logvar"):
                    gaussian_vars = torch.exp(model.gaussian_logvar * 0.5)

                elps = matplotlib.patches.Ellipse(model.gaussian_mus.data.cpu().numpy()[i, ii, :],
                                                  gaussian_vars.data.cpu().numpy()[i, ii, 0],
                                                  gaussian_vars.data.cpu().numpy()[i, ii, 1],
                                                  0, color='silver')
                # print(model.gaussian_vars.data.cpu().numpy()[i, ii, 0])
                # print(model.gaussian_vars.data.cpu().numpy()[i, ii, 1])

                elps.set_zorder(0)
                elps.set_clip_box(ax.bbox)
                elps.set_alpha(0.5)
                ax.add_artist(elps)
                if add_text:
                    plt.text(model.gaussian_mus.data.cpu().numpy()[i, ii, 0], model.gaussian_mus.data.cpu().numpy()[i, ii, 1],
                             ii, color='black', fontsize=10)

            # g.xlim(-3, +3)
            # g.ylim(-3, +3)

            # fig = g.get_figure()
            # fig.savefig(os.path.join(config.fig_dir, config.time_stamp, "S%i-B%i.png" % (i, epoch)), format='pdf', bbox_inches='tight')
            g.savefig(os.path.join(config.fig_dir, config.time_stamp, "S%i-B%i.png" % (i, epoch)),
                      )  # bbox_inches='tight'
            # g.savefig(os.path.join(config.fig_dir, config.time_stamp, "S%i-B%i.pdf" % (i, epoch)), format='pdf', ) # bbox_inches='tight'
            json.dump({'x': X.tolist(), 'y': y.tolist(), 'means': model.gaussian_mus.data.cpu().numpy()[i, :, :].tolist(),
                       'vars': gaussian_vars.data.cpu().numpy()[i, :, :].tolist()},
                      open(os.path.join(config.fig_dir, config.time_stamp, "tsne-S%i-B%i.json" % (i, epoch)), "w"))


    # all space:
    return

    X = np.concatenate(all_zs, axis=1)
    # y = np.concatenate(all_zs, axis=0)
    y = all_ys[0, :]  # 随便弄一个。。这个肯定是有问题的。。
    # print(X.shape)
    # y = np.zeros(y.shape[1])
    for k in range(1, all_ys.shape[0]):
        y += (all_ys[k, :] * (config.k ** (k)))
    # print(y)
    # exit()

    fout_test_clustering = open(os.path.join(config.fig_dir, config.time_stamp, "tsne-A-B%i.txt" % (epoch)), "w")

    if config.tsne:
        # X = all_zs
        Means = np.concatenate(model.gaussian_mus.data.cpu().numpy(), axis=1)
        X_tsne, Means_tsne = utils.tsne(X, Means, y, os.path.join(config.fig_dir, config.time_stamp,
                                                                  "tsne-A-B%i.png" % (epoch)),
                                        text_label=all_text if add_text else None)
        # Dumping points aftering TSNE
        json.dump({'text': all_text, 'x': X_tsne.tolist(), 'y': y.tolist(), 'Means': Means_tsne.tolist()},
                  open(os.path.join(config.fig_dir, config.time_stamp, "tsne-A-B%i.json" % (epoch)), "w"))

        # # Check Clustering:
        # cluster2text = defaultdict(list)
        # for it in range(len(all_text)):
        #     cluster2text[y[it]].append(all_text[it])
        # for c in cluster2text:
        #     fout_test_clustering.write("Cluster " + str(c) + "\n")
        #     for t in cluster2text[c]:
        #         fout_test_clustering.write(t + "\n")
        #     fout_test_clustering.write("\n")
    else:
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def init_parser(alg):

    if alg == 'WOLP_DDPG':

        parser = argparse.ArgumentParser(description='WOLP_DDPG')

        parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
        parser.add_argument('--gpu-ids', type=int, default=[1], nargs='+', help='GPUs to use [-1 CPU only]')
        parser.add_argument('--gpu-nums', type=int, default=1, help='#GPUs to use (default: 1)')
        parser.add_argument('--id', default='0', type=str, help='experiment id')

        # train set
        parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
        parser.add_argument('--max-episode', type=int, default=500, help='maximum #episode.')

        # test set
        # parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
        parser.add_argument('--test-episode', type=int, default=10, help='maximum testing #episode.')
        parser.add_argument('--load-model-dir', default='CacheContent_window-run1', metavar='LMD', help='folder to load window trained models from')
        # parser.add_argument('--load-model-dir', default='CacheContent-run31', metavar='LMD', help='folder to load baseline trained models from')

        # environment set
        # parser.add_argument('--env', default='CacheContent', type=str, help='Ride sharing')
        parser.add_argument('--env', default='CacheContent_window', type=str, help='Ride sharing')
        # parser.add_argument('--window_size', default=1, type=int, help='window size')
        parser.add_argument('--window_size', default="random", type=str, help='window size')

        # cache size set
        parser.add_argument('--max-actions', default=151, type=int, help='# max actions(cache size + 1)')
        parser.add_argument('--cache_capacity', default=150, type=int, help='cache capacity at base station')

        # zipf set
        parser.add_argument('--zipf_a', default=1.3, type=float, help='zipf parameter a')
        parser.add_argument('--entire_request_nb', default=2000000, type=int, help='generate entire request number')
        parser.add_argument('--split_nb', default=1000, type=int, help='total request split n part')
        parser.add_argument('--sub_request_len', default=2000, type=int, help='sub request list length')
        parser.add_argument('--max-episode-length', type=int, default=2000, metavar='M', help='maximum length of an episode (default: 20000)')

        parser.add_argument('--hidden1', default=256, type=int, help='hidden num of first fully connect layer')
        parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')
        parser.add_argument('--c-lr', default=0.001, type=float, help='critic net learning rate')
        parser.add_argument('--p-lr', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
        parser.add_argument('--warmup', default=128, type=int, help='time without training but only filling the replay memory')
        parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
        parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
        parser.add_argument('--window_length', default=1, type=int, help='')
        parser.add_argument('--tau-update', default=0.001, type=float, help='moving average for target network')
        parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
        parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
        parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
        parser.add_argument('--init_w', default=0.003, type=float, help='')
        parser.add_argument('--epsilon', default=80000, type=int, help='Linear decay of exploration policy')
        parser.add_argument('--seed', default=-1, type=int, help='')
        parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay for L2 Regularization loss')

        return parser

    else:

        raise RuntimeError('undefined algorithm {}'.format(alg))
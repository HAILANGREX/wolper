#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
# from train_test import train, test
import train_test as baseline
import train_test_window as window
import warnings
from arg_parser import init_parser
from setproctitle import setproctitle as ptitle
from normalized_env import NormalizedEnv
import gym
import cache_env as cache_env

if __name__ == "__main__":
    ptitle('test_wolp')
    warnings.filterwarnings('ignore')
    parser = init_parser('WOLP_DDPG')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]

    from util import get_output_folder, setup_logger
    from wolp_agent import WolpertingerAgent

    args.save_model_dir = get_output_folder('output', args.env)

    env = cache_env.CacheEnv(args.cache_capacity)
    # continuous = None
    continuous = False
    try:
        # continuous action
        nb_states = env.observation_space.shape[0]
        nb_actions = env.action_space.shape[0]
        action_high = env.action_space.high
        action_low = env.action_space.low
        continuous = True
        env = NormalizedEnv(env)
    except IndexError:
        # discrete action for 1 dimension
        nb_states = env.observation_space.shape[0]  # state dimension
        nb_actions = 1  # action dimension
        max_actions = env.action_space.n
        continuous = False

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    if continuous:
        agent_args = {
            'continuous':continuous,
            'max_actions':None,
            'action_low': action_low,
            'action_high': action_high,
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'args': args,
        }
    else:
        agent_args = {
            'continuous':continuous,
            'max_actions':max_actions,
            'action_low': None,
            'action_high': None,
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'args': args,
        }

    agent = WolpertingerAgent(**agent_args)

    # if args.load:
    #     agent.load_weights(args.load_model_dir)

    if args.gpu_ids[0] >= 0 and args.gpu_nums > 0:
         agent.cuda_convert()

    # set logger, log args here
    log = {}
    if args.mode == 'train':
        setup_logger('RS_log', r'{}/RS_train_log'.format(args.save_model_dir))
    elif args.mode == 'test':
        setup_logger('RS_log', r'{}/RS_test_log'.format(args.save_model_dir))
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
    log['RS_log'] = logging.getLogger('RS_log')  # singleton, return the cite of the same logger object
    d_args = vars(args)
    for k in d_args.keys():
        log['RS_log'].info('{0}: {1}'.format(k, d_args[k]))

    if args.mode == 'train':

        if args.env == "CacheContent_window":  # window experiment set
            train_args = {
                'continuous': continuous,
                'env': env,
                'agent': agent,
                'max_episode': args.max_episode,
                'warmup': args.warmup,
                'save_model_dir': args.save_model_dir,
                # 'max_episode_length': 20,
                'max_episode_length': args.max_episode_length,
                'logger': log['RS_log'],
                'WINDOW': args.window_size,
                'zipf_a': args.zipf_a,
                'entire_request_nb': args.entire_request_nb,
                'split_nb': args.split_nb,
                'sub_request_len': args.sub_request_len,
            }

            # print('window experiment!')
            log['RS_log'].info('window train experiment!')
            window.train(**train_args)
        else:
            train_args = {
                'continuous': continuous,
                'env': env,
                'agent': agent,
                'max_episode': args.max_episode,
                'warmup': args.warmup,
                'save_model_dir': args.save_model_dir,
                'max_episode_length': args.max_episode_length,
                'logger': log['RS_log'],
                'zipf_a': args.zipf_a,
                'entire_request_nb': args.entire_request_nb,
                'split_nb': args.split_nb,
                'sub_request_len': args.sub_request_len,
            }

            # print('baseline experiment!')
            log['RS_log'].info('baseline train experiment!')
            baseline.train(**train_args)

    elif args.mode == 'test':

        if args.env == "CacheContent_window":  # window experiment set
            test_args = {
                'env': env,
                'agent': agent,
                'model_path': args.load_model_dir,
                'test_episode': args.test_episode,
                'max_episode_length': args.max_episode_length,
                # 'max_episode_length': 50,
                'logger': log['RS_log'],
                'WINDOW': args.window_size,
                'zipf_a': args.zipf_a,
                'entire_request_nb': args.entire_request_nb,
                'split_nb': args.split_nb,
                'sub_request_len': args.sub_request_len,
            }
            log['RS_log'].info('window test experiment!')
            window.test(**test_args)
        else:
            test_args = {
                'env':env,
                'agent': agent,
                'model_path': args.load_model_dir,
                'test_episode':args.test_episode,
                'max_episode_length': args.max_episode_length,
                'logger': log['RS_log'],
                'zipf_a': args.zipf_a,
                'entire_request_nb': args.entire_request_nb,
                'split_nb': args.split_nb,
                'sub_request_len': args.sub_request_len,
            }
            log['RS_log'].info('baseline test experiment!')
            baseline.test(**test_args)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))

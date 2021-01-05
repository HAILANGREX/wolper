#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import random
import numpy as np
from gensim.models import word2vec


def loda_T_request():
    # return all T request list
    T_request = []
    # file = './data/ba_1000_2_5_10000.walks'  # entire simulated data
    file = './data/test_01.txt'  # entire simulated data
    # file = 'data/test.walks'
    # file = './data/test_few.walks'
    # file = './data/test_half.walks'
    with open(file, 'r') as f:
        for l in f:
            # print('l:', l)  # str
            # print('type_l:', type(l))
            l = l.strip()
            for ele in l.split(' '):
                # print('ele:', ele)
                T_request.append(int(ele))
    return T_request

def load_W2Vmodel():
    return word2vec.Word2Vec.load('./data/test_01.model')

def split_T_request(T_request, request_length):
    """
    used for entire data, every split is one episode, request_length equal max episode length
    :param T_request: all simulated data
    :param request_length: max episode length
    :return: all split list [[],[],...]
    """
    all_split_list = []
    split_numbers = int(math.ceil(len(T_request)/request_length))
    start = 0
    for _ in range(split_numbers):
        if start+request_length <= len(T_request):
            all_split_list.append(copy.deepcopy(T_request[start:start+request_length]))
            start = start + request_length
        else:
            all_split_list.append(copy.deepcopy(T_request[start:]))
            break
    return all_split_list

def zipf_data(zipf_a, entire_request_nb, split_nb, sub_request_len):
    """
    generate zipf test data
    :param a: zipf parameter float type, should large than 1
    :param entire_request_nb: total request
    :param split_nb: the number of split, every split is a episode
    :param sub_request_len: sub request number
    :return: all_split_list [[],[],...]
    """
    # Samples are drawn from a Zipf distribution with specified parameter a > 1
    s = list(map(int, np.random.zipf(zipf_a, entire_request_nb)))
    start = 0
    all_split_list = []
    for _ in range(split_nb):
        if start+sub_request_len <= entire_request_nb:
            all_split_list.append(copy.deepcopy(s[start:start+sub_request_len]))
            start = start + sub_request_len
        else:
            all_split_list.append(copy.deepcopy(s[start:]))
            break
    return all_split_list

def train(continuous, env, agent, max_episode, warmup, save_model_dir, max_episode_length, logger, WINDOW, zipf_a, entire_request_nb, split_nb, sub_request_len):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None

    # load T request
    T_request = loda_T_request()  # load all data
    all_split_request_list = split_T_request(T_request, max_episode_length)


    # zipf_args = {
    #     'zipf_a': zipf_a,
    #     'entire_request_nb': entire_request_nb,
    #     'split_nb': split_nb,
    #     'sub_request_len': sub_request_len
    # }
    # all_split_request_list = zipf_data(**zipf_args)  # load zipf data

    md = load_W2Vmodel()

    # print('all_split_request_list number:', len(all_split_request_list))
    # env.T_request = T_request
    # WINDOW = 100

    # print('T_request numbers:', len(T_request))

    # while episode < max_episode:
    while episode < len(all_split_request_list):  # every split is a episode

        env.T_request = all_split_request_list[episode]  # get list ith sub request list
        episode_t_request = all_split_request_list[episode]  # get list ith sub request list

        # print('episode_t_request:', len(episode_t_request))
        # print('episode_t_request:', episode_t_request)

        cache_hit_count = 0
        cache_hit_rate = 0
        env.reset()  # every episode reset env
        done = False
        WINDOW = 100
        num = 5
        if WINDOW == "random":
            window = random.randint(1,1000)  # random window
        else:
            window = WINDOW  # after N request execute action(update cache), every episode reset window
        action_list = []  # save action
        reward_list = []  # save reward, execute highest reward action

        # each episode(for t=1 to T)
        while True:

            window = window - 1  # N request execute one action
            env.t = episode_steps  # current request index
            current_request_id = episode_t_request[episode_steps]  # current request !!!!! not T_request!!!!!
            env.request_list.append(current_request_id)  # save request

            # if Requested content is already cached then Update cache hit rate and end decision epoch
            if current_request_id in env.cache_list:
                # update cache hit rate and end decision epoch
                cache_hit_count += 1
                # cache_hit_rate = cache_hit_count / episode_steps
                # print('cache hit rate1:', cache_hit_rate)

            # Requested content is not cached and Cache storage is not full. End decision epoch
            elif len(env.cache_list) < env.C:
                # Cache the currently requested content. Update cache state and cache hit rate
                env.cache_list.append(current_request_id)
                env.vector_list.append(md[str(current_request_id)])
                # if episode_steps != 0:
                #     cache_hit_rate = cache_hit_count / episode_steps
                # print('cache hit rate2:', cache_hit_rate)

            # Requested content is not cache and Cache storage is full
            else:
                # need execute action
                if s_t is None:
                    s_t = env.generate_state()  # according to cache(request list and cache list) get init state
                    agent.reset(s_t)  # each episode reset agent

                env.current_content_id = current_request_id

                # cache_hit_rate = cache_hit_count / episode_steps
                # print('cache hit rate3:', cache_hit_rate)

                # agent pick action ...
                # args.warmup: time without training but only filling the memory
                if step <= warmup:  # step will not reset to 0
                    action = agent.random_action()
                else:
                    action = agent.select_action(s_t)
                    # print("pre action:", action)

                # env response with next_observation, reward, terminate_info
                if not continuous:
                    # action before reshape: [[1]]  action after reshape: 1
                    action = action.reshape(1,).astype(int)[0]
                    action = action - 1
                    # print('action:', action)

                # for window experiment
                action_list.append(action)
                # r_t = env.get_simulated_reward(action)  # get simulated reward
                # reward_list.append(r_t)
                s_t1 = env.generate_state()  # cache don't update but state update, however, state change small

                # execute aciton
                if window <= 0:  # N request

                    # print("action list:", action_list)
                    # random choice
                    action = random.choice(action_list)
                    # fetch the max reward action
                    # action = action_list[reward_list.index(max(reward_list))]
                    # print('len_action_list:', len(action_list))
                    # print('execute action——————————————————————————————:', action)
                    s_t1, r_t, done = env.step(action, md, num)

                    # reset window var
                    if WINDOW == "random":
                        window = random.randint(1, 1000)
                        # print("window{0}".format(window))
                    else:
                        window = WINDOW
                    action_list = []
                    reward_list = []

                    # agent observe and update policy
                    agent.observe(r_t, s_t1, done)
                    if step > warmup:
                        agent.update_policy()

                    # update
                    step += 1  # only execute action step add 1
                    episode_reward += r_t

                s_t = s_t1

            # update
            episode_steps += 1


            if done or episode_steps >= len(episode_t_request):  # end of an episode(use = because episode+1)


                # output cache hit rate
                cache_hit_rate = cache_hit_count / episode_steps
                # print('episode final cache hit rate------------------------------:', cache_hit_rate)

                logger.info(
                    "episode {0} final cache hit rate------------------------------:{1}".format(episode, cache_hit_rate)
                )

                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(episode, episode_reward)
                )

                # print('s_t:', s_t)

                if s_t is not None:  # every request is in cache, no action execute
                    agent.memory.append(
                        s_t,
                        agent.select_action(s_t),
                        0., True
                    )

                # current episode end and reset
                s_t = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1

                # break to next episode
                break

        # [optional] save intermediate model every run through of 2 episodes
        if step > warmup and episode > 0 and episode % 2 == 0:
            agent.save_model(save_model_dir)
            logger.info("### Model Saved before Ep:{0} ###".format(episode))

def test(env, agent, model_path, test_episode, max_episode_length, logger, WINDOW, zipf_a, entire_request_nb, split_nb, sub_request_len):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    # load T request
    T_request = loda_T_request()  # load all data
    all_split_request_list = split_T_request(T_request, max_episode_length)

    # # test data
    # zipf_args = {
    #     'zipf_a': zipf_a,
    #     'entire_request_nb': entire_request_nb,
    #     'split_nb': split_nb,
    #     'sub_request_len': sub_request_len
    # }
    #
    # all_split_request_list = zipf_data(**zipf_args)  # load zipf data

    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    ave_cache_hit_rate = 0
    episode = 0
    episode_steps = 0
    episode_reward = 0.
    s_t = None
    for i in range(test_episode):

        env.T_request = all_split_request_list[episode]  # get list ith sub request list
        episode_t_request = all_split_request_list[episode]  # get list ith sub request list
        cache_hit_count = 0
        cache_hit_rate = 0
        env.reset()  # every episode reset env
        done = False
        if WINDOW == "random":
            window = random.randint(1,1000)  # random window
        else:
            window = WINDOW  # after N request execute action(update cache), every episode reset window
        action_list = []  # save action
        reward_list = []  # save reward, execute highest reward action

        # each episode(for t=1 to T)
        while True:

            window = window - 1  # N request execute one action
            env.t = episode_steps  # current request index
            current_request_id = episode_t_request[episode_steps]  # current request !!!!! not T_request!!!!!
            env.request_list.append(current_request_id)  # save request

            # if Requested content is already cached then Update cache hit rate and end decision epoch
            if current_request_id in env.cache_list:
                # update cache hit rate and end decision epoch
                cache_hit_count += 1
                # cache_hit_rate = cache_hit_count / episode_steps
                # print('cache hit rate1:', cache_hit_rate)

            # Requested content is not cached and Cache storage is not full. End decision epoch
            elif len(env.cache_list) < env.C:
                # Cache the currently requested content. Update cache state and cache hit rate
                env.cache_list.append(current_request_id)
                # if episode_steps != 0:
                #     cache_hit_rate = cache_hit_count / episode_steps
                # print('cache hit rate2:', cache_hit_rate)

            # Requested content is not cache and Cache storage is full
            else:
                # need execute action
                if s_t is None:
                    s_t = env.generate_state()  # according to cache(request list and cache list) get init state
                    agent.reset(s_t)

                env.current_content_id = current_request_id

                action = policy(s_t)
                # print("action:", action)
                # action = action[0][0]-1  # action [1]
                action = action[0]-1  # action [1]

                # for window experiment
                action_list.append(action)
                # r_t = env.get_simulated_reward(action)  # get simulated reward
                # reward_list.append(r_t)
                s_t = env.generate_state()  # cache don't update but state update, however, state change small

                # print('action:', action)
                # execute aciton
                if window <= 0:  # N request
                    # random choice
                    action = random.choice(action_list)
                    # action = action_list[reward_list.index(max(reward_list))]
                    s_t, r_t, done = env.step(action)

                    # reset window var
                    if WINDOW == "random":
                        window = random.randint(1, 1000)
                        # print("window{0}".format(window))
                    else:
                        window = WINDOW
                    action_list = []
                    # reward_list = []

                    episode_reward += r_t

            episode_steps += 1

            if max_episode_length and episode_steps >= max_episode_length:
                done = True
            if done:  # end of an episode

                # output cache hit rate
                cache_hit_rate = cache_hit_count / episode_steps
                # print('episode final cache hit rate------------------------------:', cache_hit_rate)

                logger.info(
                    "episode {0} final cache hit rate------------------------------:{1}".format(episode, cache_hit_rate)
                )

                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(i+1, episode_reward)
                )
                s_t = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
                break

        ave_cache_hit_rate += cache_hit_rate

    ave_cache_hit_rate /= test_episode
    logger.info("test average cache hit rate:{0}".format(ave_cache_hit_rate))

import gym
import numpy as np
import pickle
import copy
from gym import spaces
from gym.utils import seeding
import scipy
from gensim.matutils import blas_nrm2, blas_scal, ret_normalized_vec
from gensim.models import word2vec
import numpy
from gensim import matutils
from numpy import math
from wandb.util import np

class CacheEnv(gym.Env):

    def __init__(self, cache_capacity):

        self.C = cache_capacity
        self.SHORT_WIN = 10  # short term length
        self.MED_WIN = 100  # medium term length
        self.LONG_WIN = 1000  # long term length
        self.action_space = spaces.Discrete(self.C+1)  # action number is C+1(0,1,...,C)
        high = np.ones((self.C+1)*3)  # prob
        # high = np.array([self.LONG_WIN for _ in range((self.C+1)*3)])  # count
        low = np.zeros((self.C+1)*3)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.seed()
        self.close()
        self.cache_list = []  # simulated cache
        self.vector_list=[]  #vector list
        self.request_list = []  # save history request
        self.current_content_id = -1  # invalid value -1
        self.reward = 0.0
        self.T_request = []  # current episode all request
        self.t = 0  # current request index
        self.w = 0.01  # the weight to balance the short and long-term rewards
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        assert self.action_space.contains(action)

        # the currently requested content is not stored and the current caching space is not updated
        if action == 0:
            # cache don't update
            pass

        # the action is to store the currently requested content by replacing the υth content in the cache space.
        else:
            action = action - 1  # address cache list overflow. e.g, action is 100 will exchange cache_list[99]

            self.cache_list[action] = self.current_content_id  # replace cache content

        # after execute action get reward and new state
        self.reward = float(self.short_reward() + self.w * self.long_reward())  # get reward
        self.state = self.generate_state()  # get new state

        # print('cache:', self.cache_list)

        if self.t >= len(self.T_request)-1:  # t is the last request
            self.done = True

        return self.state, self.reward, self.done

    def step(self, action, md, num):

        assert self.action_space.contains(action)

        # the currently requested content is not stored and the current caching space is not updated
        if action == 0:
            # cache don't update
            pass

        # the action is to store the currently requested content by replacing the υth content in the cache space.
        else:
            action = action - 1
            self.replace_space(action, num, md)
            # self.cache_list[action] = self.current_content_id  # replace cache content

        # after execute action get reward and new state
        self.reward = float(self.short_reward() + self.w * self.long_reward())  # get reward
        self.state = self.generate_state()  # get new state

        # print('cache:', self.cache_list)

        if self.t >= len(self.T_request)-1:  # t is the last request
            self.done = True

        return self.state, self.reward, self.done

    def replace_space(self, action, num, md, ):
        old_vector=[]
        for i in self.vector_list:
            old_vector.append(self.unitvec(i))
        dvspace = numpy.array(old_vector)
        mean = numpy.array(old_vector[action])
        dists = numpy.dot(dvspace,mean)
        best = matutils.argsort(dists, topn=num , reverse=True)
        # ignore (don't return) words from the input
        new_vector = [self.current_content_id]
        appended =  md.wv.most_similar(str(self.current_content_id),topn=num-1)
        new_vector += [int(appended[i][0]) for i in range(len(appended))]

        for i in range(len(best)):
            self.cache_list[best[i]]=new_vector[i]

        # for b in range(len(best)):
        #     print(self.cache_list[best[b]])
        #     print(new_vector[b])
        #     self.cache_list[best[b]] = new_vector[b]

    def unitvec(self, vec, norm='l2', return_norm=False):
        """Scale a vector to unit length.

        Parameters
        ----------
        vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
            Input vector in any format
        norm : {'l1', 'l2', 'unique'}, optional
            Metric to normalize in.
        return_norm : bool, optional
            Return the length of vector `vec`, in addition to the normalized vector itself?

        Returns
        -------
        numpy.ndarray, scipy.sparse, list of (int, float)}
            Normalized vector in same format as `vec`.
        float
            Length of `vec` before normalization, if `return_norm` is set.

        Notes
        -----
        Zero-vector will be unchanged.

        """
        supported_norms = ('l1', 'l2', 'unique')
        if norm not in supported_norms:
            raise ValueError(
                "'%s' is not a supported norm. Currently supported norms are %s." % (norm, supported_norms))

        if scipy.sparse.issparse(vec):
            vec = vec.tocsr()
            if norm == 'l1':
                veclen = np.sum(np.abs(vec.data))
            if norm == 'l2':
                veclen = np.sqrt(np.sum(vec.data ** 2))
            if norm == 'unique':
                veclen = vec.nnz
            if veclen > 0.0:
                if np.issubdtype(vec.dtype, np.integer):
                    vec = vec.astype(np.float)
                vec /= veclen
                if return_norm:
                    return vec, veclen
                else:
                    return vec
            else:
                if return_norm:
                    return vec, 1.0
                else:
                    return vec

        if isinstance(vec, np.ndarray):
            if norm == 'l1':
                veclen = np.sum(np.abs(vec))
            if norm == 'l2':
                if vec.size == 0:
                    veclen = 0.0
                else:
                    veclen = blas_nrm2(vec)
            if norm == 'unique':
                veclen = np.count_nonzero(vec)
            if veclen > 0.0:
                if np.issubdtype(vec.dtype, np.integer):
                    vec = vec.astype(np.float)
                if return_norm:
                    return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
                else:
                    return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
            else:
                if return_norm:
                    return vec, 1.0
                else:
                    return vec

        try:
            first = next(iter(vec))  # is there at least one element?
        except StopIteration:
            if return_norm:
                return vec, 1.0
            else:
                return vec

        if isinstance(first, (tuple, list)) and len(first) == 2:  # gensim sparse format
            if norm == 'l1':
                length = float(sum(abs(val) for _, val in vec))
            if norm == 'l2':
                length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
            if norm == 'unique':
                length = 1.0 * len(vec)
            assert length > 0.0, "sparse documents must not contain any explicit zero entries"
            if return_norm:
                return ret_normalized_vec(vec, length), length
            else:
                return ret_normalized_vec(vec, length)
        else:
            raise ValueError("unknown input type")


    def reset(self):
        """
        reset env for each episode
        """
        self.cache_list = []  # cache clear
        self.request_list = []  # history request clear
        self.vector_list = [] # history clear
        self.current_content_id = 0
        self.reward = 0.0
        self.t = 0
        self.done = False

    def short_reward(self):
        if (self.t+1) < len(self.T_request) and self.T_request[self.t+1] in self.cache_list:
            return 1
        else:
            return 0

    def long_reward(self):
        hit_count = 0
        for i in range(1, 101):
            if (self.t+i) < len(self.T_request) and self.T_request[self.t+i] in self.cache_list:
                hit_count += 1
        return hit_count

    # def generate_state(self):
    #
    #     request_sum = len(self.request_list)
    #     long_term = []
    #     # print('request_sum:', request_sum)
    #     long_term.append(self.request_list.count(self.current_content_id)/request_sum)  # current request content index is 0
    #     for elem in self.cache_list:
    #         long_term.append(self.request_list.count(elem)/request_sum)  # normalization
    #
    #     short_term = []
    #     short_request_list = copy.deepcopy(self.request_list[request_sum-self.SHOR_WIN:request_sum])
    #     len_short_request_list = len(short_request_list)
    #     # print('len_short_request_list:', len_short_request_list)
    #     short_term.append(short_request_list.count(self.current_content_id)/len_short_request_list)  # current request content index is 0
    #     for elem in self.cache_list:
    #         short_term.append(short_request_list.count(elem)/len_short_request_list)
    #
    #     medium_term = []
    #     # print('request_list:', self.request_list)
    #     medium_request_list = copy.deepcopy(self.request_list[0:int(request_sum/2)])
    #     len_medium_request_list = len(medium_request_list)
    #     # print('len_medium_request_list:', len_medium_request_list)
    #     medium_term.append(medium_request_list.count(self.current_content_id)/len_medium_request_list)  # current request content index is 0
    #     for elem in self.cache_list:
    #         medium_term.append(medium_request_list.count(elem)/len_medium_request_list)
    #
    #     state = short_term + medium_term + long_term
    #     # print('len_state:', len(state))
    #     # print('state:', state)
    #
    #     return np.array(state)

    def generate_state(self):
        """
        short 10, medium 100, long 1000 probability
        :return: state(np.array)
        """

        request_sum = len(self.request_list)  # length >= 100(if C=100)

        # long term 1000
        long_term = []
        if request_sum < self.LONG_WIN:  # current request number < 1000
            # print('request_sum:', request_sum)
            long_term.append(
                self.request_list.count(self.current_content_id) / request_sum)  # current request content index is 0
            for elem in self.cache_list:
                long_term.append(self.request_list.count(elem) / request_sum)  # normalization
        else:
            # most recent 1000 requests
            long_request_list = copy.deepcopy(self.request_list[request_sum - self.LONG_WIN:request_sum])
            len_long_request_list = len(long_request_list)
            long_term.append(long_request_list.count(
                self.current_content_id) / len_long_request_list)  # current request content index is 0
            for elem in self.cache_list:
                long_term.append(long_request_list.count(elem) / len_long_request_list)

        # short term 10
        short_term = []
        # most recent 10 requests
        short_request_list = copy.deepcopy(self.request_list[request_sum - self.SHORT_WIN:request_sum])
        len_short_request_list = len(short_request_list)
        # print('len_short_request_list:', len_short_request_list)
        short_term.append(short_request_list.count(
            self.current_content_id) / len_short_request_list)  # current request content index is 0
        for elem in self.cache_list:
            short_term.append(short_request_list.count(elem) / len_short_request_list)

        # medium term 100
        medium_term = []
        if request_sum < self.MED_WIN:
            medium_term.append(
                self.request_list.count(self.current_content_id) / request_sum)  # current request content index is 0
            for elem in self.cache_list:
                medium_term.append(self.request_list.count(elem) / request_sum)  # normalization
        else:
            # most recent 100 request
            medium_request_list = copy.deepcopy(self.request_list[request_sum - self.MED_WIN:request_sum])
            len_medium_request_list = len(medium_request_list)
            medium_term.append(medium_request_list.count(
                self.current_content_id) / len_medium_request_list)  # current request content index is 0
            for elem in self.cache_list:
                medium_term.append(medium_request_list.count(elem) / len_medium_request_list)

        state = short_term + medium_term + long_term
        # print('len_state:', len(state))
        # print('state:', state)

        return np.array(state)

    # def generate_state(self):
    #     '''
    #     short 10, medium 100, long 1000, no normalization  count
    #     :return:
    #     '''
    #
    #     request_sum = len(self.request_list)
    #
    #     # long term
    #     long_term = []
    #     if request_sum < self.LONG_WIN:
    #         # print('request_sum:', request_sum)
    #         long_term.append(
    #             self.request_list.count(self.current_content_id))  # current request content index is 0
    #         for elem in self.cache_list:
    #             long_term.append(self.request_list.count(elem))  # normalization
    #     else:
    #         long_request_list = copy.deepcopy(self.request_list[request_sum - self.LONG_WIN:request_sum])
    #         len_long_request_list = len(long_request_list)
    #         long_term.append(long_request_list.count(
    #             self.current_content_id))  # current request content index is 0
    #         for elem in self.cache_list:
    #             long_term.append(long_request_list.count(elem))
    #
    #     # short term
    #     short_term = []
    #     short_request_list = copy.deepcopy(self.request_list[request_sum - self.SHOR_WIN:request_sum])
    #     len_short_request_list = len(short_request_list)
    #     # print('len_short_request_list:', len_short_request_list)
    #     short_term.append(short_request_list.count(
    #         self.current_content_id))  # current request content index is 0
    #     for elem in self.cache_list:
    #         short_term.append(short_request_list.count(elem))
    #
    #     # medium term
    #     medium_term = []
    #     if request_sum < self.MED_WIN:
    #         # print('request_sum:', request_sum)
    #         medium_term.append(
    #             self.request_list.count(self.current_content_id))  # current request content index is 0
    #         for elem in self.cache_list:
    #             medium_term.append(self.request_list.count(elem))  # normalization
    #     else:
    #         medium_request_list = copy.deepcopy(self.request_list[request_sum - self.MED_WIN:request_sum])
    #         len_medium_request_list = len(medium_request_list)
    #         medium_term.append(medium_request_list.count(
    #             self.current_content_id))  # current request content index is 0
    #         for elem in self.cache_list:
    #             medium_term.append(medium_request_list.count(elem))
    #
    #     state = short_term + medium_term + long_term
    #     # print('len_state:', len(state))
    #     # print('state:', state)
    #
    #     return np.array(state)

    def get_reward(self):
        self.reward = float(self.short_reward() + self.w * self.long_reward())
        return self.reward

    def get_simulated_reward(self, action):
        action = action - 1  # address cache list overflow. e.g, action is 100 will exchange cache_list[99]
        # save previous
        prev_content_id = self.cache_list[action]
        # execute action
        self.cache_list[action] = self.current_content_id

        self.reward = float(self.short_reward() + self.w * self.long_reward())

        # roll back
        self.cache_list[action] = prev_content_id

        return self.reward

    # def generate_state(self):
    #     print('random state')
    #     return np.random.rand((self.C+1)*3)
    #     # state = np.random.randint(self.LONG_WIN, size=(self.C+1)*3)
    #     # print('state:', state)
    #     return state




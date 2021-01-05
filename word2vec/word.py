import scipy
from gensim.matutils import blas_nrm2, blas_scal, ret_normalized_vec
from gensim.models import word2vec
import numpy
from numpy import math
import copy
from wandb.util import np
from gensim.models.word2vec import LineSentence
import logging

def unitvec(vec, norm='l2', return_norm=False):
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
        raise ValueError("'%s' is not a supported norm. Currently supported norms are %s." % (norm, supported_norms))

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
def write_new_data(name):
    zipf_args = {
        'zipf_a': 1.3,
        'entire_request_nb': 2000000,
        'split_nb': 1000,
        'sub_request_len': 2000
    }
    all_split_request_list = zipf_data(**zipf_args)  # load zipf data
    with open(name+'.txt','a') as file_handle:
        for i in range(len(all_split_request_list)):
            for j in range(len(all_split_request_list[i])):
                file_handle.write(str(all_split_request_list[i][j]))
                file_handle.write(' ')
        file_handle.close()

# write_new_data('test_01')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# # # sentences = word2vec.Text8Corpus('/home/deep/wolpertinger_ddpg-master/wolpertinger_ddpg-master/data/test.txt')
# # # inp = "/home/deep/wolpertinger_ddpg-master/wolpertinger_ddpg-master/data/test.txt"
inp = "./test_01.txt"
model = word2vec.Word2Vec(LineSentence(inp), size=100,min_count=0,  window=5, workers=4)
model.save('./test_01.model')
# for word in model.wv.index2word:
#     print(word,model[word])
# print(model)






md = word2vec.Word2Vec.load('./test_01.model')
# md = word2vec.Word2Vec.load('./../data/test_01.model')
# # 用于比较单个词语
# print(numpy.dot(unitvec(md["154"]),unitvec(md["862"])))
# print(md.similarity('154', '862'))
#
print(md.most_similar("26899",topn=10))
# print(type(md['511']))
# print(len(list(md.wv.index2word)))
# aa= md.wv.index2word
# for word in md.wv.index2word:
#     print(word,md[word])



# out: -0.06432766
# wv是4.0新版本后的方法，代替model.n_similartity
# n_similarity用于比较文章
# print(md.wv.n_similarity(['fox','dogs'], ['dogs', 'fox']))
# out：1.0
# most_similar找到相似度最高的词
# print(type(md))

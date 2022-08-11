import os
import json
import numpy as np
import matplotlib.pyplot as plt

pinv = np.linalg.pinv
pinv_fast = lambda x: pinv(x, hermitian=True)
fft = lambda x: np.fft.fftshift(np.fft.fft(x))
fftfreq = lambda m: np.fft.fftshift(np.fft.fftfreq(m))
calcPow = lambda x: np.abs(np.sum(x * np.conjugate(x))) / x.size

def normalize(x):
    if np.any(x):
        return x / np.max(x)
    else:
        return x

def value_to_decibel(vector):
    res = 20 * np.log10(vector / np.max(vector))
    res[res < -70] = -70
    return res

def make_dic_maker(dic):
    def maker(*funcs):
        new_dic = dic.copy()
        for func in funcs:
            func(new_dic)
        return new_dic
    return maker

def change_maker(key, value):
    def change(dic):
        dic[key] = value
    return change

def hermitian(matrix):
    return np.conjugate(matrix.T)

def calcu_cov(output):
    return np.matmul(output, hermitian(output)) / len(output)

def mvdr(output, expect_theta, steer_func, returnOutput=False):
    inv_mat = pinv(calcu_cov(output))
    steer_vector = steer_func(expect_theta)
    temp = np.matmul(inv_mat, steer_vector)
    weightVec = temp / (hermitian(steer_vector) @ inv_mat @ steer_vector).item()
    if not returnOutput:
        return weightVec
    else:
        return weightVec, np.squeeze(np.conjugate(weightVec.T) @ output)

def my_plot(*args, fig_ax_pair=(None, None), num=None, **kwargs):
    if fig_ax_pair == (None, None):
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax_pair
    if num is not None:
        fig.canvas.manager.set_window_title(num)
    ax.plot(*args, **kwargs)
    fig.show()
    return fig, ax

def plot_position(theta, fig_ax_pair=(None, None), num=None, **kwargs):
    fig_ax = my_plot((theta, theta),
            (5, -70),
            fig_ax_pair=fig_ax_pair,
            num=num,
            **kwargs)
    fig_ax[1].set_ylim((-70, 5))
    return fig_ax

def syn(weight, output):
    return np.matmul(hermitian(weight), output).squeeze()

f_then_g = lambda f, g: lambda x: f(g(x))
def funcs_cons(*funcs):
    identity = lambda x: x
    res_func = identity
    for func in funcs:
        res_func = f_then_g(res_func, func)
    return res_func

def cal_ac(expect_theta, coherent_theta, steer_func):
    matrix_ac = [steer_func(expect_theta)]
    if isinstance(coherent_theta, (tuple, list)):
        for theta in coherent_theta:
            matrix_ac.append(steer_func(theta))
    else:
        matrix_ac.append(steer_func(coherent_theta))
    matrix_ac = np.concatenate(matrix_ac, axis=1)
    return matrix_ac

def cal_bc(coherent_theta, steer_func):
    matrix_bc = cal_ac(0, coherent_theta, steer_func)
    matrix_bc[:, 0] = 0
    return matrix_bc

def cal_f_vec(coherent_theta):
    f_vec = [1]
    if isinstance(coherent_theta, (tuple, list)):
        for _ in coherent_theta:
            f_vec.append(0)
    else:
        f_vec.append(0)
    return np.array([[item] for item in f_vec])

def mcmv(output, expect_theta, coherent_theta, steer_func, returnOutput=False):
    inv_cov = f_then_g(pinv_fast, calcu_cov)(output)
    matrix_ac = cal_ac(expect_theta, coherent_theta, steer_func)
    vector_f = cal_f_vec(coherent_theta)
    mcmv_weight = (
            inv_cov @
            matrix_ac @
            pinv(hermitian(matrix_ac) @ inv_cov @ matrix_ac) @
            vector_f
            )
    if returnOutput:
        return mcmv_weight, np.squeeze(np.conjugate(mcmv_weight.T) @ output)
    else:
        return mcmv_weight

def exactly(expect_theta, snr, interference_theta, inr, sigma_power, steer_func):
    a0 = steer_func(expect_theta)
    cov_mat = sigma_power * np.eye(a0.size) + snr**2 * a0 @ hermitian(a0)
    for item_inr, theta in zip(inr, interference_theta):
        a = steer_func(theta)
        cov_mat += item_inr**2 * a @ hermitian(a)
    return np.linalg.pinv(cov_mat) @ a0

def ctmv(output, expect_theta, coherent_theta, steer_func, sigma_power, diagonal_load=0):
    cov_mat = calcu_cov(output)
    cov_mat_loaded = cov_mat + diagonal_load * np.eye(len(cov_mat))
    inv_loaded_cov = pinv_fast(cov_mat_loaded)
    matrix_ac = cal_ac(expect_theta, coherent_theta, steer_func)
    matrix_bc = cal_bc(coherent_theta, steer_func)
    matrix_t = (
            np.eye(len(cov_mat)) -
            (matrix_ac - matrix_bc) @
            pinv_fast(hermitian(matrix_ac) @ inv_loaded_cov @ matrix_ac) @
            hermitian(matrix_ac) @
            inv_loaded_cov
            )
    newly_created_cov = (
            matrix_t @ cov_mat @ hermitian(matrix_t) -
            sigma_power * matrix_t @ hermitian(matrix_t) +
            sigma_power * np.eye(len(cov_mat))
            )
    steer_vector = steer_func(expect_theta)
    inv_newly_created_cov = pinv_fast(newly_created_cov)
    weight = (
            inv_newly_created_cov @ steer_vector /
            (hermitian(steer_vector) @ inv_newly_created_cov @ steer_vector)
            )
    return weight

def ctp(output, expect_theta, coherent_theta, steer_func, sigma_power):
    cov_mat = calcu_cov(output)
    inv_cov = pinv_fast(cov_mat)
    matrix_ac = cal_ac(expect_theta, coherent_theta, steer_func)
    matrix_t = (
            np.eye(len(cov_mat)) -
            matrix_ac @
            pinv_fast(hermitian(matrix_ac) @ inv_cov @ matrix_ac) @
            hermitian(matrix_ac) @
            inv_cov
            )
    newly_created_cov = (
            matrix_t @ cov_mat @ hermitian(matrix_t) -
            sigma_power * matrix_t @ hermitian(matrix_t) +
            sigma_power * np.eye(len(cov_mat))
            )
    u, s, _ = np.linalg.svd(newly_created_cov, hermitian=True)
    count = 0
    refer_sum = np.sum(s) / len(s)
    for item in s:
        if item > refer_sum:
            count += 1
    matrix_u = None
    if count == 0:
        matrix_u = np.zeros((len(cov_mat), 1))
    else:
        matrix_u = u[:, :count]
    u2, _, _ = np.linalg.svd(cov_mat, hermitian=True)
    # eig_vec = u2[:, :1]
    max_index = max(range(count+1), key=lambda index: np.linalg.norm((np.eye(len(cov_mat)) - matrix_t @ hermitian(matrix_t)) @ u2[:, index: index+1]))
    return (np.eye(len(cov_mat)) - matrix_t @ hermitian(matrix_t)) @ u2[:, max_index: max_index+1]

def proposed(output, expect_theta, steer_func, retoutput=False):
    steer_vector = steer_func(expect_theta)
    output_2 = np.conjugate(steer_vector[1][0]) * output
    delta = output[:-1, :] - output_2[1:, :]
    weight = mvdr(delta, expect_theta, lambda x: steer_func(x)[:-1, :])
    if not retoutput:
        return weight
    else:
        return weight, syn(weight, output[:-1, :])

def yang_ho_chi(output, coherent_number, steer_func, expect_theta=0, retoutput=False):
    ele_num = len(output)
    d = ele_num - coherent_number
    delta = output[:-1, :] - output[1:, :]
    xm = np.conjugate(output[-1:, :])
    r = np.sum(np.multiply(delta, xm), axis=1, keepdims=True) / output.shape[1]
    matrix_rc = np.hstack([r[index: index+d] for index in range(coherent_number)])
    q, _ = np.linalg.qr(matrix_rc)
    q = q[:, :d]
    weight = (np.eye(d) - q @ hermitian(q)) @ steer_func(expect_theta)[:d, :]
    if not retoutput:
        return weight
    else:
        return weight, syn(weight, output[:d, :])

def data_generate(data_iterable, save_path):
    index = 0
    info_dic = {}
    name_func_maker = lambda x: lambda y: os.path.join(save_path, x+str(y)+'.npy')
    x_name_func = name_func_maker('x')
    y_name_func = name_func_maker('y')
    def save_func(name, data):
        with open(name, 'wb') as f:
            np.save(f, data)
    for x, y, info in data_iterable:
        save_func(x_name_func(index), x)
        save_func(y_name_func(index), y)
        info_dic[str(index)] = info
        index += 1
    with open(os.path.join(save_path, 'info.txt'), 'w') as f:
        json.dump(info_dic, f, indent=4)

def eval_power(signal):
    return np.abs(np.sum(signal * hermitian(signal))) / signal.size

def smooth(output, expect_theta, steer_func):
    cov1 = calcu_cov(output[:-1, :])
    cov2 = calcu_cov(output[1:, :])
    cov = (cov1 + cov2 ) / 2
    return np.linalg.pinv(cov) @ steer_func(expect_theta)[:-1, :]

def smooth2(output, expect_theta, subarray_num, steer_func, returnOutput=False):
    covs = []
    ele_num = output.shape[0]
    syn_num = ele_num - subarray_num + 1
    for k in range(subarray_num):
        covs.append(calcu_cov(output[k: k+syn_num, :]))
    cov = sum(covs) / subarray_num
    weightVec = np.linalg.pinv(cov) @ steer_func(expect_theta)[:syn_num, :]
    if returnOutput:
        return weightVec, np.squeeze(np.conjugate(weightVec.T) @ output[:ele_num+1-subarray_num, :])
    else:
        return weightVec

def output_noise_power(noise_power, weight):
    res = 0
    for item in weight.flatten():
        res += item * np.conjugate(item)
    return noise_power * res

def calcSINR(weight, expect, interference, noise):
    weightLen = weight.size
    synExpect = np.squeeze(np.conjugate(weight.T) @ expect[:weightLen, :])
    synInterference = np.squeeze(np.conjugate(weight.T) @ interference[:weightLen, :])
    synNoise = np.squeeze(np.conjugate(weight.T) @ noise[:weightLen, :])
    realValue = calcPow(synExpect) / (calcPow(synInterference) + calcPow(synNoise))
    return 10 * np.log(realValue)

if __name__ == '__main__':
    test_data = ((1, 2, {'name': 'zxy', 'age': 22}) for _ in range(1))
    data_generate(test_data, '/Users/zhangxingyu/new')

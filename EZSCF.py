'''
EZSCF is the name of this purely Python-based RHF/RMP2 program of atomic systems, mainly used for the bachelor's thesis of Xiao Liu @ BNU.
The name EZSCF is a homophone with "Easy SCF" and also a pun on the initial Chinese Pinyin letters of my beloved Alma Mater, Hangzhou No.2 High School (aka 杭州二中 or HZEZ).
Differing from widely used Cartesian GTO basis set, this program applies Spherical Harmonic GTO basis set so as to compute high angular momentum part more efficiently and accurately.

Developed by Xiao Liu under the guidance of Prof. Zhendong Li @ BNU. All references are marked in different places of the program.
Many thanks to Rui Li (from ZJU to Caltech; HZEZ alumnus), Jingze Li (from BNU to UCSD), Jie Feng (from HUST to SJTU; HZEZ alumnus) & Ruiyi Zhou (from HKU to UNC Chapel Hill) for helpful discussions!
A special thank goes to Sobereva & Warm_Cloud.
Last but the most important, thanks Qiming Sun (also HZEZ alumnus) for his prominent open-source program PySCF!
Feel free to contact Xiao Liu via email at the address of 201611150142@mail.bnu.edu.cn if you have any questions or advice.

Xiao Liu
May 1, 2020
'''

import numpy as np
from math import exp, cos, sin, radians
from math import sqrt as msqrt
from scipy import linalg
from scipy.special import gamma, gammainc, beta, betainc
from sympy.physics.wigner import gaunt
from sympy import *
from functools import reduce
import warnings
import operator
import re
import datetime
import copy
import basis_set

warnings.filterwarnings("ignore")
# pi = 3.1415926535897932385

def factorial(n):
    if n<0 : return 0
    if n==0: return 1
    if n==1: return 1
    if n>1: return reduce(operator.mul,range(n,0,-1))

def factor_double(n):
    if n<0 or n==0 or type(n) != int : return 1
    if n==1: return 1
    if n==2: return 2
    if n>2: return reduce(operator.mul,range(n,0,-2))

def get_l(wfn_i):
    orbital_type = re.sub('[^A-Z]', '', wfn_i)

    if orbital_type == 'S':return 0
    if orbital_type == 'P':return 1
    if orbital_type == 'D':return 2
    if orbital_type == 'F':return 3
    if orbital_type == 'G':return 4
    if orbital_type == 'H':return 5
    if orbital_type == 'I':return 6

def get_orbital_num(wfn_i):
    orbital_type = re.sub('[^A-Z]', '', wfn_i)

    if orbital_type == 'S':return 1
    if orbital_type == 'P':return 3
    if orbital_type == 'D':return 5
    if orbital_type == 'F':return 7
    if orbital_type == 'G':return 9
    if orbital_type == 'H':return 11
    if orbital_type == 'I':return 13

def get_m_array(l):
    m_list = []

    for i in range(l+1):
        m_list.append(i)
        m_list.append(-i)
    del m_list[0]

    m_array = np.array([m_list]).T

    return m_array

def phi_fn(phi, m):
    # Reference: 量子化学——基本原理和从头计算法(上册) P109
    factor = msqrt(1/(2*pi))

    exp_complex = 1
    if m>0:
        exp_complex = msqrt(2) * cos(m*phi)
    if m<0:
        exp_complex = msqrt(2) * sin(m*phi)

    value = factor * exp_complex
    return value

def legendre(cos_theta, l):
    # Reference: 量子化学——基本原理和从头计算法(上册) P116
    value = 0

    for k in range(int(l/2)+1):
        temp = pow(-1, k) * factorial(2*l-2*k) / pow(2, l) / factorial(k) / factorial(l-k) / factorial(l-2*k) * pow(cos_theta, l-2*k)
        value += temp
    return value

def associated_legendre(cos_theta, l, m):
    # Reference: 量子化学——基本原理和从头计算法(上册) P121
    t = symbols('cos_theta', real=True)

    temp = diff(legendre(t,l), t, abs(m)).subs(t, cos_theta)
    value = pow(1 - (pow(cos_theta, 2)), abs(m / 2)) * temp
    return value

def theta_fn(theta, l, m):
    # Reference: 量子化学——基本原理和从头计算法(上册) P129
    factor = pow(-1, (m+abs(m))/2) * msqrt( (2*l+1)/2 * factorial(l-abs(m)) / factorial(l+abs(m)) )

    cos_theta = cos(theta)
    as_le_fn = associated_legendre(cos_theta, l, m)

    value = factor * as_le_fn
    return value

def norm_gauss(alpha, concn, r, l):
    # Reference: Molecular Electronic Structure Theory P234（page 262 in pdf version）
    gauss = concn * 2 * pow(2* alpha, 0.75) / pow(pi, 0.25) * msqrt( pow(2,l)/factor_double(2*l +1) ) * pow(msqrt(2*alpha)*r, l) * exp(-alpha * r * r)
    return gauss

def norm_sh(theta, phi, l, m):
    value = phi_fn(phi, m) * theta_fn(theta, l, m)
    return value

def combine_sto_array(sto_array_list, num):
    wfn_array = sto_array_list[0]

    if num == 1:
        return wfn_array
    else:
        for i in range(1,num):
            wfn_array = np.vstack((wfn_array, sto_array_list[i]))
        return wfn_array

def form_wfn_array(wfn):
    sto_array_list = []

    for orbital_type in wfn:
        l = get_l(orbital_type)
        m_array = get_m_array(l)

        phi = symbols('phi')
        theta = symbols('theta')
        sh = np.frompyfunc(norm_sh, 4, 1)
        sh_array = sh(theta, phi, l, m_array)

        r = symbols('r')
        get_norm_gauss = np.frompyfunc(norm_gauss, 4, 1)
        gauss_array = get_norm_gauss(wfn[orbital_type][0], wfn[orbital_type][1], r, l)
        sto_array = np.multiply(sh_array,gauss_array.sum(axis=0)).reshape(-1,1)
        sto_array_list.append(sto_array)

    wfn_array = combine_sto_array(sto_array_list, len(sto_array_list))
    return wfn_array

def form_wfn_matrix(wfn_array):
    wfn_matrix = wfn_array.repeat(wfn_array.shape[0], axis=1)
    return wfn_matrix

def get_tuple_list(wfn):
    tuple_list = []
    index_1 = 0

    for orbital_type in wfn:
        index_2 = 0
        l = get_l(orbital_type)
        m_array = np.tile(get_m_array(l),(wfn[orbital_type][1].shape[1], 1))

        for i in m_array:
            index_1 += 1
            n = int(index_2/(2*l+1))+1
            tuple = index_1, n, l, i, orbital_type
            tuple_list.append(tuple)
            index_2 += 1

    return tuple_list

def get_alpha_concn_norm_factor(alpha, concn, l):
    # Reference: Molecular Electronic Structure Theory P234（page 262 in pdf version）
    alpha_concn_norm_factor = 2 * concn * pow(2*alpha, 0.75)/ pow(pi, 0.25) * pow(2* pow(alpha, 0.5), l) / msqrt(factor_double(2*l+1))
    return alpha_concn_norm_factor

def unnormed_S(wfn_i, tuple1, tuple2):
    index1 = tuple1[0]
    n1 = tuple1[1]
    l1 = tuple1[2]
    m1 = tuple1[3]
    index2 = tuple2[0]
    n2 = tuple2[1]
    l2 = tuple2[2]
    m2 = tuple2[3]

    S_value = 0
    if l1 != l2 or m1 != m2 or n1>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i) or n2>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i):
        return S_value

    else:
        l = get_l(wfn_i)
        alpha = wfn[wfn_i][0]
        concn_matrix = np.tile(wfn[wfn_i][1], get_orbital_num(wfn_i))
        concn_array_i = concn_matrix[:,n1-1]
        concn_array_j = concn_matrix[:,n2-1]

        for ii in range(len(alpha)):
            alpha_i = alpha[ii]
            concn_i = concn_array_i[ii]

            for jj in range(len(alpha)):
                alpha_j = alpha[jj]
                concn_j = concn_array_j[jj]

                S_temp = 0.5*pow(alpha_i+alpha_j, -(2*l+3)/2)*gamma((2*l+3)/2)

                if S_temp == [nan]: S_temp =0
                # Reference: Molecular Electronic Structure Theory P234（page 262 in pdf version）
                norm_factor_i = get_alpha_concn_norm_factor(alpha_i, concn_i, l)
                norm_factor_j = get_alpha_concn_norm_factor(alpha_j, concn_j, l)
                norm_factor = norm_factor_i * norm_factor_j

                S_norm = S_temp * norm_factor
                S_value += S_norm

    return S_value

def form_unnormed_S_matrix(tuple_list):
    dim = len(tuple_list)
    S_matrix = np.zeros((dim,dim))

    for i in tuple_list:
        for j in tuple_list:
            if i[4] == j[4]:
                index1 = i[0]
                index2 = j[0]
                S_value = unnormed_S(i[4], i, j)
                S_matrix[index1-1][index2-1] = S_value

    return S_matrix

def get_S_norm_factor(n):
    # Normalize diagonal elements of S matrix to be unitary, for the sake of getting the same output with Gaussian and PySCF
    global unnormed_S_matrix
    return 1/msqrt(unnormed_S_matrix[n][n])

def S(wfn_i, tuple1, tuple2):
    index1 = tuple1[0]
    n1 = tuple1[1]
    l1 = tuple1[2]
    m1 = tuple1[3]
    index2 = tuple2[0]
    n2 = tuple2[1]
    l2 = tuple2[2]
    m2 = tuple2[3]

    S_value = 0
    if l1 != l2 or m1 != m2 or n1>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i) or n2>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i):
        return S_value

    else:
        l = get_l(wfn_i)
        alpha = wfn[wfn_i][0]
        concn_matrix = np.tile(wfn[wfn_i][1], get_orbital_num(wfn_i))
        concn_array_i = concn_matrix[:,n1-1]
        concn_array_j = concn_matrix[:,n2-1]
        S_norm_factor = get_S_norm_factor(index1-1) * get_S_norm_factor(index2-1)

        for ii in range(len(alpha)):
            alpha_i = alpha[ii]
            concn_i = concn_array_i[ii]

            for jj in range(len(alpha)):
                alpha_j = alpha[jj]
                concn_j = concn_array_j[jj]

                S_temp = 0.5*pow(alpha_i+alpha_j, -(2*l+3)/2)*gamma((2*l+3)/2)

                if S_temp == [nan]: S_temp =0
                # Reference: Molecular Electronic Structure Theory P234（page 262 in pdf version）
                norm_factor_i = get_alpha_concn_norm_factor(alpha_i, concn_i, l)
                norm_factor_j = get_alpha_concn_norm_factor(alpha_j, concn_j, l)
                norm_factor = norm_factor_i * norm_factor_j * S_norm_factor

                S_norm = S_temp * norm_factor
                S_value += S_norm

    return S_value

def form_S_matrix(tuple_list):
    dim = len(tuple_list)
    S_matrix = np.zeros((dim,dim))

    for i in tuple_list:
        for j in tuple_list:
            if i[4] == j[4]:
                index1 = i[0]
                index2 = j[0]
                S_value = S(i[4], i, j)
                S_matrix[index1-1][index2-1] = S_value

    return S_matrix

def T(wfn_i, tuple1, tuple2):
    index1 = tuple1[0]
    n1 = tuple1[1]
    l1 = tuple1[2]
    m1 = tuple1[3]
    index2 = tuple2[0]
    n2 = tuple2[1]
    l2 = tuple2[2]
    m2 = tuple2[3]

    T_value = 0
    if l1 != l2 or m1 != m2 or n1>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i) or n2>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i):
        return T_value

    else:
        l = get_l(wfn_i)
        alpha = wfn[wfn_i][0]
        concn_matrix = np.tile(wfn[wfn_i][1], get_orbital_num(wfn_i))
        concn_array_i = concn_matrix[:,n1-1]
        concn_array_j = concn_matrix[:,n2-1]
        S_norm_factor = get_S_norm_factor(index1-1) * get_S_norm_factor(index2-1)

        for ii in range(len(alpha)):
            alpha_i = alpha[ii]
            concn_i = concn_array_i[ii]

            for jj in range(len(alpha)):
                alpha_j = alpha[jj]
                concn_j = concn_array_j[jj]

                # Reference: 量子化学——基本原理和从头计算法(中册) P26
                T_temp = alpha_i*alpha_j*pow(alpha_i+alpha_j, -(2*l+5)/2)*gamma((2*l+5)/2)
                if T_temp == [nan]: T_temp = 0
                # Reference: Molecular Electronic Structure Theory P234（page 262 in pdf version）
                norm_factor_i = get_alpha_concn_norm_factor(alpha_i, concn_i, l)
                norm_factor_j = get_alpha_concn_norm_factor(alpha_j, concn_j, l)
                norm_factor = norm_factor_i * norm_factor_j * S_norm_factor

                T_norm = T_temp * norm_factor
                T_value += T_norm

    return T_value

def form_T_matrix(tuple_list):
    dim = len(tuple_list)
    T_matrix = np.zeros((dim,dim))

    for i in tuple_list:
        for j in tuple_list:
            if i[4] == j[4]:
                index1 = i[0]
                index2 = j[0]
                T_value = T(i[4], i, j)
                T_matrix[index1 - 1][index2 - 1] = T_value

    return T_matrix

def V(atomic_num, wfn_i, tuple1, tuple2):
    Z = atomic_num
    index1 = tuple1[0]
    n1 = tuple1[1]
    l1 = tuple1[2]
    m1 = tuple1[3]
    index2 = tuple2[0]
    n2 = tuple2[1]
    l2 = tuple2[2]
    m2 = tuple2[3]

    V_value = 0
    if l1 != l2 or m1 != m2 or n1>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i) or n2>wfn[wfn_i][1].shape[1]*get_orbital_num(wfn_i):
        return V_value

    else:
        l = get_l(wfn_i)
        alpha = wfn[wfn_i][0]
        concn_matrix = np.tile(wfn[wfn_i][1], get_orbital_num(wfn_i))
        concn_array_i = concn_matrix[:,n1-1]
        concn_array_j = concn_matrix[:,n2-1]
        S_norm_factor = get_S_norm_factor(index1-1) * get_S_norm_factor(index2-1)

        for ii in range(len(alpha)):
            alpha_i = alpha[ii]
            concn_i = concn_array_i[ii]

            for jj in range(len(alpha)):
                alpha_j = alpha[jj]
                concn_j = concn_array_j[jj]

                # Reference: 量子化学——基本原理和从头计算法(中册) P28
                V_temp = -0.5*Z*pow(alpha_i+alpha_j, -l-1)*gamma(l+1)
                if V_temp == [nan]: V_temp = 0
                # Reference: Molecular Electronic Structure Theory P234（page 262 in pdf version）
                norm_factor_i = get_alpha_concn_norm_factor(alpha_i, concn_i, l)
                norm_factor_j = get_alpha_concn_norm_factor(alpha_j, concn_j, l)
                norm_factor = norm_factor_i * norm_factor_j * S_norm_factor

                V_norm = V_temp * norm_factor
                V_value += V_norm

    return V_value

def form_V_matrix(atomic_num, tuple_list):
    Z = atomic_num
    dim = len(tuple_list)
    V_matrix = np.zeros((dim,dim))

    for i in tuple_list:
        for j in tuple_list:
            if i[4] == j[4]:
                index1 = i[0]
                index2 = j[0]
                V_value = V(Z, i[4], i, j)
                V_matrix[index1 - 1][index2 - 1] = V_value

    return V_matrix

def H(atomic_num, wfn_i, tuple1, tuple2):
    Z = atomic_num
    H_value = T(wfn_i, tuple1, tuple2) + V(Z, wfn_i, tuple1, tuple2)

    return H_value

def form_H_core_matrix(T_matrix, V_matrix):
    return T_matrix + V_matrix

def get_2e_integral_type(tuple_I, tuple_J, tuple_K, tuple_L):
    # Reference1: Atomic Structure Theory Lectures on Atomic Physics P82 (page 92 in pdf version)
    # Reference2: Atomic Structure Theory Lectures on Atomic Physics P19 (page 29 in pdf version)
    # Reference3: 量子化学——基本原理和从头计算法(中册) P28-30

    index_I = tuple_I[0]
    n_I = int(tuple_I[1])
    l_I = int(tuple_I[2])
    m_I = int(tuple_I[3])
    type_I = tuple_I[4]
    tuple_quantum_I = l_I, m_I, type_I

    index_J = tuple_J[0]
    n_J = int(tuple_J[1])
    l_J = int(tuple_J[2])
    m_J = int(tuple_J[3])
    type_J = tuple_J[4]
    tuple_quantum_J = l_J, m_J, type_J

    index_K = tuple_K[0]
    n_K = int(tuple_K[1])
    l_K = int(tuple_K[2])
    m_K = int(tuple_K[3])
    type_K = tuple_K[4]
    tuple_quantum_K = l_K, m_K, type_K

    index_L = tuple_L[0]
    n_L = int(tuple_L[1])
    l_L = int(tuple_L[2])
    m_L = int(tuple_L[3])
    type_L = tuple_L[4]
    tuple_quantum_L = l_L, m_L, type_L

    if tuple_quantum_I == tuple_quantum_J:
        if tuple_quantum_K == tuple_quantum_L:
            return 'abab'
        else:
            return '0'

    elif tuple_quantum_I == tuple_quantum_K:
        if tuple_quantum_J == tuple_quantum_L:
            return 'abba'
        else:
            return '0'
    elif tuple_quantum_I == tuple_quantum_L:
        if tuple_quantum_J == tuple_quantum_K:
            return 'abba'
        else:
            return '0'

    else:
        return '0'

def get_ak(tuple_I, tuple_J, tuple_K, tuple_L):
    # angular part of [IJ,KL]
    # Reference1: Atomic Structure Theory Lectures on Atomic Physics P82 (page 92 in pdf version)
    # Reference2: Atomic Structure Theory Lectures on Atomic Physics P19 (page 29 in pdf version)
    # Reference3: 量子化学——基本原理和从头计算法(中册) P28-30

    index_I = tuple_I[0]
    n_I = int(tuple_I[1])
    l_I = int(tuple_I[2])
    m_I = int(tuple_I[3])
    type_I = tuple_I[4]

    index_J = tuple_J[0]
    n_J = int(tuple_J[1])
    l_J = int(tuple_J[2])
    m_J = int(tuple_J[3])
    type_J = tuple_J[4]

    index_K = tuple_K[0]
    n_K = int(tuple_K[1])
    l_K = int(tuple_K[2])
    m_K = int(tuple_K[3])
    type_K = tuple_K[4]

    index_L = tuple_L[0]
    n_L = int(tuple_L[1])
    l_L = int(tuple_L[2])
    m_L = int(tuple_L[3])
    type_L = tuple_L[4]

    compute = False
    tuple_ak_list = []

    integral_type = get_2e_integral_type(tuple_I, tuple_J, tuple_K, tuple_L)
    if integral_type == 'abab' or integral_type == 'abba':
        compute = True
    else:
        pass

    if compute == True:

        if l_I>l_J:
            l_I, l_J = l_J, l_I
            m_I, m_J = m_J, m_I
        elif l_I == l_J and abs(m_I) < abs(m_J):
            l_I, l_J = l_J, l_I
            m_I, m_J = m_J, m_I

        k_IJ_lower_bound = abs(l_I-l_J)
        k_IJ_upper_bound = l_I+l_J
        tuple_IJ_list = []
        for k_IJ in range(k_IJ_lower_bound, k_IJ_upper_bound+1):
            if (l_I + l_J + k_IJ) % 2 == 0:
                gaunt_IJ_sum = 0
                for q_IJ in range(-k_IJ, k_IJ + 1):
                    gaunt_IJ_temp = gaunt(l_I,k_IJ,l_J,-m_I,-q_IJ,m_J) * pow(-1, 2*l_I-m_I)
                    gaunt_IJ_sum += gaunt_IJ_temp
                tuple_IJ = l_I,m_I,l_J,m_J,k_IJ,gaunt_IJ_sum
                tuple_IJ_list.append(tuple_IJ)

        if l_K>l_L:
            l_K, l_L = l_L, l_K
            m_K, m_L = m_L, m_K
        elif l_K == l_L and abs(m_K) < abs(m_L):
            l_K, l_L = l_L, l_K
            m_K, m_L = m_L, m_K

        k_KL_lower_bound = abs(l_K-l_L)
        k_KL_upper_bound = l_K+l_L
        tuple_KL_list = []
        for k_KL in range(k_KL_lower_bound, k_KL_upper_bound+1):
            if (l_K + l_L + k_KL) % 2 == 0:
                gaunt_KL_sum = 0
                for q_KL in range(-k_KL, k_KL + 1):
                    gaunt_KL_sum += gaunt(l_K,k_KL,l_L,-m_K,q_KL,m_L) * pow(-1, 2*l_K-m_K)
                tuple_KL = l_I,m_I,l_J,m_J,k_KL,gaunt_KL_sum
                tuple_KL_list.append(tuple_KL)

        for tuple_IJ in tuple_IJ_list:
            k_of_IJ = tuple_IJ[4]
            gaunt_of_IJ = tuple_IJ[5]
            for tuple_KL in tuple_KL_list:
                k_of_KL = tuple_KL[4]
                gaunt_of_KL = tuple_KL[5]
                if k_of_IJ == k_of_KL:
                    k = k_of_IJ
                    ak = 4*pi/(2*k+1)*gaunt_of_IJ*gaunt_of_KL
                    if type(ak) != type(1) and type(ak) != type(1.0):
                        ak_float = ak.evalf()
                        tuple_ak = index_I, index_J, index_K, index_L, k, ak_float
                        tuple_ak_list.append(tuple_ak)
                    else:
                        tuple_ak = index_I, index_J, index_K, index_L, k, ak
                        tuple_ak_list.append(tuple_ak)
    else:
        # tuple_nan = index_I, index_J, index_K, index_L, 0, 0
        tuple_nan = 'nan'
        tuple_ak_list.append(tuple_nan)

    return tuple_ak_list

def get_Rk(tuple_I, tuple_J, tuple_K, tuple_L, k):
    # radical part of [IJ,KL]
    # Reference1: Relativistic calculations for atoms: self-consistent treatment of Breit interaction and nuclear volume effect eq.(1.34)
    # Reference2: Molecular Electronic Structure Theory P234 (page 262 in pdf version)
    # Reference3: 量子化学——基本原理和从头计算法(中册) P28-30

    index_I = tuple_I[0]
    n_I = int(tuple_I[1])
    l_I = int(tuple_I[2])
    m_I = int(tuple_I[3])
    type_I = tuple_I[4]

    index_J = tuple_J[0]
    n_J = int(tuple_J[1])
    l_J = int(tuple_J[2])
    m_J = int(tuple_J[3])
    type_J = tuple_J[4]

    index_K = tuple_K[0]
    n_K = int(tuple_K[1])
    l_K = int(tuple_K[2])
    m_K = int(tuple_K[3])
    type_K = tuple_K[4]

    index_L = tuple_L[0]
    n_L = int(tuple_L[1])
    l_L = int(tuple_L[2])
    m_L = int(tuple_L[3])
    type_L = tuple_L[4]

    Rk = 0

    alpha_I = wfn[type_I][0]
    concn_matrix_I = np.tile(wfn[type_I][1], get_orbital_num(type_I))
    concn_array_I = concn_matrix_I[:, n_I - 1]

    alpha_J = wfn[type_J][0]
    concn_matrix_J = np.tile(wfn[type_J][1], get_orbital_num(type_J))
    concn_array_J = concn_matrix_J[:, n_J - 1]

    alpha_K = wfn[type_K][0]
    concn_matrix_K = np.tile(wfn[type_K][1], get_orbital_num(type_K))
    concn_array_K = concn_matrix_K[:, n_K - 1]

    alpha_L = wfn[type_L][0]
    concn_matrix_L = np.tile(wfn[type_L][1], get_orbital_num(type_L))
    concn_array_L = concn_matrix_L[:, n_L - 1]

    S_norm_factor = get_S_norm_factor(index_I-1) * get_S_norm_factor(index_J-1) * get_S_norm_factor(index_K-1) * get_S_norm_factor(index_L-1)

    for index_I in range(len(alpha_I)):
        alpha_I_i = alpha_I[index_I]
        concn_I_i = concn_array_I[index_I]

        for index_J in range(len(alpha_J)):
            alpha_J_j = alpha_J[index_J]
            concn_J_j = concn_array_J[index_J]

            for index_K in range(len(alpha_K)):
                alpha_K_k = alpha_K[index_K]
                concn_K_k = concn_array_K[index_K]

                for index_L in range(len(alpha_L)):
                    alpha_L_l = alpha_L[index_L]
                    concn_L_l = concn_array_L[index_L]

                    norm_factor_I_i = get_alpha_concn_norm_factor(alpha_I_i, concn_I_i, l_I)
                    norm_factor_J_j = get_alpha_concn_norm_factor(alpha_J_j, concn_J_j, l_J)
                    norm_factor_K_k = get_alpha_concn_norm_factor(alpha_K_k, concn_K_k, l_K)
                    norm_factor_L_l = get_alpha_concn_norm_factor(alpha_L_l, concn_L_l, l_L)
                    common_factor = norm_factor_I_i * norm_factor_J_j * norm_factor_K_k * norm_factor_L_l * S_norm_factor

                    # Rk_IJKL_part1 is the part of r2<r1，Rk_IJKL_part2 is the part of r2>r1. Note that the defination of detainc function in Reference paper is different from scipy.special.betainc.
                    alpha1 = alpha_I_i + alpha_J_j
                    alpha2 = alpha_K_k + alpha_L_l
                    n1_part1 = l_I + l_J + 1 - k
                    n2_part1 = l_K + l_L + 2 + k
                    # Reference: Relativistic calculations for atoms: self-consistent treatment of Breit interaction and nuclear volume effect eq.(1.34)
                    Rk_part1 = common_factor * gamma(0.5 * (n1_part1 + n2_part1 + 2)) / (4 * pow(alpha2, 0.5 * (n2_part1 + 1)) * pow(alpha1, 0.5 * (n1_part1 + 1)) ) * betainc(
                            0.5 * (n2_part1 + 1), 0.5 * (n1_part1 + 1), alpha2 / (alpha1 + alpha2)) * beta(0.5 * (n2_part1 + 1), 0.5 * (n1_part1 + 1))

                    n1_part2 = l_I + l_J + 2 + k
                    n2_part2 = l_K + l_L + 1 - k
                    # Reference: Relativistic calculations for atoms: self-consistent treatment of Breit interaction and nuclear volume effect eq.(1.34)
                    Rk_part2 = common_factor * ( 0.25 * pow(alpha1, -0.5 * (n1_part2 + 1)) * gamma(0.5 * (n1_part2 + 1)) * pow(alpha2, -0.5 * (
                            n2_part2 + 1)) * gamma(0.5 * (n2_part2 + 1)) - gamma(0.5 * (n1_part2 + n2_part2 + 2)) / (4 * pow(alpha2, 0.5 * (n2_part2 + 1)) * pow(alpha1, 0.5 * (
                            n1_part2 + 1))) * betainc(0.5 * (n2_part2 + 1), 0.5 * (n1_part2 + 1), alpha2 / (alpha1 + alpha2)) * beta(0.5 * (n2_part2 + 1), 0.5 * (n1_part2 + 1)) )

                    Rk += (Rk_part1 + Rk_part2)

    return Rk

def get_2e_integral(tuple_I, tuple_J, tuple_K, tuple_L):
    integral_value = 0

    tuple_ak_list = get_ak(tuple_I, tuple_J, tuple_K, tuple_L)

    if tuple_ak_list[0] != 'nan':
        for tuple_ak in tuple_ak_list:
            k = tuple_ak[4]
            ak = tuple_ak[5]
            rk = get_Rk(tuple_I, tuple_J, tuple_K, tuple_L, k)
            integral_value += (ak*rk)

        return integral_value

    else:
        return 0

def get_k_dict(tuple_list):
    # [IJ|KL]
    k_dict = {}
    compute = False

    for orbital_L in tuple_list:
        index_L, n_L, l_L, m_L = orbital_L[0], orbital_L[1], orbital_L[2], orbital_L[3]
        for orbital_K in tuple_list:
            index_K, n_K, l_K, m_K = orbital_K[0], orbital_K[1], orbital_K[2], orbital_K[3]
            for orbital_J in tuple_list:
                index_J, n_J, l_J, m_J = orbital_J[0], orbital_J[1], orbital_J[2], orbital_J[3]
                for orbital_I in tuple_list:
                    index_I, n_I, l_I, m_I = orbital_I[0], orbital_I[1], orbital_I[2], orbital_I[3]

                    dict_index = str(l_I) + ' ' + str(l_J) + ' ' + str(l_K) + ' ' + str(l_L)
                    integral_type = get_2e_integral_type(orbital_I, orbital_J, orbital_K, orbital_L)

                    if integral_type == 'abab' or integral_type == 'abba':
                        compute = True
                    else:
                        pass

                    if compute == True:

                        k_IJ_lower_bound = abs(l_I - l_J)
                        k_IJ_upper_bound = l_I + l_J
                        k_IJ_list = []
                        for k_IJ in range(k_IJ_lower_bound, k_IJ_upper_bound + 1):
                            if (l_I + l_J + k_IJ) % 2 == 0:
                                k_IJ_list.append(k_IJ)

                        k_KL_lower_bound = abs(l_K - l_L)
                        k_KL_upper_bound = l_K + l_L
                        k_KL_list = []
                        for k_KL in range(k_KL_lower_bound, k_KL_upper_bound + 1):
                            if (l_K + l_L + k_KL) % 2 == 0:
                                k_KL_list.append(k_KL)

                        k_list = []
                        for k in k_IJ_list:
                            if k in k_KL_list:
                                k_list.append(k)
                        if k_list != []:
                            if dict_index not in k_dict:
                                k_dict[dict_index] = k_list

    return k_dict

def get_ak_dict(tuple_list):
    # [IJ|KL]
    ak_dict ={}

    for orbital_L in tuple_list:
        index_L, n_L, l_L, m_L = orbital_L[0], orbital_L[1], orbital_L[2], orbital_L[3]
        for orbital_K in tuple_list:
            index_K, n_K, l_K, m_K = orbital_K[0], orbital_K[1], orbital_K[2], orbital_K[3]
            for orbital_J in tuple_list:
                index_J, n_J, l_J, m_J = orbital_J[0], orbital_J[1], orbital_J[2], orbital_J[3]
                for orbital_I in tuple_list:
                    index_I, n_I, l_I, m_I = orbital_I[0], orbital_I[1], orbital_I[2], orbital_I[3]

                    dict_index_IJKL = str(l_I) + str(m_I) + ' ' + str(l_J) + str(m_J) + ' ' + str(l_K) + str(m_K) + ' ' + str(l_L) + str(m_L)
                    tuple_ak_list = get_ak(orbital_I, orbital_J, orbital_K, orbital_L)

                    if tuple_ak_list[0] == 'nan':
                        pass
                    else:
                        ak_dict_input = []
                        if dict_index_IJKL not in ak_dict:
                            # [IJ|KL] = [IJ|LK] = [JI|KL] = [JI|LK] = [KL|IJ] = [KL|JI] = [LK|IJ] = [LK|JI]
                            dict_index_IJLK = str(l_I) + str(m_I) + ' ' + str(l_J) + str(m_J) + ' ' + str(l_L) + str(m_L) + ' ' + str(l_K) + str(m_K)
                            dict_index_JIKL = str(l_J) + str(m_J) + ' ' + str(l_I) + str(m_I) + ' ' + str(l_K) + str(m_K) + ' ' + str(l_L) + str(m_L)
                            dict_index_JILK = str(l_J) + str(m_J) + ' ' + str(l_I) + str(m_I) + ' ' + str(l_L) + str(m_L) + ' ' + str(l_K) + str(m_K)
                            dict_index_KLIJ = str(l_K) + str(m_K) + ' ' + str(l_L) + str(m_L) + ' ' + str(l_I) + str(m_I) + ' ' + str(l_J) + str(m_J)
                            dict_index_KLJI = str(l_K) + str(m_K) + ' ' + str(l_L) + str(m_L) + ' ' + str(l_J) + str(m_J) + ' ' + str(l_I) + str(m_I)
                            dict_index_LKIJ = str(l_L) + str(m_L) + ' ' + str(l_K) + str(m_K) + ' ' + str(l_I) + str(m_I) + ' ' + str(l_J) + str(m_J)
                            dict_index_LKJI = str(l_L) + str(m_L) + ' ' + str(l_K) + str(m_K) + ' ' + str(l_J) + str(m_J) + ' ' + str(l_I) + str(m_I)
                            if dict_index_IJLK in ak_dict:
                                ak_dict[dict_index_IJKL] = ak_dict[dict_index_IJLK]
                            elif dict_index_JIKL in ak_dict:
                                ak_dict[dict_index_IJKL] = ak_dict[dict_index_JIKL]
                            elif dict_index_JILK in ak_dict:
                                ak_dict[dict_index_IJKL] = ak_dict[dict_index_JILK]
                            elif dict_index_KLIJ in ak_dict:
                                ak_dict[dict_index_IJKL] = ak_dict[dict_index_KLIJ]
                            elif dict_index_KLJI in ak_dict:
                                ak_dict[dict_index_IJKL] = ak_dict[dict_index_KLJI]
                            elif dict_index_LKIJ in ak_dict:
                                ak_dict[dict_index_IJKL] = ak_dict[dict_index_LKIJ]
                            elif dict_index_LKJI in ak_dict:
                                ak_dict[dict_index_IJKL] = ak_dict[dict_index_LKJI]
                            else:
                                for tuple_ak in tuple_ak_list:
                                    tuple_ak_k = tuple_ak[5], tuple_ak[4]
                                    ak_dict_input.append(tuple_ak_k)
                                ak_dict[dict_index_IJKL] = ak_dict_input

    return ak_dict

def get_Rk_dict(tuple_list):
    # [IJ|KL]
    Rk_dict ={}

    for orbital_L in tuple_list:
        index_L, n_L, l_L, m_L = orbital_L[0], orbital_L[1], orbital_L[2], orbital_L[3]
        for orbital_K in tuple_list:
            index_K, n_K, l_K, m_K = orbital_K[0], orbital_K[1], orbital_K[2], orbital_K[3]
            for orbital_J in tuple_list:
                index_J, n_J, l_J, m_J = orbital_J[0], orbital_J[1], orbital_J[2], orbital_J[3]
                for orbital_I in tuple_list:
                    index_I, n_I, l_I, m_I = orbital_I[0], orbital_I[1], orbital_I[2], orbital_I[3]

                    global k_dict
                    k_dict_index = str(l_I) + ' ' + str(l_J) + ' ' + str(l_K) + ' ' + str(l_L)
                    if k_dict_index in k_dict:
                        for k in k_dict[k_dict_index]:
                            dict_index_IJKL = str(n_I) + str(l_I) + ' ' + str(n_J) + str(l_J) + ' ' + str(n_K) + str(l_K) + ' ' + str(n_L) + str(l_L) + '+' + str(k)
                            if dict_index_IJKL not in Rk_dict:
                                # [IJ|KL] = [IJ|LK] = [JI|KL] = [JI|LK] = [KL|IJ] = [KL|JI] = [LK|IJ] = [LK|JI]
                                dict_index_IJLK = str(n_I) + str(l_I) + ' ' + str(n_J) + str(l_J) + ' ' + str(n_L) + str(l_L) + ' ' + str(n_K) + str(l_K) + '+' + str(k)
                                dict_index_JIKL = str(n_J) + str(l_J) + ' ' + str(n_I) + str(l_I) + ' ' + str(n_K) + str(l_K) + ' ' + str(n_L) + str(l_L) + '+' + str(k)
                                dict_index_JILK = str(n_J) + str(l_J) + ' ' + str(n_I) + str(l_I) + ' ' + str(n_L) + str(l_L) + ' ' + str(n_K) + str(l_K) + '+' + str(k)
                                dict_index_KLIJ = str(n_K) + str(l_K) + ' ' + str(n_L) + str(l_L) + ' ' + str(n_I) + str(l_I) + ' ' + str(n_J) + str(l_J) + '+' + str(k)
                                dict_index_KLJI = str(n_K) + str(l_K) + ' ' + str(n_L) + str(l_L) + ' ' + str(n_J) + str(l_J) + ' ' + str(n_I) + str(l_I) + '+' + str(k)
                                dict_index_LKIJ = str(n_L) + str(l_L) + ' ' + str(n_K) + str(l_K) + ' ' + str(n_I) + str(l_I) + ' ' + str(n_J) + str(l_J) + '+' + str(k)
                                dict_index_LKJI = str(n_L) + str(l_L) + ' ' + str(n_K) + str(l_K) + ' ' + str(n_J) + str(l_J) + ' ' + str(n_I) + str(l_I) + '+' + str(k)
                                if dict_index_IJLK in Rk_dict:
                                    Rk_dict[dict_index_IJKL] = Rk_dict[dict_index_IJLK]
                                elif dict_index_JIKL in Rk_dict:
                                    Rk_dict[dict_index_IJKL] = Rk_dict[dict_index_JIKL]
                                elif dict_index_JILK in Rk_dict:
                                    Rk_dict[dict_index_IJKL] = Rk_dict[dict_index_JILK]
                                elif dict_index_KLIJ in Rk_dict:
                                    Rk_dict[dict_index_IJKL] = Rk_dict[dict_index_KLIJ]
                                elif dict_index_KLJI in Rk_dict:
                                    Rk_dict[dict_index_IJKL] = Rk_dict[dict_index_KLJI]
                                elif dict_index_LKIJ in Rk_dict:
                                    Rk_dict[dict_index_IJKL] = Rk_dict[dict_index_LKIJ]
                                elif dict_index_LKJI in Rk_dict:
                                    Rk_dict[dict_index_IJKL] = Rk_dict[dict_index_LKJI]
                                else:
                                    Rk_dict[dict_index_IJKL] = get_Rk(orbital_I, orbital_J, orbital_K, orbital_L, k)

    return Rk_dict

def get_2e_integral_dict(tuple_list):
    # [IJ|KL]
    two_e_integral_dict = {}

    for orbital_L in tuple_list:
        index_L, n_L, l_L, m_L = orbital_L[0], orbital_L[1], orbital_L[2], orbital_L[3]
        for orbital_K in tuple_list:
            index_K, n_K, l_K, m_K = orbital_K[0], orbital_K[1], orbital_K[2], orbital_K[3]
            for orbital_J in tuple_list:
                index_J, n_J, l_J, m_J = orbital_J[0], orbital_J[1], orbital_J[2], orbital_J[3]
                for orbital_I in tuple_list:
                    index_I, n_I, l_I, m_I = orbital_I[0], orbital_I[1], orbital_I[2], orbital_I[3]

                    global k_dict, ak_dict, Rk_dict

                    ak_dict_index_IJKL = str(l_I) + str(m_I) + ' ' + str(l_J) + str(m_J) + ' ' + str(l_K) + str(m_K) + ' ' + str(l_L) + str(m_L)
                    dict_index_IJKL = str(index_I) + ' ' + str(index_J) + ' ' + str(index_K) + ' ' + str(index_L)

                    if dict_index_IJKL not in two_e_integral_dict:
                        dict_index_IJLK = str(index_I) + ' ' + str(index_J) + ' ' + str(index_L) + ' ' + str(index_K)
                        if ak_dict_index_IJKL in ak_dict:
                            two_e_integral_value = 0

                            for tuple_ak_k in ak_dict[ak_dict_index_IJKL]:
                                ak, k = tuple_ak_k[0], tuple_ak_k[1]
                                Rk_dict_index_IJKL = str(n_I) + str(l_I) + ' ' + str(n_J) + str(l_J) + ' ' + str(n_K) + str(l_K) + ' ' + str(n_L) + str(l_L) + '+' + str(k)
                                if Rk_dict_index_IJKL in Rk_dict:
                                    Rk = Rk_dict[Rk_dict_index_IJKL]
                                    two_e_integral_value += (ak*Rk)

                            two_e_integral_dict[dict_index_IJKL] = two_e_integral_value

    return two_e_integral_dict

def get_2e_integral_array(tuple_list):
    dim = len(tuple_list)
    global two_e_integral_dict

    two_e_integral_array = np.zeros((dim,dim,dim,dim))
    for two_e_integral_index in two_e_integral_dict:
        i, j, k, l = map( int(), two_e_integral_index.split() )
        two_e_integral_array[i-1][j-1][k-1][l-1] = two_e_integral_dict[two_e_integral_index]

    return two_e_integral_array

def G(tuple1, tuple2, P, tuple_list):
    dim = len(tuple_list)
    G_value = 0
    global two_e_integral_dict
    # Reference: 量子化学——基本原理和从头计算法(中册) P227
    # for P matrix, elements could be represented as P_st, s = row_index, t = column_index
    for row_index in range(dim):
        for column_index in range(dim):
            if P[row_index][column_index] == 0:
                pass
            else:
                J_index = str(tuple1[0]) + ' ' + str(tuple2[0]) + ' ' + str(tuple_list[row_index][0]) + ' ' + str(tuple_list[column_index][0])
                K_index = str(tuple1[0]) + ' ' + str(tuple_list[column_index][0]) + ' ' + str(tuple2[0]) + ' ' + str(tuple_list[row_index][0])
                if J_index in two_e_integral_dict and K_index in two_e_integral_dict:
                    J = two_e_integral_dict[J_index]
                    K = two_e_integral_dict[K_index]
                    G_value += P[row_index][column_index] * (2 * J - K)
    return G_value

def form_G_matrix(P, tuple_list):
    dim = len(tuple_list)
    G_matrix = np.zeros((dim, dim))

    for i in tuple_list:
        for j in tuple_list:
            index1 = i[0]
            index2 = j[0]
            G_value = G(i, j, P, tuple_list)
            G_matrix[index1 - 1][index2 - 1] = G_value

    return G_matrix

def F(atomic_num, wfn_i, tuple1, tuple2, P, tuple_list):
    H_value = H(atomic_num, wfn_i, tuple1, tuple2)
    G_value = G(tuple1, tuple2, P, tuple_list)
    F_value = H_value + G_value

    return F_value

def form_F_matrix(H_core, G_matrix):
    return H_core + G_matrix

def get_A_matrix(S_matrix):
    # In order to do canonical orthogonalization, we need to get Transform matrix A from S (via Unitary matrix U and diagonal matrix s).
    # Reference: 量子化学——基本原理和从头计算法(中册) P228
    U_matrix = linalg.eigh(S_matrix)[1]
    print('\nUnitary Matrix U\n', U_matrix)
    s_matrix = np.around(np.dot(np.dot(linalg.inv(U_matrix),S_matrix), U_matrix), decimals=8, out=None)
    print('\nOrthogonal Matrix s\n', s_matrix)
    A_matrix = np.dot(U_matrix, np.where(s_matrix != 0, pow(s_matrix, -0.5), 0))

    return A_matrix

def get_initial_P_matrix(n, tuple_list):
    dim = len(tuple_list)
    P_matrix = np.zeros((dim, dim))

    for n_i in range(1,int(n/2)+1):
        P_matrix[n_i-1][n_i-1] = 1

    return P_matrix

def update_P(P_old, n, C):
    # Note that deepcopy must be used here and afterwards to decouple the change of P_new & P_old
    dim = np.shape(P_old)[0]
    P_new = np.zeros((dim,dim))

    for row_index in range(dim):
        for column_index in range(dim):
            for n_i in range(1,int(n/2)+1):
                # Reference: 量子化学——基本原理和从头计算法(中册) P227
                P_new[row_index][column_index] += (C[row_index][n_i-1] * C[column_index][n_i-1])

    return P_new

def sorted_eigenvector(F, rev=False):
    # Note that only when the eigenvalues and eigenvectors are in good sequence can we start SCF.
    initial_energy, C_prime = linalg.eigh(F)
    temp_list = []

    for eigen_index in range(len(initial_energy)):
        temp_tuple = initial_energy[eigen_index], C_prime[...,eigen_index]
        temp_list.append(temp_tuple)

    initial_energy_sorted = sorted(initial_energy, reverse=rev)
    temp_list_sorted = sorted(temp_list, key=lambda x:x[0], reverse=rev)
    C_prime_repack = np.array([])

    for index in range(len(temp_list_sorted)):
        C_prime_vector = np.array(temp_list_sorted[index][1]).reshape((len(initial_energy), 1))
        if index == 0:
            C_prime_repack = C_prime_vector
        else:
            C_prime_repack = np.hstack( (C_prime_repack, C_prime_vector) )

    return initial_energy_sorted, C_prime_repack

def Hartree_Fock(A_matrix, Initial_P_matrix, initial_F_matrix, H_core, tuple_list, n, E_convergence_limit=1e-10, P_convergence_limit=1e-6, circle_limit=1000):
    circle_done = 0
    ground_state_energy_list = []
    total_SCF_energy_list = []
    P_list = []
    global final_energy, final_C, final_total_SCF_energy

    print('\n********Start Hartree-Fock calculation!********')
    # Reference1: 量子化学——基本原理和从头计算法(中册) P228-229
    initial_F_prime_matrix = np.dot(np.dot(A_matrix.T, initial_F_matrix), A_matrix)
    print('\nInitial F\'\n', initial_F_prime_matrix)
    initial_energy = sorted_eigenvector(initial_F_prime_matrix)[0]
    print('')
    initial_ground_state_energy = initial_energy[0]
    ground_state_energy_list.append(initial_ground_state_energy)
    for seq in range(len(tuple_list)):
        print('Initial Energy of Orbital', seq + 1, initial_energy[seq], 'hartree')
    initial_total_SCF_energy = 0
    initial_PH = np.dot(Initial_P_matrix, H_core)
    for index in range(1, int(n / 2) + 1):
        initial_total_SCF_energy += initial_energy[index - 1]
    initial_total_SCF_energy += np.trace(initial_PH)
    print('Initial Total SCF energy', initial_total_SCF_energy, 'hartree')
    total_SCF_energy_list.append(initial_total_SCF_energy)

    C_prime = sorted_eigenvector(initial_F_prime_matrix)[1]
    print('\nEigen Matrix C\'\n', C_prime)
    C = np.dot(A_matrix, C_prime)
    print('\nCoefficient Matrix C\n', C)
    P = update_P(Initial_P_matrix, n, C)
    print('\n Density Matrix P\n', P)
    P_list.append(copy.deepcopy(P))

    print('\n********Start SCF Circles!********')
    while circle_done<circle_limit:
        print('\n****Circle****',circle_done+1)
        G_new = form_G_matrix(P_list[circle_done], tuple_list)
        print('\nTwo-Electron Potential Matrix G\n', G_new)

        F_new = form_F_matrix(H_core, G_new)
        print('\nFock Matrix F\n', F_new)
        F_prime_new = np.dot(np.dot(A_matrix.T, F_new), A_matrix)
        print('\nF\'\n', F_prime_new)
        new_energy = sorted_eigenvector(F_prime_new)[0]
        print('')
        new_ground_state_energy = new_energy[0]
        ground_state_energy_list.append(new_ground_state_energy)
        for seq in range(len(tuple_list)):
            print('Energy of Orbital', seq + 1, new_energy[seq], 'hartree')
        new_total_SCF_energy = 0
        new_PH = np.dot(P, H_core)
        for index in range(1, int(n / 2) + 1):
            new_total_SCF_energy += new_energy[index - 1]
        new_total_SCF_energy += np.trace(new_PH)
        print('Total SCF energy', new_total_SCF_energy, 'hartree')
        total_SCF_energy_list.append(new_total_SCF_energy)

        C_prime_new = sorted_eigenvector(F_prime_new)[1]
        print('\nEigen Matrix C\'\n', C_prime_new)
        C_new = np.dot(A_matrix, C_prime_new)
        print('\nCoefficient Matrix C\n', C_new)
        P_new = update_P(P_list[circle_done], n, C_new)
        print('\nDensity Matrix P\n', P_new)
        P_list.append(copy.deepcopy(P_new))

        delta_E_ground = ground_state_energy_list[circle_done]-ground_state_energy_list[circle_done+1]
        print('\nDelta E ground', delta_E_ground)
        delta_P = msqrt( np.sum( np.square(P_list[circle_done]-P_list[circle_done+1]) ) )
        print('Delta P', delta_P)

        if abs(delta_E_ground) < E_convergence_limit and delta_P < P_convergence_limit:
            print('\nConvergence reached in', circle_done+1,'circles!\n')

            final_energy = new_energy
            for seq in range(len(tuple_list)):
                print('Final Energy of Orbital', seq + 1, final_energy[seq], 'hartree')
            final_F = F_new
            print('\nFinal Fock Matrix F\n', final_F)
            final_P = P_new
            print('\nFinal Density Matrix P\n', final_P)
            final_C = C_new
            print('\nFinal Coefficient Matrix C\n', final_C)

            # Reference2: 量子化学——基本原理和从头计算法(中册) P229
            final_total_SCF_energy = 0
            final_PH = np.dot(final_P, H_core)
            for index in range(1,int(n/2)+1):
                final_total_SCF_energy += new_energy[index-1]
            final_total_SCF_energy += np.trace(final_PH)
            print('\nFinal total SCF energy of restricted close shell system E(RHF)', final_total_SCF_energy, 'hartree')

            break

        else:
            circle_done += 1

    return None

def g_abcd(tuple_a, tuple_b, tuple_c, tuple_d):
    g_value = 0
    global two_e_integral_dict, final_C
    dim = np.shape(final_C)[0]

    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    two_e_integral_index = str(a+1) + ' ' + str(b+1) + ' ' + str(c+1) + ' ' + str(d+1)
                    if two_e_integral_index in two_e_integral_dict:
                        g_value += (final_C[a][tuple_a[0]-1] * final_C[b][tuple_b[0]-1] * final_C[c][tuple_c[0]-1] * final_C[d][tuple_d[0]-1] * two_e_integral_dict[two_e_integral_index])
                    else:
                        pass
    return g_value

def get_g_abcd_dict(n, tuple_list):
    g_abcd_dict = {}
    occupied_list = []
    unoccupied_list = []
    global final_C

    for e in range(int(n/2)):
        occupied_list.append(tuple_list[e])
    for e in range(int(n/2),len(tuple_list)):
        unoccupied_list.append(tuple_list[e])

    # Assume that I & K are occupied orbitals while J & L are unoccupied orbitals.
    for orbital_L in unoccupied_list:
        index_L, n_L, l_L, m_L = orbital_L[0], orbital_L[1], orbital_L[2], orbital_L[3]
        for orbital_K in occupied_list:
            index_K, n_K, l_K, m_K = orbital_K[0], orbital_K[1], orbital_K[2], orbital_K[3]
            for orbital_J in unoccupied_list:
                index_J, n_J, l_J, m_J = orbital_J[0], orbital_J[1], orbital_J[2], orbital_J[3]
                for orbital_I in occupied_list:
                    index_I, n_I, l_I, m_I = orbital_I[0], orbital_I[1], orbital_I[2], orbital_I[3]

                    dict_index_IJKL = str(index_I) + ' ' + str(index_J) + ' ' + str(index_K) + ' ' + str(index_L)
                    dict_index_IJLK = str(index_I) + ' ' + str(index_J) + ' ' + str(index_L) + ' ' + str(index_K)
                    dict_index_JIKL = str(index_J) + ' ' + str(index_I) + ' ' + str(index_K) + ' ' + str(index_L)
                    dict_index_JILK = str(index_J) + ' ' + str(index_I) + ' ' + str(index_L) + ' ' + str(index_K)
                    dict_index_KLIJ = str(index_K) + ' ' + str(index_L) + ' ' + str(index_I) + ' ' + str(index_J)
                    dict_index_KLJI = str(index_K) + ' ' + str(index_L) + ' ' + str(index_J) + ' ' + str(index_I)
                    dict_index_LKIJ = str(index_L) + ' ' + str(index_K) + ' ' + str(index_I) + ' ' + str(index_J)
                    dict_index_LKJI = str(index_L) + ' ' + str(index_J) + ' ' + str(index_J) + ' ' + str(index_I)

                    if dict_index_IJKL not in g_abcd_dict:
                        # [IJ|KL] = [IJ|LK] = [JI|KL] = [JI|LK] = [KL|IJ] = [KL|JI] = [LK|IJ] = [LK|JI]
                        if dict_index_IJLK in g_abcd_dict:
                            g_abcd_dict[dict_index_IJKL] = g_abcd_dict[dict_index_IJLK]
                        elif dict_index_JIKL in g_abcd_dict:
                            g_abcd_dict[dict_index_IJKL] = g_abcd_dict[dict_index_JIKL]
                        elif dict_index_JILK in g_abcd_dict:
                            g_abcd_dict[dict_index_IJKL] = g_abcd_dict[dict_index_JILK]
                        elif dict_index_KLIJ in g_abcd_dict:
                            g_abcd_dict[dict_index_IJKL] = g_abcd_dict[dict_index_KLIJ]
                        elif dict_index_KLJI in g_abcd_dict:
                            g_abcd_dict[dict_index_IJKL] = g_abcd_dict[dict_index_KLJI]
                        elif dict_index_LKIJ in g_abcd_dict:
                            g_abcd_dict[dict_index_IJKL] = g_abcd_dict[dict_index_LKIJ]
                        elif dict_index_LKJI in g_abcd_dict:
                            g_abcd_dict[dict_index_IJKL] = g_abcd_dict[dict_index_LKJI]
                        else:
                            g_abcd_dict[dict_index_IJKL] = g_abcd(orbital_I, orbital_J, orbital_K, orbital_L)

                    if dict_index_JILK not in g_abcd_dict:
                        #  [JI|LK] = [KL|IJ] = [KL|JI] = [LK|IJ] = [LK|JI] = [IJ|KL] = [IJ|LK] = [JI|KL]
                        if dict_index_KLIJ in g_abcd_dict:
                            g_abcd_dict[dict_index_JILK] = g_abcd_dict[dict_index_KLIJ]
                        elif dict_index_KLJI in g_abcd_dict:
                            g_abcd_dict[dict_index_JILK] = g_abcd_dict[dict_index_KLJI]
                        elif dict_index_LKIJ in g_abcd_dict:
                            g_abcd_dict[dict_index_JILK] = g_abcd_dict[dict_index_LKIJ]
                        elif dict_index_LKJI in g_abcd_dict:
                            g_abcd_dict[dict_index_JILK] = g_abcd_dict[dict_index_LKJI]
                        elif dict_index_IJKL in g_abcd_dict:
                            g_abcd_dict[dict_index_JILK] = g_abcd_dict[dict_index_IJKL]
                        elif dict_index_IJLK in g_abcd_dict:
                            g_abcd_dict[dict_index_JILK] = g_abcd_dict[dict_index_IJLK]
                        elif dict_index_JIKL in g_abcd_dict:
                            g_abcd_dict[dict_index_JILK] = g_abcd_dict[dict_index_JIKL]
                        else:
                            g_abcd_dict[dict_index_JILK] = g_abcd(orbital_J, orbital_I, orbital_L, orbital_K)

    return g_abcd_dict

def MP2(n, tuple_list):
    print('\n********Start MP2 calculation!********')
    # Reference3: 量子化学——基本原理和从头计算法(中册) P403-406
    # Note that ab represent a virtual orbital respectively, while kl represent an occupied orbital respectively (For He, k=1, l=1).
    # MP2 energy could be calculated by eq.(14.5.34) in 量子化学——基本原理和从头计算法(中册) P406
    second_order_energy = 0
    dim = len(tuple_list)
    global final_energy, final_C, final_total_SCF_energy, g_abcd_dict

    for a in range(int(n / 2) + 1, dim + 1):
        Energy_a = final_energy[a - 1]
        for b in range(int(n / 2) + 1, dim + 1):
            Energy_b = final_energy[b - 1]
            for k in range(1, int(n / 2) + 1):
                Energy_k = final_energy[k - 1]
                for l in range(1, int(n / 2) + 1):
                    Energy_l = final_energy[l - 1]

                    # Reference4: 量子化学——基本原理和从头计算法(中册) P226
                    MP2_part1 = g_abcd_dict[str(k) + ' ' + str(a) + ' ' + str(l) + ' ' + str(b)]
                    MP2_part2 = g_abcd_dict[str(a) + ' ' + str(k) + ' ' + str(b) + ' ' + str(l)]
                    MP2_part3 = g_abcd_dict[str(b) + ' ' + str(k) + ' ' + str(a) + ' ' + str(l)]
                    second_order_energy += (MP2_part1 * (2 * MP2_part2 - MP2_part3) / (Energy_k + Energy_l - Energy_a - Energy_b))

    print('\nSecond order perturbation energy E2', second_order_energy, 'hartree')

    total_MP2_energy = final_total_SCF_energy + second_order_energy
    print('\nTotal MP2 energy E(MP2)', total_MP2_energy, 'hartree')

    return None


'''
Hopefully the name of intermediate variables can show how the program runs to some extent. 
The commands are as follows. Set the basis set, calculation level and atomic number of element before running your task.
'''
# Commands
total_s_time = datetime.datetime.now()

task = basis_set.basis_set('He', 'cc-pVDZ')
wfn = task.wfn()
atom = task.atom
basis_set_name = task.basis_set_name
task_name = atom + ' MP2/' + basis_set_name
a_n = 2
e_n = 2
print('Task Name:')
print(task_name,' Atomic number:', a_n,' Electron number:', e_n,'\n')
print('Wavefunctions:')
# print(wfn,'\n')
for i in wfn:
    print(i, '\nalpha\n' , wfn[i][0], '\nconcn\n', wfn[i][1])

# wfn_array = form_wfn_array(wfn)
# print('\nwfn_array\n', wfn_array)
# wfn_matrix = form_wfn_matrix(wfn_array)

tuple_list = get_tuple_list(wfn)
print('Orbitals\n', tuple_list)


# Get all integrals that two-electron integral requires in dictionaries.
unnormed_S_matrix = form_unnormed_S_matrix(tuple_list)

print('')
k_s_time = datetime.datetime.now()
k_dict = get_k_dict(tuple_list)
k_e_time = datetime.datetime.now()
# print('\n', k_dict)
print('k takes', (k_e_time - k_s_time).seconds, 'seconds.')

ak_s_time = datetime.datetime.now()
ak_dict = get_ak_dict(tuple_list)
ak_e_time = datetime.datetime.now()
# print('\n', ak_dict)
print('ak takes', (ak_e_time - ak_s_time).seconds, 'seconds.')

Rk_s_time = datetime.datetime.now()
Rk_dict = get_Rk_dict(tuple_list)
Rk_e_time = datetime.datetime.now()
# print('\n', Rk_dict)
print('Rk takes', (Rk_e_time - Rk_s_time).seconds, 'seconds.')

two_e_integral_s_time = datetime.datetime.now()
two_e_integral_dict = get_2e_integral_dict(tuple_list)
two_e_integral_e_time = datetime.datetime.now()
# print('\n', two_e_integral_dict)
print('Two-Electron integral takes', (two_e_integral_e_time-two_e_integral_s_time).seconds, 'seconds.')


# # Get all matrices.
print('\nUnnormed Overlap Integral S\n', unnormed_S_matrix)
S_matrix = form_S_matrix(tuple_list)
print('\nOverlap Integral S\n', S_matrix)
T_matrix = form_T_matrix(tuple_list)
print('\nKinetic Integral T\n', T_matrix)
V_matrix = form_V_matrix(a_n, tuple_list)
print('\nPotential Energy V\n', V_matrix)
H_core = form_H_core_matrix(T_matrix, V_matrix)
print('\nCore Hamiltonian H core\n', H_core)
Initial_P = get_initial_P_matrix(e_n, tuple_list)
print('\nInitial Density Matrix P\n', Initial_P)

Initial_G = form_G_matrix(Initial_P, tuple_list)
print('\nInitial Two-Electron Potential Matrix G\n', Initial_G)
Initial_F = form_F_matrix(H_core, Initial_G)
print('\nInitial Fock Matrix F\n', Initial_F)

A_matrix = get_A_matrix(S_matrix)
print('\nTransform Matrix A\n', A_matrix)


# SCF
SCF_s_time = datetime.datetime.now()
Hartree_Fock(A_matrix, Initial_P, Initial_F, H_core, tuple_list, e_n, E_convergence_limit=1e-10, P_convergence_limit=1e-6, circle_limit=1000)
SCF_e_time = datetime.datetime.now()
print('\nSCF takes', (SCF_e_time-SCF_s_time).seconds, 'seconds.')


# MP2
MP2_s_time = datetime.datetime.now()
g_abcd_s_time = datetime.datetime.now()
g_abcd_dict = get_g_abcd_dict(e_n, tuple_list)
g_abcd_e_time = datetime.datetime.now()
print('\ng_abcd takes', (g_abcd_e_time - g_abcd_s_time).seconds, 'seconds.')
MP2(e_n, tuple_list)
MP2_e_time = datetime.datetime.now()
print('\nMP2 takes', (MP2_e_time-MP2_s_time).seconds, 'seconds.')
# print(g_abcd(tuple_list[1], tuple_list[1], tuple_list[0], tuple_list[0], final_C))


total_e_time = datetime.datetime.now()
print('\nThe whole task takes', (total_e_time-total_s_time).seconds, 'seconds.')

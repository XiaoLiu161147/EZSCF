'''
EZSCF is the name of this purely Python-based RHF/RMP2 program of atomic systems, mainly used for the bachelor's thesis of Xiao Liu @ BNU.
The name EZSCF is a homophone with "Easy SCF" and also a pun on the initial Chinese Pinyin letters of Xiao's beloved Alma Mater, Hangzhou No.2 High School (aka 杭州二中 or HZEZ).
Differing from widely used Cartesian GTO basis set, this program applies Spherical Harmonic GTO basis set so as to compute high angular momentum part more efficiently and accurately.

Developed by Xiao Liu under the guidance of Prof. Zhendong Li @ BNU. All references are marked in different places of the program.
Many thanks to Rui Li (from ZJU to Caltech; HZEZ alumnus), Jingze Li (from BNU to UCSD), Jie Feng (from HUST to SJTU; HZEZ alumnus) & Ruiyi Zhou (from HKU to UNC Chapel Hill) for helpful discussions!
A special thank goes to Sobereva & Warm_Cloud.
Last but the most important, thanks Qiming Sun (also HZEZ alumnus) for his prominent open-source program PySCF!

Feel free to contact Xiao Liu via email at the address of 201611150142@mail.bnu.edu.cn if you have any questions or advice.

Xiao Liu
May 1, 2020
'''

import re
import numpy as np

periodic_table = [
'H', 'He',
'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br','Kr',
'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te','I', 'Xe',
'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm','Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr','Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

def atomic_num(atom):
    return periodic_table.index(atom)+1

def get_l(orbital_name):
    orbital_type = re.sub('[^A-Z]', '', orbital_name)

    if orbital_type == 'S':return 0
    if orbital_type == 'P':return 1
    if orbital_type == 'D':return 2
    if orbital_type == 'F':return 3
    if orbital_type == 'G':return 4
    if orbital_type == 'H':return 5
    if orbital_type == 'I':return 6

# for Dunning basis set only
class basis_set:

    cc_pVnZ = ['cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z', 'cc-pV6Z']
    aug_cc_pVnZ = ['aug-cc-pVDZ', 'aug-cc-pVTZ', 'aug-cc-pVQZ', 'aug-cc-pV5Z', 'aug-cc-pV6Z']

    def __init__(self, atom, basis_set_name):
        self.atom = atom
        self.basis_set_name = re.sub('\*', '_st_', basis_set_name)

    def atom_bs(self):
        return self.atom + '_' + re.sub('-', '_', self.basis_set_name)

    def zeta(self):
        if self.basis_set_name[-2] == 'D': return 2
        if self.basis_set_name[-2] == 'T': return 3
        if self.basis_set_name[-2] == 'Q': return 4
        if self.basis_set_name[-2] == '5': return 5
        if self.basis_set_name[-2] == '6': return 6

    def layer_num(self):
        if self.atom in ['H', 'He']: return 1
        if self.atom in ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']: return 2
        if self.atom in ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']: return 3
        if self.atom in ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br','Kr']: return 4
        if self.atom in ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te','I', 'Xe']: return 5
        if self.atom in ['Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm','Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',]: return 6
        if self.atom in ['Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr','Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']: return 7

    def wfn(self):
        text = open(self.atom_bs() + '.txt')
        wfn = {}

        gto_num = 0
        sto_num = 0
        temp_gto_num_1 = 0
        temp_gto_num_2 = 0
        line_num = 0

        for line in text:
            line_split = line.split()
            line_num += 1

            if line_num == 1:
                continue

            elif line_split[0].isalpha() == True:
                sto_type = line_split[0]
                alpha_list = []
                concn_list = []
                gto_num = int(line_split[1])
                sto_num = int(line_split[2])
                temp_gto_num_1 = 0
                temp_gto_num_2 = 0

            elif temp_gto_num_1 != gto_num:
                temp_gto_num_1 += 1
                alpha_list.append(float(line_split[0].replace('D', 'E')))
                alpha_array = np.array([alpha_list]).T
            elif temp_gto_num_1 == gto_num:
                temp_gto_num_2 += 1
                for i in range(sto_num):
                    concn_list.append(float(line_split[i].replace('D', 'E')))
            if temp_gto_num_2 == gto_num:
                concn_array = np.array(concn_list).reshape(gto_num, sto_num)
                wfn[sto_type] = alpha_array, concn_array

        return wfn

class orbital:
    def __init__(self, tuple_list):
        self.index = tuple_list[0]
        self.n = tuple_list[1]
        self.l = tuple_list[2]
        self.m = tuple_list[3]
        self.orbital_type = tuple_list[4]

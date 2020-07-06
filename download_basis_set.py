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

from requests_html import HTMLSession
import re

periodic_table = [
'H', 'He',
'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br','Kr',
'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te','I', 'Xe',
'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm','Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr','Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

session = HTMLSession()

def download_basis_set(atom_num, basis_set_name):
    atom_name = periodic_table[atom_num-1]
    basis_set_name_in_url = re.sub('\*', '_st_', basis_set_name)
    api = 'http://www.basissetexchange.org/api/basis/' + basis_set_name_in_url + '/format/bdf/?version=1&elements=' + str(atom_num)
    r = session.get(api)
    content = r.html.html

    pattern = '\*\*\*\*' + '(?:.|\n)*'
    text = re.findall(pattern, content)

    txt_name = atom_name + '_' + re.sub('-', '_', basis_set_name_in_url) + '.txt'
    open_txt = open(txt_name, mode='w')

    if text == []:
        print(atom_name + ' does not have the definition of ' + basis_set_name + ' basis set.')

    else:
        for i in text:
            open_txt.write(i.strip('\*\*\*\*\n'))
            print(txt_name, 'is finished!')

    return 0

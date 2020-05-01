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
    api = 'http://www.basissetexchange.org/api/basis/' + basis_set_name + '/format/bdf/?version=1&elements=' + str(atom_num)
    r = session.get(api)
    content = r.html.html

    pattern = '\*\*\*\*' + '(?:.|\n)*'
    text = re.findall(pattern, content)

    txt_name = atom_name + '_' + re.sub('-', '_', basis_set_name) + '.txt'
    open_txt = open(txt_name, mode='w')

    if text == []:
        print(atom_name + ' does not have the definition of ' + basis_set_name + ' basis set.')

    else:
        for i in text:
            open_txt.write(i.strip('\*\*\*\*\n'))
            print(txt_name, 'is finished!')

    return 0
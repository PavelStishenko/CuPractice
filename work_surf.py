from newscripts.cu2o_bulk_colab import cu2o_bulk, cu2o111, CuD_FCC111, STO_FCC111, CuDO_FCC111, py111
from newscripts.cu2o_bulk_colab import cu2o100, Oterm1x1, Cuterm1x1, slab3011, dimer1x1, c2x2, cu2o_halfCu_100

from newscripts.cu2o_bulk_colab import bridge_Cl_ST111, Cl_atop_satCu_ST111, Cl_atop_unsatCu_ST111

from ase.spacegroup import get_spacegroup
import numpy as np
from ase.io import read, write

vacuum_thicknes = 15
b = cu2o_bulk()


if 1:
  for make_slab in [cu2o111, CuD_FCC111, STO_FCC111, CuDO_FCC111, py111]:
  #for make_slab in [cu2o_halfCu_100, cu2o100, Oterm1x1, Cuterm1x1, slab3011, dimer1x1, c2x2]:
    for n_layers in range(3,  11):
      slab = make_slab(cu2o_bulk(), n_layers, vacuum=vacuum_thicknes)
      sg = get_spacegroup(slab)
      nCu = sum(slab.symbols=='Cu')
      nO = sum(slab.symbols=='O')
      print (make_slab.__name__, f"{n_layers:4d} {nCu/n_layers:.2f} {nO/n_layers:.2f} {nO - nCu/2:.2f} {(np.max(slab.positions[:,2]) - np.min(slab.positions[:,2])):7.4f} {len(slab):4d}",  sg.centrosymmetric)
      write(f'{make_slab.__name__}_{n_layers:02d}.xyz', [slab])
    print('')




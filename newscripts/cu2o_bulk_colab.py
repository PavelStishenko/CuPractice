from ase.io import read, write
from ase import Atoms
from ase.build import surface
from ase.build import bulk
from ase.build.supercells import make_supercell
from ase.geometry import get_distances
import numpy as np

def cu2o_bulk():
    bulk=read('9007497.cif')
    return bulk

def cu2o111(bulk, n_layers, vacuum):
    bulk.translate(np.array([1.0, 1.0, 1.0])*1.0)
    bulk.wrap()
    slab = surface(bulk, (1,1,1), n_layers, vacuum=vacuum, periodic=True) 
    return slab 
  
def CuD_FCC111(bulk, n_layers, vacuum): #oxygens too high, an easy fix though!
    slab = cu2o111(bulk, n_layers, vacuum)
    unsat_Cu_z = np.max(slab[slab.symbols=='Cu'].positions[:,2]) - 1e-3
    mask_surf_Cu=(slab.positions[:, 2] >= unsat_Cu_z) & (slab.symbols=='Cu')
    assert sum(mask_surf_Cu) == 4
    surf_O_z = unsat_Cu_z - 0.9
    mask_surf_O=(slab.positions[:, 2] >= surf_O_z) & (slab.symbols=='O')
    assert sum(mask_surf_O) == 2
    _,d = get_distances(slab[mask_surf_Cu].positions, slab[mask_surf_O].positions, slab.cell, slab.pbc)
    assert d.shape == (4,2)
    d = np.min(d, axis=1)
    Cu_cus = np.argmax(d)
    del slab[Cu_cus]
    return slab

def STO_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(superslab[superslab.symbols=='O'].positions[:,2])-1e-3
    mask2 = (superslab.positions[:, 2] >= STO_z) & (superslab.symbols=='O')
    assert sum(mask2) == 3
    index_to_remove = np.argmax(mask2)
    del superslab[index_to_remove]
    return superslab
 
def CuDO_FCC111(bulk, n_layers, vacuum): #not checked thoroughly but seems close if not quite perfect
    slab = CuD_FCC111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    
    STO_z = np.max(superslab[superslab.symbols=='O'].positions[:,2])-1e-3
    mask2 = (superslab.positions[:, 2] >= STO_z) & (superslab.symbols=='O')
    assert sum(mask2) == 3
    index_to_remove = np.argmax(mask2)
    del superslab[index_to_remove]
    return superslab
    
def py111(bulk, n_layers, vacuum):
    slab = CuD_FCC111(bulk, n_layers, vacuum)
    slab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]])
    
    surf_O_z = np.max(slab[slab.symbols=='O'].positions[:,2])-1e-3
    mask_surf_O=(slab.positions[:, 2] >= surf_O_z) & (slab.symbols=='O')
    assert sum(mask_surf_O) == 3
    
    pos_O1 =slab.positions[mask_surf_O][0]
    pos_O1[2] = 0
    slab.translate(-pos_O1)
    
    pyr = Atoms(symbols='Cu3O', positions=[[1.513,0.874,surf_O_z+1.0],[3.026,3.495,surf_O_z+1.0], [4.540,0.874,surf_O_z+1.0], [3.026,1.747,surf_O_z+2.0]])
    
    slab = slab + pyr

    return slab

def cu2o100(bulk, n_layers, vacuum):
  slab = surface(bulk, (1,0,0), n_layers, vacuum=vacuum, periodic=True) 
  return slab 
  
def Oterm1x1(bulk, n_layers, vacuum): ##
    slab = cu2o100(bulk, n_layers, vacuum)
    CuMax = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= CuMax) & (slab.symbols=='Cu')
    del slab[mask2]
    return slab
    
def Cuterm1x1 (bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    OMax = np.max(slab[slab.symbols=='O'].positions[:,2])
    mask2=(slab.positions[:, 2] >= OMax) & (slab.symbols=='O')
    del slab[mask2]
    return slab

def slab3011(bulk,n_layers,vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [1,1, 0], [0,0,1]] )
    #Max_Cu_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2]) - 2.0
    #mask2=(superslab.positions[:, 2] >= Max_Cu_z) & (superslab.symbols=='Cu')
    #del superslab[mask2]
    return superslab

def dimer1x1(bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    Cu_max = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask3=(slab.positions[:, 2] >= Cu_max) & (slab.symbols=='Cu')
    indices=list()
    for i in range(len(slab)):
        if mask3[i]==True:
            indices.append(i)
    midpoint = np.mean(slab.positions[indices[:]],axis=0)
    v=(midpoint-slab.positions[indices[0]])/2
    slab.positions[indices[0]]+=-v
    slab.positions[indices[1]]+=v
    return slab

def c2x2(bulk, n_layers, vacuum):
    slab = Oterm1x1(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[1,1,0], [-1,1, 0],  [0,0,1]] )
    Max_Cu_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2]) - 2.0
    mask2=(superslab.positions[:, 2] >= Max_Cu_z) & (superslab.symbols=='Cu')
    del superslab[mask2]
    Max_O_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2]) - 0.5
    mask3=(superslab.positions[:, 2] >= Max_O_z) & (superslab.symbols=='O')
    del superslab[mask3]
    return superslab

def bridge_Cl_ST111(bulk,n_layers,vacuum, sc_size,Cl_X_position,Cl_Y_position,Cl_Z_position):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, sc_size)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads


def bridge_STCl_CuCu(bulk,n_layers,vacuum, sc_size,Cl_Z_position):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    Cl_X_position=-1.25
    Cl_Y_position=0.9
    slab= make_supercell(slab_initial, sc_size)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def NO3_ST111(bulk,n_layers,vacuum, sc_size,Cl_X_position,Cl_Y_position,Cl_Z_position): ##untested function
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, sc_size)
    N_X_position=0
    N_Y_position=0
    N_Z_position=10
    O_X_position=1.4
    O_Y_position=0
    O_Z_position=10

    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    N_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [N_X_position,N_Y_position, N_Z_position] 
    O_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [O_X_position,O_Y_position, O_Z_position] 
    N = Atoms(symbols='N', positions = [N_pos])
    O_ads = Atoms(symbols='O', positions = [O_pos])

    Ob= O_ads.copy()
    Ob.translate([-1.4,1.3,0])
    Oc= O_ads.copy()
    Oc.translate([-2.4,-1,0])

    slabads = slab + O_ads + Ob + Oc + N
    return slabads
    
def Cl_atop_satCu_ST111(bulk,n_layers,vacuum, sc_size,Cl_Z_position):
    Cl_X_position=2.3
    Cl_Y_position=-0.5
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, sc_size)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def Cl_atop_unsatCu_ST111(bulk,n_layers,vacuum, sc_size,Cl_Z_position):
    Cl_X_position=0.75
    Cl_Y_position=-3
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, sc_size)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads

def CuObridge1_STCl(bulk,n_layers,vacuum, sc_size):
    slab = cu2o111(bulk, n_layers, vacuum)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [-1.343,-0.2885, 4]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def CuObridge2_STCl(bulk,n_layers,vacuum, sc_size, Cl_X_position,Cl_Y_position,Cl_Z_position):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, sc_size)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads

    
def hollow_Cuterm100(bulk,n_layers,vacuum, sc_size, Cl_X_position,Cl_Y_position,Cl_Z_position):
    slab_initial= Cuterm1x1 (bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, sc_size)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads


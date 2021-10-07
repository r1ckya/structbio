from Bio.pairwise2 import align
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import Superimposer
from Bio.SeqUtils import seq1


def stringify(chain):
    return "".join(map(lambda r: seq1(r.get_resname()), chain.get_residues()))


def main():
    parser = PDBParser()
    structure = parser.get_structure("5ggs", "5ggs.pdb")

    chain_y = structure[0]["Y"]
    chain_z = structure[0]["Z"]

    res_y = stringify(chain_y).strip("X")
    res_z = stringify(chain_z).strip("X")

    alignment = align.globalxx(res_y, res_z)[0]

    r_y = chain_y.get_residues()
    r_z = chain_z.get_residues()

    fixed_atoms = []
    moving_atoms = []

    for a_y, a_z in zip(alignment.seqA, alignment.seqB):
        if a_y == "-":
            next(r_z)
        elif a_z == "-":
            next(r_y)
        else:
            atoms_y = Selection.unfold_entities(next(r_y), "A")
            atoms_z = Selection.unfold_entities(next(r_z), "A")

            ids_y = set(map(lambda a: a.id, atoms_y))
            ids_z = set(map(lambda a: a.id, atoms_z))

            atoms_y = [a for a in atoms_y if a.id in ids_z]
            atoms_z = [a for a in atoms_z if a.id in ids_y]

            fixed_atoms.extend(atoms_y)
            moving_atoms.extend(atoms_z)

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)

    print(f"Rotran: {sup.rotran}")
    print(f"RMS: {sup.rms}")


if __name__ == "__main__":
    main()

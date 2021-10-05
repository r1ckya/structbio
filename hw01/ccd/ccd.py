import copy
import sys

import numpy as np
from Bio import SeqIO
from Bio.pairwise2 import align, format_alignment
from Bio.PDB import Selection
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from scipy.spatial.transform import Rotation as R

DIST = dict(CN=1.32, NCA=1.47, CAC=1.53)
EPS = 1e-9


def normalize(v):
    return v / np.sqrt(np.dot(v, v))


def rmsd(chain, anchor):
    assert len(chain) >= 3 and len(anchor) == 3
    mse = 0
    for a, b in zip(chain[-3:], anchor):
        d = a.coord - b.coord
        mse += np.dot(d, d)
    return np.sqrt(mse) / 3


def rotation_axis(N, Ca):
    return dict(xyz=N, abc=Ca - N)


def central_point(axis, N, Ca, M):
    v = N - Ca
    t = (np.dot(v, M) - np.dot(v, axis["xyz"])) / np.dot(v, axis["abc"])
    return axis["xyz"] + axis["abc"] * t


def rotate(M, O, alpha, theta):
    v = M - O
    if np.linalg.norm(v) < EPS:
        return O
    v_norm = np.linalg.norm(v)
    v_cap = v / v_norm
    s_cap = np.cross(v_cap, theta)
    return O + v_norm * (np.cos(alpha) * v_cap + np.sin(alpha) * s_cap)


def ccd(chain, anchor, tol=0.08, max_iter=1000):
    chain = copy.deepcopy(chain)

    n_iter = 0

    while True:
        err = rmsd(chain, anchor)

        # if n_iter % 100 == 0:
        #     print(f"iter {n_iter}:, rmsd={err}", file=sys.stderr)

        if err < tol:
            return True, chain, err, n_iter

        if n_iter == max_iter:
            return False, chain, err, n_iter

        n_iter += 1

        for i in range(len(chain) - 2):
            # skip peptide bond, angle \theta = \pi
            if chain[i].name == "C" and chain[i + 1].name == "N":
                continue

            N = chain[i].coord
            Ca = chain[i + 1].coord

            axis = rotation_axis(N, Ca)

            theta = normalize(Ca - N)

            b, c = 0.0, 0.0

            for j in range(1, 4):
                M = chain[-j].coord
                F = anchor[-j].coord

                O = central_point(axis, N, Ca, M)

                r = M - O
                f = F - O
                r_norm = np.linalg.norm(r)

                if r_norm < EPS:
                    continue

                r_cap = r / r_norm
                s_cap = np.cross(r_cap, theta)

                b += 2 * r_norm * np.dot(f, r_cap)
                c += 2 * r_norm * np.dot(f, s_cap)

            x = b / np.sqrt(b ** 2 + c ** 2)
            y = c / np.sqrt(b ** 2 + c ** 2)

            alpha = np.arctan2(y, x)

            for j in range(i + 2, len(chain)):
                O = central_point(axis, N, Ca, chain[j].coord)
                chain[j].set_coord(rotate(chain[j].coord, O, alpha, theta))


def random_rotation(v):
    rot = R.from_euler("xyz", [np.random.uniform(-1, 1) for _ in range(3)])
    return rot.apply(v)


def init_chain(res, hole_begin, hole_len):
    last_res = res[hole_begin - 1]

    v_unit = normalize(last_res["C"].coord - last_res["CA"].coord)

    moving_chain = [last_res[c].copy() for c in ["CA", "C"]]
    coords = last_res["C"].coord.copy()

    for _ in range(hole_len + 1):
        coords += v_unit * DIST["CN"]
        moving_chain.append(
            Atom("N", coords.copy(), 0, 0, "", "N", None, element="N")
        )

        v_unit = random_rotation(v_unit)
        coords += v_unit * DIST["NCA"]
        moving_chain.append(
            Atom("CA", coords.copy(), 0, 0, "", "CA", None, element="C")
        )

        v_unit = random_rotation(v_unit)
        coords += v_unit * DIST["CAC"]
        moving_chain.append(
            Atom("C", coords.copy(), 0, 0, "", "C", None, element="C")
        )

    assert len(moving_chain) == hole_len * 3 + 2 + 3, len(moving_chain)

    next_res = res[hole_begin]
    anchor = [next_res[c].copy() for c in ["N", "CA", "C"]]

    return moving_chain, anchor


def main():
    parser = PDBParser()
    structure = parser.get_structure("unk", "hw_1_hole.pdb")
    chain = structure[0]["Z"]

    res = Selection.unfold_entities(chain, "R")
    seq_w_hole = "".join(seq1(r.resname) for r in res)
    seq = next(SeqIO.parse("PD1.fasta", "fasta")).seq

    alignment = align.globalxx(seq_w_hole, seq)[1]
    print(format_alignment(*alignment))

    hole_begin = alignment.seqA.find("-")
    hole_end = alignment.seqA.rfind("-")
    hole_len = hole_end - hole_begin + 1
    assert hole_len == 7, hole_end - hole_begin + 1

    n_trials = 100
    max_iter = 500
    tol = 0.08
    n_succ = 0
    total_err = 0

    for trial in range(n_trials):
        moving_chain, anchor = init_chain(res, hole_begin, hole_len)
        converged, chain, err, n_iter = ccd(
            moving_chain, anchor, tol, max_iter
        )
        n_succ += converged
        total_err += err
        print(f"trial {trial}, n_iter {n_iter}, err {err}", file=sys.stderr)

    p_succ = np.round(n_succ / n_trials, 3)
    mean_err = np.round(total_err / n_trials, 3)
    print(f"p(converge)={p_succ}, mean_err={mean_err}")


if __name__ == "__main__":
    main()

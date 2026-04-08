# main_diffusion_1d.py
import argparse
import numpy as np
'''
from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_1d_mesh,
    prepare_quadrature_and_basis, get_jacobians, end_dofs_from_nodes
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_fe_solution_high_order, setup_interactive_figure

import argparse
import numpy as np'''
import matplotlib.pyplot as plt

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_2d_wall_mesh,
    prepare_quadrature_and_basis, get_jacobians, border_dofs_from_tags
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import setup_interactive_figure, plot_mesh_2d, plot_fe_solution_2d


'''def main():
    parser = argparse.ArgumentParser(description="Diffusion 1D with theta-scheme (Gmsh high-order FE)")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-cl1", type=float, default=0.05)
    parser.add_argument("-cl2", type=float, default=0.05)
    parser.add_argument("-L", type=float, default=1.0)

    parser.add_argument("--theta", type=float, default=1.0, help="1: implicit Euler, 0.5: Crank-Nicolson, 0: explicit")
    parser.add_argument("--dt", type=float, default=1.0e-04)
    parser.add_argument("--nsteps", type=int, default=500)
    args = parser.parse_args()

    gmsh_init("diffusion_1d")

    L = args.L

    _, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_1d_mesh(
        L=args.L, cl1=args.cl1, cl2=args.cl2, order=args.order
    )

    unique_dofs_tags = np.unique(elemNodeTags)

    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    def kappa(x): return 1.0
    def f_source(x): return 0.0
    def u0(x): return 1.0*np.sin(np.pi*x/L) + 2.0*np.sin(8*np.pi*x/L)

    K_lil, F = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, f_source, tag_to_dof
    )
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

    K = K_lil.tocsr()
    M = M_lil.tocsr()

    n = len(F)
    U = np.array([u0(x) for x in nodeCoords[::3]], dtype=float)

    left, right = end_dofs_from_nodes(nodeCoords)
    dir_dofs = [left, right]
    dir_vals = np.array([0.0, 0.0], dtype=float)

    fig, ax = setup_interactive_figure(xlim=(0.0, args.L))
    u_min = float(np.min(U))
    u_max = float(np.max(U))
    pad = 0.05 * (u_max - u_min + 1e-14)
    ylim = (u_min - pad, u_max + pad)

    import matplotlib.pyplot as plt

    for step in range(args.nsteps):
        U = theta_step(M, K, F, F, U, dt=args.dt, theta=args.theta, dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals)

        ax.clear()
        ax.set_xlim(0.0, args.L)
        ax.set_ylim(*ylim)

        plot_fe_solution_high_order(
            elemType=elemType,
            elemNodeTags=elemNodeTags,
            nodeCoords=nodeCoords,
            U=U,
            M=120,
            show_nodes=False,
            ax=ax
        )

        ax.set_title(f"t = {step * args.dt:.4f}   (theta={args.theta})")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u_h(x,t)$")
        ax.grid(True)

        plt.pause(0.03)

    gmsh_finalize()'''

def main():
    parser = argparse.ArgumentParser(description="Diffusion 2D Multicouche")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=3600.0) # Un pas de temps de 1 heure
    parser.add_argument("--nsteps", type=int, default=100)
    args = parser.parse_args()

    # Import modifié pour appeler notre nouvelle fonction de maillage
    from gmsh_utils import build_2d_wall_mesh
    gmsh_init("mur_multicouche")

    # On génère le mur (cl = 0.005 pour avoir un beau maillage assez fin)
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = build_2d_wall_mesh(
        order=args.order, cl=0.005, H=0.1
    )

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # --- NOUVEAU : Propriétés physiques en fonction de X ---
    def get_material_props(x_coord):
        if x_coord <= 0.015 + 1e-6: # Plâtre
            return 0.35, 1000 * 1000
        elif x_coord <= 0.015 + 0.140 + 1e-6: # Béton
            return 1.80, 2300 * 880
        elif x_coord <= 0.015 + 0.140 + 0.150 + 1e-6: # Isolant
            return 0.035, 20 * 1030
        else: # Brique
            return 1.00, 1900 * 800

    def kappa(x): return get_material_props(x[0])[0]
    def rho_cp(x): return get_material_props(x[0])[1]
    def f_source(x, t): return 0.0

    # Assemblage avec les nouvelles propriétés
    # Attention, on passe coords (qui est xphys) à assemble_mass maintenant !
    M_lil = assemble_mass(elemTags, elemNodeTags, jac, det, coords, w, N, rho_cp, tag_to_dof)
    K_lil, F0 = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, lambda x: f_source(x, 0), tag_to_dof)

    M = M_lil.tocsr()
    K = K_lil.tocsr()

    # Initialisation de la température à 10°C partout au début
    U = np.array([10.0 for x in dof_coords], dtype=float)

    # Conditions de Dirichlet des DEUX cotés
    inner_dofs = border_dofs_from_tags(bnds_tags[0], tag_to_dof)
    outer_dofs = border_dofs_from_tags(bnds_tags[1], tag_to_dof)
    dir_dofs = np.concatenate([inner_dofs, outer_dofs])

    _, ax = setup_interactive_figure()

    for step in range(args.nsteps):
        t = step * args.dt

        # On impose 20°C à l'intérieur, 5°C à l'extérieur
        val_inner = 20.0 * np.ones_like(inner_dofs)
        val_outer = 5.0 * np.ones_like(outer_dofs)
        dir_vals_np1 = np.concatenate([val_inner, val_outer])

        # Calcul du pas de temps (F0 reste 0 car il n'y a pas de sources de chaleur internes type micro-onde)
        U = theta_step(M, K, F0, F0, U, dt=args.dt, theta=args.theta, dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals_np1)

        ax.clear()
        plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=U,
            tag_to_dof=tag_to_dof,
            show_mesh=False,
            ax=ax
        )
        
        # Affichage du temps en heures
        ax.set_title(f"t = {t/3600:.1f} heures")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')

        plt.pause(0.05)

    gmsh_finalize()


if __name__ == "__main__":
    main()

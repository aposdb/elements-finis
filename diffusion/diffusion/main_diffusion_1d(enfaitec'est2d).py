# main_diffusion_1d.py
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_2d_wall_mesh,
    prepare_quadrature_and_basis, get_jacobians, border_dofs_from_tags
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import setup_interactive_figure, plot_mesh_2d, plot_fe_solution_2d


# ============================================================
# NOUVEAU : Température extérieure sinusoïdale (cycle 24h)
# Varie entre 5°C et 25°C, minimum la nuit, maximum l'après-midi
# ============================================================
T_PERIOD = 24 * 3600.0   # 24 heures en secondes

def T_ext(t):
    return 15.0 + 10.0 * np.sin(2 * np.pi * t / T_PERIOD - np.pi / 2)


def main():
    parser = argparse.ArgumentParser(description="Diffusion 2D Multicouche - Cycle 24h")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--dt",     type=float, default=3600.0)  # 1 heure par pas
    parser.add_argument("--nsteps", type=int,   default=72)      # 3 jours = 72 pas
    args = parser.parse_args()

    from gmsh_utils import build_2d_wall_mesh
    gmsh_init("mur_multicouche")

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

    def get_material_props(x_coord):
        if x_coord <= 0.015 + 1e-6:                      # Plâtre
            return 0.35, 1000 * 1000
        elif x_coord <= 0.015 + 0.140 + 1e-6:            # Béton
            return 1.80, 2300 * 880
        elif x_coord <= 0.015 + 0.140 + 0.150 + 1e-6:   # Isolant
            return 0.035, 20 * 1030
        else:                                              # Brique
            return 1.00, 1900 * 800

    def kappa(x):   return get_material_props(x[0])[0]
    def rho_cp(x):  return get_material_props(x[0])[1]
    def f_source(x, t): return 0.0

    M_lil = assemble_mass(elemTags, elemNodeTags, jac, det, coords, w, N, rho_cp, tag_to_dof)
    K_lil, F0 = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN,
        kappa, lambda x: f_source(x, 0), tag_to_dof
    )

    M = M_lil.tocsr()
    K = K_lil.tocsr()

    # Initialisation à 10°C partout
    U = np.array([10.0 for _ in dof_coords], dtype=float)

    inner_dofs = border_dofs_from_tags(bnds_tags[0], tag_to_dof)
    outer_dofs = border_dofs_from_tags(bnds_tags[1], tag_to_dof)
    dir_dofs = np.concatenate([inner_dofs, outer_dofs])

    _, ax = setup_interactive_figure()

    # ============================================================
    # NOUVEAU : Listes pour stocker l'historique des températures
    # (placées AVANT la boucle pour être vides au départ)
    # ============================================================
    T_inner_history = []   # T côté intérieur du mur au fil du temps
    T_outer_history = []   # T extérieure imposée au fil du temps
    time_axis = []         # axe des temps en heures

    # ============================================================
    # BOUCLE TEMPORELLE
    # ============================================================
    for step in range(args.nsteps):
        t     = step * args.dt
        t_np1 = (step + 1) * args.dt  # NOUVEAU : temps au pas suivant

        # Température intérieure fixe : 20°C
        val_inner = 20.0 * np.ones_like(inner_dofs)

        # NOUVEAU : température extérieure sinusoïdale au lieu de 5°C fixe
        val_outer = T_ext(t_np1) * np.ones_like(outer_dofs)

        dir_vals_np1 = np.concatenate([val_inner, val_outer])

        U = theta_step(
            M, K, F0, F0, U,
            dt=args.dt, theta=args.theta,
            dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals_np1
        )

        # ============================================================
        # NOUVEAU : Enregistrement des températures après chaque pas
        # (placé juste après theta_step, toujours dans la boucle)
        # ============================================================
        T_inner_history.append(float(U[inner_dofs[0]]))  # 1er nœud intérieur
        T_outer_history.append(T_ext(t_np1))             # T extérieure imposée
        time_axis.append(t_np1 / 3600.0)                 # conversion en heures

        # Visualisation 2D du champ de température
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
        ax.set_title(f"t = {t_np1/3600.0:.1f} heures  |  T_ext = {T_ext(t_np1):.1f}°C")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')
        plt.pause(0.05)

    gmsh_finalize()

    # ============================================================
    # NOUVEAU : Graphe du déphasage thermique
    # (placé APRÈS la boucle, en dehors de celle-ci)
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(time_axis, T_outer_history, label="T extérieure (imposée)",
             linestyle='--', color='royalblue')
    ax2.plot(time_axis, T_inner_history, label="T face intérieure du mur",
             color='tomato')
    ax2.set_xlabel("Temps (heures)")
    ax2.set_ylabel("Température (°C)")
    ax2.set_title("Déphasage thermique à travers le mur multicouche")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
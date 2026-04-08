# main_vague.py
"""
Diffusion thermique 2D dans un mur multicouche à interfaces ondulées.

Géométrie
---------
  x : direction de l'épaisseur (de l'intérieur vers l'extérieur)
  y : direction de la hauteur

  Les frontières gauche (x=0, intérieur) et droite (x=L, extérieur)
  sont des droites verticales avec des conditions de Dirichlet.
  Les interfaces entre les couches suivent une sinusoïde :

      x_interface(y) = x_nominal + A * sin(2π * freq * y / H)

  Ce profil ondulé crée des effets 2D absents d'un modèle 1D :
  le flux de chaleur est dévié aux interfaces, ce qui modifie le
  comportement thermique local par rapport au cas rectiligne.

Usage
-----
  python main_vague.py
  python main_vague.py --amplitude 0.04 --frequence 3 --nsteps 200
  python main_vague.py --theta 0.5 --dt 1800

Arguments
---------
  --amplitude  Demi-amplitude des vagues [m]  (doit être < min_épaisseur / 2)
  --frequence  Nombre de périodes sinusoïdales complètes sur la hauteur H
  --theta      Schéma en temps : 1=Euler implicite, 0.5=Crank-Nicolson
  --dt         Pas de temps [s]
  --nsteps     Nombre de pas de temps
  -order       Ordre polynomial des éléments finis (1 ou 2)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_wavy_wall_mesh,
    prepare_quadrature_and_basis, get_jacobians, border_dofs_from_tags,
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d


# ---------------------------------------------------------------------------
# Matériaux : (kappa [W/m/K],  rho*cp [J/m³/K])
# ---------------------------------------------------------------------------
# 3 couches d'épaisseur égale pour une géométrie symétrique et lisible.
LAYER_THICKNESSES = [0.15, 0.15, 0.15]   # béton | isolant | brique   [m]
MATERIALS = [
    (1.80,  2300 * 880),    # Béton       – bonne conductivité
    (0.035,   20 * 1030),   # Isolant     – très peu conducteur  ← couche clé
    (1.00,  1900 * 800),    # Brique      – conductivité intermédiaire
]
LAYER_NAMES = ["Béton", "Isolant", "Brique"]

# Conditions aux limites
T_INNER =  20.0   # °C  – côté intérieur (chaud)
T_OUTER =  -5.0   # °C  – côté extérieur (froid)
T_INIT  =  10.0   # °C  – température initiale uniforme


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def build_dof_mapping(nodeTags, nodeCoords, elemNodeTags):
    """
    Construit tag_to_dof (tableau de taille max_tag+1) et dof_coords (N×3).
    Les dofs sont numérotés dans l'ordre des tags uniques trouvés dans la
    connectivité.  On construit un index rapide tag->rang dans nodeCoords
    pour éviter les décalages si GMSH ne numérote pas à partir de 1.
    """
    unique_tags = np.unique(elemNodeTags.astype(np.int64))
    num_dofs    = len(unique_tags)
    max_tag     = int(np.max(nodeTags))

    tag_to_dof = np.full(max_tag + 1, -1, dtype=np.int64)

    # Index rapide tag  → position dans le tableau nodeCoords
    node_tag_to_idx = np.zeros(max_tag + 1, dtype=np.int64)
    for idx, tag in enumerate(nodeTags):
        node_tag_to_idx[int(tag)] = idx

    all_coords = nodeCoords.reshape(-1, 3)
    dof_coords = np.zeros((num_dofs, 3), dtype=float)
    for i, tag in enumerate(unique_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i]        = all_coords[node_tag_to_idx[int(tag)]]

    return tag_to_dof, dof_coords


def plot_solution(ax, cax, elemNodeTags, nodeCoords, nodeTags,
                  U, tag_to_dof, layer_thicknesses, H,
                  vmin, vmax, show_mesh=False, title=""):
    """
    Tracé d'une solution FE 2D avec tricontourf.
    Renvoie le mappable pour la colorbar.
    """
    import matplotlib.tri as mtri

    num_dofs   = len(U)
    max_tag    = int(np.max(nodeTags))
    all_coords = nodeCoords.reshape(-1, 3)

    # Index rapide tag → rang dans nodeCoords
    node_tag_to_idx = np.zeros(max_tag + 1, dtype=np.int64)
    for idx, tag in enumerate(nodeTags):
        node_tag_to_idx[int(tag)] = idx

    coords_mapped = np.zeros((num_dofs, 2))
    for tag in nodeTags:
        d = tag_to_dof[int(tag)]
        if d != -1:
            coords_mapped[d] = all_coords[node_tag_to_idx[int(tag)], :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    # Connectivité – on garde uniquement les 3 nœuds d'angle du triangle
    total = len(elemNodeTags)
    nodes_per_elem = 3
    for nposs in [3, 6, 10, 15]:
        if total % nposs == 0:
            nodes_per_elem = nposs
            break
    conn      = elemNodeTags.reshape(-1, nodes_per_elem)
    triangles = tag_to_dof[conn[:, :3].astype(np.int64)]

    triang = mtri.Triangulation(x, y, triangles)
    cf = ax.tricontourf(triang, U, levels=100, cmap='RdBu_r',
                        vmin=vmin, vmax=vmax)

    if show_mesh:
        ax.triplot(triang, color='white', linewidth=0.2, alpha=0.3)

    # Interfaces nominales en pointillés blancs
    x_ifaces = np.cumsum(layer_thicknesses[:-1])
    for xn in x_ifaces:
        ax.axvline(xn, color='white', lw=0.9, ls='--', alpha=0.6)

    # Annotations des couches
    x_interfaces = np.concatenate([[0.0], np.cumsum(layer_thicknesses)])
    for k, name in enumerate(LAYER_NAMES):
        xc = (x_interfaces[k] + x_interfaces[k + 1]) / 2.0
        ax.text(xc, H * 1.02, name, ha='center', va='bottom',
                fontsize=8, color='0.35')

    ax.set_aspect('equal')
    ax.set_xlabel("Épaisseur x [m]")
    ax.set_ylabel("Hauteur y [m]")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    return cf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diffusion 2D – Mur à interfaces ondulées"
    )
    parser.add_argument("-order",      type=int,   default=1)
    parser.add_argument("--theta",     type=float, default=1.0,
                        help="1=Euler impl., 0.5=Crank-Nicolson, 0=Euler expl.")
    parser.add_argument("--dt",        type=float, default=3600.0,
                        help="Pas de temps [s] (défaut : 1 heure)")
    parser.add_argument("--nsteps",    type=int,   default=120,
                        help="Nombre de pas de temps")
    parser.add_argument("--amplitude", type=float, default=0.03,
                        help="Demi-amplitude des vagues [m]")
    parser.add_argument("--frequence", type=int,   default=2,
                        help="Nombre de périodes sur la hauteur H")
    parser.add_argument("--show-mesh", action="store_true",
                        help="Afficher le maillage par-dessus la solution")
    args = parser.parse_args()

    H = sum(LAYER_THICKNESSES)          # domaine 0.45 × 0.45 m

    # ------------------------------------------------------------------ #
    # 1.  Maillage GMSH avec interfaces sinusoïdales                      #
    # ------------------------------------------------------------------ #
    gmsh_init("mur_vague")

    (elemType, nodeTags, nodeCoords,
     elemTags, elemNodeTags,
     bnds, bnds_tags) = build_wavy_wall_mesh(
        layer_thicknesses = LAYER_THICKNESSES,
        H         = H,
        amplitude = args.amplitude,
        frequence = args.frequence,
        n_pts     = 80,
        order     = args.order,
        cl        = 0.007,
    )

    # Affichage du maillage avant simulation
    plot_mesh_2d(elemType, nodeTags, nodeCoords,
                 elemTags, elemNodeTags, bnds, bnds_tags)

    # ------------------------------------------------------------------ #
    # 2.  Mapping dofs                                                     #
    # ------------------------------------------------------------------ #
    tag_to_dof, dof_coords = build_dof_mapping(nodeTags, nodeCoords, elemNodeTags)
    num_dofs = int(tag_to_dof.max() + 1)

    # ------------------------------------------------------------------ #
    # 3.  Quadrature / bases                                               #
    # ------------------------------------------------------------------ #
    xi, w, N, gN      = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords  = get_jacobians(elemType, xi)

    # ------------------------------------------------------------------ #
    # 4.  Propriétés matériaux (en fonction de x)                         #
    # ------------------------------------------------------------------ #
    x_interfaces = np.concatenate([[0.0], np.cumsum(LAYER_THICKNESSES)])

    def get_layer(x_coord):
        for k in range(len(LAYER_THICKNESSES)):
            if x_coord <= x_interfaces[k + 1] + 1e-9:
                return k
        return len(LAYER_THICKNESSES) - 1

    def kappa(x):    return MATERIALS[get_layer(x[0])][0]
    def rho_cp(x):   return MATERIALS[get_layer(x[0])][1]
    def zero_src(x): return 0.0

    # ------------------------------------------------------------------ #
    # 5.  Assemblage                                                       #
    # ------------------------------------------------------------------ #
    print("Assemblage des matrices…")
    M_lil = assemble_mass(
        elemTags, elemNodeTags, jac, det, coords, w, N, rho_cp, tag_to_dof
    )
    K_lil, F0 = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN,
        kappa, zero_src, tag_to_dof
    )
    M = M_lil.tocsr()
    K = K_lil.tocsr()
    print(f"  → {num_dofs} degrés de liberté, {len(elemTags)} éléments.")

    # ------------------------------------------------------------------ #
    # 6.  Condition initiale & conditions aux limites                      #
    # ------------------------------------------------------------------ #
    U = np.full(num_dofs, T_INIT, dtype=float)

    inner_dofs = border_dofs_from_tags(bnds_tags[0], tag_to_dof)
    outer_dofs = border_dofs_from_tags(bnds_tags[1], tag_to_dof)
    dir_dofs   = np.concatenate([inner_dofs, outer_dofs])
    dir_vals   = np.concatenate([
        T_INNER * np.ones(len(inner_dofs)),
        T_OUTER * np.ones(len(outer_dofs)),
    ])

    # ------------------------------------------------------------------ #
    # 7.  Boucle temporelle + visualisation                                #
    # ------------------------------------------------------------------ #
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={'width_ratios': [1, 0.03]}
    )
    plt.ion()

    cbar = None   # colorbar créée au 1er affichage
    plt.tight_layout()

    for step in range(args.nsteps):

        U = theta_step(
            M, K, F0, F0, U,
            dt             = args.dt,
            theta          = args.theta,
            dirichlet_dofs = dir_dofs,
            dir_vals_np1   = dir_vals,
        )

        # Affichage toutes les 5 itérations (sauf le tout dernier pas)
        if step % 5 != 0 and step != args.nsteps - 1:
            continue

        ax.cla()
        t_h = (step + 1) * args.dt / 3600.0
        title = (
            f"Mur à vagues  |  t = {t_h:.1f} h  "
            f"(A = {args.amplitude*100:.1f} cm,  freq = {args.frequence},  "
            f"θ = {args.theta})"
        )
        cf = plot_solution(
            ax, cax,
            elemNodeTags, nodeCoords, nodeTags,
            U, tag_to_dof, LAYER_THICKNESSES, H,
            vmin      = T_OUTER,
            vmax      = T_INNER,
            show_mesh = args.show_mesh,
            title     = title,
        )

        if cbar is None:
            cbar = fig.colorbar(cf, cax=cax)
            cbar.set_label("Température [°C]")

        
        
        plt.pause(0.02)

    print("Simulation terminée.")
    gmsh_finalize()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
# gmsh_utils.py
import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    gmsh.finalize()


def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Build and mesh a 1D segment [0,L] with different characteristic lengths.
    Returns (line_tag, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags).
    """
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType = gmsh.model.mesh.getElementType("line", order)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags


def prepare_quadrature_and_basis(elemType, order):
    """
    Returns:
      xi (flattened uvw), w (ngp), N (flattened bf), gN (flattened gbf)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN


def get_jacobians(elemType, xi, tag=-1):
    """
    Wrapper around gmsh.getJacobians.
    Returns (jacobians, dets, coords)
    """
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jacobians, dets, coords


def end_dofs_from_nodes(nodeCoords):
    """
    Robustly identify first/last node dofs from coordinates (x-min, x-max).
    nodeCoords is flattened [x0,y0,z0, x1,y1,z1, ...]
    Returns (left_dof, right_dof) as 0-based indices.
    """
    X = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right

def border_dofs_from_tags(l_tags, tag_to_dof):
    """
    Converts a list of GMSH node tags into the corresponding 
    compact matrix indices (DoFs).
    """
    # Ensure tags are integers
    l_tags = np.asarray(l_tags, dtype=int)
    
    # Filter out any tags that might not be in our DoF mapping (like geometry points)
    # then map them to our 0...N-1 indices
    valid_mask = (tag_to_dof[l_tags] != -1)
    l_dofs = tag_to_dof[l_tags[valid_mask]]
    return l_dofs

def getPhysical(name):
    """
    Get the physical group elements and nodes for a given name and dimension.
    """
    
    dimTags = gmsh.model.getEntitiesForPhysicalName(name)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=dimTags[0][0], tag=dimTags[0][1])
    elemType = elemTypes[0]  # Assuming one element type per physical group
    elemTags = elemTags[0]
    elemNodeTags = elemNodeTags[0]
    entityTag = dimTags[0][1]
    return elemType, elemTags, elemNodeTags, entityTag
    

'''def open_2d_mesh(msh_filename, order=1):
    """
    Load a .msh file.

    Parameters
    ----------
    msh_filename : str
        Path to the .msh file
    order : int
        Polynomial order of elements

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """

    import gmsh

    # --- load geometry
    gmsh.open(msh_filename)

    # --- high order
    gmsh.model.mesh.setOrder(order)

    # --- element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    # --- nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

    # --- elements
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    surf = gmsh.model.getEntities(2)[0][1]

    curve_tags = gmsh.model.getBoundary([(2, surf)], oriented=False)
    
    gmsh.model.addPhysicalGroup(1, [curve_tags[0][1]], tag=1)
    gmsh.model.setPhysicalName(1, 1, "OuterBoundary")

    gmsh.model.addPhysicalGroup(1, [curve_tags[1][1]], tag=2)
    gmsh.model.setPhysicalName(1, 2, "InnerBoundary")

    bnds = [('OuterBoundary', 1),('InnerBoundary', 1)]

    bnds_tags = []
    for name, dim in bnds:
        tag = -1
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                tag = t[1]
                break
        if tag == -1:
            raise ValueError(f"Physical group '{name}' not found in mesh.")
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags'''


def build_2d_wall_mesh(order=1, cl=0.01, H=0.1):
    """
    Génère un maillage 2D rectangulaire pour un mur multicouche.
    H est la hauteur arbitraire du mur en 2D.
    cl est la taille caractéristique des éléments (finesse du maillage).
    """
    import gmsh
    
    # Épaisseurs des 4 couches (Plâtre, Béton, Isolant, Brique)
    L1, L2, L3, L4 = 0.015, 0.140, 0.150, 0.090
    X = [0, L1, L1+L2, L1+L2+L3, L1+L2+L3+L4]
    
    # Création des points
    p = []
    for i in range(5):
        p.append(gmsh.model.geo.addPoint(X[i], 0, 0, cl))
        p.append(gmsh.model.geo.addPoint(X[i], H, 0, cl))
        
    # Création des lignes et surfaces (les 4 rectangles)
    surfaces = []
    for i in range(4):
        l_bottom = gmsh.model.geo.addLine(p[2*i], p[2*i+2])
        l_right  = gmsh.model.geo.addLine(p[2*i+2], p[2*i+3])
        l_top    = gmsh.model.geo.addLine(p[2*i+3], p[2*i+1])
        l_left   = gmsh.model.geo.addLine(p[2*i+1], p[2*i])
        
        cloop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
        surfaces.append(gmsh.model.geo.addPlaneSurface([cloop]))
        
        if i == 0: inner_line = l_left
        if i == 3: outer_line = l_right

    gmsh.model.geo.synchronize()
    
    # On définit les Groupes Physiques pour les frontières
    gmsh.model.addPhysicalGroup(1, [abs(inner_line)], tag=1)
    gmsh.model.setPhysicalName(1, 1, "InnerBoundary")
    
    gmsh.model.addPhysicalGroup(1, [abs(outer_line)], tag=2)
    gmsh.model.setPhysicalName(1, 2, "OuterBoundary")
    
    # Maillage
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    
    # Extraction des données pour Python
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)
    
    bnds = [('InnerBoundary', 1), ('OuterBoundary', 1)]
    bnds_tags = []
    for name, dim in bnds:
        tag = -1
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                tag = t[1]
                break
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])
        
    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags


def build_wavy_wall_mesh(layer_thicknesses=None, H=0.45,
                          amplitude=0.03, frequence=2,
                          n_pts=60, order=1, cl=0.008):
    """
    Build a 2D FEM mesh for a multilayer wall with wavy internal interfaces.

    The wall geometry:
      - x axis : thickness direction (diffusion direction, from InnerBoundary to OuterBoundary)
      - y axis : height direction
      - outer boundaries (x=0 and x=L) : straight vertical lines
      - internal interfaces : x_interface(y) = x_nominal + A * sin(2π * freq * y / H)

    This creates a sinusoidal waviness at each layer interface, which affects the
    local heat flux and creates 2-D effects invisible in a 1-D model.

    Parameters
    ----------
    layer_thicknesses : list of float
        Thickness [m] of each layer, left to right.
        Default: [0.15, 0.15, 0.15]  (3 equal layers, total 0.45 m)
    H : float
        Wall height [m].
    amplitude : float
        Half-amplitude of the sinusoidal interface [m].
        Must be strictly less than min(layer_thicknesses)/2 to avoid overlaps.
    frequence : int
        Number of full sine periods along H.  Using an integer ensures that the
        interface returns exactly to x_nominal at y=0 and y=H (clean corners).
    n_pts : int
        Number of control points on each wavy spline.
    order : int
        Polynomial order of the FE elements (1 = P1 triangles).
    cl : float
        Target mesh size [m].

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags
        Same layout as build_2d_wall_mesh.
    """
    import gmsh
    import numpy as np

    if layer_thicknesses is None:
        layer_thicknesses = [0.15, 0.15, 0.15]

    nb_layers = len(layer_thicknesses)
    x_bases = np.concatenate([[0.0], np.cumsum(layer_thicknesses)])

    # Safety check
    min_thick = min(layer_thicknesses)
    if amplitude >= min_thick / 2.0:
        raise ValueError(
            f"amplitude={amplitude} >= min_layer_thickness/2={min_thick/2:.4f}. "
            "Interfaces would overlap. Reduce amplitude."
        )

    y_pts = np.linspace(0.0, H, n_pts)

    # ------------------------------------------------------------------
    # 1.  Create interface curves
    #     - outer boundaries (i=0 and i=nb_layers): straight addLine
    #     - inner interfaces                       : wavy addSpline
    # ------------------------------------------------------------------
    all_bottom_pts  = []   # GMSH tag of the bottom point of each interface
    all_top_pts     = []   # GMSH tag of the top    point of each interface
    interface_curves = []  # GMSH curve tag for each interface

    for i, x_base in enumerate(x_bases):
        is_outer = (i == 0 or i == nb_layers)

        if is_outer:
            # Straight vertical boundary
            pt_b = gmsh.model.geo.addPoint(float(x_base), 0.0,   0.0, cl)
            pt_t = gmsh.model.geo.addPoint(float(x_base), float(H), 0.0, cl)
            curve = gmsh.model.geo.addLine(pt_b, pt_t)
            all_bottom_pts.append(pt_b)
            all_top_pts.append(pt_t)

        else:
            # Wavy interface – spline through n_pts control points
            # sin(2π·freq·y/H) is 0 at y=0 and y=H when freq is an integer
            pts = []
            for yj in y_pts:
                xj = float(x_base) + amplitude * np.sin(
                    2.0 * np.pi * frequence * yj / H
                )
                pts.append(gmsh.model.geo.addPoint(xj, float(yj), 0.0, cl))
            curve = gmsh.model.geo.addSpline(pts)
            all_bottom_pts.append(pts[0])
            all_top_pts.append(pts[-1])

        interface_curves.append(curve)

    # ------------------------------------------------------------------
    # 2.  Create one surface per layer
    #     Each layer i is bounded by:
    #       l_bot : bottom horizontal segment (left-bottom → right-bottom)
    #       interface_curves[i+1]  : right interface (bottom → top)
    #       l_top : top    horizontal segment (right-top  → left-top )
    #      -interface_curves[i]   : left  interface traversed backwards
    # ------------------------------------------------------------------
    for i in range(nb_layers):
        l_bot = gmsh.model.geo.addLine(all_bottom_pts[i], all_bottom_pts[i + 1])
        l_top = gmsh.model.geo.addLine(all_top_pts[i + 1], all_top_pts[i])

        cloop = gmsh.model.geo.addCurveLoop([
            l_bot,
             interface_curves[i + 1],
             l_top,
            -interface_curves[i],
        ])
        gmsh.model.geo.addPlaneSurface([cloop])

    gmsh.model.geo.synchronize()

    # ------------------------------------------------------------------
    # 3.  Physical groups for boundary conditions
    # ------------------------------------------------------------------
    gmsh.model.addPhysicalGroup(1, [interface_curves[0]],         tag=1)
    gmsh.model.setPhysicalName(1, 1, "InnerBoundary")

    gmsh.model.addPhysicalGroup(1, [interface_curves[nb_layers]], tag=2)
    gmsh.model.setPhysicalName(1, 2, "OuterBoundary")

    # ------------------------------------------------------------------
    # 4.  Mesh & extract data
    # ------------------------------------------------------------------
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    elemType  = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    bnds = [('InnerBoundary', 1), ('OuterBoundary', 1)]
    bnds_tags = []
    for name, dim in bnds:
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                bnds_tags.append(
                    gmsh.model.mesh.getNodesForPhysicalGroup(dim, t[1])[0]
                )
                break

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags
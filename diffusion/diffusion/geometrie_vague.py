import numpy as np

def generer_mur_vague(longueur, hauteur, nb_couches, amplitude, frequence, resolution=200):
    """
    Génère les coordonnées des interfaces d'un mur où les couches sont séparées par des vagues.
    
    Paramètres :
    - longueur   : La longueur totale du mur (axe X)
    - hauteur    : La hauteur totale du mur (axe Y)
    - nb_couches : Le nombre de couches distinctes
    - amplitude  : La hauteur maximale de la vague (creux à crête)
    - frequence  : Le nombre de "vagues" sur la longueur totale
    - resolution : Le nombre de points pour dessiner la courbe (plus c'est élevé, plus c'est lisse)
    """
    # Création des points sur l'axe X
    x = np.linspace(0, longueur, resolution)
    lignes_interfaces = []

    # Calcul de la hauteur de base pour chaque ligne de séparation (y compris la base et le sommet)
    hauteurs_base = np.linspace(0, hauteur, nb_couches + 1)

    for i, h_base in enumerate(hauteurs_base):
        # La base (i=0) et le sommet (i=nb_couches) restent plats
        if i == 0 or i == nb_couches:
            y = np.full_like(x, h_base)
        else:
            # Pour les couches internes, on applique l'équation de la vague (sinusoïde)
            # On normalise X par la longueur pour que la fréquence corresponde exactement au nombre de vagues
            onde = amplitude * np.sin(2 * np.pi * frequence * (x / longueur))
            
            # Optionnel : décaler les vagues d'une couche à l'autre
            # onde = amplitude * np.sin(2 * np.pi * frequence * (x / longueur) + i) 
            
            y = h_base + onde

        lignes_interfaces.append((x, y))

    return lignes_interfaces
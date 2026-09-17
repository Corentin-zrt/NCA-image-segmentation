# NCA Image Segmentation

Projet de recherche pour apprendre la segmentation d'images avec un Neural Cellular Automaton (NCA).

## Organisation

- `main.py` : lance le panneau de controle.
- `src/nca_segmentation/control_panel/` : interface Pygame pour piloter les experiences.
- `src/nca_segmentation/data/` : generation et sauvegarde du dataset de formes simples.
- `src/nca_segmentation/nca/` : modele NCA et fonctions de perte.
- `src/nca_segmentation/training/` : boucle de fine-tuning.
- `tests/` : tests unitaires des briques deterministes.

## Installation

Python 3.10 ou plus recent est recommande.

L'environnement virtuel est pratique mais pas obligatoire. Si Python, NumPy,
Pygame et Pillow sont deja installes sur la machine, le projet peut etre lance
directement depuis la racine :

```powershell
python main.py
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Lancement

```powershell
python main.py
```

Dans le panneau :

1. choisissez `MULTI` pour créer plusieurs formes sur chaque image, ou une forme simple ;
2. cliquez sur `GENERATE DATASET` pour creer les exemples ;
3. utilisez `PREVIEW NEXT` pour parcourir les images et masques ;
4. cliquez sur `TRAIN NCA` pour apprendre les classes ;
5. cliquez sur `SAVE MODEL` pour sauvegarder les poids dans `artifacts/nca_model.npz`.

Tous les datasets utilisent le meme espace de labels : `0 = fond`,
`1 = cercle`, `2 = carre`, `3 = triangle`. Un dataset de cercles seuls ne
supprime donc pas les autres sorties du modele : il entraine uniquement les
pixels des classes presentes et conserve un modele generaliste a quatre classes.

Le modele courant est conserve quand un nouveau dataset est genere. Pour
tester la generalisation : entrainez sur `MULTI`, cliquez sur `SHAPE / CIRCLE`,
generez le nouveau dataset, puis lancez quelques etapes de training ou utilisez
la prediction courante.

Les donnees generees sont ecrites dans `artifacts/dataset/`.

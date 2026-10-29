# Conception d’un système d’analyse faciale multi-tâches basé sur le Deep Learning pour l’estimation du genre, de l’âge et des émotions

Système d’analyse faciale multitâche capable de détecter plusieurs visages et de prédire simultanément le **genre, l’âge apparent et les émotions faciales** à partir d’images, de vidéos et de flux webcam. Développé avec PyTorch, RetinaFace, ResNet50 et Streamlit, ce projet regroupe trois tâches d’analyse faciale au sein d’un modèle unique de Deep Learning.

## Présentation du projet

Les systèmes traditionnels d’analyse faciale utilisent souvent des modèles distincts pour la classification du genre, l’estimation de l’âge et la reconnaissance des émotions. Ce projet adopte une approche d’**apprentissage multitâche (Multi-Task Learning — MTL)**, dans laquelle les trois tâches partagent un même extracteur de caractéristiques afin de réduire les calculs redondants et de simplifier le déploiement.

Le système prend en charge les images statiques, les fichiers vidéo et les flux webcam. Il détecte et aligne les visages, prédit leurs caractéristiques faciales et peut suivre les personnes d’une image vidéo à l’autre.

### Fonctionnalités principales

- **Détection de plusieurs visages** dans une même image ou séquence vidéo.
- **Classification du genre** : prédiction Homme/Femme avec un score de confiance.
- **Estimation de l’âge apparent** en années.
- **Reconnaissance des émotions** : classification en sept catégories.
- **Alignement facial** à partir de cinq points de repère (landmarks).
- **Suivi des personnes** dans les vidéos grâce à SORT.
- **Interface interactive** permettant l’analyse d’images, de vidéos et de flux webcam avec Streamlit.
- **Export au format ONNX** pour faciliter l’inférence optimisée.
- **Pipeline de Deep Learning personnalisé**, implémenté directement avec PyTorch, sans dépendre d’une API d’analyse faciale tout-en-un.

## Architecture du système

Le système suit les étapes suivantes :

1. **Entrée :** chargement d’une image, d’une vidéo ou d’un flux webcam.
2. **Détection faciale :** RetinaFace localise les visages et retourne les boîtes englobantes, les scores de confiance et cinq landmarks.
3. **Alignement facial :** OpenCV utilise les landmarks pour aligner chaque visage et extraire une région faciale standardisée.
4. **Prétraitement :** redimensionnement à 112 × 112 pixels et normalisation selon les statistiques ImageNet.
5. **Inférence multitâche :** le backbone ResNet50 extrait les caractéristiques communes, ensuite transmises à trois têtes spécialisées.
6. **Suivi :** SORT peut associer les détections entre les images successives d’une vidéo.
7. **Visualisation :** affichage des boîtes englobantes, des landmarks, des prédictions, des scores de confiance et des identifiants de suivi lorsque cette fonctionnalité est activée.

### Architecture du modèle

Le modèle principal, `MTLFaceModel`, comprend un backbone ResNet50 partagé et trois têtes spécialisées.

| Composant | Rôle |
|---|---|
| RetinaFace | Détection des visages et extraction des landmarks |
| ResNet50 | Extraction des caractéristiques faciales communes |
| Tête Genre | Classification binaire Homme/Femme |
| Tête Âge | Régression de l’âge apparent |
| Tête Émotion | Classification de sept émotions |
| SORT | Suivi des personnes dans les séquences vidéo |
| ONNX Runtime | Exécution optimisée du modèle |
| Streamlit | Interface interactive de démonstration |

Le backbone ResNet50 utilise des poids préentraînés sur ImageNet. Ses caractéristiques sont partagées entre les trois têtes, ce qui permet de produire les trois prédictions au cours d’un seul passage avant (*forward pass*).

### Émotions reconnues

Le modèle distingue les sept classes suivantes :

- Surprise
- Peur
- Dégoût
- Joie
- Tristesse
- Colère
- Neutralité

## Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.10 | Langage principal |
| PyTorch | Implémentation, entraînement et inférence du modèle |
| Torchvision | Backbone ResNet50 préentraîné et transformations d’images |
| RetinaFace / InsightFace | Détection faciale et extraction des landmarks |
| OpenCV | Traitement d’images, alignement facial et gestion des vidéos |
| NumPy et Pandas | Calcul numérique et préparation des données |
| Scikit-learn | Découpage des données, pondération des classes et métriques |
| ONNX | Export portable du modèle |
| ONNX Runtime | Inférence optimisée |
| Streamlit | Interface interactive |
| Matplotlib et Seaborn | Visualisation des résultats et matrices de confusion |

Le développement et l’entraînement ont été réalisés sur **Kaggle Notebooks**, avec un GPU NVIDIA Tesla T4 disposant de 16 Go de mémoire vidéo.

## Jeux de données

Aucun jeu de données utilisé ne fournit simultanément toutes les annotations nécessaires aux trois tâches. Deux pipelines de données distincts ont donc été mis en place : un pour l’âge et le genre, et un autre pour les émotions.

### Estimation de l’âge et classification du genre

- **UTKFace :** source principale des annotations d’âge et de genre.
- **IMDB-Wiki :** source complémentaire, filtrée pour éliminer les entrées peu fiables et les images contenant plusieurs visages.

Après préparation, le jeu de données combiné pour l’âge et le genre contient **195 524 images**.

### Reconnaissance des émotions

- **RAF-DB :** source principale des expressions faciales.
- **AffectNet :** source complémentaire pour la reconnaissance des émotions.
- **CK+48 :** données supplémentaires utilisées pour l’équilibrage des classes.
- **FER-2013 / FER-jonathanoheix :** sources complémentaires intégrées au processus de préparation et d’équilibrage.

Le jeu de données final consacré aux émotions contient **52 440 images**.

Les données sont réparties entre les ensembles d’entraînement, de validation et de test selon un ratio de 70 % / 15 % / 15 %. Une répartition stratifiée est utilisée pour la tâche de reconnaissance des émotions.

**Remarque :** les jeux de données sont distincts en raison de leurs annotations différentes. Deux chargeurs de données (*DataLoaders*) sont utilisés pendant l’entraînement : l’un pour l’âge et le genre, l’autre pour les émotions.

### Prétraitement et augmentation des données

Toutes les images faciales sont redimensionnées à 112 × 112 pixels et normalisées selon la moyenne et l’écart-type d’ImageNet.

Les augmentations appliquées uniquement aux données d’entraînement comprennent :

- Retournement horizontal aléatoire.
- Rotation aléatoire jusqu’à ±20°.
- Conversion en niveaux de gris.
- Modification de la luminosité, du contraste, de la saturation et de la teinte.
- Flou gaussien.
- Effacement aléatoire de régions pour simuler des occultations partielles.

La validation et le test utilisent uniquement le redimensionnement et la normalisation afin de conserver des conditions d’évaluation cohérentes.

## Méthodologie d’entraînement

L’entraînement repose sur une stratégie de transfert d’apprentissage en deux phases.

**Phase 1 — Entraînement des têtes spécialisées**

- Le backbone ResNet50 préentraîné est gelé.
- Les trois têtes sont entraînées pendant 30 époques.
- Le taux d’apprentissage initial des têtes est fixé à 0,001.

**Phase 2 — Ajustement fin du backbone**

- Les deux derniers blocs résiduels de ResNet50 sont dégelés.
- L’entraînement se poursuit pendant 75 époques.
- Le taux d’apprentissage du backbone est fixé à 0,00001 et celui des têtes à 0,0001.
- Le clipping des gradients et un scheduler adaptatif contribuent à stabiliser l’entraînement.

L’optimisation utilise Adam, une régularisation par *weight decay*, la fonction Cross-Entropy pondérée pour les tâches de classification et la fonction Huber Loss pour la régression de l’âge.

La durée totale de l’entraînement est d’environ **3 h 30 sur un GPU NVIDIA Tesla T4 de Kaggle**.

## Résultats et évaluation

Le modèle final a été évalué sur les jeux de test à l’aide de métriques adaptées à chaque tâche.

| Tâche | Métrique | Résultat obtenu | Objectif initial | État |
|---|---|---:|---:|---|
| Genre | Accuracy | 91,5 % | ≥ 90 % | Atteint |
| Genre | F1-macro | 0,914 | — | — |
| Âge | Erreur absolue moyenne (MAE) | 6,74 ans | ≤ 5 ans | Non atteint |
| Âge | Erreur absolue médiane | 4,55 ans | — | — |
| Âge | Prédictions avec une erreur inférieure à 10 ans | 77,3 % | — | — |
| Émotion | Accuracy | 65,2 % | — | — |
| Émotion | F1-macro | 0,628 | ≥ 0,60 | Atteint |
| Pipeline complet | Débit annoncé | 26,1 FPS | ≥ 25 FPS | Atteint |

Les objectifs concernant la classification du genre et la reconnaissance des émotions sont atteints. En revanche, la MAE de l’estimation de l’âge reste légèrement supérieure à l’objectif de cinq ans. Les erreurs les plus importantes concernent notamment les enfants en bas âge et les personnes âgées, moins représentés dans les données d’entraînement.

### Comparaison des backbones

Trois architectures ont été évaluées dans des conditions expérimentales comparables.

| Backbone | Paramètres | Accuracy genre | MAE âge | Accuracy émotion | F1-macro émotion |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 26,2 M | 91,5 % | 6,74 ans | 65,2 % | 0,628 |
| MobileNetV3-Large | 4,3 M | 86,0 % | 7,83 ans | 50,2 % | 0,475 |
| EfficientNet-B0 | 5,7 M | 85,2 % | 8,28 ans | 46,6 % | 0,439 |

ResNet50 obtient les meilleurs résultats globaux parmi les architectures évaluées. Malgré son nombre de paramètres plus élevé, sa capacité d’extraction des caractéristiques est plus adaptée à l’apprentissage simultané des trois tâches.

### Comparaison des détecteurs de visages

| Détecteur | Taux de détection sur 15 images | Cinq landmarks |
|---|---:|---|
| MTCNN | 86,7 % | Oui |
| RetinaFace | 100 % | Oui |
| YOLOv8n-face | 86,7 % | Non |

RetinaFace a été retenu parce qu’il a détecté tous les visages du petit échantillon d’évaluation et fournit les landmarks nécessaires à l’alignement facial. Ces résultats restent indicatifs, compte tenu du nombre limité d’images utilisées dans ce benchmark.

### Export et optimisation ONNX

Le modèle entraîné a été exporté au format ONNX, puis comparé au modèle PyTorch original sur des tenseurs d’entrée identiques.

- Taille du modèle PyTorch : environ 100,3 Mo.
- Taille du modèle ONNX : environ 99,9 Mo.
- Latence d’inférence MTL annoncée sur Tesla T4 : 7,4 ms avec PyTorch contre 6,6 ms avec ONNX Runtime.
- Les différences maximales observées entre les sorties des deux modèles sont inférieures à 1,7 × 10⁻⁶.

Les mesures de débit du pipeline complet et celles de l’inférence isolée du modèle proviennent de benchmarks différents et ne doivent pas être confondues.

## Application interactive

L’interface Streamlit propose trois modes d’utilisation.

### 1. Analyse d’images

L’utilisateur charge une image JPG ou PNG pour obtenir :

- L’image originale et l’image annotée.
- Les boîtes englobantes et les landmarks des visages détectés.
- Le genre prédit, son score de confiance, l’âge estimé et l’émotion dominante.
- Les visages alignés extraits par le détecteur.
- Un graphique représentant les probabilités des sept émotions.

### 2. Analyse vidéo

L’utilisateur charge une vidéo aux formats MP4, AVI ou MOV. Le système traite les images successives et affiche les prédictions faciales. Le suivi SORT peut être activé pour attribuer des identifiants aux personnes entre les images. La vidéo annotée peut être téléchargée après le traitement.

### 3. Analyse par webcam

Ce mode permet d’analyser les visages en temps réel à partir d’une webcam locale, avec possibilité d’activer le suivi. Il nécessite d’exécuter l’application dans un environnement ayant accès à la caméra.


## Limites du système

Malgré les résultats obtenus, plusieurs limites subsistent :

- **Estimation de l’âge :** les erreurs augmentent pour les tranches d’âge sous-représentées, notamment les enfants en bas âge et les personnes âgées.
- **Reconnaissance des émotions :** la peur et le dégoût restent difficiles à reconnaître en raison du déséquilibre des classes et des similitudes entre certaines expressions.
- **Poses extrêmes :** les rotations importantes peuvent réduire la précision des landmarks et dégrader l’alignement facial.
- **Occultations et éclairage :** les masques, les lunettes sombres, la faible luminosité et les autres occultations peuvent diminuer la fiabilité des prédictions.
- **Coût de calcul :** RetinaFace représente une part importante du temps total de traitement.
- **Périmètre de l’évaluation :** certaines mesures du détecteur et du pipeline complet ont été obtenues sur de petits échantillons et ne constituent pas des garanties universelles de performance.

Les prédictions représentent des caractéristiques faciales apparentes et des expressions estimées. Elles ne doivent pas être considérées comme des informations certaines sur l’identité, l’âge réel, l’identité de genre ou l’état émotionnel d’une personne.

## Perspectives d’amélioration

Plusieurs pistes peuvent être explorées :

- Pondération dynamique des fonctions de perte et attention croisée entre les tâches.
- Enrichissement des jeux de données, en particulier pour les âges extrêmes et les émotions sous-représentées.
- Quantification INT8 et utilisation de détecteurs plus légers pour accélérer l’inférence.
- Intégration de DeepSORT pour améliorer le suivi en présence d’occultations.
- Exploration des architectures CNN-Transformer et Vision Transformer.
- Évaluation plus large sur différents groupes démographiques, conditions d’acquisition et matériels.


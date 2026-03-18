# Intégration du Microservice de Recherche Sémantique (IA)

Ce document explique l'implémentation de la nouvelle fonctionnalité de recherche globale propulsée par l'Intelligence Artificielle au sein du projet NakamaNet.

## Objectif
L'objectif principal est de remplacer la recherche classique par mots-clés par une recherche sémantique intelligente. Cela permet aux utilisateurs de trouver des animes, des mangas, des membres ou des posts en décrivant des concepts ou des intrigues (ex: "un garçon avec un chapeau de paille") sans avoir besoin de connaître le titre exact.

## Architecture et Décisions Techniques
Afin de ne pas surcharger les APIs existantes (AdonisJS et Laravel) et pour gérer efficacement les bibliothèques d'apprentissage automatique, nous avons opté pour la création d'un microservice dédié.

Le service `ai_service` a été développé en Python (FastAPI).
Le modèle NLP (Natural Language Processing) sélectionné est `all-MiniLM-L6-v2` (Sentence-Transformers), optimisé pour la performance CPU et la génération de vecteurs d'embeddings.

### Architecture à Zéro Temps d'Arrêt (Zero-Downtime)
La base de données contenant plus de 84 000 entrées, le calcul des vecteurs à la volée ralentissait de façon critique le serveur Web. La solution mise en place sépare le système en deux processus exécutés en parallèle au sein du même conteneur Docker :
1. **Worker d'indexation (`build_index.py`)** : Un script exécuté en arrière-plan qui se connecte en lecture seule aux bases de données `catalog` et `laravel`, encode toutes les entités textuelles, et sauvegarde la matrice localement dans un fichier binaire compact (`ai_index.pkl`). Ce cache est reconstruit de manière asynchrone toutes les 24 heures.
2. **Serveur API (`main.py`)** : Le serveur Web Uvicorn qui répond instantanément aux requêtes de l'application front-end. Il charge le fichier d'index en mémoire. Si l'index est encore en construction (lors du tout premier démarrage), il renvoie un message d'attente à l'interface sans bloquer le thread principal.

Afin d'éviter les problèmes d'interblocages (deadlocks) liés à l'environnement conteneurisé, le parallélisme du Tokenizer Rust et le multi-threading OpenMP de PyTorch ont été strictement limités à un seul thread sur l'event-loop principal du serveur HTTP.

## Branches Git Créées
Deux branches ont été utilisées pour isoler cette fonctionnalité et assurer un merge propre :
1. **Backend / Racine (`searchBar`)** : Héberge le microservice `ai_service`, l'intégration Docker, et les scripts de Machine Learning.
2. **Frontend (`dev-searchBar`)** : Contient l'intégration de la modale de recherche dans le projet React / Next.js.

## Fichiers Modifiés et Ajoutés

* **Frontend** :
  * `frontend/src/app/components/layout/Navbar.tsx` : Modification de l'icône de recherche pour déclencher l'ouverture de la modale globale.
  * `frontend/src/app/components/SearchModal.tsx` : [NOUVEAU] Fenêtre modale comprenant un système de filtres par catégories et une pagination par bouton "Charger plus". L'état de l'interface gère les requêtes vers le microservice Python en envoyant les arguments `skip` et `limit`.

* **Backend (AI Microservice)** :
  * `ai_service/Dockerfile` & `ai_service/docker-compose.yml` : Configuration de l'environnement virtuel Python, montage des volumes, et connexion aux réseaux existants (`api_default`, `backend_default`).
  * `ai_service/requirements.txt` : Déclarations des dépendances (FastAPI, Uvicorn, psycopg2-binary, sentence-transformers).
  * `ai_service/build_index.py` : [NOUVEAU] Logique de lecture SQL par batch et encodage NLP en background.
  * `ai_service/main.py` : [NOUVEAU] API publique. Les fonctions embarquent un algorithme de "Cosine Similarity" (Similarité Cosinus) fusionné avec une vérification exacte sur les chaînes de texte, pour attribuer un score de pertinence aux éléments envoyés vers le front-end.

## Instructions de Déploiement
Le microservice se déploie via Docker. 
Positionnez-vous dans le dossier `ai_service` et exécutez `docker compose up -d`. Le système de traitement traitera de manière silencieuse l'ensemble des données en arrière-plan pendant une quinzaine de minutes au premier démarrage. Pour toute la durée de vie du serveur, les recherches renverront un code HTTP 200 quasi-instantané.

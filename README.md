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

### Optimisations et Nouvelles Fonctionnalités (Mises à jour récentes)

Afin d'améliorer l'expérience utilisateur et la performance du service, plusieurs optimisations majeures ont été implémentées :

1. **Recherche Personnalisée (Scor de Compatibilitate Dual)** : 
   - Le service accepte désormais un paramètre `user_id`.
   - Il calcule séparément la compatibilité pour les **Animes** et les **Mangas**.
   - **Logique de Score** : Chaque élément commun rapporte **1.0 point**, plafonné à **5.0** par catégorie.
   - Les résultats sont boostés et triés en priorité par ce score de compatibilité.
   - Deux badges distincts (⭐ X.X anime, ⭐ X.X manga) s'affichent dans l'interface pour une transparence totale.

2. **Indexation Dynamique & Sync de Bibliothèque** : 
   - Le script `build_index.py` utilise un cache d'embeddings pour éviter de re-calculer les vecteurs du catalogue (80k+ items).
   - **Mise à jour fréquente** : La synchronisation des bibliothèques se fait toutes les **5 minutes**.
   - **Fiabilité** : Les métadonnées des bibliothèques sont désormais sauvegardées à chaque itération, garantissant que les ajouts récents d'un utilisateur apparaissent quasi-instantanément dans les scores de ses amis.

3. **Stratégie "User-First"** : 
   - Les utilisateurs et les posts sont indexés en priorité absolue au démarrage (< 10 secondes). 
   - L'index est sauvegardé partiellement dès que les membres sont traités, les rendant cherchables immédiatement.

## Fichiers Modifiés et Ajoutés

* **Frontend** :
  * `frontend/src/app/components/layout/Navbar.tsx` : Modification de l'icône de recherche.
  * `frontend/src/app/components/SearchModal.tsx` : Fenêtre modale avec système de filtres, gestion du `user_id` et affichage des badges de compatibilité (steluțe).
  * `frontend/src/app/lib/library.ts` & `frontend/src/app/lib/catalogue.ts` : Optimisation des requêtes et gestion des limites de débit (Rate Limiting).

* **Backend (AI Microservice)** :
  * `ai_service/build_index.py` : Logique d'indexation incrémentale par batchs, gestion du cache d'embeddings et priorité aux utilisateurs.
  * `ai_service/main.py` : API FastAPI mise à jour pour supporter la personnalisation par `user_id` et le calcul de similarité inter-utilisateurs.

## Instructions de Déploiement
Le microservice se déploie via Docker. 
Positionnez-vous à la racine et exécutez `docker compose up -d`. Grâce au nouveau système de cache, le système est opérationnel en quelques secondes si un index existe déjà. Au premier démarrage, les utilisateurs sont disponibles en < 1 minute, tandis que le catalogue complet se synchronise progressivement.

---
*Dernière mise à jour : 12 Mai 2026*

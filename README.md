Vous êtes ingénieur IA chez MIC (Marketing Intelligence Consulting), une entreprise de conseil spécialisée sur les problématiques de marketing digital.

Dans deux semaines, vous avez rendez-vous avec Mme Aline, directrice marketing de la compagnie aérienne “Air Paradis”.

 

Air Paradis a missionné votre cabinet pour créer un produit IA permettant d’anticiper les bad buzz sur les réseaux sociaux. Il est vrai que “Air Paradis” n’a pas toujours bonne presse sur les réseaux…

En sortant d’un rendez-vous de cadrage avec les équipes de Air Paradis, vous avez noté les éléments suivants :

Air Paradis veut un prototype d’un produit IA permettant de prédire le sentiment associé à un tweet.
Données : pas de données clients chez Air Paradis. Solution : utiliser des données Open Source (ou en téléchargement direct à ce lien)
Description des données : des informations sur les tweets (utilisateur ayant posté, contenu, moment du post) et un label binaire (tweet exprimant un sentiment négatif ou non). 
TO-DO :
Préparer un prototype fonctionnel du modèle. Le modèle est exposé via une API déployée sur le Cloud, appelée par une interface locale (notebook ou application Streamlit) qui envoie un tweet à l’API et récupère la prédiction de sentiment. 
Préparer un support de présentation explicitant les méthodologies utilisées pour les différentes approches (attention : audience non technique).
Après avoir reçu votre compte-rendu, Marc, votre manager, vous a contacté pour, selon ses mots, “faire d’une pierre deux coups”.

De : Marc

Envoyé : hier 17:14

À : vous 

Objet : Air Paradis : complément

Salut 

Merci pour ton récap du meeting avec Air Paradis. J’ai l’impression que ça s’est bien passé !

Je me disais… Puisque tu vas faire un proto pour ce client, j’ai l’intuition que ce produit pourrait se généraliser à d’autres cas d’usage. 

Tu voudrais bien en profiter pour tester plusieurs approches ?

approche “Modèle sur mesure simple”, pour développer rapidement un modèle classique (ex : régression logistique) permettant de prédire le sentiment associé à un tweet.
approche “Modèle sur mesure avancé” pour développer un modèle basé sur des réseaux de neurones profonds pour prédire le sentiment associé à un tweet. => C’est ce modèle que tu devras déployer et montrer à Air Paradis.

Pour cette 2ème approche, tu penseras bien à essayer au moins deux word embeddings différents et à garder celui qui permet d’obtenir les meilleures performances. En complément, pourrais-tu également regarder l’apport en performance d’un modèle BERT ? Cela nous permettra de voir si nous devons investir dans ce type de modèle.


Et en même ce serait top si tu pouvais mettre en oeuvre un bon exemple de démarche orientée MLOps, tu sais c’est la nouvelle priorité de notre directeur !

J’aimerais que tu puisses démontrer à l’occasion de l’élaboration de ton prototype tout l’apport du MLOps, afin d’assurer une diffusion aux autres équipes : 

d’abord réaliser une présentation synthétique des principes du MLOps et ses apports, 
ensuite utiliser l’outil MLFlow, future référence pour notre société, pour assurer la gestion des expérimentations des modèles : tracking et reporting de l’entraînement des modèles, centralisation du stockage des modèles, et test du serving proposé par MLFlow, 
mettre en œuvre un pipeline de déploiement continu du modèle que tu auras choisi via une API (Git + Github + plateforme Cloud au choix), qui intègre également des tests unitaires automatisés,
et enfin initier un suivi de la performance du modèle en production. Pour cela tu utiliseras un service Azure Application Insight que tu auras créé pour l‘occasion : 
Pour remonter des traces des tweets qui seraient considérés par l’utilisateur comme mal prédits : le texte du tweet et la prédiction.
Pour déclencher une alerte (envoi SMS ou mail) dans le cas d’un nombre trop important de tweet mal prédits (par exemple 3 tweets mal prédits en l’espace de 5 minutes).
Présenter une démarche qui pourrait être mise en oeuvre pour l’analyse de ces statistiques et l’amélioration du modèle dans le temps.

Nous souhaitons limiter les coûts de mise en production de ce prototype, donc peux-tu privilégier une solution gratuite Cloud pour le déploiement de l’API de prédiction, par exemple Azure webapp (ASP F1 gratuit), PythonAnywhere, Heroku avec le package “student” de Github ou tout autre solution ?


Si le modèle avancé est trop lourd et induit un dépassement des limites de taille des solutions gratuites, tu pourras tester le déploiement avec le modèle classique, ou bien utiliser des techniques de réduction de taille de ton modèle TensorFlow-Keras via une conversion en TensorFlow Lite.


Merci d’avance !

Marc

PS : Ah au fait, tant que tu y es, tu pourras rédiger un petit article pour le blog à partir de ton travail de modélisation et de ta démarche orientée MLOps ?
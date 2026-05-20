# 📢 07_communication.py — Data Storytelling & Communication

import os
import pandas as pd
import matplotlib.pyplot as plt

print("📢 Communication & Storytelling démarrés")

# =====================================================
# 1. CHEMINS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

evaluation_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "evaluation_predictions.csv"
)

report_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_report.txt"
)

# =====================================================
# 2. CHARGEMENT DES RÉSULTATS
# =====================================================

if not os.path.exists(evaluation_path):
    raise FileNotFoundError(
        f"❌ Résultats introuvables : {evaluation_path}"
    )

df = pd.read_csv(evaluation_path)

print(f"✔ Résultats chargés : {df.shape}")

# =====================================================
# 3. ANALYSE GLOBALE
# =====================================================

mean_real = df["Valeur_Reelle"].mean()
mean_pred = df["Valeur_Predite"].mean()

mean_error = df["Erreur"].mean()
abs_error = df["Erreur"].abs().mean()

print("\n📊 RÉSUMÉ GLOBAL")

print(f"Prix moyen réel     : {mean_real:,.2f} $")
print(f"Prix moyen prédit   : {mean_pred:,.2f} $")
print(f"Erreur moyenne      : {mean_error:,.2f}")
print(f"Erreur absolue moy. : {abs_error:,.2f}")

# =====================================================
# 4. INSIGHTS MÉTIER
# =====================================================

insights = []

insights.append(
    "La qualité globale des maisons influence fortement le prix de vente."
)

insights.append(
    "Les maisons possédant une grande surface habitable obtiennent des prix plus élevés."
)

insights.append(
    "Les maisons récentes ont tendance à être valorisées davantage."
)

insights.append(
    "Le modèle Random Forest atteint une excellente précision prédictive."
)

insights.append(
    "Les zones premium augmentent légèrement la valeur immobilière."
)

print("\n💡 INSIGHTS STRATÉGIQUES")

for i, insight in enumerate(insights, start=1):
    print(f"{i}. {insight}")

# =====================================================
# 5. RECOMMANDATIONS BUSINESS
# =====================================================

recommendations = []

recommendations.append(
    "Investir dans l'amélioration de la qualité intérieure des biens."
)

recommendations.append(
    "Privilégier les projets immobiliers dans les quartiers premium."
)

recommendations.append(
    "Mettre en avant la surface habitable dans les annonces immobilières."
)

recommendations.append(
    "Utiliser le modèle ML pour automatiser l'estimation des prix."
)

recommendations.append(
    "Développer un dashboard temps réel pour les agences immobilières."
)

print("\n📈 RECOMMANDATIONS")

for i, reco in enumerate(recommendations, start=1):
    print(f"{i}. {reco}")

# =====================================================
# 6. VISUALISATION STORYTELLING
# =====================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Valeur_Reelle"],
    df["Valeur_Predite"],
    alpha=0.5
)

plt.xlabel("Prix Réel")
plt.ylabel("Prix Prédit")

plt.title("Performance du Modèle : Réel vs Prédit")

min_val = min(
    df["Valeur_Reelle"].min(),
    df["Valeur_Predite"].min()
)

max_val = max(
    df["Valeur_Reelle"].max(),
    df["Valeur_Predite"].max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val]
)

plt.tight_layout()
plt.show()

# =====================================================
# 7. GÉNÉRATION RAPPORT TEXTE
# =====================================================

report_lines = []

report_lines.append("===== RAPPORT FINAL DATA SCIENCE =====\n")

report_lines.append("Projet : Prédiction des prix immobiliers\n")

report_lines.append(f"Prix moyen réel : {mean_real:,.2f} $\n")
report_lines.append(f"Prix moyen prédit : {mean_pred:,.2f} $\n")
report_lines.append(f"Erreur absolue moyenne : {abs_error:,.2f}\n")

report_lines.append("\n===== INSIGHTS =====\n")

for insight in insights:
    report_lines.append(f"- {insight}\n")

report_lines.append("\n===== RECOMMANDATIONS =====\n")

for reco in recommendations:
    report_lines.append(f"- {reco}\n")

with open(report_path, "w", encoding="utf-8") as f:
    f.writelines(report_lines)

print(f"\n💾 Rapport sauvegardé : {report_path}")

# =====================================================
# 8. CONCLUSION
# =====================================================

print("\n🎯 CONCLUSION")

print(
    "Le pipeline Data Science a permis de construire un modèle performant "
    "de prédiction des prix immobiliers à partir de variables structurelles "
    "et géographiques."
)

print(
    "Le modèle peut être utilisé comme outil d'aide à la décision "
    "pour les agences immobilières ou investisseurs."
)

print("\n🚀 Communication terminée avec succès")
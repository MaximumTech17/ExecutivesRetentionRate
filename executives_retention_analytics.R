# ==============================================================================
# PEOPLE ANALYTICS: COMMERCIAL EXECUTIVE RETENTION & FLIGHT-RISK MODELING
# Consolidated Analytical Pipeline (Original Academic Research in R)
# 
# Author: Gabriel Andrés Salazar Espinosa
# Academic Program: Specialized Program in Data Management for Decision-Making
# Institution: Universidad del Tolima (Nov 2024)
# Domain: Commercial Sales Workforce Analytics (Insurance & Savings)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SETUP & ENVIRONMENT INITIALIZATION
# ------------------------------------------------------------------------------

# Required libraries
required_packages <- c(
  "ggplot2", "dplyr", "FactoMineR", "factoextra", "corrplot",
  "rpart", "rpart.plot", "cluster", "randomForest", "fmsb", "knitr"
)

# Install missing packages if necessary
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed) {
    install.packages(pkg, dependencies = TRUE)
  }
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(FactoMineR)
  library(factoextra)
  library(corrplot)
  library(rpart)
  library(rpart.plot)
  library(cluster)
  library(randomForest)
  library(fmsb)
  library(knitr)
})

# Detect data paths (supports local root and data/ subfolder)
data_dir <- if (dir.exists("data")) "data" else "."
file_hist <- file.path(data_dir, "Ejecutivos_Fuga_Voluntaria_Involuntaria.csv")
file_act  <- file.path(data_dir, "Ejecutivos_Activos.csv")

# Load datasets
E <- read.csv(file_hist, header = TRUE, sep = ";", stringsAsFactors = FALSE)
act <- read.csv(file_act, header = TRUE, sep = ";", stringsAsFactors = FALSE)

cat(sprintf("Historical cohort loaded: %d rows, %d columns\n", nrow(E), ncol(E)))
cat(sprintf("Active workforce loaded:   %d rows, %d columns\n\n", nrow(act), ncol(act)))

# ------------------------------------------------------------------------------
# 2. DATA HYGIENE & PREPROCESSING
# ------------------------------------------------------------------------------

# Format categorical factors
E$Fuga <- as.factor(E$Fuga)
E$Negocio <- as.factor(E$Negocio)
act$Negocio <- as.factor(act$Negocio)

# Identify numerical variables
num_vars <- sapply(E, is.numeric)
num_cols <- names(E)[num_vars]

# Impute median for missing values in numerical columns
for (col in num_cols) {
  if (any(is.na(E[[col]]))) {
    med_val <- median(E[[col]], na.rm = TRUE)
    E[[col]][is.na(E[[col]])] <- med_val
  }
  if (col %in% names(act) && any(is.na(act[[col]]))) {
    med_val <- median(E[[col]], na.rm = TRUE)
    act[[col]][is.na(act[[col]])] <- med_val
  }
}

# Standardized dataset for unsupervised techniques
E_scaled <- E
E_scaled[, num_vars] <- scale(E[, num_vars])

# ------------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) & SECTOR ASYMMETRY
# ------------------------------------------------------------------------------

# 3.1 Overall Attrition Breakdown
cat("--- Overall Attrition Breakdown ---\n")
print(table(E$Fuga))
cat("\nProportions:\n")
print(prop.table(table(E$Fuga)) * 100)

# 3.2 Tenure Distribution by Attrition Type
p_tenure <- ggplot(E, aes(x = Fuga, y = MesesServicio, fill = Fuga)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 21) +
  geom_jitter(color = "black", alpha = 0.15, width = 0.2) +
  geom_hline(yintercept = 5, linetype = "dashed", color = "darkorange", linewidth = 1) +
  scale_fill_manual(values = c("Involuntaria" = "#EF4444", "Voluntaria" = "#3B82F6")) +
  labs(
    title = "Tenure Distribution by Attrition Type",
    subtitle = "Early voluntary departure hazard (<= 5 months) vs. performance dismissal",
    x = "Attrition Type",
    y = "Tenure (Months of Service)"
  ) +
  theme_minimal()

print(p_tenure)

# 3.3 Attrition Proportions by Business Line (Ahorro vs. Seguros)
df_negocio <- as.data.frame(table(E$Negocio, E$Fuga))
names(df_negocio) <- c("Negocio", "Fuga", "Count")
df_negocio <- df_negocio %>%
  group_by(Negocio) %>%
  mutate(Pct = Count / sum(Count) * 100)

p_sector <- ggplot(df_negocio, aes(x = Negocio, y = Pct, fill = Fuga)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.9, color = "white") +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", Pct, Count)), 
            position = position_dodge(width = 0.9), vjust = -0.3, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Involuntaria" = "#EF4444", "Voluntaria" = "#3B82F6")) +
  ylim(0, 85) +
  labs(
    title = "Turnover Asymmetry by Business Line",
    subtitle = "Ahorros: dominated by dismissals (68%) | Seguros: dominated by voluntary resignation (61%)",
    x = "Business Line",
    y = "Percentage within Sector (%)",
    fill = "Attrition Type"
  ) +
  theme_minimal()

print(p_sector)

# ------------------------------------------------------------------------------
# 4. DIMENSIONALITY REDUCTION: PCA & BIPLOT
# ------------------------------------------------------------------------------

# Filter numeric subset
datos_num <- na.omit(E[, num_vars])
fuga_filtrada <- E$Fuga[as.numeric(rownames(datos_num))]

# Fit PCA
pca_model <- PCA(datos_num, scale.unit = TRUE, ncp = 5, graph = FALSE)

cat("\n--- PCA Explained Variance ---\n")
print(pca_model$eig[1:5, ])

# Generate Biplot
p_biplot <- fviz_pca_biplot(
  pca_model,
  label = "var",
  habillage = fuga_filtrada,
  addEllipses = TRUE,
  palette = c("#EF4444", "#2563EB"),
  arrowsize = 0.8,
  labelsize = 3.5,
  repel = TRUE,
  col.var = "#0F172A",
  alpha.ind = 0.35,
  ggtheme = theme_minimal()
) +
  labs(
    title = "PCA Biplot: Commercial Executive Attrition Space",
    x = "Principal Component 1 (27.6% Variance - Commercial Output & Tenure)",
    y = "Principal Component 2 (14.4% Variance - Pay Structure Ratio)"
  )

print(p_biplot)

# ------------------------------------------------------------------------------
# 5. UNSUPERVISED CLUSTERING: K-MEANS & EXECUTIVE ARCHETYPES
# ------------------------------------------------------------------------------

# Reproducible sample (20% sample matching original Workshop 2 setup)
set.seed(123)
E_sample <- E %>% sample_frac(size = 0.2)
E_sample_scaled <- as.data.frame(scale(E_sample[, num_vars]))

# 5.1 Elbow Curve (Optimal K)
p_elbow <- fviz_nbclust(E_sample_scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = "dashed", color = "red") +
  labs(title = "Elbow Method for Optimal K (Inertia / WSS)", x = "Clusters (K)", y = "Total Within WSS") +
  theme_minimal()

print(p_elbow)

# 5.2 Fit K-Means with K = 3
km_res <- kmeans(E_sample_scaled, centers = 3, nstart = 25)
E_sample$Cluster <- as.factor(km_res$cluster)
E_sample_scaled$Cluster <- as.factor(km_res$cluster)

# 5.3 Silhouette Evaluation
sil_res <- silhouette(km_res$cluster, dist(E_sample_scaled[, num_vars]))
plot(sil_res, col = c("#EF4444", "#3B82F6", "#10B981"), main = "Silhouette Plot for K = 3 Archetypes")

# 5.4 Cluster Summary Table (Standardized KPI Centroids)
cluster_summary <- E_sample_scaled %>%
  group_by(Cluster) %>%
  summarise(across(c(CumpliProm, SalarioFijoProm, MesesServicio, NoSupervisores), mean))

cat("\n--- Standardized Cluster Centroid Profiles (Table 4) ---\n")
print(kable(cluster_summary, digits = 3))

# 5.5 Radar Chart
cluster_radar_data <- as.data.frame(cluster_summary[, -1])
radar_bound <- rbind(
  rep(1.2, ncol(cluster_radar_data)),   # Max bound
  rep(-1.2, ncol(cluster_radar_data)),  # Min bound
  cluster_radar_data
)

radarchart(
  radar_bound,
  axistype = 1,
  pcol = c("#EF4444", "#2563EB", "#10B981"),
  plwd = 2.5,
  title = "Executive Behavioral Archetypes (Radar Centroids)",
  cglcol = "#CBD5E1",
  cglty = 1,
  axislabcol = "black",
  caxislabels = seq(-1.2, 1.2, 0.6)
)
legend(
  "topright",
  legend = c("Cluster 1: Struggling Novices", "Cluster 2: Balanced Core", "Cluster 3: Star Veterans"),
  col = c("#EF4444", "#2563EB", "#10B981"),
  lty = 1,
  lwd = 2.5,
  bty = "n",
  cex = 0.8
)

# 5.6 Random Forest Cluster Importance
rf_cluster <- randomForest(Cluster ~ ., data = E_sample_scaled, importance = TRUE)
cat("\n--- Random Forest Cluster Importance ---\n")
varImpPlot(rf_cluster, main = "Key Features Defining Executive Archetypes")

# ------------------------------------------------------------------------------
# 6. SUPERVISED MODELING: LOGISTIC REGRESSION & DECISION TREE
# ------------------------------------------------------------------------------

# 6.1 Econometric Logistic Regression
logit_model <- glm(
  Fuga ~ SalarioFijoProm + CumpliProm + MesesServicio + NoSupervisores,
  family = binomial(link = "logit"),
  data = E
)
cat("\n--- Logistic Regression Summary ---\n")
print(summary(logit_model))

cat("\n--- Odds Ratios (e^Beta) ---\n")
print(exp(coef(logit_model)))

# 6.2 Decision Tree Classifier
variables_arbol <- c("Fuga", "MesesServicio", "SalarioTotalProm", "CumpliProm", "PropEjecuFugadosJefe", "Negocio")
datos_arbol <- E[, variables_arbol]

arbol <- rpart(
  Fuga ~ .,
  data = datos_arbol,
  minsplit = 50,
  minbucket = 20,
  cp = 0.01
)

cat("\n--- Decision Tree Rules ---\n")
print(arbol)

# Visualize Decision Tree
rpart.plot(
  arbol,
  type = 2,
  extra = 104,
  under = TRUE,
  box.palette = "RdBu",
  main = "Decision Tree: Attrition Classification Rules"
)

# ------------------------------------------------------------------------------
# 7. PRODUCTION SCORING: ACTIVE WORKFORCE FLIGHT RISK
# ------------------------------------------------------------------------------

# Predict voluntary flight probability on active workforce
pred_probs <- predict(arbol, newdata = act, type = "prob")
act$Prob_Voluntaria <- pred_probs[, "Voluntaria"]

# Distribution histogram
p_hist_act <- ggplot(act, aes(x = Prob_Voluntaria)) +
  geom_histogram(bins = 20, fill = "#38BDF8", color = "white", alpha = 0.9) +
  geom_vline(aes(xintercept = 0.85), linetype = "dashed", color = "#EF4444", linewidth = 1.2) +
  geom_vline(aes(xintercept = 0.90), linetype = "solid", color = "#B91C1C", linewidth = 1.5) +
  labs(
    title = "Predicted Voluntary Flight Risk Distribution (Active Workforce N = 554)",
    subtitle = "Red dashed: >85% threshold | Red solid: >90% critical flight risk",
    x = "Predicted Resignation Probability",
    y = "Active Executive Count"
  ) +
  theme_minimal()

print(p_hist_act)

# Summary counts of high-risk active executives
high_risk_85 <- sum(act$Prob_Voluntaria > 0.85)
high_risk_90 <- sum(act$Prob_Voluntaria > 0.90)

cat(sprintf("\nActive Executives with Flight Risk > 85%%: %d\n", high_risk_85))
cat(sprintf("Active Executives with Flight Risk > 90%%: %d\n\n", high_risk_90))

# 7.1 Top 10 High Flight-Risk Active Executives (Replicating Table 2)
top10_act <- act %>%
  arrange(desc(Prob_Voluntaria)) %>%
  head(10) %>%
  select(ID_Ejecutivo, Negocio, MesesServicio, CumpliProm, SalarioTotalProm, Prob_Voluntaria)

cat("--- Top 10 Active Executives with Highest Resignation Probability ---\n")
print(kable(top10_act, digits = 4, col.names = c("ID Ejecutivo", "Negocio", "Meses", "Cumplimiento", "Salario Total", "Prob. Fuga")))


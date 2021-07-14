library('TraMineR')
library('cluster')
library('factoextra')
library('NbClust')
library('caret')
library('RcmdrMisc')
library('ggplot2')

# Variables
case_name = 'TMall'
case_name = 'Cainiao'
preprocess_folderpath <- sprintf('%s/%s_preprocessed', case_name, case_name)
output_folderpath <- sprintf('%s/%s_output', case_name, case_name)
produce_date <- '20210715'
start_day <- 178
end_day <- 184
label <- 'none'
sent_day <- 10

# Main
# Assign case name
if (case_name == 'TMall') {
  file_name <- sprintf('%s_V3.2-duration_%s_%s-label_%s', produce_date, start_day, end_day, label)
  data <- read.csv(sprintf('%s/TMall_user_state_sequence_table_%s.csv', preprocess_folderpath, file_name))
  # Compare with traditional distance vector for clustering input
  action_counts <- read.csv(sprintf('%s/action_counts-%s.csv', preprocess_folderpath, file_name))
  # euclidean_distance <- read.csv(sprintf('%s/euclidean_distance-%s.csv', preprocess_folderpath, file_name))
  # cosine_similarity <- read.csv(sprintf('%s/cosine_similarity-%s.csv', preprocess_folderpath, file_name))
  colorset <- c("gray", "forestgreen", "darkorange3", "green", "gold", "white")
} else {
  file_name <- sprintf('%s-sentday_%s', produce_date, sent_day)
  data <- read.csv(sprintf('%s/order_logistic_states-%s.csv', preprocess_folderpath, file_name))
  colorset <- c("white","aquamarine3","azure1","azure2","dodgerblue1","aquamarine1","azure3","aquamarine4","green","green3")
}

# Sequence format
data.seq <- seqdef(data, 2:8, xtstep=6)

# Transition rates between states
data.trate <- round(seqtrate(data.seq, time.varying = FALSE), 2)
View(data.trate)
write.csv(data.trate, sprintf('%s/transition_rate_%s.csv', preprocess_folderpath, file_name))

# Substitution Cost & Indel Cost
data.seq.cost <- seqcost(data.seq, method='TRATE')
View(data.seq.cost$sm)
data.seq.cost$indel
write.csv(data.seq.cost$sm, sprintf('%s/substitution_cost_matrix_%s.csv', preprocess_folderpath, file_name))


# Optimal Matching: computing distances between sequences
# OM
data.om <- seqdist(data.seq, method = 'OM', sm = 'TRATE', indel = 'auto')

# OMloc
data.om <- seqdist(data.seq, method = 'OMloc', sm = 'TRATE', indel = 'auto')

# OMslen
data.om <- seqdist(data.seq, method = 'OMslen', sm = 'TRATE', indel = 'auto')

# OMspell
data.om <- seqdist(data.seq, method = 'OMspell', sm = 'TRATE', indel = "auto")

# OMstran
data.om <- seqdist(data.seq, method = 'OMstran', sm = 'TRATE', indel = "auto", otto = 0.5, with.missing = TRUE)

# CHI
data.om <- seqdist(data.seq, method = 'CHI2')

# SVR
data.om <- seqdist(data.seq, method = 'SVRspell')

# NMS
data.om <- seqdist(data.seq, method = 'NMS')
data.om <- seqdist(data.seq, method = 'NMSMST')

# TWED
data.om <- seqdist(data.seq, method = 'TWED', nu = 0.5, sm = 'INDELSLOG')

# output - dissimilarity matrix
write.csv(data.om, 
          sprintf('%s/dissimilarity_matrix-%s-seqdist_%s-sm_%s-indel_%s.csv', 
                  preprocess_folderpath, file_name, 'OMstran', 'TRATE', 'auto'), 
          row.names = FALSE)


# Find optimal number of clusters, method = "silhouette", "wss"
fviz_nbclust(data.om, FUNcluster = hcut, method = "silhouette", k.max = 8, print.summary = TRUE)

# Clustering: Hierarchical Clustering
# (1) With origin data
clusterward <- agnes(action_counts, diss = FALSE, method = 'ward')
# (2) With dissimilarity matrix
clusterward <- agnes(data.om, diss = TRUE, method = 'ward')
clusterward <- agnes(euclidean_distance, diss = TRUE, method = 'ward')

cluster.result <- cutree(clusterward, k=8)
cluster.label.hcut <- factor(cluster.result, labels = paste("Cluster"), 1:8)

# Clustering: K-means
cluster.result <- kmeans(data.om, centers = 2)$cluster
cluster.label.kmeans <- factor(cluster.result, labels = paste("Cluster"), 1:2)

# output - cluster labels add back to user state sequences data
data$hcut_cluster <- cluster.label.hcut
data$kmeans_cluster <- cluster.label.kmeans
action_counts$hcut_cluster <- cluster.label.hcut
action_counts$kmeans_cluster <- cluster.label.kmeans
euclidean_distance$hcut_cluster <- cluster.label.hcut
euclidean_distance$kmeans_cluster <- cluster.label.kmeans
write.csv(data, 
          sprintf('%s/clustered-%s-seqdist_%s-sm_%s-indel_%s-method_%s-center_%s.csv',
                  output_folderpath, file_name, 'OMstran', 'TRATE', 'auto', 'hcut', '8'), 
          row.names = FALSE)
write.csv(action_counts, 
          sprintf('%s/clustered-%s-seqdist_%s-method_%s-center_%s.csv',
                  output_folderpath, file_name, 'action_counts', 'hcut', '8'), 
          row.names = FALSE)
write.csv(euclidean_distance, 
          sprintf('%s/clustered-%s-seqdist_%s-method_%s-center_%s.csv',
                  output_folderpath, file_name, 'euclidean_distance', 'hcut', '8'), 
          row.names = FALSE)

# Plots
plot_name <- sprintf('%s/cluster_distribution-%s-seqdist_%s-sm_%s-indel_%s-method_%s-center_%s', 
                     output_folderpath, file_name, 'OMstran', 'TRATE', 'auto', 'hcut', '8')
plot_name

# State distribution plot
seqdplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)
seqfplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)
seqiplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)
seqIplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)
seqHtplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)  # transversal entropy from seqstatd
seqmsplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)  # modal state sequence from seqmodst
seqmtplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)  # mean time spent from seqmeant
seqrplot(data.seq, group = cluster.label.hcut, border=NA, cpal=colorset)  # representative sequence
seqpcplot(data.seq, group = cluster.label.hcut, border=NA, order.align = "time", cpal=colorset)  # decorated parallel coordinate, sequences are displayed as jittered frequency-weighted parallel lines
seqiplot(data.seq, border=NA, withlegend='right', cpal=colorset)


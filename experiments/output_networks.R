INPUT_RDA_PATH <- "input.rda"

e <- new.env(parent = emptyenv())
load(INPUT_RDA_PATH, envir=e)

corr <- e$corr
genes <- e$genes
donors <- e$donors

row.names(donors) <- NULL

for (d in unique(donors$Donor)) {
  for (c in unique(donors$Cell_type)) {
    donors_subset <- subset(subset(donors, Donor==d), Cell_type==c)
    time_points <- donors_subset$time
    for (ridx in rownames(donors_subset)){
      i <- strtoi(ridx)
      M <- matrix(1, nrow=nrow(genes), ncol=nrow(genes));
      M[upper.tri(M)] = corr[,i]
      M[lower.tri(M)] = corr[,i]
      t <- donors_subset[ridx,]$time
      filename <- paste(paste("corr", c, d, t, sep="-"), ".csv", sep="")
      write.csv(M, file=filename, row.names=FALSE, col.names=FALSE)
    }
  }
}

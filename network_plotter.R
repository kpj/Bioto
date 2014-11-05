library(network)
library(ggplot2)
library(sna)
library(ergm)


plotg <- function(net, fname, value=NULL) {
  m <- as.matrix.network.adjacency(net) # get sociomatrix

  plotcord <- data.frame(gplot.layout.fruchtermanreingold(m, NULL)) # Fruchterman and Reingold's force-directed placement algorithm
  #plotcord <- data.frame(gplot.layout.kamadakawai(m, NULL)) # Kamada-Kawai's algorithm

  colnames(plotcord) = c("X1","X2")
  edglist <- as.matrix.network.edgelist(net)
  edges <- data.frame(plotcord[edglist[,1],], plotcord[edglist[,2],])
  plotcord$elements <- as.factor(get.vertex.attribute(net, "elements"))
  colnames(edges) <- c("X1", "Y1", "X2", "Y2")
  edges$midX <- (edges$X1 + edges$X2) / 2
  edges$midY <- (edges$Y1 + edges$Y2) / 2

  pnet <- ggplot() +
    geom_segment(aes(x=X1, y=Y1, xend=X2, yend=Y2), data=edges, size=0.5, colour="grey") +
    geom_point(aes(x=X1, y=X2, col=elements, fill=5), data=plotcord, position=position_jitterdodge(dodge.width=3)) +
    geom_text(aes(x=X1, y=X2, label=elements), data=plotcord) +
    scale_x_continuous(breaks=NULL) +
    scale_y_continuous(breaks=NULL) +
    theme(legend.position="NONE") +
    ggsave(filename=fname, scale=5)
  return(print(pnet))
}


graph_file <- "adja_m.txt"
names_file <- "node_names.txt"

m <- read.csv(graph_file, header=FALSE, sep=" ")
g <- network(m)

n <- as.character(read.table(names_file)[[1]])
set.vertex.attribute(g, "elements", n)

fname <- commandArgs(trailingOnly=TRUE)
print(paste("Saving to", fname))

plotg(g, fname)

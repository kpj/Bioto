library(ggplot2)
library(reshape2)
library(grid)


foutname <- commandArgs(trailingOnly=TRUE)
print(paste("Saving to", foutname))

data = read.csv("geo_org_distr.csv", header=FALSE)
colnames(data) <- c("organism", "count")

data$bar_label_alpha = ifelse(grepl("coli", data$organism), 1, 0.3)

p <- ggplot(data, aes(x=organism, y=count, fill=factor(organism))) +
  geom_bar(stat="identity") +
  geom_text(aes(label=count, vjust=-0.25), size=4, alpha=data$bar_label_alpha) +
  scale_y_log10(expand=c(0, 0.03), limits=c(1, 2500)) +
  theme(
    legend.position="NONE",
    axis.text.x=element_text(angle=90, hjust=1, vjust=0.35, face=ifelse(grepl("coli", data$organism),"bold","plain")),
    panel.background=element_blank(),
    axis.ticks.margin=unit(1, "points")
  )
print(p)

if(length(foutname) != 0) {
  ggsave(p, filename=foutname)
}

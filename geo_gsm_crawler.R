library(GEOquery)


# helper functions
extract_sample_id <- function(gsmids) {
  return(strsplit(gsmids[[1]], ",")[[1]][[1]])
}

handle_gds <- function(df, gds_name) {
  gds <- getGEO(gds_name)
  gsm_name <- extract_sample_id(Meta(gds)$sample_id)
  gsm <- getGEO(gsm_name)
  
  val_typ <- Meta(gds)$value_type
  desc <- Meta(gsm)$description
  datpro <- ifelse(!is.null(Meta(gsm)$data_processing), Meta(gsm)$data_processing, "undef")
  
  df <- rbind(
    df,
    data.frame(
      gds=gds_name,
      gsm=gsm_name,
      value_type=val_typ,
      #description=desc,
      data_processing=datpro
    )
  )
  return(df)
}


# generate data
df <- data.frame(
  gds=character(0),
  gsm=character(0),
  value_type=character(0),
  #description=character(0),
  data_processing=character(0)
)

gds_files <- list.files("../data/concentrations")
for(gds_name in gds_files) {
  gds <- sub(".soft", "", gds_name)
  print(paste("Handling", gds))
  df <- handle_gds(df, gds)
}


# save data
write.table(df, file="gds_summary.txt")

sink(file="foo.txt")
print.data.frame(df)
sink()
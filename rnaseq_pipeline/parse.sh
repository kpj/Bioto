# Required software: yaourt -S tophat bowtie2 samtools python2-htseq

set -u
set -e


if [ $# -ne 1 ] ; then
    echo "Usage: $(basename $0) <sra file>"
    exit 1
fi

sra_file="$1"
tmp_dir="./tmp"

cd "$( dirname "${BASH_SOURCE[0]}" )"
PATH="$PATH:./bin"


# sra parsing
echo " > Convert sra to fastq"
fastq_file="${sra_file/%.sra/.fastq}"
fastq_dump "$sra_file" -O "$tmp_dir"

# bowtie2 genome mapping
genome_dir="$tmp_dir/genome"
if [ ! -d "$genome_dir" ] ; then
    echo " > Index genome"
    mkdir -p "$genome_dir"
    (
        cd "$genome_dir"
        bowtie2-build "../../data/ecoli_genome.fa" "ecoli_genome"
    )
fi
echo " > Map fastq to genome"
tophat_out="$tmp_dir/${sra_file%%.sra}_tophat_out"
tophat2 --num-threads 8 --output-dir "$tophat_out" "$genome_dir/ecoli_genome" "$tmp_dir/$fastq_file"
bam_file="$tophat_out/accepted_hits.bam"

# count reads
echo " > Convert bam to sam"
sam_file="${sra_file/%.sra/.sam}"
samtools view "$bam_file" > "$tmp_dir/$sam_file"

echo " > Fix chromosome labels"
sed -i 's/NC_000913.3/Chromosome/g' "$tmp_dir/$sam_file"

results_dir="results"
if [ ! -d "$results_dir" ] ; then
    mkdir -p "$results_dir"
fi

echo " > Parse sam file"
count_file="${sra_file/%.sra/.count}"
htseq-count --stranded=no "$tmp_dir/$sam_file" "data/ecoli_genome_annotations.gtf" > "$results_dir/$count_file"

echo " > Map gene ids to gene names"
mapped_count_file="${sra_file/%.sra/_mapped.count}"
python "bin/id_mapper.py" "$results_dir/$count_file" > "$results_dir/$mapped_count_file"

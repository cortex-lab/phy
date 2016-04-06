OUTPUT_DIR = $HOME/phy
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $OUTPUT_DIR
export PATH="$OUTPUT_DIR/bin:$PATH"
conda env create -n phy --force

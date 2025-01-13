#!/bin/bash
set -e

#--------------------------------------
: ${PROJ_ID:=""}
: ${QUEUE:="lustre_scaling"}
: ${NEKBONE_INSTALL_DIR:=./install}

### Don't touch anything that follows this line. ###
if [ $# -ne 4 ]; then
  echo "Usage: [PROJ_ID=value] $0 <backend> <order> <max_iter> <hh:mm:ss>"
  exit 1
fi

if [ -z "$PROJ_ID" ]; then
  echo "ERROR: PROJ_ID is empty"
  exit 1
fi

if [ -z "$QUEUE" ]; then
  echo "ERROR: QUEUE is empty"
  exit 1
fi

if [ -z "$NEKBONE_INSTALL_DIR" ]; then
  echo "ERROR: NEKBONE_INSTALL_DIR is empty"
  exit 1
fi

bin=${NEKBONE_INSTALL_DIR}/bin/nekbone-driver
if [ ! -f $bin ]; then
  echo "ERROR: Cannot find ${bin}"
  exit 1
fi

backend=$1
order=$2
max_iter=$3
time=$4

gpus_per_node=6
tiles_per_gpu=2

chk_case $TOTAL_RANKS

#--------------------------------------
# Generate the submission script
SFILE=s.bin
echo "#!/bin/bash" > $SFILE
echo "#PBS -A $PROJ_ID" >>$SFILE
echo "#PBS -N nekbone_${backend}" >>$SFILE
echo "#PBS -l walltime=${time}:00" >>$SFILE
echo "#PBS -l select=$qnodes" >>$SFILE
echo "#PBS -l place=scatter" >>$SFILE
echo "#PBS -k doe" >>$SFILE
echo "#PBS -j oe" >>$SFILE

echo "export TZ='/usr/share/zoneinfo/US/Central'" >> $SFILE

# job to "run" from your submission directory
echo "cd \$PBS_O_WORKDIR" >> $SFILE

echo "echo Jobid: \$PBS_JOBID" >>$SFILE
echo "echo Running on host \`hostname\`" >>$SFILE
echo "echo Running on nodes \`cat \$PBS_NODEFILE\`" >>$SFILE
echo "module list" >> $SFILE

CMD=.lhelper
echo "#!/bin/bash" > $CMD
echo "gpu_id=\$(((PALS_LOCAL_RANKID / ${tiles_per_gpu}) % ${gpus_per_node}))" >> $CMD
echo "tile_id=\$((PALS_LOCAL_RANKID % ${tiles_per_gpu}))" >> $CMD
echo "export ZE_AFFINITY_MASK=\$gpu_id.\$tile_id" >> $CMD
echo "\"\$@\"" >> $CMD
chmod u+x $CMD

for element in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
  echo "mpiexec --no-vni -n 1 -ppn 1 -- ./${CMD} $bin --nekbone-backend=${backend} " \
    "--nekbone-max-iter=${max_iter} --nekbone-order ${order} --nekbone-verbose=1 " \
    "--nekbone-nelems $element" >>$SFILE
  echo "sleep 5" >>$SFILE
done

qsub -q $QUEUE $SFILE

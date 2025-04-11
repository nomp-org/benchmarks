#!/bin/bash
set -e

: ${PROJ_ID:="Performance"}
: ${QUEUE:="debug"}
: ${NEKBONE_INSTALL_DIR:=./install}
: ${NOMP_INSTALL_DIR:-$HOME/.nomp}

### Don't touch anything that follows this line. ###
if [[ $# -gt 6 ]]; then
  echo "Usage: [PROJ_ID=value] $0 <backend> <order> <profile=0/1> <num_trials> <max_iter> <hh:mm>"
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

backend=${1:-"sycl"}
order=${2:-7}
profile=${3:-0}
num_trials=${4:-5}
max_iter=${5:-500}
time=${6:-1:00}

gpus_per_node=6
tiles_per_gpu=2
qnodes=1
backend_=${backend/:/_}

#--------------------------------------
# Generate the submission script
SFILE=s.bin
echo "#!/bin/bash" > $SFILE
echo "#PBS -A $PROJ_ID" >>$SFILE
echo "#PBS -q $QUEUE" >>$SFILE
echo "#PBS -N nekbone_${backend_}" >>$SFILE
#echo "#PBS -l filesystems=home" >>$SFILE
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
echo "sycl-ls" >> $SFILE
echo "ulimit -s unlimited " >>$SFILE

if [ $profile -eq 1 ]; then
  echo "module load thapi" >> $SFILE
fi
echo "module list" >> $SFILE

# Nomp flags in case we are running nomp
echo "export NOMP_INSTALL_DIR=$NOMP_INSTALL_DIR" >>$SFILE

# OCCA flags in case we are running OCCA
echo "export OCCA_DPCPP_COMPILER_FLAGS=\"-fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma\"" >>$SFILE

CMD=.lhelper
echo "#!/bin/bash" > $CMD
echo "gpu_id=\$(((PALS_LOCAL_RANKID / ${tiles_per_gpu}) % ${gpus_per_node}))" >> $CMD
echo "tile_id=\$((PALS_LOCAL_RANKID % ${tiles_per_gpu}))" >> $CMD
echo "export ZE_AFFINITY_MASK=\$gpu_id.\$tile_id" >> $CMD
echo "\"\$@\"" >> $CMD
chmod u+x $CMD

DBGCMD=
if [ $profile -eq 1 ]; then
  DBGCMD="iprof --"
fi

for element in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do
for ((i=0; i<num_trials; i++)); do
  echo "mpiexec --no-vni -n 1 -ppn 1 -- ./${CMD} ${DBGCMD} $bin --nekbone-backend=${backend} " \
    "--nekbone-max-iter=${max_iter} --nekbone-order ${order} --nekbone-verbose=1 " \
    "--nekbone-scripts-dir=${NEKBONE_INSTALL_DIR}/scripts --nekbone-nelems $element" >>$SFILE
  echo "sleep 5" >>$SFILE
done
done

qsub $SFILE

#!/bin/bash
set -e

: ${PROJ_ID:="Performance"}
: ${QUEUE:="debug"}
: ${NEKBONE_INSTALL_DIR:=./install}
: ${SPACK_DIR:=${HOME}/workspace/anl/thapi/spack}

### Don't touch anything that follows this line. ###
if [[ $# -lt 1 && $# -gt 4 ]]; then
  echo "Usage: [PROJ_ID=value] $0 <backend> <order> <max_iter> <hh:mm>"
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

backend=${1:-"cuda"}
order=${2:-7}
max_iter=${3:-500}
time=${4:-1:00}
num_trials=${5:-5}

profile=1
gpus_per_node=4
tiles_per_gpu=1
qnodes=1
backend_=${backend/:/_}

#--------------------------------------
# Generate the submission script
# sbatch
SFILE=s.bin
echo "#!/bin/bash" > $SFILE
echo "#PBS -A $PROJ_ID" >>$SFILE
echo "#PBS -q $QUEUE" >>$SFILE
echo "#PBS -N nekbone_${backend_}" >>$SFILE
echo "#PBS -l walltime=${time}:00" >>$SFILE
echo "#PBS -l filesystems=home:eagle:grand" >>$SFILE
echo "#PBS -l select=$qnodes:system=polaris" >>$SFILE
echo "#PBS -l place=scatter" >>$SFILE
echo "#PBS -k doe" >>$SFILE
echo "#PBS -j eo" >>$SFILE

echo "export TZ='/usr/share/zoneinfo/US/Central'" >> $SFILE

# job to "run" from your submission directory
echo "cd \$PBS_O_WORKDIR" >> $SFILE

echo "echo Jobid: \$PBS_JOBID" >>$SFILE
echo "echo Running on host \`hostname\`" >>$SFILE
echo "echo Running on nodes \`cat \$PBS_NODEFILE\`" >>$SFILE
echo "nvidia-smi" >> $SFILE
echo "ulimit -s unlimited " >>$SFILE

echo "module use /soft/modulefiles" >> $SFILE
echo "module use /opt/cray/pe/lmod/modulefiles/mix_compilers" >> $SFILE
echo "module load libfabric" >> $SFILE
echo "module load PrgEnv-gnu" >> $SFILE
echo "module load nvhpc-mixed" >> $SFILE
echo "module load craype-x86-milan craype-accel-nvidia80" >> $SFILE
echo "module load spack-pe-base cmake" >> $SFILE
if [ $profile -eq 1 ]; then
  echo ". ${SPACK_DIR}/share/spack/setup-env.sh" >> $SFILE
  echo "spack load thapi" >> $SFILE
fi
echo "module list" >> $SFILE

# OCCA flags in case we are running OCCA
echo "export OCCA_CUDA_COMPILER_FLAGS=\"-w -O3 -lineinfo --use_fast_math\"" >>$SFILE
echo "export OCCA_VERBOSE=1" >>$SFILE

CMD=.lhelper
echo "#!/bin/bash" >$CMD
echo "gpu_id=\$((${gpus_per_node} - 1 - \${PMI_LOCAL_RANK} % ${gpus_per_node}))" >>$CMD
echo "export CUDA_VISIBLE_DEVICES=\$gpu_id" >>$CMD
echo "\$*" >>$CMD
chmod u+x $CMD

DBGCMD=
if [ $profile -eq 1 ]; then
  DBGCMD="iprof --"
fi

for element in 512 1024 2048 4096 8192 16384 32768 65536; do
for ((i=0; i<num_trials; i++)); do
  echo "mpiexec --no-vni -n 1 -ppn 1 -- ./${CMD} ${DBGCMD} $bin --nekbone-backend=${backend} " \
    "--nekbone-max-iter=${max_iter} --nekbone-order ${order} --nekbone-verbose=1 " \
    "--nekbone-scripts-dir=${NEKBONE_INSTALL_DIR}/scripts --nekbone-nelems $element" >>$SFILE
  echo "sleep 5" >>$SFILE
done
done

qsub $SFILE

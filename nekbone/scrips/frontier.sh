#!/bin/bash

: ${PROJ_ID:="CSC262"}
: ${NEKBONE_INSTALL_DIR:=./install}
: ${QUEUE:="batch"}

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
time_fmt=`echo $time|tr ":" " "|awk '{print NF}'`
if [ "$time_fmt" -ne "3" ]; then
  echo "Error: time is not in the format <hh:mm:ss>, got: ${time}"
  exit 1
fi

cpus_per_task=8

# sbatch
SFILE=s.bin
echo "#!/bin/bash" >$SFILE
echo "#SBATCH -A $PROJ_ID" >>$SFILE
echo "#SBATCH -J nekbone_${backend}" >>$SFILE
echo "#SBATCH -o %x-%j.out" >>$SFILE
echo "#SBATCH -t $time" >>$SFILE
echo "#SBATCH -N 1" >>$SFILE
echo "#SBATCH -p $QUEUE" >>$SFILE
echo "#SBATCH -C nvme" >>$SFILE
echo "#SBATCH --exclusive" >>$SFILE
echo "#SBATCH --ntasks-per-node=1" >>$SFILE
echo "#SBATCH --gpus-per-task=1" >>$SFILE
echo "#SBATCH --gpu-bind=closest" >>$SFILE
echo "#SBATCH --cpus-per-task=$cpus_per_task" >>$SFILE
echo "" >>$SFILE

echo "module load PrgEnv-gnu" >>$SFILE
echo "module load craype-accel-amd-gfx90a" >>$SFILE
echo "module load cray-mpich" >>$SFILE
echo "module load rocm" >>$SFILE
echo "module unload cray-libsci" >>$SFILE
echo "module list" >>$SFILE
echo "" >>$SFILE

echo "rocm-smi" >>$SFILE
echo "rocm-smi --showpids" >>$SFILE
echo "" >>$SFILE

echo "# which nodes am I running on?" >>$SFILE
echo "squeue -u \$USER" >>$SFILE
echo "" >>$SFILE

echo "export MPICH_GPU_SUPPORT_ENABLED=0" >>$SFILE
echo "export MPICH_OFI_NIC_POLICY=NUMA" >>$SFILE
echo "" >>$SFILE

echo "ulimit -s unlimited " >>$SFILE
echo "" >>$SFILE

echo "date" >>$SFILE
echo "" >>$SFILE

for element in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
  echo "srun -N 1 -n 1 $bin --nekbone-backend=${backend} --nekbone-max-iter=${max_iter} " \
    "--nekbone-order ${order} --nekbone-verbose=1 --nekbone-nelems $element" >>$SFILE
  echo "sleep 5" >>$SFILE
done

sbatch $SFILE

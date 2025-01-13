#!/bin/bash

backend=$1
elements=${2:-1024}
order=${3:-7}
max_iter=${4:-100}
scripts_dir=${5:-`pwd`/install/scripts}

./install/bin/nekbone-driver --nekbone-backend=${backend} \
  --nekbone-max-iter=${max_iter} \
  --nekbone-order ${order} \
  --nekbone-nelems $elements \
  --nekbone-verbose=1 \
  --nekbone-scripts-dir="${scripts_dir}"

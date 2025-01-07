#!/bin/bash

backend=$1
elements=${2:-1024}
order=${3:-7}
max_iter=${4:-100}

./install/bin/nekbone-driver --nekbone-backend=${backend} \
  --nekbone-max-iter=${max_iter} \
  --nekbone-order ${order} \
  --nekbone-nelems $elements \
  --nekbone-verbose=1 

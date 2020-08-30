#!/bin/bash

SPATIAL_READ="553467,1055260,0.524484"
SPATIAL_WRITE="161168,1055260,0.152728"
TEMPORAL_READ="133342,1055260,0.126359"
TEMPORAL_WRITE="0,1055260,0"

cd samples/bfs
make clean
make
bash run_sampling.sh

S=`tail -n 1 spatial_read_t0.csv`
if [ "$S" != "$SPATIAL_READ" ]; then
  echo "SPATIAL_READ "$S" vs "$SPATIAL_READ
fi

S=`tail -n 1 spatial_write_t0.csv`
if [ "$S" != "$SPATIAL_WRITE" ]; then
  echo "SPATIAL_WRITE "$S" vs "$SPATIAL_WRITE
fi

S=`tail -n 1 temporal_read_t0.csv`
if [ "$S" != "$TEMPORAL_READ" ]; then
  echo "TEMPORAL_READ "$S" vs "$TEMPORAL_READ
fi

S=`tail -n 1 temporal_write_t0.csv`
if [ "$S" != "$TEMPORAL_WRITE" ]; then
  echo "TEMPORAL_WRITE "$S" vs "$TEMPORAL_WRITE
fi

cd ../..

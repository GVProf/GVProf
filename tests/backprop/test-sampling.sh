#!/bin/bash

SPATIAL_READ="84039,400160,0.210013"
SPATIAL_WRITE="21009,400160,0.0525015"
TEMPORAL_READ="63058,400160,0.157582"
TEMPORAL_WRITE="0,400160,0"

cd samples/backprop
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

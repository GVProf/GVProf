#!/bin/bash

SPATIAL_READ="4194507,19988592,0.209845"
SPATIAL_WRITE="1048623,19988592,0.0524611"
TEMPORAL_READ="3149872,19988592,0.157583"
TEMPORAL_WRITE="0,19988592,0"

cd samples/backprop
make clean
make
bash ./run.sh

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

GVPROF_source_path=/root/GVProf
rodinia_path=/root/GVProf/samples
target_log_file=/root/GVProf/overhead.txt
iteration_num=1

cd $rodinia_path

run() {
    cur_path=$1
    block_sampling=$2
    kernel_sampling=$3
    EXEC_AND_PARAMS=${@:4}
    echo ${EXEC_AND_PARAMS}
    cd ${cur_path}
    echo ${cur_path} >>  ${target_log_file}
    rm -rf time*.txt gvprof*
    for i in {1..${iteration_num}}; do
        { time $EXEC_AND_PARAMS; } 2>>time.txt
    done

    gvprof_overhead -i ${iteration_num} -v -e data_flow -ck HPCRUN_SANITIZER_READ_TRACE_IGNORE=1 -ck HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=131072 -ck HPCRUN_SANITIZER_GPU_ANALYSIS_BLOCKS=1 $EXEC_AND_PARAMS

    rm -rf gvprof*

    gvprof_overhead -i ${iteration_num} -v -e value_pattern -s ${block_sampling} -ck HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=${kernel_sampling} -ck HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=131072 -ck HPCRUN_SANITIZER_WHITELIST=w.txt $EXEC_AND_PARAMS
    
    /usr/bin/python3 ${GVPROF_source_path}/python/filter_time.py >> ${target_log_file}
}

run $rodinia_path/bfs 20 20 ./bfs ../data/graph1MW_6.txt
run $rodinia_path/backprop 20 20 ./backprop 65536
run $rodinia_path/srad_v1 20 20 ./srad 1 0.5 502 458
run $rodinia_path/hotspot 20 20 ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out
run $rodinia_path/pathfinder 20 20 ./pathfinder 100000 100 20
run $rodinia_path/cfd 20 20 ./euler3d ../data/fvcorr.domn.097K
run $rodinia_path/huffman 20 20 ./pavle ../data/test1024_H2.206587175259.in
run $rodinia_path/lavaMD 100 100 ./lavaMD -boxes1d 10
run $rodinia_path/hotspot3D 20 20 ./3D 512 8 100 ../data/power_512x8 ../data/temp_512x8 output.out
run $rodinia_path/streamcluster 100 100 ./sc_gpu 10 20 256 65536 65536 1000 none output.txt 1

# # real applications
# #darknet
# run /root/gpuapps/darknet ./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights data/dog.jpg -i 0 -thresh 0.25
# # castro
# run /root/gpuapps/Castro/Exec/hydro_tests/Sedov ./Castro2d.gnu.CUDA.ex inputs.2d.cyl_in_cartcoords
# # barracuda
# run /root/gpuapps/barracuda ./bin/barracuda aln sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa sample_data/sample_reads.fastq >quicktest.sai
# # pytorch
# eval "$(/root/anaconda3/bin/conda shell.bash hook)"
# conda activate pytorch

# cd /root/gpuapps/pytorch_vp/pytorch
# rm -rf gvprof*
# rm w.txt
# ln -s w_resnet.txt w.txt
# run /root/gpuapps/pytorch_vp/pytorch python3 1-resnet50-unit.py
# rm w.txt
# ln -s w_deepwave.txt w.txt
# run /root/gpuapps/pytorch_vp/pytorch python3 2-deepwave-unit.py
# rm w.txt
# ln -s w_bert.txt w.txt
# run /root/gpuapps/pytorch_vp/pytorch python3 3-bert-unit.py

# conda deactivate

# # namd
# run /root/gpuapps/NAMD/Linux-x86_64-g++ ./namd3 ../src/alanin

# # qmcpack

# #lammps
# run /root/gpuapps/lammps/bench ../build/lmp -k on g 1 -sf kk -in in.lj

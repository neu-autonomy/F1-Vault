for i in 0 1 2 5 10 20; do
    python play_h5_file.py --file_path dynamics_data_0000.h5 --env_id $i --mode static
    mv trajectory_plot.png env_${i}.png
done